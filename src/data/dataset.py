import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class FederatedPPMIDataset(Dataset):
    """
    Multimodal PyTorch Dataset for Fed-PhenoGraft.
    Returns features and explicit mask flags for missing modalities.
    Accepts DataFrames or Series with any index — alignment is done by position.
    """
    def __init__(self, clinical_df, mri_df, pet_df, genetic_df, targets,
                 diagnosis=None, client_id=None):
        self.client_id = client_id
        
        # Convert to numpy arrays, aligning by position (not index)
        self.clinical = np.asarray(clinical_df.values if hasattr(clinical_df, 'values') else clinical_df, dtype=np.float32)
        self.mri = np.asarray(mri_df.values if hasattr(mri_df, 'values') else mri_df, dtype=np.float32)
        self.pet = np.asarray(pet_df.values if hasattr(pet_df, 'values') else pet_df, dtype=np.float32)
        self.genetic = np.asarray(genetic_df.values if hasattr(genetic_df, 'values') else genetic_df, dtype=np.float32)
        
        # Handle targets: could be Series, DataFrame, or ndarray
        if isinstance(targets, pd.DataFrame):
            self.targets = targets.values.astype(np.float32)
        elif isinstance(targets, pd.Series):
            self.targets = targets.values.astype(np.float32).reshape(-1, 1)
        else:
            self.targets = np.asarray(targets, dtype=np.float32).reshape(-1, 1)
        
        # Diagnosis label (PD=1 / HC=0) for the classification head.
        # Optional: defaults to zeros when not provided (regression-only use).
        if diagnosis is None:
            self.diagnosis = np.zeros(len(self.clinical), dtype=np.float32)
        else:
            diag_vals = diagnosis.values if hasattr(diagnosis, 'values') else diagnosis
            self.diagnosis = np.nan_to_num(
                np.asarray(diag_vals, dtype=np.float32).reshape(-1)
            )

        # Validate lengths
        n = len(self.clinical)
        assert len(self.mri) == n, f"MRI length {len(self.mri)} != clinical length {n}"
        assert len(self.pet) == n, f"PET length {len(self.pet)} != clinical length {n}"
        assert len(self.genetic) == n, f"Genetic length {len(self.genetic)} != clinical length {n}"
        assert len(self.targets) == n, f"Targets length {len(self.targets)} != clinical length {n}"
        
    def __len__(self):
        return len(self.targets)
        
    def __getitem__(self, idx):
        clin = torch.tensor(self.clinical[idx], dtype=torch.float32)
        
        mri_feats = self.mri[idx]
        mri_mask = torch.tensor(1.0 if np.all(mri_feats == 0) else 0.0, dtype=torch.float32)
        mri = torch.tensor(mri_feats, dtype=torch.float32)
        
        pet_feats = self.pet[idx]
        pet_mask = torch.tensor(1.0 if np.all(pet_feats == 0) else 0.0, dtype=torch.float32)
        pet = torch.tensor(pet_feats, dtype=torch.float32)
        
        gen_feats = self.genetic[idx]
        gen_mask = torch.tensor(1.0 if np.all(gen_feats == 0) else 0.0, dtype=torch.float32)
        genetic = torch.tensor(gen_feats, dtype=torch.float32)
        
        y = torch.tensor(self.targets[idx].flatten()[0], dtype=torch.float32)
        
        return {
            'clinical': clin,
            'mri': mri,
            'mri_mask': mri_mask,
            'pet': pet,
            'pet_mask': pet_mask,
            'genetic': genetic,
            'genetic_mask': gen_mask,
            'target': y,
            'diagnosis': torch.tensor(self.diagnosis[idx], dtype=torch.float32)
        }

def _subset_dataset(dataset, client_indices, client_id):
    return FederatedPPMIDataset(
        dataset.clinical[client_indices],
        dataset.mri[client_indices],
        dataset.pet[client_indices],
        dataset.genetic[client_indices],
        dataset.targets[client_indices],
        diagnosis=dataset.diagnosis[client_indices],
        client_id=client_id,
    )


def _iid_partition(n, num_clients, rng):
    indices = rng.permutation(n)
    split_sizes = [n // num_clients] * num_clients
    for i in range(n % num_clients):
        split_sizes[i] += 1
    parts, cur = [], 0
    for size in split_sizes:
        parts.append(indices[cur:cur + size])
        cur += size
    return parts


def _dirichlet_partition(strata, num_clients, alpha, rng, min_size=20):
    """
    Label-skew non-IID partition (Hsu et al., 2019): for every stratum, sample
    client proportions from Dirichlet(alpha) and assign its samples accordingly.
    Lower alpha → more heterogeneous clients. Redraws until every client holds
    at least `min_size` samples so local training stays well-posed.
    """
    strata = np.asarray(strata)
    n = len(strata)
    for _ in range(100):
        client_indices = [[] for _ in range(num_clients)]
        for s in np.unique(strata):
            s_idx = np.where(strata == s)[0]
            rng.shuffle(s_idx)
            props = rng.dirichlet([alpha] * num_clients)
            cuts = (np.cumsum(props) * len(s_idx)).astype(int)[:-1]
            for cid, chunk in enumerate(np.split(s_idx, cuts)):
                client_indices[cid].extend(chunk.tolist())
        if min(len(c) for c in client_indices) >= min(min_size, n // (num_clients * 2)):
            return [np.array(sorted(c)) for c in client_indices]
    # Could not satisfy min_size after redraws — fall back to IID
    return _iid_partition(n, num_clients, rng)


def _site_partition(site_labels, num_clients, rng):
    """
    Groups real acquisition sites into `num_clients` clients via greedy
    bin-packing (largest site first onto the currently smallest client), so
    every client is a union of whole sites — no site is split across clients.
    """
    site_labels = np.asarray(site_labels)
    sites, counts = np.unique(site_labels[~pd.isna(site_labels)], return_counts=True)
    order = np.argsort(-counts)
    client_sites = [[] for _ in range(num_clients)]
    client_loads = np.zeros(num_clients)
    for j in order:
        target = int(np.argmin(client_loads))
        client_sites[target].append(sites[j])
        client_loads[target] += counts[j]

    client_indices = []
    for cid in range(num_clients):
        mask = np.isin(site_labels, client_sites[cid])
        client_indices.append(np.where(mask)[0])
    # Subjects with unknown site go to the smallest client
    unknown = np.where(pd.isna(site_labels))[0]
    if len(unknown):
        smallest = int(np.argmin([len(c) for c in client_indices]))
        client_indices[smallest] = np.sort(
            np.concatenate([client_indices[smallest], unknown]))
    return client_indices


def create_federated_splits(dataset, num_clients=3, seed=42, partition="iid",
                            dirichlet_alpha=0.5, strata=None, site_labels=None):
    """
    Partitions the FederatedPPMIDataset into `num_clients` clients.

    partition:
      'iid'       — random shards (original behavior).
      'dirichlet' — label-skew non-IID shards via Dirichlet(alpha) over
                    `strata` (defaults to the diagnosis label). Standard
                    federated-learning benchmark for cross-site heterogeneity.
      'site'      — clients are unions of REAL acquisition sites taken from
                    `site_labels` (one label per dataset row). Falls back to
                    'dirichlet' when no site labels are available.
    """
    rng = np.random.default_rng(seed)
    n = len(dataset)

    if partition == "site":
        if site_labels is None or pd.isna(np.asarray(site_labels)).all():
            partition = "dirichlet"
        else:
            parts = _site_partition(site_labels, num_clients, rng)

    if partition == "dirichlet":
        if strata is None:
            strata = dataset.diagnosis
        parts = _dirichlet_partition(strata, num_clients, dirichlet_alpha, rng)
    elif partition != "site":
        parts = _iid_partition(n, num_clients, rng)

    return [_subset_dataset(dataset, idx, cid) for cid, idx in enumerate(parts)]


def load_site_labels(raw_dir):
    """
    Reads the PPMI Center-Subject list (PATNO → acquisition site) when present.
    Returns a pd.Series indexed by PATNO, or None if no such CSV exists.
    """
    from pathlib import Path
    raw_dir = Path(raw_dir)
    candidates = ["Center-Subject_List.csv", "Center_Subject_List.csv",
                  "Center-Subject_List__PPMI_.csv", "Site_List.csv"]
    for fname in candidates:
        fpath = raw_dir / fname
        if fpath.exists():
            df = pd.read_csv(fpath, low_memory=False)
            site_col = next((c for c in ["CNO", "SITE", "SITE_APRV", "CENTER"]
                             if c in df.columns), None)
            if site_col and "PATNO" in df.columns:
                df = df.drop_duplicates(subset=["PATNO"], keep="last")
                return pd.Series(df[site_col].values, index=df["PATNO"].values)
    return None
