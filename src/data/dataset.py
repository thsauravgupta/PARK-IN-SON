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
    def __init__(self, clinical_df, mri_df, pet_df, genetic_df, targets, client_id=None):
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
            'target': y
        }

def create_federated_splits(dataset, num_clients=3, seed=42):
    """
    Partitions the FederatedPPMIDataset into `num_clients` random splits,
    simulating isolated clinical sites.
    """
    np.random.seed(seed)
    indices = np.random.permutation(len(dataset))
    split_sizes = [len(dataset) // num_clients] * num_clients
    for i in range(len(dataset) % num_clients):
        split_sizes[i] += 1
        
    client_datasets = []
    current_idx = 0
    for i, size in enumerate(split_sizes):
        client_indices = indices[current_idx:current_idx + size]
        
        client_ds = FederatedPPMIDataset(
            dataset.clinical[client_indices],
            dataset.mri[client_indices],
            dataset.pet[client_indices],
            dataset.genetic[client_indices],
            dataset.targets[client_indices],
            client_id=i
        )
        client_datasets.append(client_ds)
        current_idx += size
        
    return client_datasets
