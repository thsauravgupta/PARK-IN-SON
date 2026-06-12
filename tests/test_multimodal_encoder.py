# -*- coding: utf-8 -*-
"""
Tests for the MultimodalEncoder architecture.

Covers:
  * Each modality encoder independently (shape, no NaN, device)
  * FusionTransformer: full + partial modalities
  * PatientGNN: k-NN graph construction + GATv2 forward
  * MultimodalEncoder end-to-end: all modalities, missing modalities
  * from_config() factory
  * CohortDataset: single sample, batch collate, missing modality collate
  * forward_with_reconstruction: shape of recon outputs
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.encoders.clinical_encoder import ClinicalEncoder
from src.models.encoders.mri_encoder import ROITransformerEncoder
from src.models.encoders.pet_encoder import PETEncoder
from src.models.encoders.genetic_encoder import GeneticEncoder
from src.models.fusion.fusion_transformer import FusionTransformer
from src.models.fusion.patient_gnn import PatientGNN, GATv2Layer
from src.models.multimodal_encoder import MultimodalEncoder

# ---------------------------------------------------------------------------
# Shared fixtures / constants
# ---------------------------------------------------------------------------

B = 8           # batch size
LATENT = 64     # shared latent dim (small for fast tests)
FUSED  = 128    # fused dim
N_ROIS = 20     # small for test speed
N_GENES = 9
PET_DIM = 10
CLIN_DIM = 25


def make_model(use_gnn: bool = True) -> MultimodalEncoder:
    """Build a small MultimodalEncoder for testing."""
    modality_names = ["clinical", "mri", "pet", "genetic"]
    encoders = {
        "clinical": ClinicalEncoder(input_dim=CLIN_DIM, latent_dim=LATENT),
        "mri":      ROITransformerEncoder(n_rois=N_ROIS, feat_per_roi=1, latent_dim=LATENT, d_model=LATENT, n_heads=2, n_layers=2),
        "pet":      PETEncoder(input_dim=PET_DIM, latent_dim=LATENT),
        "genetic":  GeneticEncoder(n_genes=N_GENES, gene_embed_dim=8, latent_dim=LATENT),
    }
    fusion = FusionTransformer(
        modality_names=modality_names,
        latent_dim=LATENT,
        fused_dim=FUSED,
        n_heads=2,
        n_layers=2,
        ffn_dim=128,
    )
    gnn = PatientGNN(node_dim=FUSED, out_dim=FUSED, k_neighbours=3, n_heads=2, n_layers=2)
    return MultimodalEncoder(encoders=encoders, fusion=fusion, gnn=gnn, use_gnn=use_gnn)


def make_x_dict(batch_size: int = B) -> dict:
    """Synthetic batch with all modalities present."""
    return {
        "clinical": torch.randn(batch_size, CLIN_DIM),
        "mri":      torch.randn(batch_size, N_ROIS),
        "pet":      torch.randn(batch_size, PET_DIM),
        "genetic":  torch.randint(0, 3, (batch_size, N_GENES)).float(),
    }


# ===========================================================================
# 1. Individual encoder tests
# ===========================================================================

class TestClinicalEncoder:
    def test_encode_shape(self):
        enc = ClinicalEncoder(input_dim=CLIN_DIM, latent_dim=LATENT)
        x = torch.randn(B, CLIN_DIM)
        z = enc.encode(x)
        assert z.shape == (B, LATENT), f"Expected ({B}, {LATENT}), got {z.shape}"

    def test_no_nan(self):
        enc = ClinicalEncoder(input_dim=CLIN_DIM, latent_dim=LATENT)
        z = enc.encode(torch.randn(B, CLIN_DIM))
        assert not torch.isnan(z).any(), "NaN in ClinicalEncoder output"

    def test_decode_shape(self):
        enc = ClinicalEncoder(input_dim=CLIN_DIM, latent_dim=LATENT)
        z = enc.encode(torch.randn(B, CLIN_DIM))
        x_hat = enc.decode(z)
        assert x_hat.shape == (B, CLIN_DIM)

    def test_forward_equals_encode(self):
        enc = ClinicalEncoder(input_dim=CLIN_DIM, latent_dim=LATENT)
        enc.eval()
        x = torch.randn(B, CLIN_DIM)
        assert torch.allclose(enc(x), enc.encode(x))

    def test_latent_dim_property(self):
        enc = ClinicalEncoder(input_dim=CLIN_DIM, latent_dim=LATENT)
        assert enc.latent_dim == LATENT

    def test_batch_size_1(self):
        """BatchNorm should handle batch_size=1 in eval mode."""
        enc = ClinicalEncoder(input_dim=CLIN_DIM, latent_dim=LATENT)
        enc.eval()
        z = enc.encode(torch.randn(1, CLIN_DIM))
        assert z.shape == (1, LATENT)


class TestROITransformerEncoder:
    def test_encode_flat_input(self):
        """Should accept flat (B, n_rois) input."""
        enc = ROITransformerEncoder(n_rois=N_ROIS, latent_dim=LATENT, n_heads=2, n_layers=2)
        x = torch.randn(B, N_ROIS)
        z = enc.encode(x)
        assert z.shape == (B, LATENT)

    def test_encode_2d_input(self):
        """Should accept (B, n_rois, feat_per_roi) input."""
        enc = ROITransformerEncoder(n_rois=N_ROIS, feat_per_roi=2, latent_dim=LATENT, d_model=LATENT, n_heads=2, n_layers=2)
        x = torch.randn(B, N_ROIS, 2)
        z = enc.encode(x)
        assert z.shape == (B, LATENT)

    def test_no_nan(self):
        enc = ROITransformerEncoder(n_rois=N_ROIS, latent_dim=LATENT, n_heads=2, n_layers=2)
        z = enc.encode(torch.randn(B, N_ROIS))
        assert not torch.isnan(z).any()

    def test_decode_shape(self):
        enc = ROITransformerEncoder(n_rois=N_ROIS, feat_per_roi=1, latent_dim=LATENT, n_heads=2, n_layers=2)
        z = enc.encode(torch.randn(B, N_ROIS))
        x_hat = enc.decode(z)
        assert x_hat.shape == (B, N_ROIS, 1)

    def test_latent_dim_property(self):
        enc = ROITransformerEncoder(n_rois=N_ROIS, latent_dim=LATENT, n_heads=2, n_layers=2)
        assert enc.latent_dim == LATENT


class TestPETEncoder:
    def test_encode_shape(self):
        enc = PETEncoder(input_dim=PET_DIM, latent_dim=LATENT)
        z = enc.encode(torch.randn(B, PET_DIM))
        assert z.shape == (B, LATENT)

    def test_decode_shape(self):
        enc = PETEncoder(input_dim=PET_DIM, latent_dim=LATENT)
        z = enc.encode(torch.randn(B, PET_DIM))
        assert enc.decode(z).shape == (B, PET_DIM)

    def test_no_nan(self):
        enc = PETEncoder(input_dim=PET_DIM, latent_dim=LATENT)
        enc.eval()
        assert not torch.isnan(enc.encode(torch.randn(B, PET_DIM))).any()

    def test_batch_size_1_eval(self):
        enc = PETEncoder(input_dim=PET_DIM, latent_dim=LATENT)
        enc.eval()
        z = enc.encode(torch.randn(1, PET_DIM))
        assert z.shape == (1, LATENT)


class TestGeneticEncoder:
    def test_encode_shape(self):
        enc = GeneticEncoder(n_genes=N_GENES, gene_embed_dim=8, latent_dim=LATENT)
        x = torch.randint(0, 3, (B, N_GENES)).float()
        z = enc.encode(x)
        assert z.shape == (B, LATENT)

    def test_accepts_float_input(self):
        """Should round & clamp float inputs."""
        enc = GeneticEncoder(n_genes=N_GENES, gene_embed_dim=8, latent_dim=LATENT)
        x_float = torch.zeros(B, N_GENES)
        x_float[:, 0] = 1.0
        z = enc.encode(x_float)
        assert z.shape == (B, LATENT)

    def test_decode_logits_shape(self):
        enc = GeneticEncoder(n_genes=N_GENES, gene_embed_dim=8, latent_dim=LATENT)
        z = enc.encode(torch.randint(0, 3, (B, N_GENES)).float())
        logits = enc.decode_logits(z)
        assert logits.shape == (B, N_GENES, 3)

    def test_no_nan(self):
        enc = GeneticEncoder(n_genes=N_GENES, gene_embed_dim=8, latent_dim=LATENT)
        z = enc.encode(torch.randint(0, 3, (B, N_GENES)).float())
        assert not torch.isnan(z).any()

    def test_unknown_state(self):
        """State=2 (unknown) should encode without error."""
        enc = GeneticEncoder(n_genes=N_GENES, gene_embed_dim=8, latent_dim=LATENT)
        x = torch.full((B, N_GENES), 2.0)
        z = enc.encode(x)
        assert not torch.isnan(z).any()


# ===========================================================================
# 2. FusionTransformer tests
# ===========================================================================

class TestFusionTransformer:
    def make_fusion(self):
        return FusionTransformer(
            modality_names=["clinical", "mri", "pet", "genetic"],
            latent_dim=LATENT,
            fused_dim=FUSED,
            n_heads=2,
            n_layers=2,
            ffn_dim=128,
        )

    def make_z_dict(self, batch_size: int = B):
        return {
            "clinical": torch.randn(batch_size, LATENT),
            "mri":      torch.randn(batch_size, LATENT),
            "pet":      torch.randn(batch_size, LATENT),
            "genetic":  torch.randn(batch_size, LATENT),
        }

    def test_output_shape_all_present(self):
        fusion = self.make_fusion()
        z_dict = self.make_z_dict()
        fused, mod_tokens = fusion(z_dict)
        assert fused.shape == (B, FUSED)
        assert mod_tokens.shape == (B, 4, LATENT)

    def test_no_nan_all_present(self):
        fusion = self.make_fusion()
        fused, _ = fusion(self.make_z_dict())
        assert not torch.isnan(fused).any()

    def test_missing_one_modality(self):
        """Forward should work when one modality is absent for some patients."""
        fusion = self.make_fusion()
        z_dict = self.make_z_dict()
        # Patient 0 and 1 missing MRI
        mask = {
            "clinical": torch.ones(B, dtype=torch.bool),
            "mri":      torch.tensor([False, False] + [True] * (B - 2)),
            "pet":      torch.ones(B, dtype=torch.bool),
            "genetic":  torch.ones(B, dtype=torch.bool),
        }
        fused, _ = fusion(z_dict, modality_mask=mask)
        assert fused.shape == (B, FUSED)
        assert not torch.isnan(fused).any()

    def test_missing_two_modalities(self):
        """Forward with 2 missing modalities for all patients."""
        fusion = self.make_fusion()
        z_dict = {"clinical": torch.randn(B, LATENT), "pet": torch.randn(B, LATENT)}
        mask = {
            "clinical": torch.ones(B, dtype=torch.bool),
            "pet":      torch.ones(B, dtype=torch.bool),
        }
        fused, _ = fusion(z_dict, modality_mask=mask)
        assert fused.shape == (B, FUSED)
        assert not torch.isnan(fused).any()

    def test_entirely_absent_modality(self):
        """Modality not in z_dict at all → fully missing for all patients."""
        fusion = self.make_fusion()
        # Only clinical and genetic
        z_dict = {
            "clinical": torch.randn(B, LATENT),
            "genetic":  torch.randn(B, LATENT),
        }
        fused, _ = fusion(z_dict)
        assert fused.shape == (B, FUSED)
        assert not torch.isnan(fused).any()

    def test_different_fused_dims(self, fused_dim=64):
        fusion = FusionTransformer(
            modality_names=["clinical", "mri", "pet", "genetic"],
            latent_dim=LATENT, fused_dim=fused_dim, n_heads=2, n_layers=1, ffn_dim=64
        )
        fused, _ = fusion(self.make_z_dict())
        assert fused.shape == (B, fused_dim)


# ===========================================================================
# 3. PatientGNN tests
# ===========================================================================

class TestPatientGNN:
    def make_gnn(self, k=3):
        return PatientGNN(node_dim=FUSED, out_dim=FUSED, k_neighbours=k, n_heads=2, n_layers=2)

    def test_output_shape(self):
        gnn = self.make_gnn()
        x = torch.randn(B, FUSED)
        z, edge_idx = gnn(x)
        assert z.shape == (B, FUSED)

    def test_no_nan(self):
        gnn = self.make_gnn()
        x = torch.randn(B, FUSED)
        z, _ = gnn(x)
        assert not torch.isnan(z).any()

    def test_edge_index_shape(self):
        gnn = self.make_gnn(k=3)
        x = torch.randn(B, FUSED)
        _, edge_idx = gnn(x)
        assert edge_idx.shape[0] == 2        # (2, E)
        assert edge_idx.shape[1] > 0

    def test_knn_graph_symmetry(self):
        """Symmetrised k-NN graph should have no self-loops."""
        gnn = self.make_gnn(k=3)
        x = torch.randn(B, FUSED)
        edge_idx = gnn.build_knn_graph(x)
        # No self-loops
        assert not (edge_idx[0] == edge_idx[1]).any()

    def test_prebuilt_edge_index(self):
        """Should accept a pre-built edge_index and skip k-NN construction."""
        gnn = self.make_gnn()
        x = torch.randn(B, FUSED)
        # Manual 2-cycle graph
        edge_idx = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        z, returned_idx = gnn(x, edge_index=edge_idx)
        assert z.shape == (B, FUSED)
        assert torch.equal(returned_idx, edge_idx)

    def test_batch_size_1_neighbours_clamped(self):
        """k is clamped to N-1, so single-node graph should not error."""
        gnn = PatientGNN(node_dim=FUSED, out_dim=FUSED, k_neighbours=5, n_heads=1, n_layers=1)
        x = torch.randn(2, FUSED)  # Only 2 patients — k clamped to 1
        z, _ = gnn(x)
        assert z.shape == (2, FUSED)


# ===========================================================================
# 4. MultimodalEncoder end-to-end tests
# ===========================================================================

class TestMultimodalEncoder:
    def test_full_forward_shape(self):
        model = make_model()
        x_dict = make_x_dict()
        z, per_mod, edge_idx = model(x_dict)
        assert z.shape == (B, FUSED)
        assert "clinical" in per_mod
        assert per_mod["clinical"].shape == (B, LATENT)

    def test_no_nan_full(self):
        model = make_model()
        z, per_mod, _ = model(make_x_dict())
        assert not torch.isnan(z).any()
        for name, zmod in per_mod.items():
            assert not torch.isnan(zmod).any(), f"NaN in per_mod[{name}]"

    def test_missing_mri(self):
        """Model handles batch where some patients have no MRI."""
        model = make_model()
        x_dict = make_x_dict()
        mask = {
            "clinical": torch.ones(B, dtype=torch.bool),
            "mri":      torch.tensor([False, True, True, False, True, True, True, False]),
            "pet":      torch.ones(B, dtype=torch.bool),
            "genetic":  torch.ones(B, dtype=torch.bool),
        }
        z, _, _ = model(x_dict, modality_mask=mask)
        assert z.shape == (B, FUSED)
        assert not torch.isnan(z).any()

    def test_missing_all_imaging(self):
        """Only clinical + genetic present — should still produce valid embedding."""
        model = make_model()
        x_dict = {
            "clinical": torch.randn(B, CLIN_DIM),
            "genetic":  torch.randint(0, 3, (B, N_GENES)).float(),
        }
        z, per_mod, _ = model(x_dict)
        assert z.shape == (B, FUSED)
        assert not torch.isnan(z).any()
        assert "mri" not in per_mod
        assert "pet" not in per_mod

    def test_gnn_disabled(self):
        model = make_model(use_gnn=False)
        z, _, edge_idx = model(make_x_dict())
        assert z.shape == (B, FUSED)
        assert edge_idx is None

    def test_forward_with_reconstruction(self):
        model = make_model()
        x_dict = make_x_dict()
        z, per_mod, recon = model.forward_with_reconstruction(x_dict)
        assert z.shape == (B, FUSED)
        assert "clinical" in recon
        # Clinical recon should have same shape as input
        assert recon["clinical"].shape == x_dict["clinical"].shape

    def test_count_parameters(self):
        model = make_model()
        counts = model.count_parameters()
        assert "total" in counts
        assert counts["total"] > 0
        assert "encoder_clinical" in counts
        assert "fusion_transformer" in counts
        assert "patient_gnn" in counts

    def test_from_config(self, tmp_path):
        """from_config() should build a valid model from a minimal config."""
        config = {
            "multimodal_encoder": {
                "latent_dim": 64,
                "fused_dim": 128,
                "dropout": 0.1,
                "fusion_transformer": {"n_heads": 2, "n_layers": 2, "ffn_dim": 128},
                "patient_gnn": {"k_neighbours": 3, "gat_heads": 2, "gat_layers": 2},
            },
            "mri": {"n_rois": 20, "feat_per_roi": 1},
            "clinical": {
                "updrs_cols": ["updrs_i", "updrs_ii", "updrs_iii"],
                "extra_behavioral": [],
                "demographic_cols": ["age", "sex"],
            },
            "genetic": {
                "target_genes": ["LRRK2", "GBA", "SNCA"],
                "explicit_features": ["n_variants"],
            },
        }
        model = MultimodalEncoder.from_config(config)
        assert isinstance(model, MultimodalEncoder)
        assert "clinical" in model.encoders

    def test_add_encoder(self):
        model = make_model()
        new_enc = ClinicalEncoder(input_dim=5, latent_dim=LATENT)
        model.add_encoder("eeg", new_enc)
        assert "eeg" in model.encoders
        assert "eeg" in model.modality_names


# ===========================================================================
# 5. CohortDataset tests
# ===========================================================================

class TestCohortDataset:
    def _make_cohort_df(self, n=20) -> pd.DataFrame:
        patnos = list(range(1000, 1000 + n))
        df = pd.DataFrame({
            "has_clinical": [True]  * n,
            "has_mri":      [True]  * (n // 2) + [False] * (n - n // 2),
            "has_pet":      [True]  * n,
            "has_genetic":  [True]  * n,
            "n_modalities": [3 if i >= n // 2 else 4 for i in range(n)],
            "diagnosis":    (["PD", "HC"] * (n // 2))[:n],
            "split":        ["train"] * n,
            "updrs_iii_v04": np.random.randn(n).tolist(),
        }, index=patnos)
        return df

    def _make_modality_dfs(self, n=20) -> dict:
        patnos = list(range(1000, 1000 + n))
        return {
            "clinical": pd.DataFrame(
                np.random.randn(n, CLIN_DIM),
                index=patnos,
                columns=[f"clin_{i}" for i in range(CLIN_DIM)],
            ),
            "mri": pd.DataFrame(
                np.random.randn(n // 2, N_ROIS),
                index=patnos[:n // 2],
                columns=[f"roi_{i}" for i in range(N_ROIS)],
            ),
            "pet": pd.DataFrame(
                np.random.randn(n, PET_DIM),
                index=patnos,
                columns=[f"pet_{i}" for i in range(PET_DIM)],
            ),
            "genetic": pd.DataFrame(
                np.random.randint(0, 3, (n, N_GENES)).astype(float),
                index=patnos,
                columns=[f"gene_{i}" for i in range(N_GENES)],
            ),
        }

    def test_len(self):
        from src.data.cohort import CohortDataset
        ds = CohortDataset(self._make_cohort_df(), self._make_modality_dfs())
        assert len(ds) == 20

    def test_getitem_present_modalities(self):
        from src.data.cohort import CohortDataset
        ds = CohortDataset(self._make_cohort_df(), self._make_modality_dfs())
        sample = ds[0]   # patient 1000 — has all modalities
        assert sample["mask"]["clinical"] is True
        assert sample["x"]["clinical"].shape == (CLIN_DIM,)
        assert sample["label"].item() >= 0

    def test_getitem_missing_mri(self):
        from src.data.cohort import CohortDataset
        ds = CohortDataset(self._make_cohort_df(), self._make_modality_dfs())
        # Patient at index 10 (PATNO 1010) has no MRI
        sample = ds[10]
        assert sample["mask"]["mri"] is False
        assert sample["x"]["mri"] is None

    def test_collate_fn_shapes(self):
        from src.data.cohort import CohortDataset
        from torch.utils.data import DataLoader
        ds = CohortDataset(self._make_cohort_df(), self._make_modality_dfs())
        loader = DataLoader(ds, batch_size=4, collate_fn=CohortDataset.collate_fn)
        batch = next(iter(loader))
        assert batch["x"]["clinical"].shape == (4, CLIN_DIM)
        assert batch["mask"]["clinical"].shape == (4,)
        assert batch["label"].shape == (4,)

    def test_collate_missing_modality_mask(self):
        from src.data.cohort import CohortDataset
        from torch.utils.data import DataLoader
        ds = CohortDataset(self._make_cohort_df(), self._make_modality_dfs())
        loader = DataLoader(ds, batch_size=20, collate_fn=CohortDataset.collate_fn)
        batch = next(iter(loader))
        mri_mask = batch["mask"]["mri"]
        # First 10 patients have MRI, last 10 don't
        assert mri_mask[:10].all()
        assert not mri_mask[10:].any()

    def test_fit_normalisation(self):
        from src.data.cohort import CohortDataset
        ds = CohortDataset(self._make_cohort_df(), self._make_modality_dfs(), normalise=True)
        train_patnos = list(range(1000, 1010))
        ds.fit_normalisation(train_patnos)
        assert "clinical" in ds._means
        assert ds._stds["clinical"].shape == (CLIN_DIM,)

    def test_repr(self):
        from src.data.cohort import CohortDataset
        ds = CohortDataset(self._make_cohort_df(), self._make_modality_dfs())
        r = repr(ds)
        assert "CohortDataset" in r
        assert "clinical" in r


# ===========================================================================
# 6. Gradient flow tests
# ===========================================================================

class TestGradientFlow:
    def test_gradients_flow_through_full_model(self):
        """Encoder params should receive gradients; decoder params only during pretraining."""
        model = make_model()
        model.train()
        x_dict = make_x_dict()
        z, _, _ = model(x_dict)
        loss = z.mean()
        loss.backward()

        for name, enc in model.encoders.items():
            for pname, param in enc.named_parameters():
                # decoder_net / decoder / gene_heads are NOT in the inference path
                # (only used in forward_with_reconstruction for pretraining)
                if "decoder" in pname or "gene_heads" in pname:
                    continue
                assert param.grad is not None, \
                    f"No gradient for encoder '{name}' param '{pname}'"

    def test_gradients_with_missing_modality(self):
        """Gradients should flow through present-modality encoder params."""
        model = make_model()
        model.train()
        x_dict = make_x_dict()
        mask = {
            "clinical": torch.ones(B, dtype=torch.bool),
            "mri":      torch.zeros(B, dtype=torch.bool),   # All MRI missing
            "pet":      torch.ones(B, dtype=torch.bool),
            "genetic":  torch.ones(B, dtype=torch.bool),
        }
        z, _, _ = model(x_dict, modality_mask=mask)
        z.mean().backward()
        # Clinical encoder encode path must have grads (it was used)
        for pname, param in model.encoders["clinical"].named_parameters():
            if "decoder" in pname:
                continue
            assert param.grad is not None, f"No grad for clinical encoder param '{pname}'"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
