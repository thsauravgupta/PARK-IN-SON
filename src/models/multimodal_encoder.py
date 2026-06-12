# -*- coding: utf-8 -*-
"""
MultimodalEncoder — top-level orchestrator for the PARK-IN-SON model.

Pipeline (per forward call):
    1.  Per-modality encoders  → latent vectors  (B, latent_dim) each
    2.  FusionTransformer      → fused embedding (B, fused_dim)
    3.  PatientGNN             → GNN-refined     (B, out_dim = fused_dim)

Design principles:
  * Adding a new modality is one line:
        model.add_encoder("eeg", EEGEncoder(...))
    The FusionTransformer must be reinitialised to include the new slot
    (use ``from_config()`` which wires everything automatically).

  * Missing modality handling is first-class: pass ``modality_mask`` to
    mark which samples have which modalities.  The FusionTransformer
    replaces absent slots with learned [MISSING] tokens and masks them
    from cross-modal attention.

  * GNN is optional (``use_gnn=False``).  When disabled, the fused
    embedding from the FusionTransformer is returned directly.

Usage
-----
    # Build from config
    model = MultimodalEncoder.from_config(config)

    # Forward (all modalities present)
    z, per_mod, edge_idx = model(x_dict, modality_mask=None)

    # Forward (some patients missing MRI)
    mask = {"clinical": all_present, "mri": has_mri, "pet": ..., "genetic": ...}
    z, per_mod, edge_idx = model(x_dict, modality_mask=mask)

    # Forward with reconstruction (for pretraining loss)
    z, per_mod, recon = model.forward_with_reconstruction(x_dict, mask)
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.models.encoders.base_encoder import BaseEncoder
from src.models.fusion.fusion_transformer import FusionTransformer
from src.models.fusion.patient_gnn import PatientGNN


class MultimodalEncoder(nn.Module):
    """Top-level multimodal encoder: encoders + FusionTransformer + GATv2 GNN.

    Args:
        encoders:  Ordered dict of modality name → BaseEncoder subclass.
                   All encoders must have the same ``latent_dim``.
        fusion:    FusionTransformer configured with the same modality names
                   and ``latent_dim``.
        gnn:       PatientGNN operating on the fused embedding space.
        use_gnn:   If False, skip the GNN stage (useful for ablations).
    """

    def __init__(
        self,
        encoders: Dict[str, BaseEncoder],
        fusion: FusionTransformer,
        gnn: Optional[PatientGNN] = None,
        use_gnn: bool = True,
    ):
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
        self.fusion = fusion
        self.gnn = gnn
        self.use_gnn = use_gnn and (gnn is not None)
        self.modality_names = list(encoders.keys())

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def encode_modalities(
        self,
        x_dict: Dict[str, Optional[torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Run each present modality encoder independently.

        Args:
            x_dict: ``dict[modality_name -> (B, ...) tensor | None]``.
                    Entries of ``None`` are silently skipped.

        Returns:
            ``dict[modality_name -> (B, latent_dim)]``.
        """
        z_dict: Dict[str, torch.Tensor] = {}
        for name, encoder in self.encoders.items():
            x = x_dict.get(name)
            if x is not None:
                z_dict[name] = encoder.encode(x)
        return z_dict

    def forward(
        self,
        x_dict: Dict[str, Optional[torch.Tensor]],
        modality_mask: Optional[Dict[str, torch.Tensor]] = None,
        edge_index: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        """Full forward pass through encoder → fusion → GNN.

        Args:
            x_dict:        ``dict[modality_name -> (B, ...) | None]``.
            modality_mask: ``dict[modality_name -> (B,) bool]``.
                           ``True`` = modality present for that sample.
                           ``None`` → all supplied modalities assumed present.
            edge_index:    Pre-built ``(2, E)`` patient graph edge index.
                           ``None`` → k-NN graph built automatically from
                           fused embeddings.

        Returns:
            z_final:   ``(B, out_dim)`` — final patient embedding
                       (digital twin seed).
            z_per_mod: ``dict[modality_name -> (B, latent_dim)]``.
            edge_index: ``(2, E)`` used in GNN (``None`` if GNN disabled).
        """
        # 1. Per-modality encoding
        z_per_mod = self.encode_modalities(x_dict)

        # 2. Cross-modal fusion
        z_fused, _ = self.fusion(z_per_mod, modality_mask)

        # 3. GNN refinement
        edge_out: Optional[torch.Tensor] = None
        if self.use_gnn:
            z_final, edge_out = self.gnn(z_fused, edge_index)
        else:
            z_final = z_fused

        return z_final, z_per_mod, edge_out

    def forward_with_reconstruction(
        self,
        x_dict: Dict[str, Optional[torch.Tensor]],
        modality_mask: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Forward pass that also returns per-modality reconstructions.

        Used during pretraining to compute reconstruction losses alongside
        the fused patient embedding.

        Args:
            x_dict:        Same as ``forward()``.
            modality_mask: Same as ``forward()``.

        Returns:
            z_final:    ``(B, out_dim)``.
            z_per_mod:  ``dict[modality_name -> (B, latent_dim)]``.
            recon_dict: ``dict[modality_name -> reconstructed tensor]``
                        — same shape as the corresponding input tensor.
        """
        z_per_mod: Dict[str, torch.Tensor] = {}
        recon_dict: Dict[str, torch.Tensor] = {}

        for name, encoder in self.encoders.items():
            x = x_dict.get(name)
            if x is not None:
                z, x_hat = encoder.forward_with_reconstruction(x)
                z_per_mod[name] = z
                recon_dict[name] = x_hat

        z_fused, _ = self.fusion(z_per_mod, modality_mask)

        if self.use_gnn:
            z_final, _ = self.gnn(z_fused)
        else:
            z_final = z_fused

        return z_final, z_per_mod, recon_dict

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    def add_encoder(self, name: str, encoder: BaseEncoder) -> None:
        """Register a new modality encoder at runtime.

        Note: The FusionTransformer must be reinitialised (or fine-tuned)
        to accommodate the new modality slot.  Call ``from_config()`` with
        an updated config for a clean rebuild.
        """
        self.encoders[name] = encoder
        if name not in self.modality_names:
            self.modality_names.append(name)

    @classmethod
    def from_config(cls, config: dict) -> "MultimodalEncoder":
        """Build a fully configured MultimodalEncoder from config.yaml.

        Expected config structure::

            multimodal_encoder:
              latent_dim: 128
              fused_dim: 256
              dropout: 0.1
              fusion_transformer:
                n_heads: 8
                n_layers: 4
                ffn_dim: 512
              patient_gnn:
                k_neighbours: 10
                gat_heads: 4
                gat_layers: 2

        Modality-specific input dims are inferred from the data config
        sections (``clinical``, ``mri``, ``pet``, ``genetic``).
        """
        from src.models.encoders.clinical_encoder import ClinicalEncoder
        from src.models.encoders.mri_encoder import ROITransformerEncoder
        from src.models.encoders.pet_encoder import PETEncoder
        from src.models.encoders.genetic_encoder import GeneticEncoder

        mc = config.get("multimodal_encoder", {})
        latent_dim: int = mc.get("latent_dim", 128)
        fused_dim: int = mc.get("fused_dim", 256)
        dropout: float = mc.get("dropout", 0.1)

        ft_cfg = mc.get("fusion_transformer", {})
        gnn_cfg = mc.get("patient_gnn", {})

        # ---- Clinical dims -----------------------------------------------
        clin_cfg = config.get("clinical", {})
        clin_cols = (
            clin_cfg.get("updrs_cols", [])
            + clin_cfg.get("extra_behavioral", [])
            + clin_cfg.get("demographic_cols", [])
        )
        clinical_input_dim = len(clin_cols) if clin_cols else 30

        # ---- Genetic dims ------------------------------------------------
        gen_cfg = config.get("genetic", {})
        target_genes = gen_cfg.get("target_genes", [])
        explicit_feats = gen_cfg.get("explicit_features", [])
        genetic_n_genes = len(target_genes) + len(explicit_feats) if target_genes else 9

        # ---- MRI dims ----------------------------------------------------
        mri_cfg = config.get("mri", {})
        n_rois: int = mri_cfg.get("n_rois", 100)
        feat_per_roi: int = mri_cfg.get("feat_per_roi", 1)

        # ---- PET dims ----------------------------------------------------
        pet_input_dim: int = 10  # Fixed — PETPreprocessor produces exactly 10 features

        modality_names = ["clinical", "mri", "pet", "genetic"]

        encoders: Dict[str, BaseEncoder] = {
            "clinical": ClinicalEncoder(
                input_dim=clinical_input_dim,
                latent_dim=latent_dim,
                hidden_dims=[256, 128],
                dropout=dropout,
            ),
            "mri": ROITransformerEncoder(
                n_rois=n_rois,
                feat_per_roi=feat_per_roi,
                latent_dim=latent_dim,
                d_model=latent_dim,
                n_heads=4,
                n_layers=4,
                dropout=dropout,
            ),
            "pet": PETEncoder(
                input_dim=pet_input_dim,
                latent_dim=latent_dim,
                dropout=dropout,
            ),
            "genetic": GeneticEncoder(
                n_genes=genetic_n_genes,
                gene_embed_dim=16,
                latent_dim=latent_dim,
                dropout=dropout,
                gene_names=(target_genes + explicit_feats) or None,
            ),
        }

        fusion = FusionTransformer(
            modality_names=modality_names,
            latent_dim=latent_dim,
            fused_dim=fused_dim,
            n_heads=ft_cfg.get("n_heads", 8),
            n_layers=ft_cfg.get("n_layers", 4),
            ffn_dim=ft_cfg.get("ffn_dim", 512),
            dropout=ft_cfg.get("dropout", dropout),
        )

        gnn = PatientGNN(
            node_dim=fused_dim,
            out_dim=fused_dim,
            k_neighbours=gnn_cfg.get("k_neighbours", 10),
            n_heads=gnn_cfg.get("gat_heads", 4),
            n_layers=gnn_cfg.get("gat_layers", 2),
            dropout=gnn_cfg.get("dropout", dropout),
        )

        return cls(encoders=encoders, fusion=fusion, gnn=gnn, use_gnn=True)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def count_parameters(self) -> Dict[str, int]:
        """Return parameter counts per component (useful for debugging)."""
        counts = {}
        for name, enc in self.encoders.items():
            counts[f"encoder_{name}"] = sum(p.numel() for p in enc.parameters())
        counts["fusion_transformer"] = sum(p.numel() for p in self.fusion.parameters())
        if self.gnn is not None:
            counts["patient_gnn"] = sum(p.numel() for p in self.gnn.parameters())
        counts["total"] = sum(p.numel() for p in self.parameters())
        return counts

    def __repr__(self) -> str:
        counts = self.count_parameters()
        enc_lines = "\n    ".join(
            f"{k}: {v:,}" for k, v in counts.items() if k != "total"
        )
        return (
            f"MultimodalEncoder(\n"
            f"  modalities={self.modality_names},\n"
            f"  use_gnn={self.use_gnn},\n"
            f"  parameters=\n    {enc_lines}\n"
            f"  total={counts['total']:,}\n"
            f")"
        )
