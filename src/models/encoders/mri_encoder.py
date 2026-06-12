# -*- coding: utf-8 -*-
"""
MRI ROI Transformer Encoder.

Brain MRI data is represented as a set of region-of-interest (ROI)
measurements (cortical thickness, subcortical volume, etc.) from a
parcellation atlas (e.g. Schaefer-100).

WHY a Transformer over ROI tokens, not a flat MLP?
  * Each ROI is a meaningful anatomical unit — it has a spatial identity.
  * Disease affects coordinated networks of brain regions, not regions in
    isolation.  Self-attention learns *which ROIs co-vary* — exactly the
    inter-regional coordination that matters in Parkinson's disease.
  * The architecture generalises: later, the functional connectivity matrix
    can be injected as an attention bias (ROI₁ attends more to ROI₂ if
    they are functionally connected), turning this into a brain-graph
    Transformer with zero architectural changes.

Architecture:
    Input (B, n_rois [, feat_per_roi])
      → Linear projection → (B, n_rois, d_model)
      → + learnable ROI positional embeddings
      → Prepend [CLS] token
      → 4-layer Transformer Encoder (pre-LN, GELU, batch_first)
      → CLS token → Linear → (B, latent_dim)

Decoder (pretraining only):
    (B, latent_dim) → Linear → GELU → Linear → (B, n_rois × feat_per_roi)
"""

import torch
import torch.nn as nn

from src.models.encoders.base_encoder import BaseEncoder


class ROITransformerEncoder(BaseEncoder):
    """Transformer encoder over parcellated brain ROI features.

    Args:
        n_rois:        Number of atlas parcels (default 100, Schaefer-100).
        feat_per_roi:  Number of features per ROI (e.g. 1 for thickness only,
                       2 for thickness + volume).  If the input tensor is
                       flat ``(B, n_rois)``, it is reshaped to
                       ``(B, n_rois, 1)`` automatically.
        latent_dim:    Output embedding dimensionality (shared across all
                       modality encoders).
        d_model:       Internal Transformer width.  Should be ≥ latent_dim.
        n_heads:       Number of attention heads (must divide d_model).
        n_layers:      Number of Transformer encoder layers.
        dropout:       Dropout probability.
    """

    def __init__(
        self,
        n_rois: int = 100,
        feat_per_roi: int = 1,
        latent_dim: int = 128,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self._latent_dim = latent_dim
        self.n_rois = n_rois
        self.feat_per_roi = feat_per_roi
        self.d_model = d_model

        # Project each ROI's features to the Transformer d_model width
        self.input_proj = nn.Linear(feat_per_roi, d_model)

        # Learnable positional embedding — one per ROI (atlas-parcel identity)
        self.roi_pos_embed = nn.Embedding(n_rois, d_model)

        # Learnable [CLS] token that aggregates the global brain state
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Transformer encoder (Pre-LayerNorm for stable training)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # Pre-LN: more stable than Post-LN
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )

        # Project CLS output → latent_dim
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, latent_dim),
        )

        # Decoder for pretraining (reconstructs ROI features from latent)
        self.decoder_net = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_rois * feat_per_roi),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.input_proj.weight, std=0.02)
        nn.init.zeros_(self.input_proj.bias)
        # ROI positional embeddings: small normal
        nn.init.trunc_normal_(self.roi_pos_embed.weight, std=0.02)

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape flat ``(B, n_rois)`` to ``(B, n_rois, feat_per_roi)``."""
        if x.dim() == 2:
            # Flat ROI vector — each ROI has a single feature value
            return x.unsqueeze(-1).expand(-1, -1, self.feat_per_roi)
        return x  # Already (B, n_rois, feat_per_roi)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode ROI features to brain embedding.

        Args:
            x: ``(B, n_rois)`` or ``(B, n_rois, feat_per_roi)``.

        Returns:
            ``(B, latent_dim)`` brain embedding.
        """
        x = self._prepare_input(x)          # (B, n_rois, feat_per_roi)
        B = x.shape[0]

        tokens = self.input_proj(x)          # (B, n_rois, d_model)

        # Add ROI positional embeddings
        roi_ids = torch.arange(self.n_rois, device=x.device)
        tokens = tokens + self.roi_pos_embed(roi_ids)   # broadcast over B

        # Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)          # (B, 1, d_model)
        tokens = torch.cat([cls, tokens], dim=1)         # (B, 1+n_rois, d_model)

        # Transformer self-attention over ROI tokens + CLS
        out = self.transformer(tokens)                   # (B, 1+n_rois, d_model)

        # Extract CLS and project to latent_dim
        cls_out = out[:, 0, :]                           # (B, d_model)
        return self.output_proj(cls_out)                 # (B, latent_dim)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct ROI features from latent embedding (pretraining).

        Args:
            z: ``(B, latent_dim)``.

        Returns:
            ``(B, n_rois, feat_per_roi)`` reconstructed ROI features.
        """
        flat = self.decoder_net(z)           # (B, n_rois * feat_per_roi)
        return flat.view(-1, self.n_rois, self.feat_per_roi)

    @property
    def latent_dim(self) -> int:
        return self._latent_dim
