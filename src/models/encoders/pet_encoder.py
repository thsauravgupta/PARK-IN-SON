# -*- coding: utf-8 -*-
"""
PET / DaTscan Lightweight MLP Encoder.

DaTscan data consists of a small number of engineered scalar features
(4 raw SBR values + 6 derived: means, asymmetry indices, putamen/caudate
ratios — 10 total, already computed by PETPreprocessor).

WHY lightweight?  Only 10 input features — a deep network would massively
overfit.  Two linear layers with a BatchNorm + GELU non-linearity are
sufficient to capture the non-linear relationships in DaTscan data while
remaining stable with small cohort sizes.

Architecture:
    (B, 10) → Linear(10→64) → BN → GELU → Dropout
            → Linear(64→latent_dim)

Decoder (pretraining only):
    (B, latent_dim) → Linear → GELU → Linear(→10)
"""

import torch
import torch.nn as nn

from src.models.encoders.base_encoder import BaseEncoder


class PETEncoder(BaseEncoder):
    """Lightweight MLP encoder for DaTscan / PET features.

    Args:
        input_dim:  Number of PET features (10 after PETPreprocessor).
        latent_dim: Output embedding dimensionality (shared with all modalities).
        dropout:    Dropout probability.
    """

    def __init__(
        self,
        input_dim: int = 10,
        latent_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self._latent_dim = latent_dim
        self.input_dim = input_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, latent_dim),
        )

        self.decoder_net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, input_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode DaTscan features to latent embedding.

        Args:
            x: ``(B, input_dim)`` normalised PET/DaTscan features.

        Returns:
            ``(B, latent_dim)`` PET embedding.
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct DaTscan features from latent (pretraining).

        Args:
            z: ``(B, latent_dim)``.

        Returns:
            ``(B, input_dim)`` reconstructed features.
        """
        return self.decoder_net(z)

    @property
    def latent_dim(self) -> int:
        return self._latent_dim
