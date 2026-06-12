# -*- coding: utf-8 -*-
"""
Clinical MLP Autoencoder Encoder.

Tabular clinical data (UPDRS scores, cognitive tests, demographics) is
heterogeneous — a mix of continuous, ordinal and binary columns.  A deep
MLP with BatchNorm handles this naturally by learning per-feature scale
transformations internally.

Architecture (encoder path):
    input_dim → [256 → BN → GELU → Dropout]
              → [128 → BN → GELU → Dropout]
              → latent_dim

The decoder mirrors this path and is only used during reconstruction
pretraining.  After pretraining, only the encoder is fine-tuned together
with the FusionTransformer.
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from src.models.encoders.base_encoder import BaseEncoder


def _mlp_block(in_dim: int, out_dim: int, dropout: float = 0.1) -> nn.Sequential:
    """A single MLP block: Linear → BatchNorm → GELU → Dropout."""
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.GELU(),
        nn.Dropout(dropout),
    )


class ClinicalEncoder(BaseEncoder):
    """MLP autoencoder encoder for tabular clinical features.

    Input:  ``(B, input_dim)`` — *z-score normalised* clinical features
    Output: ``(B, latent_dim)`` — clinical embedding

    Pretraining signal: MSE reconstruction loss.
    Fine-tuning: encoder weights updated jointly with FusionTransformer.

    Args:
        input_dim:   Number of clinical features after preprocessing.
        latent_dim:  Dimensionality of the output embedding (shared across
                     all modality encoders — must match FusionTransformer).
        hidden_dims: Intermediate MLP layer widths (encoder direction).
        dropout:     Dropout probability in each MLP block.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 128,
        hidden_dims: List[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        self._latent_dim = latent_dim
        self.input_dim = input_dim

        # ---- Encoder: input_dim → hidden → latent_dim ----
        enc_layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers.append(_mlp_block(prev, h, dropout))
            prev = h
        enc_layers.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # ---- Decoder: latent_dim → hidden (reversed) → input_dim ----
        dec_layers: List[nn.Module] = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers.append(_mlp_block(prev, h, dropout))
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder_net = nn.Sequential(*dec_layers)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map clinical features to latent embedding.

        Args:
            x: ``(B, input_dim)`` normalised clinical tensor.

        Returns:
            ``(B, latent_dim)`` embedding.
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct clinical features from latent embedding.

        Args:
            z: ``(B, latent_dim)`` embedding.

        Returns:
            ``(B, input_dim)`` reconstructed features.
        """
        return self.decoder_net(z)

    @property
    def latent_dim(self) -> int:
        return self._latent_dim
