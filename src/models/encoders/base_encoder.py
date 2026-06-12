# -*- coding: utf-8 -*-
"""
Abstract base class for all modality-specific deep encoders.

All encoders must:
  * Be a torch.nn.Module
  * Implement encode(x) -> z  of shape (B, latent_dim)
  * Implement decode(z) -> x_hat for reconstruction pretraining
  * Expose latent_dim property

Design note: separating encode / decode from PyTorch's forward() lets
the fusion pipeline call encode() without triggering the full AE
(decoder not needed during joint training / inference).
"""

from abc import abstractmethod
from typing import Tuple

import torch
import torch.nn as nn


class BaseEncoder(nn.Module):
    """Abstract base for all modality encoders in PARK-IN-SON.

    Subclasses implement:
        encode(x)  → z         latent representation
        decode(z)  → x_hat     reconstruction for pretraining
        latent_dim             output dimensionality (property)
    """

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map input to latent vector.

        Args:
            x: Modality-specific input tensor, shape varies per encoder.

        Returns:
            z: Latent embedding, shape ``(B, latent_dim)``.
        """
        raise NotImplementedError

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct input from latent vector (for pretraining).

        Args:
            z: Latent embedding, shape ``(B, latent_dim)``.

        Returns:
            x_hat: Reconstructed input in the same space as the original input.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def latent_dim(self) -> int:
        """Dimensionality of the latent embedding produced by encode()."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Default forward = encode-only (inference mode).

        WHY: The fusion transformer only calls encode().  Having forward()
        delegate to encode() means the encoder can be used as a plain
        nn.Module in standard training loops while still being composable
        inside MultimodalEncoder.
        """
        return self.encode(x)

    def forward_with_reconstruction(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode and decode in one call — used during pretraining.

        Args:
            x: Input tensor.

        Returns:
            (z, x_hat): Latent embedding and reconstruction.
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        return z, x_hat

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(latent_dim={self.latent_dim})"
