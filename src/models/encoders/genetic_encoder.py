# -*- coding: utf-8 -*-
"""
Genetic Gene-Token Transformer Encoder.

WHY treat genes as tokens, not as a flat binary vector?

  1. **Expressiveness**: A flat binary vector pushed through an MLP treats
     gene co-occurrences as purely additive.  A per-gene embedding lets the
     model learn rich, non-additive gene–gene interactions (e.g. LRRK2+GBA
     compound heterozygosity has a different effect than either alone).

  2. **Scalability to WGS**: When you upgrade from 9 candidate-gene flags to
     whole-genome sequencing (millions of variants), you can replace the binary
     per-gene embedding with a variant-level tokenisation scheme (e.g. one token
     per functional variant) without changing the downstream architecture.

  3. **Handles unknown status naturally**: A 3-state embedding
     (0=WT, 1=mutation, 2=unknown/not-tested) is cleaner than imputing zeros.

Architecture:
    Input: (B, n_genes)  integer tensor  {0, 1, 2}
      → per-gene nn.Embedding(3, gene_embed_dim)  × n_genes
      → stack → (B, n_genes, gene_embed_dim)
      → Prepend [CLS] token
      → 2-layer Transformer Encoder (small — only n_genes tokens)
      → CLS output → Linear → (B, latent_dim)

Decoder (pretraining, cross-entropy per gene):
    (B, latent_dim) → Linear → GELU → per-gene Linear(gene_embed_dim→3)
"""

from typing import List, Optional

import torch
import torch.nn as nn

from src.models.encoders.base_encoder import BaseEncoder


class GeneticEncoder(BaseEncoder):
    """Gene-token embedding Transformer for genetic variant data.

    Each gene is treated as a discrete token with states:
        0 = wild-type (no mutation detected)
        1 = mutation present
        2 = unknown / not tested

    Args:
        n_genes:        Number of genes (default 9 for PPMI panel).
        gene_embed_dim: Embedding dimension per gene token.
        latent_dim:     Output embedding dimensionality.
        dropout:        Dropout probability.
        gene_names:     Optional list of gene names for __repr__.
    """

    # Gene states
    STATE_WT = 0
    STATE_MUT = 1
    STATE_UNKNOWN = 2

    def __init__(
        self,
        n_genes: int = 9,
        gene_embed_dim: int = 16,
        latent_dim: int = 128,
        dropout: float = 0.1,
        gene_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self._latent_dim = latent_dim
        self.n_genes = n_genes
        self.gene_embed_dim = gene_embed_dim
        self.gene_names = gene_names or [f"gene_{i}" for i in range(n_genes)]

        # Per-gene embedding tables (3 discrete states each)
        self.gene_embeddings = nn.ModuleList(
            [nn.Embedding(3, gene_embed_dim) for _ in range(n_genes)]
        )

        # Learnable [CLS] gene-aggregation token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, gene_embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Transformer over gene tokens
        # Small: only n_genes tokens, so few layers + heads suffice
        n_heads = max(1, gene_embed_dim // 8)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=gene_embed_dim,
            nhead=n_heads,
            dim_feedforward=gene_embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=2, enable_nested_tensor=False
        )

        # CLS → latent_dim projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(gene_embed_dim),
            nn.Linear(gene_embed_dim, latent_dim),
        )

        # Decoder: latent → per-gene feature space (for pretraining)
        self.decoder_net = nn.Sequential(
            nn.Linear(latent_dim, n_genes * gene_embed_dim),
            nn.GELU(),
            nn.Unflatten(1, (n_genes, gene_embed_dim)),
        )
        # Per-gene 3-class classification heads (used during pretraining)
        self.gene_heads = nn.ModuleList(
            [nn.Linear(gene_embed_dim, 3) for _ in range(n_genes)]
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for emb in self.gene_embeddings:
            nn.init.normal_(emb.weight, std=0.02)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode gene mutation flags to genetic embedding.

        Args:
            x: ``(B, n_genes)`` integer tensor with values in {0, 1, 2}.
               Float inputs are rounded and clamped automatically.

        Returns:
            ``(B, latent_dim)`` genetic embedding.
        """
        B = x.shape[0]
        # Sanitise: convert float→int and clamp to valid range
        x_int = x.long().clamp(self.STATE_WT, self.STATE_UNKNOWN)

        # Embed each gene: list of (B, gene_embed_dim)
        gene_tokens = [emb(x_int[:, i]) for i, emb in enumerate(self.gene_embeddings)]
        gene_tokens = torch.stack(gene_tokens, dim=1)  # (B, n_genes, gene_embed_dim)

        # Prepend [CLS]
        cls = self.cls_token.expand(B, -1, -1)                    # (B, 1, gene_embed_dim)
        tokens = torch.cat([cls, gene_tokens], dim=1)              # (B, 1+n_genes, gene_embed_dim)

        # Transformer over gene tokens
        out = self.transformer(tokens)                             # (B, 1+n_genes, gene_embed_dim)

        # CLS output → latent
        return self.output_proj(out[:, 0, :])                      # (B, latent_dim)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to per-gene feature vectors (pretraining).

        Args:
            z: ``(B, latent_dim)``.

        Returns:
            ``(B, n_genes, gene_embed_dim)`` reconstructed gene features.
        """
        return self.decoder_net(z)

    def decode_logits(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to per-gene classification logits (3 classes each).

        Used during pretraining with CrossEntropyLoss.

        Args:
            z: ``(B, latent_dim)``.

        Returns:
            ``(B, n_genes, 3)`` classification logits.
        """
        gene_feats = self.decode(z)                    # (B, n_genes, gene_embed_dim)
        logits = torch.stack(
            [head(gene_feats[:, i, :]) for i, head in enumerate(self.gene_heads)],
            dim=1,
        )                                              # (B, n_genes, 3)
        return logits

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def __repr__(self) -> str:
        return (
            f"GeneticEncoder(n_genes={self.n_genes}, "
            f"gene_embed_dim={self.gene_embed_dim}, "
            f"latent_dim={self.latent_dim}, "
            f"genes={self.gene_names})"
        )
