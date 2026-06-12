# -*- coding: utf-8 -*-
"""
Cross-Modal Fusion Transformer.

Each modality embedding is treated as a token.  A learnable [FUSION] CLS
token aggregates all present modalities via Transformer self-attention and
produces the fused patient embedding.

Missing Modality Strategy
--------------------------
Patients frequently lack one or more modalities (e.g. a PPMI subject may
have clinical + genetic data but no MRI scan at baseline).

Two mechanisms handle this gracefully:

  1. **Learned [MISSING] token** — when a modality is absent for a patient,
     its embedding slot is filled with a per-modality learned vector
     (``missing_tokens``).  This is *not* zeros; it is a trainable parameter
     that the model can learn to "ignore" or to use as a prior.

  2. **Attention masking** — the ``src_key_padding_mask`` argument prevents
     the [FUSION] CLS token (and other modality tokens) from attending to the
     [MISSING] placeholder tokens.  The CLS token therefore only aggregates
     signal from *present* modalities.

  3. **Presence embedding** — a binary ``nn.Embedding(2, latent_dim)``
     (0=missing, 1=present) is added to each token so the model can
     explicitly condition on which modalities it is seeing.

Tokens layout (position order matters for the attention mask):
    pos 0 : [FUSION] CLS  — always unmasked
    pos 1 : modality_0 token
    pos 2 : modality_1 token
    ...
    pos M : modality_{M-1} token
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class FusionTransformer(nn.Module):
    """Cross-modal Transformer that fuses per-modality embeddings.

    Args:
        modality_names: Ordered list of modality identifiers (e.g.
                        ``["clinical", "mri", "pet", "genetic"]``).
                        The order is fixed across training and inference.
        latent_dim:     Dimensionality of each per-modality input embedding.
                        Must match the ``latent_dim`` of all modality encoders.
        fused_dim:      Dimensionality of the output [FUSION] CLS embedding.
        n_heads:        Number of Transformer attention heads.
        n_layers:       Number of Transformer encoder layers.
        ffn_dim:        Feed-forward network width inside each Transformer layer.
        dropout:        Dropout probability.
    """

    def __init__(
        self,
        modality_names: List[str],
        latent_dim: int = 128,
        fused_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.modality_names = modality_names
        self.n_modalities = len(modality_names)
        self.latent_dim = latent_dim
        self.fused_dim = fused_dim

        # Map modality name → canonical index (for embedding lookups)
        self._mod_to_idx = {name: i for i, name in enumerate(modality_names)}

        # Learnable [FUSION] CLS token (1 per model, broadcast over batch)
        self.fusion_cls = nn.Parameter(torch.zeros(1, 1, latent_dim))
        nn.init.trunc_normal_(self.fusion_cls, std=0.02)

        # Modality-type positional embeddings (one per modality slot)
        self.modality_type_embed = nn.Embedding(self.n_modalities, latent_dim)

        # Per-modality learned [MISSING] tokens
        # WHY ParameterDict?  Named access for easy inspection / freezing.
        self.missing_tokens = nn.ParameterDict(
            {name: nn.Parameter(torch.randn(latent_dim) * 0.02) for name in modality_names}
        )

        # Binary presence embedding: 0=missing, 1=present
        self.presence_embed = nn.Embedding(2, latent_dim)

        # Transformer encoder (Pre-LN for stability)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )

        # Project [FUSION] CLS output → fused_dim
        self.output_proj = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, fused_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.modality_type_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.presence_embed.weight, std=0.02)

    def forward(
        self,
        modality_embeddings: Dict[str, torch.Tensor],
        modality_mask: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuse per-modality embeddings into a single patient representation.

        Args:
            modality_embeddings:
                ``dict[modality_name -> (B, latent_dim)]`` — output of each
                modality encoder.  Modalities not in this dict are treated as
                fully absent for all samples.
            modality_mask:
                ``dict[modality_name -> (B,) bool tensor]`` where ``True``
                means the modality is present for that sample.
                If ``None``, all supplied modalities are assumed present.

        Returns:
            fused:          ``(B, fused_dim)`` — fused patient embedding.
            modality_tokens:``(B, n_modalities, latent_dim)`` — individual
                            modality token representations (post-attention,
                            useful for interpretability / reconstruction).
        """
        # Infer batch size and device from the first available embedding
        ref_tensor = next(iter(modality_embeddings.values()))
        B, device = ref_tensor.shape[0], ref_tensor.device

        # ---- Build token sequence ----------------------------------------
        # Sequence layout: [CLS | mod_0 | mod_1 | ... | mod_{M-1}]
        # key_padding_mask shape: (B, 1 + M)
        # True = "ignore this position in attention"
        # CLS (pos 0) is NEVER masked.
        key_padding_mask = torch.zeros(
            B, 1 + self.n_modalities, dtype=torch.bool, device=device
        )

        tokens: List[torch.Tensor] = []

        for i, name in enumerate(self.modality_names):
            if name in modality_embeddings:
                z = modality_embeddings[name]               # (B, latent_dim)

                # Determine per-sample presence
                if modality_mask is not None and name in modality_mask:
                    present = modality_mask[name].to(device=device, dtype=torch.bool)
                else:
                    present = torch.ones(B, dtype=torch.bool, device=device)
            else:
                # Entire modality absent — missing for all samples
                z = torch.zeros(B, self.latent_dim, device=device)
                present = torch.zeros(B, dtype=torch.bool, device=device)

            # Replace absent samples' embeddings with learned [MISSING] token
            missing_tok = self.missing_tokens[name].unsqueeze(0).expand(B, -1)
            z = torch.where(present.unsqueeze(-1), z, missing_tok)

            # Mark absent samples in the attention mask (pos i+1 because pos 0 = CLS)
            key_padding_mask[:, i + 1] = ~present

            # Add modality-type positional embedding
            mod_type_id = torch.tensor(i, device=device)
            z = z + self.modality_type_embed(mod_type_id)

            # Add binary presence embedding
            z = z + self.presence_embed(present.long())

            tokens.append(z)

        # Stack: (B, n_modalities, latent_dim)
        modality_tokens = torch.stack(tokens, dim=1)

        # Prepend [FUSION] CLS token
        cls = self.fusion_cls.expand(B, -1, -1)             # (B, 1, latent_dim)
        all_tokens = torch.cat([cls, modality_tokens], dim=1)  # (B, 1+M, latent_dim)

        # ---- Transformer with missing-modality masking -------------------
        out = self.transformer(all_tokens, src_key_padding_mask=key_padding_mask)
        # out: (B, 1+M, latent_dim)

        # Extract [FUSION] CLS (pos 0) and project to fused_dim
        cls_out = out[:, 0, :]                               # (B, latent_dim)
        fused = self.output_proj(cls_out)                    # (B, fused_dim)

        return fused, modality_tokens
