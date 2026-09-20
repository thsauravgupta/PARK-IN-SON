
# -*- coding: utf-8 -*-

"""
Fed-PhenoGraft multimodal Parkinson's progression model.

Modalities:
    Clinical
    FreeSurfer-derived structural MRI
    DaTScan/PET
    Genetics

MRI:
    Regional FreeSurfer measurements are encoded using an
    anatomy-informed GNN.

Global FreeSurfer measurements are encoded using an MLP.

The MRI representation then participates in the existing
phenotype-guided cross-modal attention and shared/private
latent decomposition.
"""

import torch
import torch.nn as nn

from src.models.attention import AsymmetricCrossAttention
from src.models.hsic import HSICLoss
from src.models.mri_encoder import MRIEncoder


class SharedPrivateEncoder(nn.Module):

    """
    Generic shared/private encoder for PET and genetics.
    """

    def __init__(
        self,
        input_dim,
        shared_dim,
        private_dim,
        dropout=0.2,
    ):

        super().__init__()

        self.shared_mlp = nn.Sequential(

            nn.Linear(
                input_dim,
                64,
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                64,
                shared_dim,
            ),
        )

        self.private_mlp = nn.Sequential(

            nn.Linear(
                input_dim,
                64,
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                64,
                private_dim,
            ),
        )

    def forward(
        self,
        x,
    ):

        return (
            self.shared_mlp(x),
            self.private_mlp(x),
        )


class FedPhenoGraft(nn.Module):

    """
    Main Fed-PhenoGraft model.

    MRI changes:
        Previous:
            FreeSurfer vector → MLP

        New:
            FreeSurfer regional features
                    ↓
            anatomy-informed GNN
                    +
            global MRI MLP
                    ↓
              MRI embedding

    Missing modalities are handled through learned mask tokens.

    HSIC is calculated only on subjects where the modality is actually
    observed.
    """

    def __init__(
        self,
        input_dims,
        embed_dim=32,
        num_heads=4,
        dropout=0.35,
        use_attention=True,
        mri_feature_names=None,
        mri_graph_features=None,
        mri_gnn_hidden=64,
    ):

        super().__init__()

        self.use_attention = use_attention

        # ================================================================
        # Clinical phenotype encoder
        # ================================================================

        self.clin_encoder = nn.Sequential(

            nn.Linear(
                input_dims["clinical"],
                64,
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                64,
                embed_dim,
            ),
        )

        # ================================================================
        # MRI GNN encoder
        # ================================================================

        self.mri_enc = MRIEncoder(

            input_dim=input_dims["mri"],

            embed_dim=embed_dim,

            dropout=dropout,

            feature_names=mri_feature_names,

            graph_features=mri_graph_features,

            hidden_dim=mri_gnn_hidden,
        )

        # ================================================================
        # PET
        # ================================================================

        self.pet_enc = SharedPrivateEncoder(

            input_dims["pet"],

            embed_dim,

            embed_dim,

            dropout,
        )

        # ================================================================
        # Genetics
        # ================================================================

        self.gen_enc = SharedPrivateEncoder(

            input_dims["genetic"],

            embed_dim,

            embed_dim,

            dropout,
        )

        # ================================================================
        # Missing-modality tokens
        # ================================================================

        self.mri_mask_token = nn.Parameter(
            torch.randn(
                1,
                1,
                embed_dim,
            ) * 0.02
        )

        self.pet_mask_token = nn.Parameter(
            torch.randn(
                1,
                1,
                embed_dim,
            ) * 0.02
        )

        self.gen_mask_token = nn.Parameter(
            torch.randn(
                1,
                1,
                embed_dim,
            ) * 0.02
        )

        # ================================================================
        # Phenotype-guided asymmetric attention
        # ================================================================

        self.mri_attn = AsymmetricCrossAttention(
            embed_dim,
            num_heads,
            dropout,
        )

        self.pet_attn = AsymmetricCrossAttention(
            embed_dim,
            num_heads,
            dropout,
        )

        self.gen_attn = AsymmetricCrossAttention(
            embed_dim,
            num_heads,
            dropout,
        )

        # ================================================================
        # Prediction heads
        # ================================================================

        fused_dim = embed_dim * 4

        self.prediction_head = nn.Sequential(

            nn.Linear(
                fused_dim,
                128,
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                128,
                1,
            ),
        )

        self.classification_head = nn.Sequential(

            nn.Linear(
                fused_dim,
                64,
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                64,
                1,
            ),
        )

        # ================================================================
        # HSIC
        # ================================================================

        self.hsic_loss = HSICLoss(
            sigma=1.0
        )

    def enable_mc_dropout(self):

        """
        Enable dropout during inference for MC-dropout uncertainty analysis.
        """

        for module in self.modules():

            if isinstance(
                module,
                nn.Dropout,
            ):

                module.train()

    @staticmethod
    def _apply_mask(
        shared,
        mask,
        token,
    ):

        """
        Replace missing modality representations with a learned token.
        """

        batch_size = shared.size(0)

        mask = mask.view(
            batch_size,
            1,
            1,
        ).to(
            shared.dtype
        )

        return (
            shared * (1.0 - mask)
            +
            token.expand(
                batch_size,
                -1,
                -1,
            ) * mask
        )

    @staticmethod
    def _masked_hsic(
        loss_fn,
        shared,
        private,
        missing_mask,
    ):

        """
        Calculate HSIC only on observed modality samples.

        This prevents missing-modality mask-token representations from
        being treated as real MRI/PET/genetic observations.
        """

        observed = (
            missing_mask.view(-1) < 0.5
        )

        if int(observed.sum().item()) < 2:

            return shared.new_tensor(
                0.0
            )

        return loss_fn(
            shared[observed],
            private[observed],
        )

    def forward(
        self,
        batch,
    ):

        n = batch["clinical"].size(0)

        # ================================================================
        # Clinical query
        # ================================================================

        clinical_query = self.clin_encoder(
            batch["clinical"]
        ).unsqueeze(1)

        # ================================================================
        # Encode all modalities
        # ================================================================

        mri_shared_raw, mri_private = self.mri_enc(
            batch["mri"]
        )

        pet_shared_raw, pet_private = self.pet_enc(
            batch["pet"]
        )

        gen_shared_raw, gen_private = self.gen_enc(
            batch["genetic"]
        )

        # ================================================================
        # Apply missing modality tokens
        # ================================================================

        mri_shared = self._apply_mask(

            mri_shared_raw.unsqueeze(1),

            batch["mri_mask"],

            self.mri_mask_token,
        )

        pet_shared = self._apply_mask(

            pet_shared_raw.unsqueeze(1),

            batch["pet_mask"],

            self.pet_mask_token,
        )

        gen_shared = self._apply_mask(

            gen_shared_raw.unsqueeze(1),

            batch["genetic_mask"],

            self.gen_mask_token,
        )

        # ================================================================
        # Phenotype-guided multimodal attention
        # ================================================================

        if self.use_attention:

            mri_out, mri_attn_weights = self.mri_attn(
                clinical_query,
                mri_shared,
            )

            pet_out, pet_attn_weights = self.pet_attn(
                clinical_query,
                pet_shared,
            )

            gen_out, gen_attn_weights = self.gen_attn(
                clinical_query,
                gen_shared,
            )

        else:

            mri_out = mri_shared
            pet_out = pet_shared
            gen_out = gen_shared

            mri_attn_weights = None
            pet_attn_weights = None
            gen_attn_weights = None

        # ================================================================
        # Multimodal fusion
        # ================================================================

        fused = torch.cat(

            [
                clinical_query.squeeze(1),

                mri_out.squeeze(1),

                pet_out.squeeze(1),

                gen_out.squeeze(1),
            ],

            dim=1,
        )

        # ================================================================
        # Outputs
        # ================================================================

        prediction = self.prediction_head(
            fused
        )

        classification_logit = self.classification_head(
            fused
        )

        # ================================================================
        # HSIC
        # ================================================================

        hsic = (

            self._masked_hsic(
                self.hsic_loss,
                mri_shared_raw,
                mri_private,
                batch["mri_mask"],
            )

            +

            self._masked_hsic(
                self.hsic_loss,
                pet_shared_raw,
                pet_private,
                batch["pet_mask"],
            )

            +

            self._masked_hsic(
                self.hsic_loss,
                gen_shared_raw,
                gen_private,
                batch["genetic_mask"],
            )
        )

        return {

            "pred": prediction.squeeze(-1),

            "cls_logit": classification_logit.squeeze(-1),

            "loss_hsic": hsic,

            "attn_weights": {

                "mri": mri_attn_weights,

                "pet": pet_attn_weights,

                "genetic": gen_attn_weights,
            },
        }