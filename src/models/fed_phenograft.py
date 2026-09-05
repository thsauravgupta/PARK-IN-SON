import torch
import torch.nn as nn
from src.models.attention import AsymmetricCrossAttention
from src.models.hsic import HSICLoss

class SharedPrivateEncoder(nn.Module):
    def __init__(self, input_dim, shared_dim, private_dim, dropout=0.2):
        super().__init__()
        self.shared_mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, shared_dim)
        )
        self.private_mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, private_dim)
        )
        
    def forward(self, x):
        return self.shared_mlp(x), self.private_mlp(x)

class FedPhenoGraft(nn.Module):
    """
    Fed-PhenoGraft Main Model.
    Includes Mask Tokens for missing modalities, MC Dropout, Shared-Private embeddings, 
    and Asymmetric Attention guided by Clinical embeddings.
    """
    def __init__(self, input_dims, embed_dim=32, num_heads=4, dropout=0.2,
                 use_attention=True):
        super().__init__()
        self.use_attention = use_attention

        # Clinical Encoder (Primary Phenotype)
        self.clin_encoder = nn.Sequential(
            nn.Linear(input_dims['clinical'], 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, embed_dim)
        )
        
        # Decomposed Encoders
        self.mri_enc = SharedPrivateEncoder(input_dims['mri'], embed_dim, embed_dim, dropout)
        self.pet_enc = SharedPrivateEncoder(input_dims['pet'], embed_dim, embed_dim, dropout)
        self.gen_enc = SharedPrivateEncoder(input_dims['genetic'], embed_dim, embed_dim, dropout)
        
        # Modality Missing Mask Tokens
        self.mri_mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pet_mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.gen_mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Asymmetric Attention Banks
        self.mri_attn = AsymmetricCrossAttention(embed_dim, num_heads, dropout)
        self.pet_attn = AsymmetricCrossAttention(embed_dim, num_heads, dropout)
        self.gen_attn = AsymmetricCrossAttention(embed_dim, num_heads, dropout)
        
        # Final Disease Progression Regression Head
        self.prediction_head = nn.Sequential(
            nn.Linear(embed_dim * 4, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

        # PD vs HC Classification Head (multitask, shares the fused representation)
        self.classification_head = nn.Sequential(
            nn.Linear(embed_dim * 4, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        self.hsic_loss = HSICLoss(sigma=1.0)

    def enable_mc_dropout(self):
        """Active Dropout during inference for Bayesian uncertainty calculation."""
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    def forward(self, batch):
        N = batch['clinical'].size(0)
        clin_q = self.clin_encoder(batch['clinical']).unsqueeze(1) # (N, 1, E)
        
        # Extract and Mask target modalities
        mri_shared, mri_private = self.mri_enc(batch['mri'])
        mri_shared = mri_shared.unsqueeze(1)
        mri_mask = batch['mri_mask'].view(N, 1, 1).expand(-1, 1, mri_shared.size(-1))
        mri_shared = mri_shared * (1 - mri_mask) + self.mri_mask_token.expand(N, -1, -1) * mri_mask
        
        pet_shared, pet_private = self.pet_enc(batch['pet'])
        pet_shared = pet_shared.unsqueeze(1)
        pet_mask = batch['pet_mask'].view(N, 1, 1).expand(-1, 1, pet_shared.size(-1))
        pet_shared = pet_shared * (1 - pet_mask) + self.pet_mask_token.expand(N, -1, -1) * pet_mask
        
        gen_shared, gen_private = self.gen_enc(batch['genetic'])
        gen_shared = gen_shared.unsqueeze(1)
        gen_mask = batch['genetic_mask'].view(N, 1, 1).expand(-1, 1, gen_shared.size(-1))
        gen_shared = gen_shared * (1 - gen_mask) + self.gen_mask_token.expand(N, -1, -1) * gen_mask
        
        # Multi-modal fusion via Querying (ablation: plain concat when disabled)
        if self.use_attention:
            mri_out, mri_attn_weights = self.mri_attn(clin_q, mri_shared)
            pet_out, pet_attn_weights = self.pet_attn(clin_q, pet_shared)
            gen_out, gen_attn_weights = self.gen_attn(clin_q, gen_shared)
        else:
            mri_out, pet_out, gen_out = mri_shared, pet_shared, gen_shared
            mri_attn_weights = pet_attn_weights = gen_attn_weights = None
        
        fused = torch.cat([clin_q.squeeze(1), mri_out.squeeze(1), pet_out.squeeze(1), gen_out.squeeze(1)], dim=1)
        pred = self.prediction_head(fused)
        cls_logit = self.classification_head(fused)
        
        # Accumulate Orthogonality constraints
        hsic = self.hsic_loss(mri_shared.squeeze(1), mri_private) + \
               self.hsic_loss(pet_shared.squeeze(1), pet_private) + \
               self.hsic_loss(gen_shared.squeeze(1), gen_private)
                    
        return {
            'pred': pred.squeeze(-1),
            'cls_logit': cls_logit.squeeze(-1),
            'loss_hsic': hsic,
            'attn_weights': {
                'mri': mri_attn_weights,
                'pet': pet_attn_weights,
                'genetic': gen_attn_weights
            }
        }
