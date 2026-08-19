import torch
import torch.nn as nn

class AsymmetricCrossAttention(nn.Module):
    """
    Phenotype-Guided Asymmetric Cross Attention.
    Clinical Phenotypes act as the Query to fetch relevant Keys and Values from Structural/Genetic modalities.
    """
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        # Batch first ensures shape (N, L, E)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, query, kv):
        # query: (N, target_seq_len (1), E)
        # kv: (N, source_seq_len, E)
        # Weights returned are useful for Explainability (XAI)
        attn_out, attn_weights = self.mha(query, kv, kv)
        # Add & Norm
        return self.norm(query + attn_out), attn_weights
