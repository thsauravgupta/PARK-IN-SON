# -*- coding: utf-8 -*-
"""
GATv2 Patient Graph Neural Network.

WHY a GNN on top of the Fusion Transformer?

  * The FusionTransformer operates *per-patient* — each patient's embedding
    is computed independently.  The GNN introduces *population-level context*:
    patients with similar profiles share information.

  * This is especially valuable for:
      - **Rare PD subtypes** (few labelled examples — borrow signal from
        phenotypically similar patients).
      - **Missing modality imputation** (impute missing modality features
        from k-nearest neighbours whose embeddings are already rich).
      - **Digital twin concept**: your twin = your nearest neighbours in
        the patient graph.  Once the graph is built, you can ask "who
        are the patients most similar to this new patient?"

WHY GATv2 over vanilla GCN?
  * GATv2 (Brody et al., 2021) uses a *dynamic* attention mechanism:
    attention coefficients depend on both node features at training time,
    making it strictly more expressive than GATv1 (whose attention is
    limited to a linear function of the concatenated features).
  * Interpretable: edge attention weights reveal which "neighbour patients"
    most influenced each patient's final embedding.

Implementation Notes
--------------------
This implementation uses *edge-index sparse* message-passing so it is
memory-efficient for large cohorts (no dense N×N attention matrices).
The adjacency is built on-the-fly as a k-NN graph in cosine similarity
space, dynamically updated as embeddings improve during training.

For very large cohorts (N > 5 000) consider replacing with the PyTorch
Geometric GATv2Conv which supports mini-batch graph sampling.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATv2Layer(nn.Module):
    """Single GATv2 message-passing layer.

    Reference: Brody et al., "How Attentive are Graph Attention Networks?",
    ICLR 2022.  https://arxiv.org/abs/2105.14491

    Uses edge-index representation (sparse) for memory efficiency.

    Args:
        in_dim:         Input node feature dimensionality.
        out_dim:        Output node feature dimensionality per head.
        n_heads:        Number of attention heads.
        dropout:        Attention coefficient dropout probability.
        negative_slope: LeakyReLU negative slope.
        concat:         If True, concatenate head outputs (→ n_heads * out_dim).
                        If False, average head outputs (→ out_dim).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        negative_slope: float = 0.2,
        concat: bool = True,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = out_dim   # per-head output dim
        self.concat = concat
        # When concat=True  → actual output dim = n_heads * out_dim
        # When concat=False → actual output dim = out_dim (after avg projection)
        self.actual_out_dim = n_heads * out_dim if concat else out_dim

        # Linear projections for source and destination nodes
        self.W_src = nn.Linear(in_dim, n_heads * out_dim, bias=False)
        self.W_dst = nn.Linear(in_dim, n_heads * out_dim, bias=False)

        # Per-head attention vector (learnable)
        self.attn_vec = nn.Parameter(torch.empty(1, n_heads, out_dim))
        nn.init.xavier_uniform_(self.attn_vec.view(1, -1).unsqueeze(0))

        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.dropout = nn.Dropout(dropout)

        # Output projection when not concatenating
        if not concat:
            self.out_proj = nn.Linear(n_heads * out_dim, out_dim)

        nn.init.xavier_uniform_(self.W_src.weight)
        nn.init.xavier_uniform_(self.W_dst.weight)

    def forward(
        self,
        x: torch.Tensor,          # (N, in_dim)
        edge_index: torch.Tensor,  # (2, E)  [src, dst]
    ) -> torch.Tensor:             # (N, n_heads * head_dim) or (N, head_dim)
        N = x.shape[0]
        src_idx, dst_idx = edge_index[0], edge_index[1]  # each: (E,)

        # Linear transforms → (N, n_heads, head_dim)
        h_src = self.W_src(x).view(N, self.n_heads, self.head_dim)
        h_dst = self.W_dst(x).view(N, self.n_heads, self.head_dim)

        # Gather per-edge features
        h_src_e = h_src[src_idx]   # (E, n_heads, head_dim)
        h_dst_e = h_dst[dst_idx]   # (E, n_heads, head_dim)

        # GATv2 attention score: e_ij = a ⊙ LeakyReLU(W_src * h_i + W_dst * h_j)
        e = (self.attn_vec * self.leaky_relu(h_src_e + h_dst_e)).sum(-1)  # (E, n_heads)

        # Softmax over in-edges for each destination node (scatter softmax)
        # --- numerically stable: subtract per-destination max ---
        e_max = torch.full((N, self.n_heads), float("-inf"), device=x.device)
        e_max.scatter_reduce_(
            0,
            dst_idx.unsqueeze(-1).expand(-1, self.n_heads),
            e,
            reduce="amax",
            include_self=True,
        )
        e_shifted = e - e_max[dst_idx]      # (E, n_heads)
        e_exp = torch.exp(e_shifted)

        # Sum of exponentials per destination
        sum_exp = torch.zeros(N, self.n_heads, device=x.device)
        sum_exp.scatter_add_(
            0,
            dst_idx.unsqueeze(-1).expand(-1, self.n_heads),
            e_exp,
        )

        alpha = e_exp / (sum_exp[dst_idx] + 1e-8)   # (E, n_heads)
        alpha = self.dropout(alpha)

        # Aggregate messages: out_j = Σ_i  α_ij · (W_src · h_i)
        weighted = h_src_e * alpha.unsqueeze(-1)      # (E, n_heads, head_dim)
        out = torch.zeros(N, self.n_heads, self.head_dim, device=x.device)
        out.scatter_add_(
            0,
            dst_idx.view(-1, 1, 1).expand(-1, self.n_heads, self.head_dim),
            weighted,
        )

        # Reshape output
        if self.concat:
            return out.reshape(N, self.n_heads * self.head_dim)
        else:
            return self.out_proj(out.reshape(N, self.n_heads * self.head_dim))


class PatientGNN(nn.Module):
    """GATv2 Graph Neural Network over a patient similarity graph.

    **Graph construction**: a k-nearest-neighbour graph is built dynamically
    from the fused patient embeddings using cosine similarity.  Because the
    graph is rebuilt each forward pass, it evolves as embeddings improve
    during training — the patient graph is "alive."

    Args:
        node_dim:      Input node feature dimensionality (= FusionTransformer
                       output fused_dim).
        out_dim:       Output node feature dimensionality.
        k_neighbours:  Number of nearest neighbours per patient.
        n_heads:       Number of GAT attention heads.
        n_layers:      Number of GATv2 message-passing layers.
        dropout:       Dropout probability.
    """

    def __init__(
        self,
        node_dim: int = 256,
        out_dim: int = 256,
        k_neighbours: int = 10,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.k = k_neighbours
        self.node_dim = node_dim
        self.out_dim = out_dim

        # Stack of GATv2 layers — track actual dims carefully
        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        in_d = node_dim
        for layer_idx in range(n_layers):
            is_last = layer_idx == n_layers - 1
            # head_dim: per-head output size
            # - last layer: concat=False → actual out = out_dim, so head_dim = out_dim
            # - other layers: concat=True → actual out = n_heads * head_dim
            #   choose head_dim = out_dim so actual out = n_heads * out_dim
            #   (but we want actual out = out_dim, so set head_dim = out_dim // n_heads)
            if is_last:
                head_dim = out_dim   # concat=False → out_proj maps n_heads*head_dim → out_dim
                concat = False
                actual_out = out_dim
            else:
                head_dim = out_dim // n_heads   # concat=True → actual = n_heads*(out_dim//n_heads) = out_dim
                concat = True
                actual_out = n_heads * head_dim   # = out_dim (if out_dim divisible by n_heads)

            self.gat_layers.append(
                GATv2Layer(
                    in_dim=in_d,
                    out_dim=head_dim,
                    n_heads=n_heads,
                    dropout=dropout,
                    concat=concat,
                )
            )
            self.norms.append(nn.LayerNorm(actual_out))
            in_d = actual_out

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # Residual projection (if input/output dims differ)
        self.residual_proj = (
            nn.Linear(node_dim, out_dim) if node_dim != out_dim else nn.Identity()
        )

    def build_knn_graph(self, x: torch.Tensor) -> torch.Tensor:
        """Construct k-NN edge index from cosine similarity.

        Args:
            x: ``(N, d)`` node features.

        Returns:
            ``(2, E)`` edge index where E ≤ N × k.
        """
        N = x.shape[0]
        k = min(self.k, N - 1)

        x_norm = F.normalize(x.detach(), p=2, dim=-1)
        sim = x_norm @ x_norm.T          # (N, N)
        sim.fill_diagonal_(float("-inf"))

        # Top-k neighbours per node (column = source, row = destination)
        _, topk_idx = sim.topk(k, dim=1)   # (N, k)

        # Build edge list: each node i → its k nearest neighbours
        src = torch.arange(N, device=x.device).unsqueeze(1).expand(N, k).reshape(-1)
        dst = topk_idx.reshape(-1)

        # Symmetrise: add reverse edges
        edge_index = torch.cat(
            [torch.stack([src, dst], dim=0),
             torch.stack([dst, src], dim=0)],
            dim=1,
        )
        # Deduplicate
        edge_index = torch.unique(edge_index, dim=1)
        return edge_index

    def forward(
        self,
        x: torch.Tensor,                           # (N, node_dim)
        edge_index: Optional[torch.Tensor] = None,  # (2, E) or None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """GNN forward pass.

        Args:
            x:          ``(N, node_dim)`` node features (patient embeddings).
            edge_index: Pre-built edge index.  If ``None``, a k-NN graph is
                        constructed automatically.

        Returns:
            z_gnn:      ``(N, out_dim)`` GNN-refined patient embeddings.
            edge_index: ``(2, E)`` edge index used (useful for inspection).
        """
        if edge_index is None:
            edge_index = self.build_knn_graph(x)

        h = x
        residual = self.residual_proj(x)

        for gat, norm in zip(self.gat_layers, self.norms):
            h = gat(h, edge_index)
            h = norm(h)
            h = self.act(h)
            h = self.dropout(h)

        # Residual skip connection
        h = h + residual

        return h, edge_index
