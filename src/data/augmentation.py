# -*- coding: utf-8 -*-
import torch
from torch_geometric.utils import dropout_edge

def apply_dropedge(edge_index: torch.Tensor, p: float = 0.05, training: bool = True) -> torch.Tensor:
    """
    Randomly drops edges from the graph during training to prevent overfitting.
    This acts as a structural regularizer for GNNs.
    """
    if not training or p == 0.0:
        return edge_index
        
    # PyG dropout_edge returns (edge_index, edge_mask)
    edge_index, _ = dropout_edge(edge_index, p=p, force_undirected=True)
    return edge_index

def apply_node_masking(x: torch.Tensor, p: float = 0.05, training: bool = True) -> torch.Tensor:
    """
    Randomly masks node features to zero during training to force the network
    to rely on spatial neighbors rather than individual node memorization.
    """
    if not training or p == 0.0:
        return x
        
    # Create mask of same shape as x
    mask = torch.rand_like(x) > p
    return x * mask.float()
