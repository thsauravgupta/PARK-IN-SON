# -*- coding: utf-8 -*-
import torch
from torch_geometric.data import Data
import numpy as np
import networkx as nx
import pandas as pd
from scipy.spatial.distance import pdist, squareform

class GraphBuilder:
    def __init__(self, n_nodes: int):
        self.n_nodes = n_nodes

    def build_graph(self, features: np.ndarray, threshold: float = 0.5) -> Data:
        """
        Builds a PyG Data object from tabular node features.
        features: shape (n_nodes,)
        """
        raise NotImplementedError

class MRIGraphBuilder(GraphBuilder):
    def __init__(self, n_nodes: int = 100):
        super().__init__(n_nodes)
        
    def build_graph(self, features: np.ndarray, threshold: float = 0.5) -> Data:
        """
        Constructs a structural graph from MRI regional volumes/thicknesses.
        Here we assume features is a vector of 100 regional volumes for a single patient.
        We build a fully connected graph or use a correlation threshold.
        For an individual patient with just 1 value per ROI, we can connect ROIs 
        based on absolute difference in normalized volume (homophily), 
        or simply build a complete graph and let the GNN learn edge weights via attention/GCN.
        Here we use a k-NN graph based on absolute feature difference.
        """
        # Node features (n_nodes, 1)
        x = torch.tensor(features, dtype=torch.float32).view(-1, 1)
        
        # Calculate pairwise absolute differences
        diffs = squareform(pdist(features.reshape(-1, 1), metric='euclidean'))
        
        # Connect each node to its k nearest neighbors in feature space (structural similarity)
        k = 10
        edge_index = []
        for i in range(self.n_nodes):
            # get top k indices with smallest difference (excluding self)
            nearest = np.argsort(diffs[i])[1:k+1]
            for j in nearest:
                edge_index.append([i, j])
                
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        return Data(x=x, edge_index=edge_index)

class PETGraphBuilder(GraphBuilder):
    def __init__(self):
        # 4 regions: Caudate L, Caudate R, Putamen L, Putamen R
        super().__init__(n_nodes=4)
        
    def build_graph(self, features: np.ndarray) -> Data:
        """
        Constructs a spatial graph for the 4 striatum sub-regions.
        features: [caudate_l, caudate_r, putamen_l, putamen_r]
        We build edges based on anatomical connections:
        L-Caudate <-> R-Caudate
        L-Putamen <-> R-Putamen
        L-Caudate <-> L-Putamen
        R-Caudate <-> R-Putamen
        """
        x = torch.tensor(features, dtype=torch.float32).view(4, 1)
        
        edges = [
            (0, 1), (1, 0), # L-Caudate <-> R-Caudate
            (2, 3), (3, 2), # L-Putamen <-> R-Putamen
            (0, 2), (2, 0), # L-Caudate <-> L-Putamen
            (1, 3), (3, 1)  # R-Caudate <-> R-Putamen
        ]
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return Data(x=x, edge_index=edge_index)
