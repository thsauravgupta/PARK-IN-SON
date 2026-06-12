# -*- coding: utf-8 -*-
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import DataLoader
from sklearn.preprocessing import StandardScaler
import joblib

from src.embeddings.base_embedder import BaseEmbedder
from src.data.graph_builder import PETGraphBuilder
from src.data.augmentation import apply_dropedge

class PETGraphAttentionAutoencoder(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=8, out_channels=4, heads=2):
        super().__init__()
        # GAT encodes the 4 regional nodes into a richer latent space via attention
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        # 8 hidden * 2 heads = 16
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=True)
        self.decoder = nn.Linear(out_channels, in_channels)
        
    def encode(self, x, edge_index):
        x = F.elu(self.gat1(x, edge_index))
        return self.gat2(x, edge_index)
        
    def decode(self, z):
        return self.decoder(z)
        
    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        return self.decode(z), z

class PETEmbedder(BaseEmbedder):
    """
    Generates 16-dimensional PET embeddings using a Graph Attention Network (GAT).
    We model the 4 striatum subregions as a graph and extract an attention-weighted latent graph vector.
    """
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self._output_dim = 16  # 4 nodes * 4 out_channels per node = 16
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.scaler = StandardScaler()
        self.builder = PETGraphBuilder()
        self.model = PETGraphAttentionAutoencoder(in_channels=1, out_channels=4).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()

    def fit(self, X_train: pd.DataFrame) -> 'BaseEmbedder':
        self.logger.info(f"Training PET GAT Autoencoder on {self.device}...")
        
        # PET Preprocessor outputs 10 engineered features, but the Graph Builder
        # expects the 4 base regions: caudate_l, caudate_r, putamen_l, putamen_r
        # We need to extract just those 4 base features for the nodes.
        # If they aren't present (because the preprocessor already engineered them and dropped),
        # we will use the first 4 columns as a fallback
        base_cols = ["caudate_l", "caudate_r", "putamen_l", "putamen_r"]
        cols_to_use = [c for c in base_cols if c in X_train.columns]
        if len(cols_to_use) < 4:
            cols_to_use = X_train.columns[:4]
            
        X_base = X_train[cols_to_use].values
        self.scaler.fit(X_base)
        X_scaled = self.scaler.transform(X_base)
        
        epochs = self.config.get("autoencoder", {}).get("pet", {}).get("epochs", 50)
        from torch_geometric.loader import DataLoader
        
        # Pre-build all graphs
        self.logger.info("Pre-building PET graphs...")
        graph_list = [self.builder.build_graph(features) for features in X_scaled]
        loader = DataLoader(graph_list, batch_size=64, shuffle=True)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for data in loader:
                data = data.to(self.device)
                
                edge_index = apply_dropedge(data.edge_index, p=0.05, training=True)
                
                self.optimizer.zero_grad()
                reconstructed, _ = self.model(data.x, edge_index)
                
                loss = self.criterion(reconstructed, data.x)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * data.num_graphs
                
            if (epoch + 1) % 10 == 0:
                self.logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(X_scaled):.4f}")
                
        self.logger.info("PET GAT Embedder fitted.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        patnos = X.index
        base_cols = ["caudate_l", "caudate_r", "putamen_l", "putamen_r"]
        cols_to_use = [c for c in base_cols if c in X.columns]
        if len(cols_to_use) < 4:
            cols_to_use = X.columns[:4]
            
        X_base = X[cols_to_use].values
        X_scaled = self.scaler.transform(X_base)
        from torch_geometric.loader import DataLoader
        
        graph_list = [self.builder.build_graph(features) for features in X_scaled]
        loader = DataLoader(graph_list, batch_size=64, shuffle=False)
        
        self.model.eval()
        embeds = []
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                z = self.model.encode(data.x, data.edge_index)
                
                # Reshape from (batch_size*4, 4) to (batch_size, 16)
                z_reshaped = z.view(-1, 16).cpu().numpy()
                embeds.extend(z_reshaped)
                
        embeds = np.array(embeds)
        return pd.DataFrame(embeds, index=patnos, columns=[f"PET_{i}" for i in range(self._output_dim)])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path.with_suffix('.pt'))
        joblib.dump(self.scaler, path.with_suffix('.scaler.joblib'))

    @classmethod
    def load(cls, path: Path) -> 'BaseEmbedder':
        instance = cls()
        instance.model.load_state_dict(torch.load(path.with_suffix('.pt')))
        instance.scaler = joblib.load(path.with_suffix('.scaler.joblib'))
        return instance

    @property
    def output_dim(self) -> int:
        return self._output_dim
