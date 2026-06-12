# -*- coding: utf-8 -*-
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import joblib

from src.embeddings.base_embedder import BaseEmbedder
from src.data.graph_builder import MRIGraphBuilder
from src.data.augmentation import apply_dropedge, apply_node_masking

class MRIGraphAutoencoder(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=128, out_channels=64):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.decoder = nn.Linear(out_channels, in_channels)
        
    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)
        
    def decode(self, z):
        return self.decoder(z)
        
    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        return self.decode(z), z

class MRIEmbedder(BaseEmbedder):
    """
    Generates 64-dimensional MRI embeddings using a Graph Convolutional Autoencoder.
    """
    def __init__(self, mode: str = 'gae', config: dict = None):
        self.mode = mode
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self._output_dim = 64
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.scaler = StandardScaler()
        self.builder = MRIGraphBuilder(n_nodes=self.config.get("mri", {}).get("n_rois", 100))
        self.model = MRIGraphAutoencoder(in_channels=1, out_channels=self._output_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005)
        self.criterion = nn.MSELoss()

    def fit(self, X_train: pd.DataFrame) -> 'BaseEmbedder':
        self.logger.info(f"Training MRI GAE on {self.device}...")
        self.scaler.fit(X_train.values)
        X_scaled = self.scaler.transform(X_train.values)
        
        epochs = self.config.get("autoencoder", {}).get("mri", {}).get("epochs", 50)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(X_scaled)):
                features = X_scaled[i]
                data = self.builder.build_graph(features)
                data = data.to(self.device)
                
                # Apply Graph Augmentations
                edge_index = apply_dropedge(data.edge_index, p=0.1, training=True)
                x = apply_node_masking(data.x, p=0.1, training=True)
                
                self.optimizer.zero_grad()
                reconstructed, _ = self.model(x, edge_index)
                
                loss = self.criterion(reconstructed, data.x) # reconstruct original x
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                
            if (epoch + 1) % 10 == 0:
                self.logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(X_scaled):.4f}")
                
        self.logger.info("MRI GAE Embedder fitted.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        patnos = X.index
        X_scaled = self.scaler.transform(X.values)
        self.model.eval()
        
        embeds = []
        with torch.no_grad():
            for i in range(len(X_scaled)):
                features = X_scaled[i]
                data = self.builder.build_graph(features)
                data = data.to(self.device)
                z = self.model.encode(data.x, data.edge_index)
                # Global mean pooling over nodes to get graph-level embedding
                graph_emb = z.mean(dim=0).cpu().numpy()
                embeds.append(graph_emb)
                
        embeds = np.array(embeds)
        return pd.DataFrame(embeds, index=patnos, columns=[f"MRI_{i}" for i in range(self._output_dim)])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Save PyTorch model state and sklearn scaler
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
