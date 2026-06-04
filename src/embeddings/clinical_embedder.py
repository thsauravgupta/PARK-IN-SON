# -*- coding: utf-8 -*-
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import joblib

from src.embeddings.base_embedder import BaseEmbedder

class ClinicalAutoencoder(nn.Module):
    def __init__(self, input_dim: int, bottleneck_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, bottleneck_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class ClinicalEmbedder(BaseEmbedder):
    """
    Generates 32-dimensional clinical embeddings.
    """
    def __init__(self, mode: str = 'pca', config: dict = None):
        self.mode = mode
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self._output_dim = 32
        
        if self.mode == 'pca':
            self.model = Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=self._output_dim, whiten=True))
            ])
        elif self.mode == 'autoencoder':
            self.model = None
            self.scaler = StandardScaler()
        else:
            raise ValueError("Mode must be 'pca' or 'autoencoder'")

    def fit(self, X_train: pd.DataFrame) -> 'BaseEmbedder':
        if self.mode == 'pca':
            self.model.fit(X_train.values)
            self.logger.info("PCA fitted.")
        else:
            X_scaled = self.scaler.fit_transform(X_train.values)
            input_dim = X_scaled.shape[1]
            self.model = ClinicalAutoencoder(input_dim, self._output_dim)
            
            # Simple training loop
            optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
            criterion = nn.MSELoss()
            
            X_tensor = torch.FloatTensor(X_scaled)
            self.model.train()
            
            best_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(100):
                optimizer.zero_grad()
                _, decoded = self.model(X_tensor)
                loss = criterion(decoded, X_tensor)
                loss.backward()
                optimizer.step()
                
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= 10:
                    self.logger.info(f"Early stopping at epoch {epoch}. Loss: {best_loss:.4f}")
                    break
                    
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        patnos = X.index
        if self.mode == 'pca':
            embeds = self.model.transform(X.values)
        else:
            self.model.eval()
            X_scaled = self.scaler.transform(X.values)
            with torch.no_grad():
                embeds, _ = self.model(torch.FloatTensor(X_scaled))
            embeds = embeds.numpy()
            
        return pd.DataFrame(embeds, index=patnos, columns=[f"Clin_{i}" for i in range(self._output_dim)])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.mode == 'pca':
            joblib.dump(self.model, path.with_suffix('.joblib'))
        else:
            torch.save({
                'scaler': self.scaler,
                'model_state': self.model.state_dict(),
                'input_dim': self.model.encoder[0].in_features
            }, path.with_suffix('.pt'))

    @classmethod
    def load(cls, path: Path) -> 'BaseEmbedder':
        raise NotImplementedError

    @property
    def output_dim(self) -> int:
        return self._output_dim
