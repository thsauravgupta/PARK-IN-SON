# -*- coding: utf-8 -*-
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

from src.embeddings.base_embedder import BaseEmbedder

class MRIEmbedder(BaseEmbedder):
    """
    Generates 64-dimensional MRI embeddings.
    """
    def __init__(self, mode: str = 'pca', config: dict = None):
        self.mode = mode
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self._output_dim = 64
        
        if self.mode == 'pca':
            self.model = Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=self._output_dim, whiten=True))
            ])
        else:
            # For simplicity, we just fallback to PCA if torch sparse AE is not implemented here yet
            self.logger.warning("Sparse AE not fully implemented, falling back to PCA")
            self.model = Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=self._output_dim, whiten=True))
            ])

    def fit(self, X_train: pd.DataFrame) -> 'BaseEmbedder':
        self.model.fit(X_train.values)
        self.logger.info("MRI Embedder fitted.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        patnos = X.index
        embeds = self.model.transform(X.values)
        return pd.DataFrame(embeds, index=patnos, columns=[f"MRI_{i}" for i in range(self._output_dim)])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path.with_suffix('.joblib'))

    @classmethod
    def load(cls, path: Path) -> 'BaseEmbedder':
        raise NotImplementedError

    @property
    def output_dim(self) -> int:
        return self._output_dim
