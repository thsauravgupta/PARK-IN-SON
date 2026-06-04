# -*- coding: utf-8 -*-
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from src.embeddings.base_embedder import BaseEmbedder

class PETEmbedder(BaseEmbedder):
    """
    Generates 16-dimensional PET embeddings (mostly padding to standard size).
    """
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self._output_dim = 16

    def fit(self, X_train: pd.DataFrame) -> 'BaseEmbedder':
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        patnos = X.index
        # Simply pad the 10 features to 16
        vals = X.values
        if vals.shape[1] < self._output_dim:
            padded = np.pad(vals, ((0, 0), (0, self._output_dim - vals.shape[1])), mode='constant')
        else:
            padded = vals[:, :self._output_dim]
            
        return pd.DataFrame(padded, index=patnos, columns=[f"PET_{i}" for i in range(self._output_dim)])

    def save(self, path: Path) -> None:
        pass

    @classmethod
    def load(cls, path: Path) -> 'BaseEmbedder':
        return cls()

    @property
    def output_dim(self) -> int:
        return self._output_dim
