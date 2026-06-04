# -*- coding: utf-8 -*-
"""
Abstract base class for all modality-specific embedding generators.

WHY: Enforces a consistent API across modalities (clinical, MRI, PET, genetic)
so that ``fusion.py`` can treat all embedders uniformly via duck-typing.
This also enables easy swapping between PCA and autoencoder approaches
without changing downstream code.

Design decisions:
  * ``fit`` / ``transform`` / ``fit_transform`` mirrors scikit-learn's API
    so embedders compose naturally with sklearn ``Pipeline`` objects.
  * ``save`` / ``load`` are first-class methods because medical-ML pipelines
    must persist trained artefacts for audit and reproducibility.
  * ``output_dim`` is a property so fusion can pre-allocate the concatenated
    embedding matrix without running a forward pass.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypeVar

import numpy as np

# Self type for generic return annotations on classmethods
_T = TypeVar("_T", bound="BaseEmbedder")


class BaseEmbedder(ABC):
    """Abstract interface for modality-specific embedding generators.

    WHY: Enforces consistent API across modalities so ``fusion.py`` can
    treat all embedders uniformly.  Also enables easy swapping of
    PCA vs autoencoder approaches without touching consumer code.

    Subclasses must implement:
        * ``fit(X_train)`` — learn the embedding mapping
        * ``transform(X)`` — project new data into the embedding space
        * ``save(path)`` — persist trained artefacts to disk
        * ``load(path)`` — restore a previously saved embedder
        * ``output_dim`` — dimensionality of the embedding output
    """

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, X_train: np.ndarray) -> "BaseEmbedder":
        """Learn the embedding mapping from training data.

        Args:
            X_train: Training feature matrix, shape ``(n_samples, n_features)``.

        Returns:
            ``self``, for method chaining.
        """
        ...

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project data into the learned embedding space.

        Args:
            X: Feature matrix, shape ``(n_samples, n_features)``.

        Returns:
            Embedding matrix, shape ``(n_samples, output_dim)``.
        """
        ...

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Convenience: fit on *X* then transform it.

        WHY: Avoids redundant computation when the same data is used for
        both fitting and transforming (common during training).

        Args:
            X: Feature matrix, shape ``(n_samples, n_features)``.

        Returns:
            Embedding matrix, shape ``(n_samples, output_dim)``.
        """
        return self.fit(X).transform(X)

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist the trained embedder to *path*.

        Args:
            path: File or directory to write artefacts to.
        """
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseEmbedder":
        """Restore a previously saved embedder from *path*.

        Args:
            path: File or directory previously written by :meth:`save`.

        Returns:
            Fully initialised embedder ready for :meth:`transform`.
        """
        ...

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimensionality of the embedding vectors produced by :meth:`transform`.

        WHY: The fusion layer needs to know each modality's size at
        graph-construction time so it can pre-allocate the concatenated
        matrix and validate shapes before any data flows through.
        """
        ...

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(output_dim={self.output_dim})"
