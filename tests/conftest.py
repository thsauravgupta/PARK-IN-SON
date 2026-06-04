# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for the PPMI test suite.

WHY: Centralised fixtures avoid duplicating synthetic data construction
across test modules. Every test module that needs PPMI-like data imports
from here, ensuring consistent schema and making maintenance trivial.
"""

import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on sys.path so local imports work regardless of
# how pytest is invoked (IDE, CLI, CI).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, seed_everything

# ---------------------------------------------------------------------------
# Global seed for deterministic test data
# ---------------------------------------------------------------------------
seed_everything(42)


# ---------------------------------------------------------------------------
# Config fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def config() -> Dict:
    """Load project config once per test session."""
    return load_config(PROJECT_ROOT / "config.yaml")


# ---------------------------------------------------------------------------
# Clinical data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def clinical_df() -> pd.DataFrame:
    """Synthetic clinical DataFrame matching PPMI schema.

    Includes baseline visit rows with all expected clinical columns.
    Values are physiologically plausible (non-negative UPDRS, 0-30 MoCA).
    """
    rng = np.random.RandomState(42)
    n = 80

    data = {
        "PATNO": np.arange(3000, 3000 + n),
        "EVENT_ID": ["BL"] * n,
        "updrs_i": rng.randint(0, 16, n).astype(float),
        "updrs_ii": rng.randint(0, 52, n).astype(float),
        "updrs_iii": rng.randint(0, 132, n).astype(float),
        "updrs_iii_a": rng.randint(0, 44, n).astype(float),
        "updrs_iv": rng.randint(0, 24, n).astype(float),
        "moca": rng.randint(18, 31, n).astype(float),
        "age": rng.uniform(45, 80, n),
        "gender": rng.choice([0, 1], n).astype(float),
        "education": rng.uniform(8, 22, n),
        "tremor": rng.uniform(0, 4, n),
        "pigd": rng.uniform(0, 4, n),
        "se_adl": rng.uniform(50, 100, n),
        "epworth": rng.randint(0, 24, n).astype(float),
        "gds": rng.randint(0, 15, n).astype(float),
        "rbd": rng.uniform(0, 13, n),
        "scopa_aut": rng.randint(0, 69, n).astype(float),
        "semantic_fluency": rng.randint(5, 30, n).astype(float),
        "symbol_digit": rng.randint(10, 60, n).astype(float),
        "hvlt_recall": rng.randint(0, 36, n).astype(float),
        "hvlt_recognition": rng.randint(0, 12, n).astype(float),
        "hvlt_retention": rng.uniform(0, 100, n),
        "lns": rng.randint(0, 21, n).astype(float),
        "stai_state": rng.randint(20, 80, n).astype(float),
        "stai_trait": rng.randint(20, 80, n).astype(float),
        "upsit": rng.randint(0, 40, n).astype(float),
    }
    return pd.DataFrame(data)


@pytest.fixture()
def clinical_df_with_nans(clinical_df: pd.DataFrame) -> pd.DataFrame:
    """Clinical DataFrame with ~10% NaN values injected."""
    df = clinical_df.copy()
    rng = np.random.RandomState(99)
    # Only inject NaNs into numeric feature columns (not PATNO / EVENT_ID)
    feature_cols = [c for c in df.columns if c not in ("PATNO", "EVENT_ID")]
    mask = rng.random(size=(len(df), len(feature_cols))) < 0.10
    df.loc[:, feature_cols] = df[feature_cols].where(~mask)
    return df


# ---------------------------------------------------------------------------
# PET / DaTscan data fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def pet_df() -> pd.DataFrame:
    """Synthetic DaTscan DataFrame with caudate/putamen SBR values."""
    rng = np.random.RandomState(42)
    n = 80
    return pd.DataFrame({
        "PATNO": np.arange(3000, 3000 + n),
        "EVENT_ID": ["BL"] * n,
        "caudate_r": rng.uniform(1.0, 4.0, n),
        "caudate_l": rng.uniform(1.0, 4.0, n),
        "putamen_r": rng.uniform(0.5, 3.0, n),
        "putamen_l": rng.uniform(0.5, 3.0, n),
    })


# ---------------------------------------------------------------------------
# Genetic data fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def genetic_df() -> pd.DataFrame:
    """Synthetic genetic DataFrame with binary mutation flags."""
    rng = np.random.RandomState(42)
    n = 80
    genes = ["LRRK2", "GBA", "SNCA", "PINK1", "PARK2", "APOE"]
    data: Dict = {"PATNO": np.arange(3000, 3000 + n)}
    for gene in genes:
        data[gene] = rng.choice([0, 1], size=n, p=[0.85, 0.15])
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Embedding fixtures (for embedder + runner tests)
# ---------------------------------------------------------------------------

@pytest.fixture()
def random_clinical_features() -> np.ndarray:
    """Random (50, 20) array simulating preprocessed clinical features."""
    return np.random.RandomState(42).randn(50, 20)


@pytest.fixture()
def random_mri_features() -> np.ndarray:
    """Random (50, 100) array simulating MRI ROI volumes."""
    return np.random.RandomState(42).randn(50, 100)


@pytest.fixture()
def random_pet_features() -> np.ndarray:
    """Random (50, 10) array simulating PET features."""
    return np.random.RandomState(42).randn(50, 10)


@pytest.fixture()
def random_genetic_features() -> np.ndarray:
    """Random (50, 9) array simulating genetic features."""
    return np.random.RandomState(42).randint(0, 2, size=(50, 9)).astype(float)


@pytest.fixture()
def fused_embeddings() -> pd.DataFrame:
    """Synthetic 144-dim fused embeddings with PATNO index for runner tests."""
    rng = np.random.RandomState(42)
    n = 100
    dim = 144
    data = rng.randn(n, dim)
    columns = [f"emb_{i}" for i in range(dim)]
    df = pd.DataFrame(data, columns=columns)
    df.index = pd.Index(np.arange(3000, 3000 + n), name="PATNO")
    return df


@pytest.fixture()
def regression_targets() -> pd.Series:
    """Synthetic UPDRS-III year-2 regression targets."""
    rng = np.random.RandomState(42)
    return pd.Series(
        rng.uniform(5, 80, 100),
        index=pd.Index(np.arange(3000, 3100), name="PATNO"),
        name="updrs_iii_v04",
    )


@pytest.fixture()
def classification_labels() -> pd.Series:
    """Synthetic PD vs HC binary labels."""
    rng = np.random.RandomState(42)
    return pd.Series(
        rng.choice([0, 1], 100, p=[0.4, 0.6]),
        index=pd.Index(np.arange(3000, 3100), name="PATNO"),
        name="diagnosis",
    )
