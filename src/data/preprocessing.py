# -*- coding: utf-8 -*-
"""
Leak-free preprocessing and subject-level splitting for Fed-PhenoGraft.

All statistics (imputation means/medians, scaling mean/std) are fit ONLY on
the training subjects and then applied unchanged to validation and test
subjects. Missing-modality rows (all-zero) are preserved as all-zero after
scaling so the mask-token logic in FederatedPPMIDataset keeps working.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def _missing_row_mask(df: pd.DataFrame) -> np.ndarray:
    """A row is a 'missing modality' row when every value is exactly zero."""
    return (df.values == 0).all(axis=1)


class ModalityPreprocessor:
    """
    Per-modality imputer + scaler that is fit on training data only.

    Clinical: median-impute then standardize (clinical is never 'missing').
    MRI / PET / Genetic: standardize using statistics from OBSERVED training
    rows only; rows flagged as missing (all-zero) stay all-zero so the
    downstream mask tokens remain valid.
    """

    def __init__(self):
        self.clinical_imputer = SimpleImputer(strategy="median")
        self.clinical_scaler = StandardScaler()
        self.clinical_cols_ = None   # columns with >=1 observed TRAIN value
        self.modality_scalers = {}   # name -> fitted StandardScaler or None

    def fit(self, clinical: pd.DataFrame, mri: pd.DataFrame,
            pet: pd.DataFrame, genetic: pd.DataFrame) -> "ModalityPreprocessor":
        # Keep only clinical columns that have at least one observed value in
        # the TRAINING split (an all-NaN column cannot be imputed). Column
        # selection is decided on train data only, so it leaks nothing.
        self.clinical_cols_ = clinical.columns[clinical.notna().any()].tolist()
        dropped = [c for c in clinical.columns if c not in self.clinical_cols_]
        if dropped:
            logger.warning(f"Dropping clinical columns with no observed training "
                           f"values: {dropped}")
        clinical = clinical[self.clinical_cols_]

        self.clinical_imputer.fit(clinical.values)
        self.clinical_scaler.fit(self.clinical_imputer.transform(clinical.values))

        for name, df in [("mri", mri), ("pet", pet), ("genetic", genetic)]:
            observed = ~_missing_row_mask(df)
            if observed.sum() < 2:
                logger.warning(
                    f"Modality '{name}': fewer than 2 observed training rows; "
                    f"skipping scaling (identity transform)."
                )
                self.modality_scalers[name] = None
                continue
            scaler = StandardScaler()
            scaler.fit(df.values[observed])
            # Guard against zero-variance features (constant columns)
            scaler.scale_[scaler.scale_ == 0] = 1.0
            self.modality_scalers[name] = scaler
        return self

    def _transform_modality(self, name: str, df: pd.DataFrame) -> pd.DataFrame:
        scaler = self.modality_scalers.get(name)
        if scaler is None:
            return df.copy()
        missing = _missing_row_mask(df)
        values = scaler.transform(np.nan_to_num(df.values, nan=0.0))
        values[missing] = 0.0  # preserve missing-modality semantics
        return pd.DataFrame(values, index=df.index, columns=df.columns)

    def transform(self, clinical: pd.DataFrame, mri: pd.DataFrame,
                  pet: pd.DataFrame, genetic: pd.DataFrame) -> tuple:
        clinical = clinical[self.clinical_cols_]
        clin_values = self.clinical_scaler.transform(
            self.clinical_imputer.transform(clinical.values)
        )
        clin_out = pd.DataFrame(clin_values, index=clinical.index,
                                columns=clinical.columns)
        mri_out = self._transform_modality("mri", mri)
        pet_out = self._transform_modality("pet", pet)
        gen_out = self._transform_modality("genetic", genetic)
        return clin_out, mri_out, pet_out, gen_out


def create_subject_splits(diagnosis: pd.Series, val_fraction: float = 0.15,
                          test_fraction: float = 0.15, seed: int = 42,
                          stratify: bool = True) -> tuple:
    """
    Subject-level train/val/test split. Every subject (PATNO) lands in exactly
    one partition, which rules out identity leakage across splits.

    Args:
        diagnosis: Series indexed like the aligned dataset (one row per subject).
        val_fraction / test_fraction: fractions of the FULL dataset.
        stratify: stratify on diagnosis so PD/HC ratios match across splits.

    Returns:
        (train_idx, val_idx, test_idx) — positional integer indices.
    """
    n = len(diagnosis)
    indices = np.arange(n)
    strat_labels = pd.Series(diagnosis).fillna(-1).values if stratify else None

    def _split(idx, labels, test_size, seed):
        try:
            return train_test_split(idx, test_size=test_size, random_state=seed,
                                    stratify=labels)
        except ValueError:
            # Stratification impossible (a class has < 2 members) — fall back.
            logger.warning("Stratified split failed; falling back to random split.")
            return train_test_split(idx, test_size=test_size, random_state=seed)

    trainval_idx, test_idx = _split(indices, strat_labels, test_fraction, seed)

    rel_val = val_fraction / (1.0 - test_fraction)
    labels_tv = strat_labels[trainval_idx] if strat_labels is not None else None
    train_idx, val_idx = _split(trainval_idx, labels_tv, rel_val, seed + 1)

    logger.info(
        f"Subject-level split: train={len(train_idx)}, val={len(val_idx)}, "
        f"test={len(test_idx)} (no subject appears in more than one partition)"
    )
    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)
