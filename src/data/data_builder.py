# -*- coding: utf-8 -*-
"""
Unified Data Builder for Fed-PhenoGraft.
Orchestrates all modality-specific loaders into a single aligned multimodal dataset
ready for the FederatedPPMIDataset PyTorch wrapper.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

from src.data.clinical_loader import build_clinical_features
from src.data.mri_pipeline import build_mri_features
from src.data.pet_loader import load_datscan
from src.data.genetic_loader import load_genetic_data
from src.data.dataset import FederatedPPMIDataset

logger = logging.getLogger(__name__)


def build_real_dataset(config: dict) -> tuple:
    """
    Master pipeline: loads real PPMI data from CSVs and NIfTI files,
    preprocesses each modality, and returns aligned DataFrames.
    
    Falls back to synthetic data for any modality that cannot be loaded.
    
    Args:
        config: parsed config.yaml dict
        
    Returns:
        (clinical_df, mri_df, pet_df, genetic_df, targets_df, diagnosis_series)
        All indexed by PATNO with aligned rows.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    raw_dir = project_root / config["paths"]["raw"]
    mri_dir = project_root / config["paths"]["mri_raw"]
    
    baseline_visit = config.get("target", {}).get("baseline_visit", "BL")
    target_visit = config.get("target", {}).get("regression_visit", "V04")
    target_mode = config.get("target", {}).get("mode", "absolute")
    n_rois = config.get("mri", {}).get("n_rois", 100)
    use_real_mri = config.get("mri", {}).get("use_real_mri", False)

    # ── Step 1: Clinical features + targets ──────────────────────────
    logger.info("=" * 60)
    logger.info("Step 1: Loading Clinical Data...")
    clinical_df, targets_df = build_clinical_features(raw_dir, baseline_visit,
                                                      target_visit, target_mode)
    
    if clinical_df.empty:
        logger.error("Clinical data is empty. Cannot proceed without at least clinical features.")
        logger.info("Generating synthetic clinical fallback (400 samples)...")
        np.random.seed(config.get("seed", 42))
        n = 400
        clinical_df = pd.DataFrame(
            np.random.randn(n, 20),
            columns=[f"C_{i}" for i in range(20)],
            index=range(3000, 3000 + n)
        )
        clinical_df.index.name = "PATNO"
        targets_df = pd.DataFrame({
            "updrs_iii_target": clinical_df.iloc[:, 0] * 5 + np.random.randn(n) * 2 + 20,
            "diagnosis": np.random.choice([0, 1], size=n, p=[0.4, 0.6])
        }, index=clinical_df.index)
    
    patnos = clinical_df.index.tolist()
    
    # ── Step 2: MRI features ────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 2: Processing MRI Data...")
    mri_df = build_mri_features(mri_dir, patnos, n_rois=n_rois, use_real_mri=use_real_mri)
    
    # ── Step 3: PET/DaTScan features ────────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 3: Loading PET/DaTScan Data...")
    pet_df = load_datscan(raw_dir, visit=baseline_visit)
    
    if pet_df.empty:
        logger.info("Generating synthetic PET fallback...")
        np.random.seed(43)
        pet_cols = ["caudate_r", "caudate_l", "putamen_r", "putamen_l",
                    "caudate_mean", "putamen_mean", "asymmetry_caudate",
                    "asymmetry_putamen", "striatum_total", "caudate_putamen_ratio"]
        pet_df = pd.DataFrame(
            np.random.randn(len(patnos), len(pet_cols)),
            index=patnos, columns=pet_cols
        )
        # Mark 15% as missing
        mask = np.random.rand(len(patnos)) < 0.15
        pet_df.loc[mask] = 0.0
    
    # ── Step 4: Genetic features ────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 4: Loading Genetic Data...")
    genetic_df = load_genetic_data(raw_dir)
    
    if genetic_df.empty:
        logger.info("Generating synthetic Genetic fallback...")
        np.random.seed(44)
        gen_cols = ["LRRK2", "GBA", "SNCA", "PINK1", "PRKN", "APOE_e4_carrier",
                    "n_variants", "lrrk2_positive", "gba_positive"]
        genetic_df = pd.DataFrame(
            np.random.randint(0, 2, size=(len(patnos), len(gen_cols))),
            index=patnos, columns=gen_cols, dtype=float
        )
        mask = np.random.rand(len(patnos)) < 0.10
        genetic_df.loc[mask] = 0.0
    
    # ── Step 5: Align all modalities ────────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 5: Aligning modalities...")
    
    # Reindex all to clinical's PATNO (the primary modality)
    mri_df = mri_df.reindex(patnos).fillna(0)
    pet_df = pet_df.reindex(patnos).fillna(0)
    genetic_df = genetic_df.reindex(patnos).fillna(0)
    targets_df = targets_df.reindex(patnos)

    # ── Step 6: Resolve targets (NO scaling here) ───────────────────
    # NOTE: imputation and scaling are deliberately NOT done here.
    # They are fit on the training split only (src/data/preprocessing.py)
    # to prevent test-set statistics from leaking into training.
    if "updrs_iii_target" in targets_df.columns:
        regression_target = targets_df["updrs_iii_target"]
    else:
        logger.warning("No regression target found. Using zeros.")
        regression_target = pd.Series(np.zeros(len(patnos)), index=patnos)

    diagnosis = targets_df.get("diagnosis", pd.Series(np.zeros(len(patnos)), index=patnos))

    # Drop subjects with no ground-truth target instead of imputing it —
    # imputed labels are label noise and their statistics leak across splits.
    has_target = regression_target.notna()
    n_dropped = int((~has_target).sum())
    if n_dropped:
        logger.info(f"Dropping {n_dropped} subjects with missing regression target "
                    f"(no label imputation — prevents target leakage).")
        clinical_df = clinical_df.loc[has_target]
        mri_df = mri_df.loc[has_target]
        pet_df = pet_df.loc[has_target]
        genetic_df = genetic_df.loc[has_target]
        regression_target = regression_target.loc[has_target]
        diagnosis = diagnosis.loc[has_target]
        patnos = clinical_df.index.tolist()
    
    logger.info("=" * 60)
    logger.info(f"Dataset Summary:")
    logger.info(f"  Patients:  {len(patnos)}")
    logger.info(f"  Clinical:  {clinical_df.shape[1]} features")
    logger.info(f"  MRI:       {mri_df.shape[1]} ROIs")
    logger.info(f"  PET:       {pet_df.shape[1]} features")
    logger.info(f"  Genetic:   {genetic_df.shape[1]} features")
    logger.info(f"  MRI missing:     {(mri_df.sum(axis=1) == 0).sum()}/{len(patnos)}")
    logger.info(f"  PET missing:     {(pet_df.sum(axis=1) == 0).sum()}/{len(patnos)}")
    logger.info(f"  Genetic missing: {(genetic_df.sum(axis=1) == 0).sum()}/{len(patnos)}")
    
    return clinical_df, mri_df, pet_df, genetic_df, regression_target, diagnosis


def build_pytorch_dataset(config: dict) -> FederatedPPMIDataset:
    """
    Convenience wrapper: builds the full dataset and wraps it in a 
    FederatedPPMIDataset ready for DataLoader usage.
    """
    clinical_df, mri_df, pet_df, genetic_df, targets, diagnosis = build_real_dataset(config)
    
    targets_df = pd.DataFrame(targets, columns=["target"])
    
    dataset = FederatedPPMIDataset(
        clinical_df=clinical_df,
        mri_df=mri_df,
        pet_df=pet_df,
        genetic_df=genetic_df,
        targets=targets_df
    )
    
    return dataset
