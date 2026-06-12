#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build cohort_index.parquet from processed modality files.

This is the first script to run after downloading and processing PPMI data.
It:
  1. Discovers which patients have data for each modality.
  2. Creates a master cohort index with has_* availability flags.
  3. Adds train/val/test split assignments (stratified by diagnosis).
  4. Saves the result to data/processed/cohort_index.parquet.

Usage
-----
    python scripts/build_cohort_index.py

Or with a custom config:
    python scripts/build_cohort_index.py --config config.yaml --seed 42

Expected processed directory layout (created by running each modality's
pipeline / preprocessors first):

    data/processed/
    ├── clinical/baseline_features.parquet    ← ClinicalPreprocessor output
    ├── mri/roi_features.parquet              ← MRIPreprocessor output
    ├── pet/datscan_features.parquet          ← PETPreprocessor output
    └── genetic/mutation_flags.parquet        ← GeneticPreprocessor output

If a parquet file is missing, the corresponding has_* flag is set to False
for all patients (the cohort still builds — just with fewer modalities).
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------
# Project root on path
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODALITIES = ["clinical", "mri", "pet", "genetic"]

MODALITY_PATHS = {
    "clinical": "clinical/baseline_features.parquet",
    "mri":      "mri/roi_features.parquet",
    "pet":      "pet/datscan_features.parquet",
    "genetic":  "genetic/mutation_flags.parquet",
}


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_cohort_index(
    processed_dir: Path,
    seed: int = 42,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> pd.DataFrame:
    """Build PATNO-indexed cohort index with availability flags and splits.

    Args:
        processed_dir: Root of the processed data directory.
        seed:          Random seed for reproducible splits.
        val_frac:      Fraction of data for validation.
        test_frac:     Fraction of data for test.

    Returns:
        DataFrame indexed by PATNO with columns:
            has_clinical, has_mri, has_pet, has_genetic,
            n_modalities, diagnosis, split
    """
    # ---- Collect all PATNOs that appear in at least one modality ----------
    modality_patnos: dict = {}
    for mod, rel_path in MODALITY_PATHS.items():
        full_path = processed_dir / rel_path
        if full_path.exists():
            df = pd.read_parquet(full_path)
            modality_patnos[mod] = set(df.index.astype(int))
            logger.info(f"  {mod:>10s}: {len(modality_patnos[mod]):>4d} patients")
        else:
            modality_patnos[mod] = set()
            logger.warning(f"  {mod:>10s}: NOT FOUND at {full_path}")

    all_patnos = sorted(
        set().union(*[s for s in modality_patnos.values() if s])
    )

    if not all_patnos:
        raise RuntimeError(
            "No processed modality files found.  Run the preprocessing "
            "pipelines first and save outputs to data/processed/."
        )

    logger.info(f"Total unique PATNOs across all modalities: {len(all_patnos)}")

    # ---- Build has_* flags ------------------------------------------------
    rows = []
    for patno in all_patnos:
        row = {"patno": int(patno)}
        for mod in MODALITIES:
            row[f"has_{mod}"] = int(patno) in modality_patnos[mod]
        row["n_modalities"] = sum(row[f"has_{m}"] for m in MODALITIES)
        rows.append(row)

    cohort = pd.DataFrame(rows).set_index("patno")

    # ---- Try to add diagnosis label ---------------------------------------
    # Look for diagnosis in clinical data
    clin_path = processed_dir / "clinical" / "baseline_features.parquet"
    if clin_path.exists():
        clin_df = pd.read_parquet(clin_path)
        if "diagnosis" in clin_df.columns:
            cohort["diagnosis"] = clin_df["diagnosis"].reindex(cohort.index)
            logger.info("Diagnosis labels merged from clinical data.")
        else:
            logger.warning(
                "Column 'diagnosis' not found in clinical data. "
                "Assigning placeholder 'Unknown'."
            )
            cohort["diagnosis"] = "Unknown"
    else:
        cohort["diagnosis"] = "Unknown"

    # Drop patients with no modalities at all (shouldn't happen, but safeguard)
    cohort = cohort[cohort["n_modalities"] > 0].copy()

    # ---- Train / val / test split (stratified by diagnosis) ---------------
    rng = np.random.default_rng(seed)

    # Use diagnosis for stratification; fall back to unstratified if needed
    stratify_col = cohort["diagnosis"] if cohort["diagnosis"].nunique() > 1 else None

    train_val_idx, test_idx = train_test_split(
        cohort.index,
        test_size=test_frac,
        random_state=int(rng.integers(0, 2**31)),
        stratify=stratify_col,
    )
    strat_tv = stratify_col.loc[train_val_idx] if stratify_col is not None else None
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_frac / (1 - test_frac),
        random_state=int(rng.integers(0, 2**31)),
        stratify=strat_tv,
    )

    cohort["split"] = "test"
    cohort.loc[train_idx, "split"] = "train"
    cohort.loc[val_idx, "split"] = "val"

    # ---- Summary ----------------------------------------------------------
    logger.info("\n=== Cohort Index Summary ===")
    logger.info(f"  Total patients : {len(cohort)}")
    for split in ["train", "val", "test"]:
        n = (cohort["split"] == split).sum()
        logger.info(f"  {split:>5s} split : {n}")
    logger.info("")
    for mod in MODALITIES:
        col = f"has_{mod}"
        n = cohort[col].sum()
        pct = n / len(cohort) * 100
        logger.info(f"  {mod:>10s}: {n:>4d} patients ({pct:.1f}%)")

    missing_any = (cohort["n_modalities"] < len(MODALITIES)).sum()
    logger.info(
        f"\n  Patients missing ≥1 modality: {missing_any} "
        f"({missing_any / len(cohort) * 100:.1f}%)"
    )

    return cohort


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cohort_index.parquet")
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--val-frac",  type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = args.seed if args.seed is not None else config.get("seed", 42)
    processed_dir = PROJECT_ROOT / config["paths"]["processed"]

    logger.info(f"Building cohort index from: {processed_dir}")

    cohort = build_cohort_index(
        processed_dir=processed_dir,
        seed=seed,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
    )

    out_path = processed_dir / "cohort_index.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_parquet(out_path)
    logger.info(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
