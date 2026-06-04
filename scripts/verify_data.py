# -*- coding: utf-8 -*-
"""
Data verification script. Audits all downloaded CSVs and NIfTIs.
"""

import sys
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.utils import setup_logging, load_config

def main():
    logger = setup_logging(__name__)
    config = load_config()
    raw_dir = project_root / config["paths"]["raw"]
    
    required_csvs = [
        "Demographics.csv",
        "MDS_UPDRS_Part_III.csv",
        "DATScan_Analysis.csv",
        "Genetic_Testing_Results.csv",
    ]
    
    missing = []
    
    logger.info("--- Data Verification Audit ---")
    for csv in required_csvs:
        p = raw_dir / csv
        if p.exists():
            try:
                df = pd.read_csv(p, low_memory=False)
                rows, cols = df.shape
                has_patno = "PATNO" in df.columns
                logger.info(f"[OK] {csv}: {rows} rows, {cols} cols, PATNO present: {has_patno}")
            except Exception as e:
                logger.error(f"[ERROR] Could not read {csv}: {e}")
                missing.append(csv)
        else:
            logger.error(f"[MISSING] {csv}")
            missing.append(csv)
            
    mri_dir = project_root / config["paths"]["mri_raw"]
    niftis = list(mri_dir.rglob("*.nii*")) if mri_dir.exists() else []
    if niftis:
        logger.info(f"[OK] MRI: Found {len(niftis)} NIfTI files.")
    else:
        logger.warning("[WARN] MRI: No NIfTI files found. Pipeline will run in synthetic MRI mode.")
        
    if missing:
        logger.error(f"Verification failed. Missing {len(missing)} critical CSVs.")
        sys.exit(1)
        
    logger.info("All critical CSVs present. Verification passed.")
    sys.exit(0)

if __name__ == "__main__":
    main()
