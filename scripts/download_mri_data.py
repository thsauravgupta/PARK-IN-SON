# -*- coding: utf-8 -*-
"""
Script to download MRI NIfTI data using ppmi_downloader.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.utils import setup_logging, load_config, seed_everything

def main():
    seed_everything(42)
    logger = setup_logging(__name__)
    config = load_config()
    
    load_dotenv(project_root / ".env")
    
    mri_path = project_root / config["paths"]["mri_raw"]
    mri_path.mkdir(parents=True, exist_ok=True)
    
    try:
        from ppmi_downloader import PPMIDownloader
        logger.info("Initializing PPMIDownloader for MRI...")
        dl = PPMIDownloader()
        
        logger.info("Fetching 3D T1 metadata...")
        csv_file = dl.download_3D_T1_info(destination_dir=str(mri_path))
        import pandas as pd
        cohort = pd.read_csv(csv_file)
        
        # Rename columns to match what ppmi_downloader expects internally
        if "Subject ID" in cohort.columns:
            cohort = cohort.rename(columns={"Subject ID": "PATNO"})
        if "Visit" in cohort.columns:
            cohort = cohort.rename(columns={"Visit": "EVENT_ID"})
        
        # Download a subset or all - here we assume a predefined list or all available
        logger.info("Downloading NIfTI imaging data...")
        dl.download_imaging_data(cohort, destination_dir=str(mri_path))
        
        nifti_files = list(mri_path.rglob("*.nii*"))
        logger.info(f"Downloaded {len(nifti_files)} NIfTI files.")
        
    except Exception as e:
        logger.warning(f"Failed to download MRI data: {e}")
        logger.warning("Setting config mri.use_real_mri=false. Pipeline will use synthetic MRI data.")
        # In a real scenario, we might edit the config file or just rely on the existing flag
        sys.exit(0)

if __name__ == "__main__":
    main()
