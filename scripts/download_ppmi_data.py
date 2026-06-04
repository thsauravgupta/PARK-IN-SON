# -*- coding: utf-8 -*-
"""
Script to download PPMI tabular data.
Uses a robust 3-tier fallback strategy:
1. Try pypmi.fetch_studydata (may fail due to portal updates)
2. Try ppmi_downloader.PPMIDownloader.download_metadata
3. Fallback to manual download instructions
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.utils import setup_logging, load_config, seed_everything

def get_required_csvs():
    return [
        "Demographics.csv",
        "Age_at_visit.csv",
        "MDS_UPDRS_Part_I.csv",
        "MDS_UPDRS_Part_II.csv",
        "MDS_UPDRS_Part_III.csv",
        "MDS_UPDRS_Part_IV.csv",
        "DATScan_Analysis.csv",
        "Genetic_Testing_Results.csv",
        "Patient_Status.csv",
        "Screening___Demographics.csv",
        "MoCA.csv",
        "Montreal_Cognitive_Assessment__MoCA_.csv",
        "Participant_Status.csv"
    ]

def attempt_pypmi(user, password, path, logger):
    logger.info("Attempt 1: Using pypmi.fetch_studydata...")
    try:
        import pypmi
        pypmi.fetch_studydata('all', user=user, password=password, path=str(path))
        logger.info("pypmi download successful.")
        return True
    except Exception as e:
        logger.warning(f"pypmi failed: {e}")
        return False

def attempt_ppmi_downloader(user, password, path, logger):
    logger.info("Attempt 2: Using ppmi_downloader (Selenium)...")
    try:
        from ppmi_downloader import PPMIDownloader
        # Assumes username/password are in .env as expected by the library
        downloader = PPMIDownloader()
        downloader.download_metadata(get_required_csvs(), destination_dir=str(path))
        logger.info("ppmi_downloader successful.")
        return True
    except Exception as e:
        logger.warning(f"ppmi_downloader failed: {e}")
        return False

def manual_fallback(path, logger):
    logger.error("All automated download attempts failed.")
    logger.error("Please download the following CSV files manually from https://ida.loni.usc.edu:")
    for csv in get_required_csvs():
        logger.error(f"  - {csv}")
    logger.error(f"Place them inside: {path}")
    sys.exit(1)

def main():
    seed_everything(42)
    logger = setup_logging(__name__)
    config = load_config()
    
    load_dotenv(project_root / ".env")
    user = os.getenv("PPMI_USER")
    password = os.getenv("PPMI_PASSWORD")
    
    if not user or not password:
        logger.error("Credentials missing. Ensure PPMI_USER and PPMI_PASSWORD are in .env")
        sys.exit(1)
        
    data_path = project_root / config["paths"]["raw"]
    data_path.mkdir(parents=True, exist_ok=True)
    
    # 3-tier strategy
    if attempt_pypmi(user, password, data_path, logger):
        return
    if attempt_ppmi_downloader(user, password, data_path, logger):
        return
    manual_fallback(data_path, logger)

if __name__ == "__main__":
    main()
