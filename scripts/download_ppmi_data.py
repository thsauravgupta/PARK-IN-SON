# -*- coding: utf-8 -*-
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.utils import setup_logging

def main():
    logger = setup_logging(__name__)
    load_dotenv(project_root / ".env")
    
    user = os.getenv("PPMI_USER")
    password = os.getenv("PPMI_PASSWORD")
    
    data_path = project_root / "data" / "raw"
    data_path.mkdir(parents=True, exist_ok=True)
    
    if not user or not password:
        logger.error("Credentials missing. Ensure PPMI_USER and PPMI_PASSWORD are in .env")
        logger.info("Falling back to synthetic data pipeline since credentials are not available.")
        return
        
    try:
        import pypmi
        logger.info("Attempting to download PPMI tabular data via pypmi...")
        pypmi.fetch_studydata('all', user=user, password=password, path=str(data_path))
        logger.info("Download completed successfully.")
    except Exception as e:
        logger.error(f"Download failed: {e}")
        logger.info("Please download the clinical, genetic, and imaging CSVs manually from IDA LONI.")

if __name__ == "__main__":
    main()
