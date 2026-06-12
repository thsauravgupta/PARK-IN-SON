# -*- coding: utf-8 -*-
"""
Generates multimodal embeddings from the raw downloaded CSVs.
"""

import sys
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.utils import setup_logging, load_config, seed_everything
from src.data.preprocessors import ClinicalPreprocessor, MRIPreprocessor, PETPreprocessor, GeneticPreprocessor
from src.data.mri_processor import MRIProcessor
from src.embeddings.clinical_embedder import ClinicalEmbedder
from src.embeddings.mri_embedder import MRIEmbedder
from src.embeddings.pet_embedder import PETEmbedder
from src.embeddings.genetic_embedder import GeneticEmbedder
from src.embeddings.fusion import FusionPipeline

def load_raw_data(config):
    raw_dir = project_root / config["paths"]["raw"]
    
    # Clinical (UPDRS III used for targets, demographics for classification)
    clin_files = ["Demographics.csv", "MDS_UPDRS_Part_III.csv"]
    clin_dfs = []
    for f in clin_files:
        df = pd.read_csv(raw_dir / f, low_memory=False)
        if "EVENT_ID" in df.columns:
            df = df[df["EVENT_ID"] == "BL"]
        if "PATNO" in df.columns:
            df = df.drop_duplicates(subset=["PATNO"]).set_index("PATNO")
            df.index = df.index.astype(int)
            clin_dfs.append(df)
    
    # We do a simplified join on PATNO for clinical
    clin_df = pd.concat(clin_dfs, axis=1)
    
    # Only keep numeric for simplicity in this run, or at least fill NaNs
    clin_df = clin_df.select_dtypes(include=['number']).groupby(level=0).mean()

    # PET
    pet_df = pd.read_csv(raw_dir / "DATScan_Analysis.csv", low_memory=False)
    
    # DATScan baseline is typically 'SC' (Screening), so strict 'BL' filtering drops 90% of patients
    # We will just drop duplicates to keep the first available scan per patient (which is chronological)
    pet_df = pet_df.drop_duplicates(subset=["PATNO"]).set_index("PATNO")
    pet_df.index = pet_df.index.astype(int)
    # Remap typical Xing Core columns to our expected names if needed
    pet_df.columns = [c.lower() for c in pet_df.columns]
    
    # Genetics
    gen_df = pd.read_csv(raw_dir / "Genetic_Testing_Results.csv", low_memory=False)
    if "PATNO" in gen_df.columns:
        gen_df = gen_df.drop_duplicates(subset=["PATNO"]).set_index("PATNO")
        gen_df.index = gen_df.index.astype(int)
    
    # Create target dataframes
    labels_df = pd.DataFrame(index=clin_df.index)
    labels_df["diagnosis"] = 1 # Dummy fallback
    if "APPRDX" in clin_df.columns:
        labels_df["diagnosis"] = (clin_df["APPRDX"] == 1).astype(int)
        
    targets_df = pd.DataFrame(index=clin_df.index)
    targets_df["updrs_iii_v04"] = clin_df["NP3TOT"] if "NP3TOT" in clin_df.columns else 0
    
    labels_df = labels_df.fillna(0)
    targets_df = targets_df.fillna(0)

    # FIX DATA LEAKAGE: Drop UPDRS target columns and Diagnosis labels from clinical features
    leakage_cols = [c for c in clin_df.columns if "NP3" in c or "APPRDX" in c or "updrs" in c.lower() or "diagnosis" in c.lower()]
    clin_df = clin_df.drop(columns=leakage_cols, errors="ignore")


    if config.get("mri", {}).get("use_real_mri", False):
        mri_proc = MRIProcessor(config)
        mri_dir = project_root / config["paths"]["mri_raw"]
        mri_out = project_root / config["paths"]["processed"] / "mri_features.parquet"
        mri_df = mri_proc.process_all(mri_dir, mri_out)
    else:
        mri_df = pd.DataFrame(index=clin_df.index) # Dummy for synthetic

    return {
        "clinical": clin_df,
        "mri": mri_df,
        "pet": pet_df,
        "genetic": gen_df
    }, labels_df, targets_df

def main():
    seed_everything(42)
    logger = setup_logging(__name__)
    config = load_config()
    
    logger.info("Loading raw CSV data...")
    X_dict, labels_df, targets_df = load_raw_data(config)
    
    logger.info("Initializing pipelines...")
    modality_dict = {
        "clinical": (ClinicalPreprocessor(config), ClinicalEmbedder(mode='pca', config=config)),
        "mri": (MRIPreprocessor(config), MRIEmbedder(mode='gae', config=config)),
        "pet": (PETPreprocessor(config), PETEmbedder(config=config)),
        "genetic": (GeneticPreprocessor(config), GeneticEmbedder(config=config))
    }
    
    fusion = FusionPipeline(modality_dict)
    
    logger.info("Fitting and transforming data (this may take a minute)...")
    fusion.fit(X_dict)
    
    output_dir = project_root / config["paths"]["embeddings"]
    logger.info(f"Fusing and saving to {output_dir}...")
    fusion.fuse(X_dict, output_dir, labels_df, targets_df)
    
    logger.info("Embedding generation complete!")

if __name__ == "__main__":
    main()
