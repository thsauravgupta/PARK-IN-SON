# -*- coding: utf-8 -*-
"""
Genetic Data Loader for PPMI.
Loads mutation carrier status for PD-associated genes from Genetic_Testing_Results.csv
and encodes them as binary/ordinal features.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


def load_genetic_data(raw_dir: Path) -> pd.DataFrame:
    """
    Load genetic testing results from PPMI.
    
    Target genes and their significance:
        - LRRK2: Most common genetic cause of autosomal dominant PD
        - GBA: Strongest risk factor for sporadic PD
        - SNCA: Alpha-synuclein gene, causal in familial PD
        - PINK1: Mitochondrial kinase, early-onset PD
        - PARK2 (PRKN): Parkin gene, early-onset PD  
        - APOE: APOE-ε4 allele, cognitive decline risk modifier
    
    Derived features:
        - Binary carrier status per gene (0/1)
        - n_variants: total number of pathogenic variants
        - lrrk2_positive, gba_positive: explicit risk flags
    """
    raw_dir = Path(raw_dir)
    
    candidates = [
        "Genetic_Testing_Results.csv",
        "Genetic_Results.csv",
        "Genetics.csv",
        "PD_Features.csv",
    ]
    
    df = None
    for fname in candidates:
        fpath = raw_dir / fname
        if fpath.exists():
            try:
                df = pd.read_csv(fpath, low_memory=False)
            except UnicodeDecodeError:
                df = pd.read_csv(fpath, encoding='latin-1', low_memory=False)
            break
    
    if df is None:
        logger.warning("Genetic testing CSV not found. Genetic features will be zero (missing).")
        return pd.DataFrame()
    
    df = df.drop_duplicates(subset=["PATNO"], keep="last")
    
    out = pd.DataFrame(index=df["PATNO"].values)
    out.index.name = "PATNO"
    
    # Target genes to extract
    target_genes = {
        "LRRK2": ["LRRK2", "LRRK2_MUTATION", "LRRK2_STATUS"],
        "GBA": ["GBA", "GBA_MUTATION", "GBA_STATUS"],
        "SNCA": ["SNCA", "SNCA_MUTATION"],
        "PINK1": ["PINK1", "PINK1_MUTATION"],
        "PRKN": ["PRKN", "PARK2", "PARKIN", "PARK2_MUTATION"],
        "APOE": ["APOE", "APOE_GENOTYPE", "APOE4_STATUS"],
    }
    
    for gene_name, col_candidates in target_genes.items():
        found = False
        for col in col_candidates:
            if col in df.columns:
                if gene_name == "APOE":
                    # APOE is typically genotype string (e.g., "E3/E4")
                    # Convert to binary E4 carrier status
                    out[f"APOE_e4_carrier"] = (
                        df[col].astype(str)
                        .str.contains("E4|e4|4", case=False, na=False)
                        .astype(int).values
                    )
                else:
                    # PPMI records the VARIANT NAME, not a 0/1 flag: LRRK2 holds
                    # 'G2019S', 'R1441G', 'G2019S/G2019S'; GBA holds 'N409S',
                    # 'L483P', 'L29Afs*18'; SNCA holds 'A53T', 'duplication'.
                    # Non-carriers are the literal string '0'.
                    #
                    # The previous parser tried int(float(val)) and swallowed the
                    # ValueError as 0, so every genuine pathogenic variant was
                    # recorded as a NON-carrier - 1,335 carriers across five genes
                    # in this release, including 688 LRRK2 and 566 GBA. Treat any
                    # value that is neither missing nor an explicit negative as a
                    # carrier.
                    s = df[col].astype(str).str.strip().str.lower()
                    missing = s.isin(["nan", "none", "", "na", "n/a", "unknown",
                                      "not tested", "not done"])
                    negative = s.isin(["0", "0.0", "negative", "neg", "false", "no"])
                    out[gene_name] = (~missing & ~negative).astype(int).values
                    n_carriers = int(out[gene_name].sum())
                    logger.info(f"  {gene_name}: {n_carriers} carriers "
                                f"({int(missing.sum())} not tested)")
                found = True
                break
        
        if not found:
            # Gene column not found — could be absent from this dataset version
            if gene_name == "APOE":
                out["APOE_e4_carrier"] = 0
            else:
                out[gene_name] = 0
    
    # Derived features
    carrier_cols = [c for c in ["LRRK2", "GBA", "SNCA", "PINK1", "PRKN"] if c in out.columns]
    if carrier_cols:
        out["n_variants"] = out[carrier_cols].sum(axis=1)
    
    if "LRRK2" in out.columns:
        out["lrrk2_positive"] = (out["LRRK2"] > 0).astype(int)
    if "GBA" in out.columns:
        out["gba_positive"] = (out["GBA"] > 0).astype(int)
    
    logger.info(f"Loaded genetic data: {out.shape[0]} patients, {out.shape[1]} features")
    return out.fillna(0)
