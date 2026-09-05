# -*- coding: utf-8 -*-
"""
PET/DaTScan Data Loader for PPMI.
Loads Striatal Binding Ratio (SBR) values from DATScan_Analysis.csv
and engineers asymmetry and composite features.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


def load_datscan(raw_dir: Path, visit: str = "BL") -> pd.DataFrame:
    """
    Load DaTScan SBR values and engineer derived features.
    
    PPMI provides pre-computed SBR values per striatal region:
        - CAUDATE_R, CAUDATE_L (right/left caudate nucleus)
        - PUTAMEN_R, PUTAMEN_L (right/left putamen)
    
    Engineered features:
        - caudate_mean, putamen_mean (bilateral averages)
        - asymmetry_caudate, asymmetry_putamen (lateralization indices)
        - striatum_total (total dopaminergic binding)
        - caudate_putamen_ratio (caudate vs putamen binding ratio)
    """
    raw_dir = Path(raw_dir)
    
    candidates = [
        "DATScan_Analysis.csv",
        "DATscan_Analysis.csv",
        "DaTscan_Analysis.csv",
        "DatScan_Imaging.csv",
        "DATScan_SBR.csv",
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
        logger.warning("DaTScan CSV not found. PET features will be zero (missing).")
        return pd.DataFrame()
    
    # Filter to target visit. PPMI codes the baseline DaTScan as "SC"
    # (screening) — there is no "BL" row in DATScan_Analysis — so when the
    # requested visit is baseline we accept SC as well, preferring an exact
    # visit match if a patient somehow has both.
    if "EVENT_ID" in df.columns:
        accepted = [visit, "SC"] if visit == "BL" else [visit]
        df = df[df["EVENT_ID"].isin(accepted)].copy()
        df["_evt_rank"] = df["EVENT_ID"].map({e: i for i, e in enumerate(accepted)})
        df = (df.sort_values(["PATNO", "_evt_rank"])
                .drop_duplicates(subset=["PATNO"], keep="first")
                .drop(columns=["_evt_rank"]))
    else:
        df = df.drop_duplicates(subset=["PATNO"], keep="last")
    
    epsilon = 1e-8
    out = pd.DataFrame(index=df["PATNO"].values)
    out.index.name = "PATNO"
    
    # Map column names (PPMI uses various conventions)
    col_map = {
        "caudate_r": ["CAUDATE_R", "DATSCAN_CAUDATE_R", "RCAUDATE", "R_caudate", "CAUDATE_RIGHT"],
        "caudate_l": ["CAUDATE_L", "DATSCAN_CAUDATE_L", "LCAUDATE", "L_caudate", "CAUDATE_LEFT"],
        "putamen_r": ["PUTAMEN_R", "DATSCAN_PUTAMEN_R", "RPUTAMEN", "R_putamen", "PUTAMEN_RIGHT"],
        "putamen_l": ["PUTAMEN_L", "DATSCAN_PUTAMEN_L", "LPUTAMEN", "L_putamen", "PUTAMEN_LEFT"],
    }
    
    raw_vals = {}
    for target_name, candidates_list in col_map.items():
        for src_col in candidates_list:
            if src_col in df.columns:
                raw_vals[target_name] = pd.to_numeric(df[src_col].values, errors='coerce')
                break
    
    if len(raw_vals) < 4:
        logger.warning(f"Only found {len(raw_vals)}/4 DaTScan SBR columns. "
                       f"Available columns: {list(df.columns)}")
        # Return whatever raw values we have
        for k, v in raw_vals.items():
            out[k] = v
        return out.fillna(0)
    
    # Assign raw values
    out["caudate_r"] = raw_vals["caudate_r"]
    out["caudate_l"] = raw_vals["caudate_l"]
    out["putamen_r"] = raw_vals["putamen_r"]
    out["putamen_l"] = raw_vals["putamen_l"]
    
    # Engineer composite features
    out["caudate_mean"] = (out["caudate_r"] + out["caudate_l"]) / 2
    out["putamen_mean"] = (out["putamen_r"] + out["putamen_l"]) / 2
    
    out["asymmetry_caudate"] = np.abs(out["caudate_r"] - out["caudate_l"]) / (out["caudate_mean"] + epsilon)
    out["asymmetry_putamen"] = np.abs(out["putamen_r"] - out["putamen_l"]) / (out["putamen_mean"] + epsilon)
    
    out["striatum_total"] = out["caudate_r"] + out["caudate_l"] + out["putamen_r"] + out["putamen_l"]
    out["caudate_putamen_ratio"] = out["caudate_mean"] / (out["putamen_mean"] + epsilon)
    
    logger.info(f"Loaded DaTScan data: {out.shape[0]} patients, {out.shape[1]} features")
    return out.fillna(0)
