# -*- coding: utf-8 -*-
"""
Clinical Data Loader for PPMI.
Merges UPDRS-I through IV, Demographics, MoCA, and Patient Status CSVs
into a single patient-level DataFrame indexed by PATNO.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


def _safe_read(path: Path) -> pd.DataFrame:
    """Read CSV with fallback encoding."""
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding='latin-1', low_memory=False)


def _filter_visit(df: pd.DataFrame, visit: str, patno_col: str = "PATNO") -> pd.DataFrame:
    """Filter to a specific EVENT_ID visit and deduplicate by PATNO."""
    if "EVENT_ID" in df.columns:
        df = df[df["EVENT_ID"] == visit].copy()
    return df.drop_duplicates(subset=[patno_col], keep="last")


def load_updrs(raw_dir: Path, visit: str = "BL") -> pd.DataFrame:
    """
    Loads and merges MDS-UPDRS Part I–IV CSVs.
    Expected files: MDS_UPDRS_Part_I.csv, MDS_UPDRS_Part_II.csv,
                    MDS_UPDRS_Part_III.csv, MDS_UPDRS_Part_IV.csv
    """
    parts = {}
    mapping = {
        "I": "updrs_i",
        "II": "updrs_ii",
        "III": "updrs_iii",
        "IV": "updrs_iv",
    }

    for part_num, col_name in mapping.items():
        fname = f"MDS_UPDRS_Part_{part_num}.csv"
        fpath = raw_dir / fname
        if not fpath.exists():
            # Try alternate naming conventions used by PPMI
            alt_names = [
                f"MDS-UPDRS_Part_{part_num}.csv",
                f"MDS_UPDRS_Part_{part_num}__Patient_Questionnaire.csv",
                f"MDS_UPDRS_Part_{part_num}_.csv",
            ]
            for alt in alt_names:
                if (raw_dir / alt).exists():
                    fpath = raw_dir / alt
                    break

        if not fpath.exists():
            logger.warning(f"UPDRS Part {part_num} CSV not found at {fpath}, skipping.")
            continue

        df = _safe_read(fpath)
        df = _filter_visit(df, visit)

        # Prefer the official PPMI total column (NP1RTOT/NP1PTOT/NP2PTOT/
        # NP3TOT/NP4TOT). Summing raw NP* columns must EXCLUDE the total
        # (double-counting) and NHY (Hoehn & Yahr stage, not a UPDRS item).
        tot_cols = [c for c in df.columns
                    if c.upper().startswith("NP") and c.upper().endswith("TOT")]
        item_cols = [c for c in df.columns
                     if c.startswith("NP") and not c.upper().endswith("TOT")]
        if tot_cols:
            df[col_name] = pd.to_numeric(df[tot_cols].sum(axis=1), errors='coerce')
        elif item_cols:
            df[col_name] = pd.to_numeric(df[item_cols].sum(axis=1), errors='coerce')
        elif col_name.upper() in df.columns:
            df[col_name] = pd.to_numeric(df[col_name.upper()], errors='coerce')
        else:
            total_cols = [c for c in df.columns if "total" in c.lower() or "score" in c.lower()]
            if total_cols:
                df[col_name] = pd.to_numeric(df[total_cols[0]], errors='coerce')

        parts[col_name] = df[["PATNO", col_name]].set_index("PATNO")

    if not parts:
        logger.error("No UPDRS data could be loaded!")
        return pd.DataFrame()

    merged = pd.concat(parts.values(), axis=1, join="outer")
    logger.info(f"Loaded UPDRS data: {merged.shape[0]} patients, {merged.shape[1]} columns")
    return merged


def load_demographics(raw_dir: Path) -> pd.DataFrame:
    """
    Loads demographic data: age, gender, education.
    Tries multiple PPMI naming conventions.
    """
    candidates = [
        "Demographics.csv",
        "Screening___Demographics.csv",
        "Screening_Demographics.csv",
    ]

    for fname in candidates:
        fpath = raw_dir / fname
        if fpath.exists():
            df = _safe_read(fpath)
            break
    else:
        logger.warning("Demographics CSV not found.")
        return pd.DataFrame()

    df = df.drop_duplicates(subset=["PATNO"], keep="last")

    out = pd.DataFrame(index=df["PATNO"])
    out.index.name = "PATNO"

    # Age — prefer PPMI's Age_at_visit.csv (exact age at baseline visit);
    # fall back to parsing BIRTHDT, which PPMI stores as "MM/YYYY" text.
    age_path = raw_dir / "Age_at_visit.csv"
    if age_path.exists():
        age_df = _safe_read(age_path)
        age_df = _filter_visit(age_df, "BL")
        if "AGE_AT_VISIT" in age_df.columns:
            ages = pd.Series(pd.to_numeric(age_df["AGE_AT_VISIT"], errors='coerce').values,
                             index=age_df["PATNO"].values)
            out["age"] = ages.reindex(out.index).values
    if "age" not in out.columns or out["age"].isna().all():
        for col in ["BIRTHDT", "BIRTH_DATE", "AGE", "AGEAT_BL", "age_at_visit"]:
            if col in df.columns:
                if col in ["BIRTHDT", "BIRTH_DATE"]:
                    # BIRTHDT is "MM/YYYY" — extract the year component
                    birth_year = pd.to_numeric(
                        df[col].astype(str).str.extract(r"(\d{4})")[0], errors='coerce'
                    )
                    out["age"] = (2024 - birth_year).values
                else:
                    out["age"] = pd.to_numeric(df[col].values, errors='coerce')
                break

    # Gender: 0=Female, 1=Male, 2=Other
    for col in ["SEX", "GENDER", "GENDER"]:
        if col in df.columns:
            out["gender"] = pd.to_numeric(df[col].values, errors='coerce')
            break

    # Education (years)
    for col in ["EDUCYRS", "EDUCATION", "EDUC"]:
        if col in df.columns:
            out["education"] = pd.to_numeric(df[col].values, errors='coerce')
            break

    logger.info(f"Loaded demographics: {out.shape[0]} patients")
    return out


def load_moca(raw_dir: Path, visit: str = "BL") -> pd.DataFrame:
    """Loads Montreal Cognitive Assessment scores."""
    candidates = [
        "MoCA.csv",
        "Montreal_Cognitive_Assessment__MoCA_.csv",
        "Montreal_Cognitive_Assessment.csv",
    ]

    for fname in candidates:
        fpath = raw_dir / fname
        if fpath.exists():
            df = _safe_read(fpath)
            break
    else:
        logger.warning("MoCA CSV not found.")
        return pd.DataFrame()

    df = _filter_visit(df, visit)

    # MoCA total score
    score_col = None
    for col in ["MCATOT", "MOCA_TOTAL", "TOTAL", "MCATOT_TOTAL"]:
        if col in df.columns:
            score_col = col
            break

    if score_col is None:
        logger.warning("Could not find MoCA total score column.")
        return pd.DataFrame()

    out = pd.DataFrame({"moca": pd.to_numeric(df[score_col].values, errors='coerce')},
                       index=df["PATNO"].values)
    out.index.name = "PATNO"
    logger.info(f"Loaded MoCA: {out.shape[0]} patients")
    return out


def load_patient_status(raw_dir: Path) -> pd.DataFrame:
    """
    Loads diagnosis labels from Patient_Status or Participant_Status CSV.
    Returns binary label: 1 = PD, 0 = HC (Healthy Control).
    """
    candidates = [
        "Patient_Status.csv",
        "Participant_Status.csv",
    ]

    for fname in candidates:
        fpath = raw_dir / fname
        if fpath.exists():
            df = _safe_read(fpath)
            break
    else:
        logger.warning("Patient status CSV not found.")
        return pd.DataFrame()

    df = df.drop_duplicates(subset=["PATNO"], keep="last")

    # Map enrollment categories to binary
    status_col = None
    for col in ["ENROLL_CAT", "COHORT", "APPRDX", "DIAGNOSIS"]:
        if col in df.columns:
            status_col = col
            break

    if status_col is None:
        logger.warning("Could not find diagnosis/enrollment column.")
        return pd.DataFrame()

    # PD = 1, everything else (HC, SWEDD, Prodromal) = 0 for binary classification
    diagnosis_map = {
        "PD": 1, "Parkinson's Disease": 1, "Parkinson Disease": 1,
        "HC": 0, "Healthy Control": 0,
        "SWEDD": 0,  # Scans Without Evidence of Dopaminergic Deficit
        "Prodromal": 0,
    }

    out = pd.DataFrame(index=df["PATNO"].values)
    out.index.name = "PATNO"

    raw_vals = df[status_col].astype(str).values
    mapped = []
    for v in raw_vals:
        v_clean = v.strip()
        if v_clean in diagnosis_map:
            mapped.append(diagnosis_map[v_clean])
        else:
            # Try numeric: 1=PD, 2=HC in some PPMI versions
            try:
                num = int(float(v_clean))
                mapped.append(1 if num == 1 else 0)
            except ValueError:
                mapped.append(np.nan)

    out["diagnosis"] = mapped
    logger.info(f"Loaded patient status: {out.shape[0]} patients, PD={sum(d==1 for d in mapped)}")
    return out


def load_target_updrs(raw_dir: Path, visit: str = "V04") -> pd.DataFrame:
    """
    Loads the target UPDRS-III score at a future visit (default: V04 = Year 2)
    for the regression task.
    """
    fpath = raw_dir / "MDS_UPDRS_Part_III.csv"
    if not fpath.exists():
        # Try alternatives
        for alt in ["MDS-UPDRS_Part_III.csv", "MDS_UPDRS_Part_III_.csv"]:
            if (raw_dir / alt).exists():
                fpath = raw_dir / alt
                break

    if not fpath.exists():
        logger.warning("UPDRS-III CSV not found for target extraction.")
        return pd.DataFrame()

    df = _safe_read(fpath)
    df = _filter_visit(df, visit)

    # Use the official NP3TOT total when present; otherwise sum the 33
    # individual NP3 items (never both — that doubles the score).
    if "NP3TOT" in df.columns:
        df["updrs_iii_target"] = pd.to_numeric(df["NP3TOT"], errors='coerce')
        np_cols = ["NP3TOT"]
    else:
        np_cols = [c for c in df.columns
                   if c.startswith("NP3") and not c.upper().endswith("TOT")]
    if np_cols and "updrs_iii_target" not in df.columns:
        df["updrs_iii_target"] = pd.to_numeric(df[np_cols].sum(axis=1), errors='coerce')
    if "updrs_iii_target" not in df.columns:
        total_cols = [c for c in df.columns if "total" in c.lower()]
        if total_cols:
            df["updrs_iii_target"] = pd.to_numeric(df[total_cols[0]], errors='coerce')
        else:
            logger.warning("Cannot compute UPDRS-III target score.")
            return pd.DataFrame()

    out = df[["PATNO", "updrs_iii_target"]].set_index("PATNO")
    logger.info(f"Loaded regression target (UPDRS-III @ {visit}): {out.shape[0]} patients")
    return out


def build_clinical_features(raw_dir: Path, baseline_visit: str = "BL",
                            target_visit: str = "V04",
                            target_mode: str = "absolute") -> tuple:
    """
    Master function: assembles all clinical sub-tables into a single feature DataFrame
    and a separate targets DataFrame.

    Args:
        target_mode: 'absolute' → UPDRS-III score at target_visit;
                     'delta'    → UPDRS-III change (target_visit − baseline_visit).
                     Delta targets remove the baseline-score autocorrelation that
                     otherwise dominates absolute-score prediction, so the model
                     is evaluated on true progression signal.

    Returns:
        features: pd.DataFrame indexed by PATNO (clinical + demographics + MoCA)
        targets: pd.DataFrame indexed by PATNO (updrs_iii_target, diagnosis)
    """
    raw_dir = Path(raw_dir)

    updrs = load_updrs(raw_dir, visit=baseline_visit)
    demo = load_demographics(raw_dir)
    moca = load_moca(raw_dir, visit=baseline_visit)
    status = load_patient_status(raw_dir)
    target_updrs = load_target_updrs(raw_dir, visit=target_visit)

    if target_mode == "delta" and not target_updrs.empty:
        # Baseline NP3TOT from the same loader (same official-total logic) so
        # the delta is computed on directly comparable scores. Subjects missing
        # either visit get NaN and are dropped downstream — never imputed.
        baseline_updrs = load_target_updrs(raw_dir, visit=baseline_visit)
        if baseline_updrs.empty:
            logger.warning("target_mode='delta' requested but baseline UPDRS-III "
                           "could not be loaded — falling back to absolute target.")
        else:
            bl = baseline_updrs["updrs_iii_target"].reindex(target_updrs.index)
            target_updrs = target_updrs.copy()
            target_updrs["updrs_iii_target"] = target_updrs["updrs_iii_target"] - bl
            n_valid = int(target_updrs["updrs_iii_target"].notna().sum())
            logger.info(f"Delta target (UPDRS-III @ {target_visit} − @ "
                        f"{baseline_visit}): {n_valid} subjects with both visits.")

    # Merge features
    dfs = [df for df in [updrs, demo, moca] if not df.empty]
    if not dfs:
        logger.error("No clinical features could be loaded!")
        return pd.DataFrame(), pd.DataFrame()

    features = pd.concat(dfs, axis=1, join="outer")

    # Merge targets
    target_dfs = [df for df in [target_updrs, status] if not df.empty]
    if target_dfs:
        targets = pd.concat(target_dfs, axis=1, join="outer")
    else:
        targets = pd.DataFrame(index=features.index)

    # Align indices
    common = features.index.intersection(targets.index)
    features = features.loc[common]
    targets = targets.loc[common]

    logger.info(f"Final clinical dataset: {features.shape[0]} patients, "
                f"{features.shape[1]} features, {targets.shape[1]} targets")
    return features, targets
