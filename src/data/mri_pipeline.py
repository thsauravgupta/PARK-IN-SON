# -*- coding: utf-8 -*-
"""
MRI feature extraction for Fed-PhenoGraft.

Three explicit sources, selected by ``mri.source`` in config.yaml:

  freesurfer  Regional brain volumes from PPMI's FreeSurfer_MRI_Volumes.csv.
              Real, already-processed structural MRI: no registration, no
              parcellation, no scanner-intensity problems. This is the default.

  nifti       Schaefer-100 ROI parcellation of T1-weighted NIfTI volumes via
              nilearn. Requires images already normalised to MNI space (see
              ``mri.nifti_space``) because resampling an atlas onto a native
              scan changes the voxel grid without registering the anatomy.

  synthetic   Gaussian noise. Keeps the pre-existing results reproducible and
              is never a claim about real imaging.

Subjects without MRI get an all-zero row. That is the project-wide
missing-modality contract: ModalityPreprocessor preserves all-zero rows through
scaling, and FederatedPPMIDataset turns them into mask=1 so the model
substitutes its learned mask token.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# FreeSurfer columns that must NOT be divided by intracranial volume:
# two are already ICV ratios, three are topological defect counts, and the
# ICV column itself is the divisor.
_NO_ICV_SCALING = {
    "EstimatedTotalIntraCranialVol",
    "MaskVol_to_eTIV",
    "BrainSegVol_to_eTIV",
    "lhSurfaceHoles",
    "rhSurfaceHoles",
    "SurfaceHoles",
}

_ICV_CANDIDATES = ("EstimatedTotalIntraCranialVol", "eTIV", "IntraCranialVol",
                   "BrainSegVol")


# ══════════════════════════════════════════════════════════════════
# source resolution
# ══════════════════════════════════════════════════════════════════

def resolve_source(mri_cfg: dict) -> str:
    """Pick the MRI source, honouring the legacy ``use_real_mri`` flag.

    ``mri.source`` wins when present. Otherwise ``use_real_mri: true`` maps to
    'nifti' and false to 'synthetic', so older configs and the README keep
    working unchanged.
    """
    mri_cfg = mri_cfg or {}
    source = mri_cfg.get("source")
    if source:
        source = str(source).lower()
        if source not in ("freesurfer", "nifti", "synthetic"):
            raise ValueError(
                f"mri.source must be 'freesurfer', 'nifti' or 'synthetic', "
                f"got {source!r}."
            )
        return source
    if mri_cfg.get("use_real_mri"):
        logger.info("mri.source not set; legacy use_real_mri=true -> 'nifti'.")
        return "nifti"
    return "synthetic"


# ══════════════════════════════════════════════════════════════════
# caching
# ══════════════════════════════════════════════════════════════════

def _fingerprint(parts) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8", "ignore"))
    return h.hexdigest()[:16]


def _cache_paths(cache_path: Optional[Path]):
    if cache_path is None:
        return None, None
    cache_path = Path(cache_path)
    return cache_path, cache_path.with_suffix(".meta.json")


def _read_cache(cache_path, meta_path, fingerprint):
    if cache_path is None or not cache_path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("fingerprint") != fingerprint:
            return None
        df = pd.read_parquet(cache_path)
        df.index = df.index.astype(np.int64)
        logger.info(f"MRI features loaded from cache: {cache_path}")
        return df
    except Exception as exc:                       # corrupt cache is not fatal
        logger.warning(f"Ignoring unreadable MRI cache ({exc}).")
        return None


def _write_cache(df, cache_path, meta_path, fingerprint, source):
    if cache_path is None:
        return
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)
        meta_path.write_text(
            json.dumps({"fingerprint": fingerprint, "source": source,
                        "n_subjects": int(len(df)),
                        "n_features": int(df.shape[1])}, indent=2),
            encoding="utf-8",
        )
        logger.info(f"MRI features cached to {cache_path}")
    except Exception as exc:
        logger.warning(f"Could not write MRI cache ({exc}).")


# ══════════════════════════════════════════════════════════════════
# source 1 — FreeSurfer regional volumes (real MRI, no NIfTIs needed)
# ══════════════════════════════════════════════════════════════════

def load_freesurfer_volumes(csv_path: Path, visit: str = "BL",
                            normalize_by_icv: bool = True) -> pd.DataFrame:
    """Load PPMI FreeSurfer regional volumes, indexed by PATNO.

    Raw volumes scale with head size, so unless ``normalize_by_icv`` is off
    every volumetric column is divided by the subject's estimated total
    intracranial volume. Ratio columns and surface-defect counts are left
    alone (see ``_NO_ICV_SCALING``).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"mri.source='freesurfer' but the volumes CSV was not found at "
            f"{csv_path}. Download 'FreeSurfer_MRI_Volumes.csv' from IDA-LONI "
            f"(PPMI -> Study Data -> Imaging) into data/raw/mri/, or set "
            f"mri.source to 'synthetic'."
        )

    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="latin-1", low_memory=False)

    if "PATNO" not in df.columns:
        raise ValueError(f"{csv_path.name} has no PATNO column.")

    if "EVENT_ID" in df.columns and visit:
        available = set(df["EVENT_ID"].astype(str).unique())
        if visit in available:
            df = df[df["EVENT_ID"].astype(str) == visit]
        else:
            logger.warning(
                f"Visit {visit!r} absent from {csv_path.name} "
                f"(present: {sorted(available)}); using all rows."
            )

    df = df.drop_duplicates(subset=["PATNO"], keep="last")
    patno = pd.to_numeric(df["PATNO"], errors="coerce")
    df = df.loc[patno.notna()]
    out = df.drop(columns=[c for c in ("PATNO", "EVENT_ID") if c in df.columns])
    out = out.select_dtypes(include=[np.number]).copy()
    out.index = pd.Index(patno.loc[df.index].astype(np.int64), name="PATNO")

    if out.empty:
        raise ValueError(f"No numeric FreeSurfer columns found in {csv_path.name}.")

    if normalize_by_icv:
        icv_col = next((c for c in _ICV_CANDIDATES if c in out.columns), None)
        if icv_col is None:
            logger.warning("No intracranial-volume column found; skipping ICV "
                           "normalisation (head size will confound volumes).")
        else:
            icv = pd.to_numeric(out[icv_col], errors="coerce")
            valid = icv > 0
            if not valid.all():
                logger.warning(f"{int((~valid).sum())} subjects have a "
                               f"non-positive {icv_col}; left unnormalised.")
            scale_cols = [c for c in out.columns if c not in _NO_ICV_SCALING]
            out.loc[valid, scale_cols] = (
                out.loc[valid, scale_cols].div(icv[valid], axis=0)
            )
            logger.info(f"ICV-normalised {len(scale_cols)} volumetric columns "
                        f"by {icv_col}.")

    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float64)

    constant = [c for c in out.columns if out[c].nunique(dropna=False) <= 1]
    if constant:
        # Kept, not dropped: dropping on full-data statistics would be a
        # (mild) leak, and ModalityPreprocessor already neutralises
        # zero-variance columns using train-only statistics.
        logger.info(f"{len(constant)} FreeSurfer columns are constant across "
                    f"the cohort and carry no signal: {constant}")

    logger.info(f"Loaded FreeSurfer volumes: {out.shape[0]} subjects, "
                f"{out.shape[1]} features (visit={visit}).")
    return out


# ══════════════════════════════════════════════════════════════════
# source 2 — NIfTI parcellation
# ══════════════════════════════════════════════════════════════════

def _patno_from_path(path: Path, root: Path) -> Optional[int]:
    """Infer PATNO from a NIfTI path.

    Works for both the flat layout documented in DATASET_DOWNLOAD.md
    (``data/raw/mri/3000/T1w.nii.gz``) and the nested layout IDA actually
    ships (``.../PPMI/3000/MPRAGE/2011-01-01/I123456/PPMI_3000_..._.nii``).
    """
    try:
        rel_parts = path.relative_to(root).parts
    except ValueError:
        rel_parts = path.parts
    for part in rel_parts:                      # outermost integer directory
        if part.isdigit():
            return int(part)
    stem = path.name
    for token in stem.replace("-", "_").split("_"):
        if token.isdigit() and len(token) >= 4:
            return int(token)
    return None


def find_nifti_files(mri_dir: Path) -> dict:
    """Map PATNO -> chosen NIfTI path, searching recursively.

    The previous implementation globbed only one directory level, so a stock
    IDA download matched nothing and the whole modality silently became zeros.
    """
    mri_dir = Path(mri_dir)
    nifti_map: dict = {}
    if not mri_dir.exists():
        logger.warning(f"MRI directory does not exist: {mri_dir}")
        return nifti_map

    candidates: dict = {}
    for path in sorted(mri_dir.rglob("*.nii*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        patno = _patno_from_path(path, mri_dir)
        if patno is not None:
            candidates.setdefault(patno, []).append(path)

    t1_keys = ("t1", "mprage", "spgr", "structural")
    for patno, paths in candidates.items():
        t1 = [p for p in paths if any(k in str(p).lower() for k in t1_keys)]
        nifti_map[patno] = sorted(t1 or paths)[0]

    logger.info(f"Found NIfTI volumes for {len(nifti_map)} subjects under {mri_dir}")
    return nifti_map


def _build_masker(n_rois: int):
    """Fetch the Schaefer atlas once and return (masker, label_ids)."""
    from nilearn import datasets
    from nilearn.maskers import NiftiLabelsMasker
    import nibabel as nib

    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=n_rois, yeo_networks=7, resolution_mm=2)
    atlas_img = nib.load(atlas.maps) if isinstance(atlas.maps, (str, Path)) else atlas.maps
    label_ids = [int(v) for v in np.unique(np.asarray(atlas_img.dataobj)) if int(v) != 0]
    masker = NiftiLabelsMasker(labels_img=atlas_img, strategy="mean",
                               standardize=False, detrend=False, resampling_target="data")
    return masker, label_ids, atlas_img


def extract_roi_features(nifti_path, masker, label_ids, n_rois: int) -> Optional[np.ndarray]:
    """Extract one subject's parcel means, addressed by atlas label id.

    Returns a fixed-length vector where column *k* is always Schaefer label
    ``label_ids[k]``. Parcels that fall outside the image are left at zero
    rather than shifting every later parcel along, which is what the previous
    end-padding did.

    Values are divided by the subject's mean in-brain intensity, because raw
    T1 intensity is scanner- and protocol-dependent and not comparable across
    subjects.
    """
    import nibabel as nib
    from nilearn import image

    try:
        img = nib.load(str(nifti_path))
        if img.ndim == 4:
            img = image.mean_img(img)
        elif img.ndim != 3:
            logger.warning(f"Skipping {nifti_path.name}: {img.ndim}D image.")
            return None

        data = np.asarray(img.dataobj, dtype=np.float32)
        img3 = nib.Nifti1Image(data, img.affine)

        brain = data[data > 0]
        scale = float(np.mean(brain)) if brain.size else 0.0
        if scale <= 0:
            logger.warning(f"Skipping {nifti_path.name}: empty or non-positive volume.")
            return None

        signals = masker.fit_transform(nib.Nifti1Image(data[..., None], img.affine))
        signals = np.asarray(signals, dtype=np.float64).ravel()

        found = getattr(masker, "labels_", None)
        if found is None or len(found) != len(signals):
            found = label_ids[:len(signals)]
        slot = {int(lbl): i for i, lbl in enumerate(label_ids)}

        feats = np.zeros(n_rois, dtype=np.float32)
        for value, lbl in zip(signals, found):
            idx = slot.get(int(lbl))
            if idx is not None and idx < n_rois:
                feats[idx] = value / scale
        del data, img3
        return feats

    except Exception as exc:
        logger.warning(f"Failed to process {nifti_path}: {exc}")
        return None


def _build_from_nifti(mri_dir: Path, patnos: list, n_rois: int,
                      nifti_space: str) -> pd.DataFrame:
    try:
        import nibabel  # noqa: F401
        import nilearn  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "mri.source='nifti' needs nibabel and nilearn: "
            "pip install nibabel nilearn"
        ) from exc

    if str(nifti_space).lower() != "mni":
        raise ValueError(
            "mri.source='nifti' requires images already normalised to MNI "
            f"space, but mri.nifti_space is {nifti_space!r}. The Schaefer "
            "atlas is defined in MNI space; resampling it onto a native-space "
            "scan changes the voxel grid without registering the anatomy, so "
            "the parcels would land on the wrong structures. Either register "
            "the T1w volumes to MNI first (e.g. with fMRIPrep or ANTs) and set "
            "mri.nifti_space: mni, or use mri.source: freesurfer."
        )

    nifti_map = find_nifti_files(mri_dir)
    if not nifti_map:
        raise FileNotFoundError(
            f"mri.source='nifti' but no .nii/.nii.gz files were found under "
            f"{mri_dir} (searched recursively). Place the scans there - both "
            f"'{{PATNO}}/T1w.nii.gz' and the nested IDA layout "
            f"'PPMI/{{PATNO}}/{{Series}}/{{Date}}/{{ImageID}}/*.nii' are "
            f"understood - or set mri.source to 'freesurfer' or 'synthetic'. "
            f"Refusing to continue silently with an all-zero MRI modality."
        )

    masker, label_ids, _ = _build_masker(n_rois)
    wanted = [p for p in patnos if p in nifti_map]
    logger.info(f"Parcellating {len(wanted)} volumes into {n_rois} Schaefer ROIs "
                f"(atlas fetched once, features cached afterwards)...")

    rows, index, failed = [], [], 0
    for i, patno in enumerate(wanted, 1):
        feats = extract_roi_features(nifti_map[patno], masker, label_ids, n_rois)
        if feats is None:
            failed += 1
            continue
        rows.append(feats)
        index.append(int(patno))
        if i % 25 == 0 or i == len(wanted):
            logger.info(f"  {i}/{len(wanted)} processed ({failed} failed)")

    if not rows:
        raise RuntimeError(
            f"Found {len(nifti_map)} NIfTI files but every one failed to "
            f"parcellate. Check the logged warnings above."
        )

    df = pd.DataFrame(np.vstack(rows), index=pd.Index(index, name="PATNO"),
                      columns=[f"ROI_{i}" for i in range(n_rois)])
    logger.info(f"NIfTI parcellation complete: {len(df)} subjects, {failed} failed.")
    return df


# ══════════════════════════════════════════════════════════════════
# source 3 — synthetic fallback
# ══════════════════════════════════════════════════════════════════

def _build_synthetic(patnos: list, n_rois: int, seed: int = 42) -> pd.DataFrame:
    logger.warning("MRI source is SYNTHETIC: these features are Gaussian noise "
                   "and carry no imaging signal. Results must not be reported "
                   "as real structural MRI.")
    # Legacy global-seed RNG on purpose: this exact stream reproduces the
    # committed synthetic-MRI results in outputs/results/final_metrics.json.
    # Switching to default_rng would silently invalidate that baseline.
    np.random.seed(seed)
    synth = np.random.randn(len(patnos), n_rois)
    synth[np.random.rand(len(patnos)) < 0.2] = 0.0      # simulate 20% missing
    return pd.DataFrame(synth, index=pd.Index(patnos, name="PATNO"),
                        columns=[f"ROI_{i}" for i in range(n_rois)])


# ══════════════════════════════════════════════════════════════════
# entry point
# ══════════════════════════════════════════════════════════════════

def build_mri_features(mri_dir, patnos, mri_cfg=None, project_root=None,
                       n_rois: int = 100, use_real_mri: Optional[bool] = None
                       ) -> pd.DataFrame:
    """Build the MRI feature matrix for ``patnos``.

    Returns a DataFrame indexed by PATNO with one row per requested subject.
    Subjects without MRI are all-zero, which downstream code reads as
    "modality not collected".
    """
    mri_cfg = dict(mri_cfg or {})
    if use_real_mri is not None and "use_real_mri" not in mri_cfg:
        mri_cfg["use_real_mri"] = use_real_mri
    n_rois = int(mri_cfg.get("n_rois", n_rois))
    source = resolve_source(mri_cfg)
    root = Path(project_root) if project_root else Path(mri_dir).resolve().parents[2]

    logger.info(f"MRI source: {source}")

    cache_cfg = mri_cfg.get("cache")
    cache_path, meta_path = _cache_paths(root / cache_cfg if cache_cfg else None)

    if source == "freesurfer":
        csv_path = root / mri_cfg.get(
            "freesurfer_csv", "data/raw/mri/FreeSurfer_MRI_Volumes.csv")
        visit = mri_cfg.get("freesurfer_visit", "BL")
        norm = bool(mri_cfg.get("normalize_by_icv", True))
        stat = csv_path.stat() if csv_path.exists() else None
        fp = _fingerprint(["freesurfer", visit, norm,
                           stat.st_mtime_ns if stat else 0,
                           stat.st_size if stat else 0])
        features = _read_cache(cache_path, meta_path, fp)
        if features is None:
            features = load_freesurfer_volumes(csv_path, visit=visit,
                                               normalize_by_icv=norm)
            _write_cache(features, cache_path, meta_path, fp, source)

    elif source == "nifti":
        nifti_dir = Path(mri_dir)
        space = mri_cfg.get("nifti_space", "mni")
        found = find_nifti_files(nifti_dir) if nifti_dir.exists() else {}
        fp = _fingerprint(["nifti", n_rois, space, len(found)] +
                          [f"{k}:{Path(v).stat().st_mtime_ns}" for k, v in
                           sorted(found.items())])
        features = _read_cache(cache_path, meta_path, fp)
        if features is None:
            features = _build_from_nifti(nifti_dir, patnos, n_rois, space)
            _write_cache(features, cache_path, meta_path, fp, source)

    else:
        features = _build_synthetic(patnos, n_rois)

    # Align to the requested cohort; absent subjects become all-zero rows.
    index = pd.Index([int(p) for p in patnos], name="PATNO")
    features = features[~features.index.duplicated(keep="last")]
    aligned = features.reindex(index).astype(np.float64)
    aligned = aligned.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    present = int((aligned.abs().sum(axis=1) > 0).sum())
    logger.info(f"MRI matrix: {aligned.shape[0]} subjects x {aligned.shape[1]} "
                f"features | {present} with data, {len(aligned) - present} "
                f"missing (mask tokens).")
    if source != "synthetic" and present == 0:
        raise RuntimeError(
            f"MRI source '{source}' produced no usable rows for any of the "
            f"{len(index)} cohort subjects. PATNO alignment has probably "
            f"failed. Refusing to continue with an all-zero MRI modality."
        )
    return aligned
