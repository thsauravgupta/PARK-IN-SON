# -*- coding: utf-8 -*-
"""
MRI Processing Pipeline for PPMI NIfTI Brain Scans.
Uses nilearn Schaefer atlas parcellation to extract ROI-level volumetric features
from T1-weighted structural MRI images.

Expected directory layout:
    data/raw/mri/
        ├── 3001/          # PATNO directories
        │   └── *.nii.gz   # T1w NIfTI files  
        ├── 3002/
        │   └── *.nii.gz
        └── ...
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _find_nifti_files(mri_dir: Path) -> dict:
    """
    Scans the MRI directory for NIfTI files organized by PATNO.
    Returns dict: {patno: Path_to_nifti}
    """
    nifti_map = {}
    
    if not mri_dir.exists():
        logger.warning(f"MRI directory does not exist: {mri_dir}")
        return nifti_map
    
    for subdir in sorted(mri_dir.iterdir()):
        if not subdir.is_dir():
            continue
        
        try:
            patno = int(subdir.name)
        except ValueError:
            continue
        
        # Find .nii or .nii.gz files
        niftis = list(subdir.glob("*.nii.gz")) + list(subdir.glob("*.nii"))
        
        if niftis:
            # Prefer T1-weighted if multiple scans exist
            t1_files = [f for f in niftis if any(k in f.name.lower() for k in ["t1", "mprage", "spgr", "structural"])]
            chosen = t1_files[0] if t1_files else niftis[0]
            nifti_map[patno] = chosen
    
    logger.info(f"Found {len(nifti_map)} NIfTI files across patient directories")
    return nifti_map


def extract_roi_features(nifti_path: Path, n_rois: int = 100) -> Optional[np.ndarray]:
    """
    Extract regional brain features from a single NIfTI file using Schaefer atlas.
    
    Steps:
        1. Load the NIfTI image
        2. Fetch Schaefer atlas with n_rois parcels
        3. Apply NiftiLabelsMasker to extract mean signal per ROI
    
    Returns:
        np.ndarray of shape (n_rois,) or None if processing fails
    """
    try:
        import nibabel as nib
        from nilearn import datasets, image
        from nilearn.maskers import NiftiLabelsMasker
    except ImportError:
        logger.error("nibabel and nilearn are required for MRI processing. "
                      "Install via: pip install nibabel nilearn")
        return None
    
    try:
        # Load and validate
        img = nib.load(str(nifti_path))
        data = img.get_fdata()
        
        if data.ndim < 3:
            logger.warning(f"Unexpected image dimensions ({data.ndim}D) for {nifti_path}")
            return None
        
        # If 4D (fMRI-like), take mean across time
        if data.ndim == 4:
            img = image.mean_img(img)
        
        # Fetch Schaefer atlas (downloads once, cached thereafter)
        atlas = datasets.fetch_atlas_schaefer_2018(
            n_rois=n_rois,
            yeo_networks=7,
            resolution_mm=2,
        )
        
        # Resample atlas to match image space
        atlas_img = image.resample_to_img(atlas.maps, img, interpolation="nearest")
        
        # Extract mean signal per parcel
        masker = NiftiLabelsMasker(
            labels_img=atlas_img,
            standardize="zscore_sample",
            strategy="mean",
            detrend=False,
        )
        
        # For a single 3D structural image, we need to add a time dimension
        if img.ndim == 3:
            data_4d = data[..., np.newaxis]
            img_4d = nib.Nifti1Image(data_4d, img.affine, img.header)
        else:
            img_4d = img
        
        signals = masker.fit_transform(img_4d)  # shape: (1, n_rois)
        
        roi_features = signals.flatten()
        
        # Pad or truncate to expected dimension
        if len(roi_features) < n_rois:
            roi_features = np.pad(roi_features, (0, n_rois - len(roi_features)))
        elif len(roi_features) > n_rois:
            roi_features = roi_features[:n_rois]
        
        return roi_features
        
    except Exception as e:
        logger.warning(f"Failed to process {nifti_path}: {e}")
        return None


def build_mri_features(mri_dir: Path, patnos: list, n_rois: int = 100,
                       use_real_mri: bool = True) -> pd.DataFrame:
    """
    Build MRI feature matrix for all patients.
    
    Args:
        mri_dir: Path to directory containing PATNO subdirectories with NIfTI files
        patnos: List of patient IDs to process
        n_rois: Number of ROIs in Schaefer atlas
        use_real_mri: If True, attempt real NIfTI processing; if False, use synthetic
        
    Returns:
        pd.DataFrame of shape (n_patients, n_rois) indexed by PATNO.
        Patients with missing MRI will have all-zero rows (detected as missing by mask tokens).
    """
    mri_dir = Path(mri_dir)
    col_names = [f"ROI_{i}" for i in range(n_rois)]
    
    if not use_real_mri:
        logger.info("MRI set to synthetic fallback mode.")
        np.random.seed(42)
        synth = np.random.randn(len(patnos), n_rois)
        # Zero out ~20% to simulate missing
        missing = np.random.rand(len(patnos)) < 0.2
        synth[missing] = 0.0
        return pd.DataFrame(synth, index=patnos, columns=col_names)
    
    # Discover available NIfTI files
    nifti_map = _find_nifti_files(mri_dir)
    
    features = np.zeros((len(patnos), n_rois), dtype=np.float32)
    processed = 0
    missing_count = 0
    
    for i, patno in enumerate(patnos):
        if patno in nifti_map:
            roi_feats = extract_roi_features(nifti_map[patno], n_rois)
            if roi_feats is not None:
                features[i] = roi_feats
                processed += 1
            else:
                missing_count += 1
        else:
            missing_count += 1
    
    logger.info(f"MRI processing complete: {processed} processed, "
                f"{missing_count} missing (will use mask tokens)")
    
    return pd.DataFrame(features, index=patnos, columns=col_names)
