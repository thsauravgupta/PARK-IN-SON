# -*- coding: utf-8 -*-
"""
LEGACY MRI processor — kept only for scripts/generate_embeddings.py.

DO NOT USE for new work. The supported MRI path is src/data/mri_pipeline.py,
which src/main.py calls. This module has three defects that mri_pipeline.py
fixes, and they are recorded here so nobody re-adopts it by accident:

  1. PATNO is guessed with ``f.stem.split("_")[1]`` and kept as a STRING, so
     the resulting index never joins the integer PATNO index used everywhere
     else in the project.
  2. The Schaefer atlas is applied with no resampling at all, so it only works
     if the image already sits on the atlas grid.
  3. ``standardize=True`` on a single-volume structural scan is a no-op that
     nilearn warns about, leaving raw scanner-dependent intensities.

See src/data/mri_pipeline.py for the maintained implementation.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

try:
    from nilearn.maskers import NiftiLabelsMasker
    from nilearn import datasets
    NILEARN_AVAILABLE = True
except ImportError:
    NILEARN_AVAILABLE = False

class MRIProcessor:
    """
    Processes real MRI NIfTI files into ROI features using Schaefer atlas.
    """
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.n_rois = config.get("mri", {}).get("n_rois", 100)
        self.scaler = StandardScaler()
        
        if NILEARN_AVAILABLE:
            self.logger.info("Fetching Schaefer atlas...")
            # We use a dummy fetch here for structure; actual download happens automatically
            self.atlas = datasets.fetch_atlas_schaefer_2018(n_rois=self.n_rois)
            self.masker = NiftiLabelsMasker(labels_img=self.atlas.maps, standardize=True)
        else:
            self.logger.warning("Nilearn not available. Cannot process real MRI.")

    def extract_roi_features(self, nifti_path: Path) -> np.ndarray:
        """Extract ROI features for a single patient."""
        if not NILEARN_AVAILABLE:
            raise ImportError("Nilearn required for extract_roi_features")
        
        self.logger.debug(f"Extracting features from {nifti_path}")
        time_series = self.masker.fit_transform(str(nifti_path))
        # Take the mean across time (if 4D) or just return the 1D array (if 3D)
        return np.mean(time_series, axis=0) if time_series.ndim > 1 else time_series

    def process_all(self, mri_dir: Path, output_path: Path) -> pd.DataFrame:
        """Iterate over all NIfTIs and extract features."""
        nifti_files = list(mri_dir.rglob("*.nii*"))
        self.logger.info(f"Processing {len(nifti_files)} NIfTI files...")
        
        results = []
        patnos = []
        
        for f in nifti_files:
            try:
                # Naive PATNO extraction from filename (e.g., "PPMI_12345_...")
                patno = f.stem.split("_")[1] 
                feats = self.extract_roi_features(f)
                results.append(feats)
                patnos.append(patno)
            except Exception as e:
                self.logger.error(f"Failed to process {f.name}: {e}")
                
        df = pd.DataFrame(results, index=patnos, columns=[f"ROI_{i}" for i in range(self.n_rois)])
        df.index.name = "PATNO"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path)
        self.logger.info(f"Saved MRI features to {output_path}")
        return df
        
    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """StandardScaler normalization."""
        out = pd.DataFrame(self.scaler.fit_transform(df), index=df.index, columns=df.columns)
        return out
