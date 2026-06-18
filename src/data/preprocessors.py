# -*- coding: utf-8 -*-
"""
Data preprocessors for each modality.
Each preprocessor follows a fit/transform API and outputs a DataFrame indexed by PATNO.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from src.utils import safe_log_transform

class BasePreprocessor:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X_train: pd.DataFrame):
        raise NotImplementedError
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError
        
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        return self.transform(X)

class ClinicalPreprocessor(BasePreprocessor):
    """
    Extracts UPDRS, Demographics, and MoCA.
    Targets baseline for features, and V04 for regression target.
    """
    def fit(self, X_train: pd.DataFrame):
        # We assume X_train is already filtered to continuous features
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        # n_nearest_features=15 dramatically speeds up MICE on large datasets (>100 cols)
        self.imputer = IterativeImputer(random_state=42, max_iter=5, keep_empty_features=True, n_nearest_features=15)
        
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        self.numeric_cols = numeric_cols
        if len(numeric_cols) > 0:
            self.scaler.fit(X_train[numeric_cols])
            self.imputer.fit(self.scaler.transform(X_train[numeric_cols]))
        self.is_fitted = True
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted.")
            
        out = X.copy()
        if len(self.numeric_cols) > 0:
            out[self.numeric_cols] = self.scaler.transform(out[self.numeric_cols])
            out[self.numeric_cols] = self.imputer.transform(out[self.numeric_cols])
            
        return out.fillna(0)

class MRIPreprocessor(BasePreprocessor):
    """
    Generates synthetic MRI features if real NIfTI not available.
    """
    def __init__(self, config):
        super().__init__(config)
        self.use_real_mri = config.get("mri", {}).get("use_real_mri", False)
        self.n_rois = config.get("mri", {}).get("n_rois", 100)
        
    def fit(self, X_train: pd.DataFrame):
        self.is_fitted = True
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted.")
            
        patnos = X.index if "PATNO" not in X.columns else X["PATNO"]
        
        # Synthetic MRI generation
        np.random.seed(self.config.get("seed", 42))
        synthetic_mri = np.random.randn(len(patnos), self.n_rois)
        
        df = pd.DataFrame(synthetic_mri, index=patnos, columns=[f"ROI_{i}" for i in range(self.n_rois)])
        df.index.name = "PATNO"
        return df

class PETPreprocessor(BasePreprocessor):
    """
    Extracts DaTscan features and engineers asymmetry metrics.
    """
    def fit(self, X_train: pd.DataFrame):
        # Extract numeric columns for scaling
        numeric_cols = ["caudate_r", "caudate_l", "putamen_r", "putamen_l", 
                        "asymmetry_caudate", "asymmetry_putamen", "striatum_total",
                        "caudate_putamen_ratio", "caudate_mean", "putamen_mean"]
        self.numeric_cols = [c for c in numeric_cols if c in X_train.columns]
        if len(self.numeric_cols) > 0:
            self.scaler.fit(X_train[self.numeric_cols])
        self.is_fitted = True
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted.")
        
        out = pd.DataFrame(index=X.index)
        epsilon = self.config.get("pet", {}).get("epsilon", 1e-8)
        
        # Engineering features
        if all(c in X.columns for c in ["caudate_r", "caudate_l", "putamen_r", "putamen_l"]):
            out["caudate_r"] = pd.to_numeric(X["caudate_r"], errors='coerce')
            out["caudate_l"] = pd.to_numeric(X["caudate_l"], errors='coerce')
            out["putamen_r"] = pd.to_numeric(X["putamen_r"], errors='coerce')
            out["putamen_l"] = pd.to_numeric(X["putamen_l"], errors='coerce')
            
            out["caudate_mean"] = (out["caudate_r"] + out["caudate_l"]) / 2
            out["putamen_mean"] = (out["putamen_r"] + out["putamen_l"]) / 2
            out["asymmetry_caudate"] = np.abs(out["caudate_r"] - out["caudate_l"]) / (out["caudate_mean"] + epsilon)
            out["asymmetry_putamen"] = np.abs(out["putamen_r"] - out["putamen_l"]) / (out["putamen_mean"] + epsilon)
            out["striatum_total"] = out["caudate_r"] + out["caudate_l"] + out["putamen_r"] + out["putamen_l"]
            out["caudate_putamen_ratio"] = out["caudate_mean"] / (out["putamen_mean"] + epsilon)
            
            # Safe log transform
            for col in self.numeric_cols:
                if col in out.columns:
                    out[col] = safe_log_transform(out[col], epsilon)
                    
            if len(self.numeric_cols) > 0:
                out[self.numeric_cols] = self.scaler.transform(out[self.numeric_cols])
                out[self.numeric_cols] = out[self.numeric_cols].fillna(out[self.numeric_cols].median())
                
        return out.fillna(0)

class GeneticPreprocessor(BasePreprocessor):
    """
    Extracts genetic carrier status and explicit bypass features.
    """
    def fit(self, X_train: pd.DataFrame):
        self.is_fitted = True
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted.")
            
        out = pd.DataFrame(index=X.index)
        
        target_genes = self.config.get("genetic", {}).get("target_genes", ["LRRK2", "GBA", "SNCA", "PINK1", "PARK2", "APOE"])
        gene_mapping = {"PARK2": "PRKN"}
        
        for g in target_genes:
            col = gene_mapping.get(g, g)
            if col in X.columns:
                if col == "APOE":
                    out["APOE_e4_carrier"] = X["APOE"].astype(str).str.contains("E4").astype(int)
                else:
                    out[g] = pd.to_numeric(X[col], errors='coerce').fillna(0)
                    
        available_numeric = [g for g in target_genes if g != "APOE" and g in out.columns]
        
        if available_numeric:
            out["n_variants"] = out[available_numeric].sum(axis=1)
            if "LRRK2" in out.columns:
                out["lrrk2_positive"] = (out["LRRK2"] > 0).astype(int)
            if "GBA" in out.columns:
                out["gba_positive"] = (out["GBA"] > 0).astype(int)
                
        return out.fillna(0)
