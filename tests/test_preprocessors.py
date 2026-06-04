# -*- coding: utf-8 -*-
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.data.preprocessors import ClinicalPreprocessor, MRIPreprocessor, PETPreprocessor, GeneticPreprocessor

@pytest.fixture
def dummy_config():
    return {
        "mri": {"use_real_mri": False, "n_rois": 100},
        "pet": {"epsilon": 1e-8},
        "genetic": {"target_genes": ["LRRK2", "GBA", "SNCA", "PINK1", "PARK2", "APOE"]},
        "seed": 42
    }

class TestClinicalPreprocessor:
    def test_fit_transform_shape(self, dummy_config):
        data = pd.DataFrame({
            "PATNO": [1, 2, 3],
            "EVENT_ID": ["BL", "BL", "BL"],
            "updrs_i": [1, 2, 3],
            "updrs_ii": [1, 2, 3],
            "updrs_iii": [10, 20, 30],
            "moca": [25, 26, 27],
            "age": [60, 65, 70]
        }).set_index("PATNO")
        prep = ClinicalPreprocessor(dummy_config)
        out = prep.fit_transform(data)
        assert out.shape == data.shape
        assert not out.isna().any().any()

    def test_missing_imputation(self, dummy_config):
        data = pd.DataFrame({
            "PATNO": [1, 2, 3],
            "age": [60, np.nan, 70]
        }).set_index("PATNO")
        prep = ClinicalPreprocessor(dummy_config)
        out = prep.fit_transform(data)
        assert out.loc[2, "age"] == 65.0  # median

class TestMRIPreprocessor:
    def test_synthetic_mode(self, dummy_config):
        data = pd.DataFrame({"PATNO": [1, 2]}).set_index("PATNO")
        prep = MRIPreprocessor(dummy_config)
        out = prep.fit_transform(data)
        assert out.shape == (2, 100)

class TestPETPreprocessor:
    def test_feature_engineering(self, dummy_config):
        data = pd.DataFrame({
            "PATNO": [1],
            "caudate_r": [2.0],
            "caudate_l": [1.0],
            "putamen_r": [2.0],
            "putamen_l": [1.0]
        }).set_index("PATNO")
        prep = PETPreprocessor(dummy_config)
        out = prep.fit_transform(data)
        # 10 features total expected
        assert out.shape == (1, 10)

class TestGeneticPreprocessor:
    def test_explicit_features(self, dummy_config):
        data = pd.DataFrame({
            "PATNO": [1, 2],
            "LRRK2": [1, 0],
            "GBA": [0, 1]
        }).set_index("PATNO")
        prep = GeneticPreprocessor(dummy_config)
        out = prep.fit_transform(data)
        assert out.loc[1, "lrrk2_positive"] == 1
        assert out.loc[2, "gba_positive"] == 1
        assert out.loc[1, "n_variants"] == 1
