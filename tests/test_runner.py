# -*- coding: utf-8 -*-
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.baselines.models import ModelFactory

def test_model_factory_returns_expected():
    rf_reg, _ = ModelFactory.get_model('random_forest', 'regression', {})
    from sklearn.ensemble import RandomForestRegressor
    assert isinstance(rf_reg, RandomForestRegressor)
    
    xgb_cls, _ = ModelFactory.get_model('xgboost', 'classification', {})
    from xgboost import XGBClassifier
    assert isinstance(xgb_cls, XGBClassifier)

def test_custom_mlp_interface():
    from src.baselines.models import MLPRegressor
    X = np.random.randn(10, 144)
    y = np.random.randn(10)
    model = MLPRegressor(epochs=2)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (10,)
