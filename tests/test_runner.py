# # -*- coding: utf-8 -*-
# import pytest
# import numpy as np
# import pandas as pd
# from pathlib import Path

# from src.baselines.models import ModelFactory

# def test_model_factory_returns_expected():
#     rf_reg, _ = ModelFactory.get_model('random_forest', 'regression', {})
#     from sklearn.ensemble import RandomForestRegressor
#     assert isinstance(rf_reg, RandomForestRegressor)
    
#     xgb_cls, _ = ModelFactory.get_model('xgboost', 'classification', {})
#     from xgboost import XGBClassifier
#     assert isinstance(xgb_cls, XGBClassifier)

# def test_custom_mlp_interface():
#     from src.baselines.models import MLPRegressor
#     X = np.random.randn(10, 144)
#     y = np.random.randn(10)
#     model = MLPRegressor(epochs=2)
#     model.fit(X, y)
#     preds = model.predict(X)
#     assert preds.shape == (10,)


import numpy as np

from src.baselines.models import get_baseline_models
from src.baselines.runner import run_baselines


EXPECTED_MODELS = {
    "linear",
    "ridge",
    "lasso",
    "elastic_net",
    "svm",
    "knn",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "mlp",
    "xgboost",
    "lightgbm",
}


def test_get_baseline_models_returns_dict():

    models = get_baseline_models()

    assert isinstance(models, dict)
    assert len(models) == 12


def test_all_expected_baselines_are_registered():

    models = get_baseline_models()

    assert set(models.keys()) == EXPECTED_MODELS


def test_baseline_models_are_independent_instances():

    models_a = get_baseline_models()
    models_b = get_baseline_models()

    assert set(models_a.keys()) == set(models_b.keys())

    for name in models_a:

        assert models_a[name] is not models_b[name]


def test_baseline_models_can_fit_small_dataset():

    rng = np.random.default_rng(42)

    X = rng.normal(
        size=(40, 10)
    )

    y = rng.normal(
        size=40
    )

    models = get_baseline_models()

    for name, model in models.items():

        model.fit(
            X,
            y
        )

        predictions = model.predict(
            X
        )

        assert predictions.shape == (40,), (
            f"{name} returned an unexpected prediction shape"
        )

        assert np.isfinite(
            predictions
        ).all(), (
            f"{name} produced non-finite predictions"
        )


def test_run_baselines_end_to_end():

    rng = np.random.default_rng(123)

    X_train = rng.normal(
        size=(60, 8)
    )

    y_train = (
        2.0 * X_train[:, 0]
        - 1.5 * X_train[:, 1]
        + 0.5 * X_train[:, 2]
        + rng.normal(
            0,
            0.1,
            size=60,
        )
    )

    X_test = rng.normal(
        size=(20, 8)
    )

    y_test = (
        2.0 * X_test[:, 0]
        - 1.5 * X_test[:, 1]
        + 0.5 * X_test[:, 2]
        + rng.normal(
            0,
            0.1,
            size=20,
        )
    )

    results = run_baselines(
        X_train,
        y_train,
        X_test,
        y_test,
        n_splits=3,
        seed=42,
    )

    assert results is not None

    # The runner should return one result for every registered baseline.
    assert set(results.keys()) == EXPECTED_MODELS

    for name, result in results.items():

        assert result is not None, (
            f"{name} returned None"
        )


def test_run_baselines_is_reproducible():
    rng = np.random.default_rng(999)

    X_train = rng.normal(size=(50, 6))
    y_train = rng.normal(size=50)

    X_test = rng.normal(size=(15, 6))
    y_test = rng.normal(size=15)

    results_a = run_baselines(
        X_train,
        y_train,
        X_test,
        y_test,
        n_splits=3,
        seed=42,
    )

    results_b = run_baselines(
        X_train,
        y_train,
        X_test,
        y_test,
        n_splits=3,
        seed=42,
    )

    assert results_a.keys() == results_b.keys()

    for name in results_a:
        result_a = results_a[name]
        result_b = results_b[name]

        assert result_a.keys() == result_b.keys()

        for key in result_a:
            value_a = result_a[key]
            value_b = result_b[key]

            if isinstance(value_a, (list, tuple, np.ndarray)):
                np.testing.assert_allclose(
                    np.asarray(value_a, dtype=float),
                    np.asarray(value_b, dtype=float),
                    rtol=1e-10,
                    atol=1e-12,
                    err_msg=f"{name}/{key} is not reproducible",
                )
            elif isinstance(value_a, (float, np.floating)):
                np.testing.assert_allclose(
                    value_a,
                    value_b,
                    rtol=1e-10,
                    atol=1e-12,
                    err_msg=f"{name}/{key} is not reproducible",
                )
            else:
                assert value_a == value_b, (
                    f"{name}/{key} differs between repeated runs"
                )