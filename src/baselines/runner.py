# src/baselines/runner.py

from __future__ import annotations

import numpy as np

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.baselines.models import get_baseline_models
from src.evaluation.metrics import (
    compute_regression_metrics,
    concordance_correlation_coefficient,
)
from src.utils import setup_logging


def _make_pipeline(model):
    """
    Create a leak-free preprocessing + model pipeline.

    Imputation and scaling are fitted only on the data passed to
    Pipeline.fit(). During CV, this means they are fitted separately
    inside every training fold.

    During final fitting, they are fitted on the complete development
    set (train + validation), never on the held-out test set.
    """
    return Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "model",
            clone(model),
        ),
    ])


def _validate_inputs(
    X,
    y,
    name="data",
):
    """
    Basic shape/finite-value validation.

    Missing feature values are allowed because the pipeline handles
    them using median imputation. Missing target values are not allowed.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    if X.ndim != 2:
        raise ValueError(
            f"{name}: X must be 2-dimensional, got shape {X.shape}"
        )

    if y.ndim != 1:
        raise ValueError(
            f"{name}: y must be 1-dimensional, got shape {y.shape}"
        )

    if len(X) != len(y):
        raise ValueError(
            f"{name}: X and y have different numbers of samples: "
            f"{len(X)} vs {len(y)}"
        )

    if not np.isfinite(y).all():
        raise ValueError(
            f"{name}: y contains NaN or infinite values."
        )

    return X, y


def run_baselines(
    X_train,
    y_train,
    X_test=None,
    y_test=None,
    n_splits=5,
    seed=42,
):
    """
    Run all classical baselines.

    IMPORTANT:
        X_train / y_train should contain the COMPLETE DEVELOPMENT SET:
            original train + validation

        X_test / y_test should contain the untouched held-out test set.

    Protocol:
        1. 5-fold CV is performed on the development set.
        2. Each fold has its own imputer/scaler/model.
        3. CV results are used for model comparison/configuration.
        4. Each baseline is then refit ONCE on the complete development set.
        5. The final fitted model is evaluated ONCE on the held-out test set.

    The test set is never used during CV.
    """
    logger = setup_logging(__name__)

    X_dev, y_dev = _validate_inputs(
        X_train,
        y_train,
        name="development",
    )

    if X_test is not None or y_test is not None:
        if X_test is None or y_test is None:
            raise ValueError(
                "X_test and y_test must either both be provided or both be None."
            )

        X_test = np.asarray(
            X_test,
            dtype=np.float64,
        )

        y_test = np.asarray(
            y_test,
            dtype=np.float64,
        ).reshape(-1)

        if X_test.ndim != 2:
            raise ValueError(
                f"X_test must be 2-dimensional, got {X_test.shape}"
            )

        if len(X_test) != len(y_test):
            raise ValueError(
                "X_test and y_test have different numbers of samples."
            )

        if not np.isfinite(y_test).all():
            raise ValueError(
                "y_test contains NaN or infinite values."
            )

        if X_test.shape[1] != X_dev.shape[1]:
            raise ValueError(
                "Development and test feature dimensions do not match: "
                f"{X_dev.shape[1]} vs {X_test.shape[1]}"
            )

    if n_splits < 2:
        raise ValueError(
            "n_splits must be at least 2."
        )

    kf = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    models = get_baseline_models()
    results = {}

    logger.info(
        f"Running {len(models)} baselines: "
        f"{n_splits}-fold CV on development shape "
        f"{X_dev.shape} "
        f"(train + validation; preprocessing fit per fold)..."
    )

    for name, base_model in models.items():

        # ------------------------------------------------------------
        # Stage A: Cross-validation on the complete development set
        # ------------------------------------------------------------

        fold_scores = []

        for fold, (tr_idx, va_idx) in enumerate(
            kf.split(X_dev),
            start=1,
        ):
            pipe = _make_pipeline(
                base_model
            )

            pipe.fit(
                X_dev[tr_idx],
                y_dev[tr_idx],
            )

            preds = pipe.predict(
                X_dev[va_idx]
            )

            fold_ccc = concordance_correlation_coefficient(
                y_dev[va_idx],
                preds,
            )

            fold_scores.append(
                float(fold_ccc)
            )

            logger.debug(
                f"{name} fold {fold}/{n_splits}: "
                f"CCC={fold_ccc:.4f}"
            )

        cv_mean = float(
            np.mean(fold_scores)
        )

        cv_std = float(
            np.std(fold_scores)
        )

        entry = {
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "cv_scores": fold_scores,
            "test_ccc": None,
            "test_rmse": None,
            "test_mae": None,
            "test_r2": None,
            "test_pearson": None,
            "test_predictions": None,
        }

        # ------------------------------------------------------------
        # Stage B: Final fit on ALL development data
        # ------------------------------------------------------------

        if X_test is not None and y_test is not None:

            final_pipe = _make_pipeline(
                base_model
            )

            # IMPORTANT:
            # This is train + validation.
            # Test is NOT passed here.
            final_pipe.fit(
                X_dev,
                y_dev,
            )

            # Test is touched only here.
            test_preds = final_pipe.predict(
                X_test
            )

            test_metrics = compute_regression_metrics(
                y_true=y_test,
                y_pred=test_preds,
            )

            entry["test_ccc"] = float(
                test_metrics["ccc"]
            )

            entry["test_rmse"] = float(
                test_metrics["rmse"]
            )

            entry["test_mae"] = float(
                test_metrics["mae"]
            )

            entry["test_r2"] = float(
                test_metrics["r2"]
            )

            entry["test_pearson"] = float(
                test_metrics["pearson"]
            )

            # Saved for paired statistical testing against
            # Fed-PhenoGraft.
            entry["test_predictions"] = [
                float(p)
                for p in test_preds
            ]

        results[name] = entry

        if entry["test_ccc"] is not None:

            logger.info(
                f"  {name}: "
                f"CV CCC {cv_mean:.4f} ± {cv_std:.4f} | "
                f"test CCC={entry['test_ccc']:.4f} "
                f"RMSE={entry['test_rmse']:.3f} "
                f"MAE={entry['test_mae']:.3f} "
                f"R2={entry['test_r2']:.4f} "
                f"Pearson={entry['test_pearson']:.4f}"
            )

        else:

            logger.info(
                f"  {name}: "
                f"CV CCC {cv_mean:.4f} ± {cv_std:.4f}"
            )

    return results


def fit_and_evaluate_final_baseline(
    model,
    X_dev,
    y_dev,
    X_test,
    y_test,
):
    """
    Final Option-B baseline fit.

    X_dev:
        Complete development set = train + validation.

    X_test:
        Completely untouched held-out test set.

    The preprocessing and model are fitted exactly once on X_dev.

    IMPORTANT:
        This function deliberately uses _make_pipeline().
        Therefore median imputation and scaling are also learned only
        from X_dev and never from X_test.
    """
    X_dev, y_dev = _validate_inputs(
        X_dev,
        y_dev,
        name="development",
    )

    X_test = np.asarray(
        X_test,
        dtype=np.float64,
    )

    y_test = np.asarray(
        y_test,
        dtype=np.float64,
    ).reshape(-1)

    if X_test.ndim != 2:
        raise ValueError(
            f"X_test must be 2-dimensional, got {X_test.shape}"
        )

    if len(X_test) != len(y_test):
        raise ValueError(
            "X_test and y_test have different numbers of samples."
        )

    if X_test.shape[1] != X_dev.shape[1]:
        raise ValueError(
            "Development and test feature dimensions do not match."
        )

    if not np.isfinite(y_test).all():
        raise ValueError(
            "y_test contains NaN or infinite values."
        )

    final_pipe = _make_pipeline(
        model
    )

    final_pipe.fit(
        X_dev,
        y_dev,
    )

    test_preds = final_pipe.predict(
        X_test
    )

    metrics = compute_regression_metrics(
        y_true=y_test,
        y_pred=test_preds,
    )

    return {
        "model": final_pipe,
        "predictions": np.asarray(
            test_preds,
            dtype=np.float64,
        ),
        "metrics": metrics,
    }


if __name__ == "__main__":
    # Smoke test with random data if run independently.
    rng = np.random.default_rng(42)

    X = rng.standard_normal(
        (200, 150)
    )

    y = (
        rng.standard_normal(200) * 10
        + X[:, 0] * 5
    )

    # Development set = 160
    # Test set = 40
    results = run_baselines(
        X_train=X[:160],
        y_train=y[:160],
        X_test=X[160:],
        y_test=y[160:],
    )

    print(
        "\nBaseline smoke test completed."
    )

    for name, result in results.items():
        print(
            f"{name}: "
            f"CV CCC={result['cv_mean']:.4f}, "
            f"test CCC={result['test_ccc']:.4f}"
        )