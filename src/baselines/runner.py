import numpy as np
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.baselines.models import get_baseline_models
from src.evaluation.metrics import (
    concordance_correlation_coefficient, mae, pearson_r, r2_score, rmse,
)
from src.utils import setup_logging


def _make_pipeline(model):
    """Imputation and scaling live INSIDE the pipeline so they are re-fit on
    each CV fold's training portion only — no statistics leak across folds."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", clone(model)),
    ])


def run_baselines(X_train, y_train, X_test=None, y_test=None, n_splits=5, seed=42):
    """
    Evaluates baseline models with leak-free K-Fold CV on the training data,
    then (optionally) refits on the full training set and scores ONCE on the
    held-out test set. Target metric is CCC.

    Args:
        X_train: raw (unscaled) feature matrix for training subjects (N, D)
        y_train: training targets (N,)
        X_test / y_test: held-out test subjects — never seen during CV
        n_splits: number of CV folds

    Returns:
        dict: {model_name: {"cv_mean", "cv_std", "cv_scores",
                            "test_ccc", "test_rmse", "test_mae",
                            "test_r2", "test_pearson"}}
    """
    logger = setup_logging(__name__)
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64).ravel()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    models = get_baseline_models()
    results = {}

    logger.info(f"Running {len(models)} baselines: {n_splits}-fold CV on train "
                f"shape {X_train.shape} "
                f"(preprocessing fit per-fold, models cloned per-fold)...")

    for name, base_model in models.items():
        fold_scores = []
        for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
            pipe = _make_pipeline(base_model)
            pipe.fit(X_train[tr_idx], y_train[tr_idx])
            preds = pipe.predict(X_train[va_idx])
            fold_scores.append(concordance_correlation_coefficient(y_train[va_idx], preds))

        entry = {
            "cv_mean": float(np.mean(fold_scores)),
            "cv_std": float(np.std(fold_scores)),
            "cv_scores": [float(s) for s in fold_scores],
            "test_ccc": None, "test_rmse": None, "test_mae": None,
            "test_r2": None, "test_pearson": None,
        }

        if X_test is not None and y_test is not None:
            final_pipe = _make_pipeline(base_model)
            final_pipe.fit(X_train, y_train)
            test_preds = final_pipe.predict(np.asarray(X_test, dtype=np.float64))
            yt = np.asarray(y_test, dtype=np.float64).ravel()
            entry["test_ccc"] = concordance_correlation_coefficient(yt, test_preds)
            entry["test_rmse"] = rmse(yt, test_preds)
            entry["test_mae"] = mae(yt, test_preds)
            entry["test_r2"] = r2_score(yt, test_preds)
            entry["test_pearson"] = pearson_r(yt, test_preds)
            # Kept for paired significance testing against Fed-PhenoGraft
            entry["test_predictions"] = [float(p) for p in test_preds]

        results[name] = entry
        test_str = (f" | test CCC={entry['test_ccc']:.4f} "
                    f"RMSE={entry['test_rmse']:.3f} R2={entry['test_r2']:.4f}"
                    if entry["test_ccc"] is not None else "")
        logger.info(f"  {name}: CV CCC {entry['cv_mean']:.4f} "
                    f"± {entry['cv_std']:.4f}{test_str}")

    return results


if __name__ == "__main__":
    # Smoke test with random data if run independently
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 150))
    y = rng.standard_normal(200) * 10 + X[:, 0] * 5
    run_baselines(X[:160], y[:160], X[160:], y[160:])
