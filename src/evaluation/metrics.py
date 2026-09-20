
# src/evaluation/metrics.py

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score as r2_score_sklearn,
    roc_auc_score,
)


def concordance_correlation_coefficient(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Lin's Concordance Correlation Coefficient (CCC).

    CCC = 2 * covariance(y_true, y_pred) /
          (variance(y_true) + variance(y_pred)
           + (mean(y_true) - mean(y_pred))^2)

    Returns NaN if the input is invalid or the denominator is zero.
    """
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) < 2:
        return float("nan")

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    var_true = np.var(y_true)
    var_pred = np.var(y_pred)

    covariance = np.mean(
        (y_true - mean_true) *
        (y_pred - mean_pred)
    )

    denominator = (
        var_true
        + var_pred
        + (mean_true - mean_pred) ** 2
    )

    if denominator <= 0:
        return float("nan")

    return float(
        2.0 * covariance / denominator
    )


def _safe_pearson(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Calculate Pearson correlation safely."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    if len(y_true) < 2:
        return float("nan")

    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")

    return float(np.corrcoef(y_true, y_pred)[0, 1])

def rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Root Mean Squared Error."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have equal length: "
            f"{len(y_true)} != {len(y_pred)}"
        )

    mask = np.isfinite(y_true) & np.isfinite(y_pred)

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return float("nan")

    return float(
        np.sqrt(
            mean_squared_error(y_true, y_pred)
        )
    )


def mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Mean Absolute Error."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have equal length: "
            f"{len(y_true)} != {len(y_pred)}"
        )

    mask = np.isfinite(y_true) & np.isfinite(y_pred)

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return float("nan")

    return float(
        mean_absolute_error(y_true, y_pred)
    )


def r2_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Coefficient of determination R²."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have equal length: "
            f"{len(y_true)} != {len(y_pred)}"
        )

    mask = np.isfinite(y_true) & np.isfinite(y_pred)

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return float("nan")

    return float(
        r2_score_sklearn(
            y_true,
            y_pred,
        )
    )


def pearson_r(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Pearson correlation coefficient."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have equal length: "
            f"{len(y_true)} != {len(y_pred)}"
        )

    mask = np.isfinite(y_true) & np.isfinite(y_pred)

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    return _safe_pearson(y_true, y_pred)


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate the complete progression-regression metric set.

    Metrics:
        CCC
        RMSE
        MAE
        R²
        Pearson r
    """
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have equal length: "
            f"{len(y_true)} != {len(y_pred)}"
        )

    mask = np.isfinite(y_true) & np.isfinite(y_pred)

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        raise ValueError("No finite observations available.")

    mse = mean_squared_error(y_true, y_pred)

    return {
        "ccc": concordance_correlation_coefficient(
            y_true,
            y_pred,
        ),
        "rmse": float(np.sqrt(mse)),
        "mae": float(
            mean_absolute_error(y_true, y_pred)
        ),
        "r2": float(
            r2_score_sklearn(y_true, y_pred)
        ),
        "pearson": _safe_pearson(
            y_true,
            y_pred,
        ),
    }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Calculate binary classification metrics.

    y_prob must contain the probability of the positive class.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float64).reshape(-1)

    if len(y_true) != len(y_prob):
        raise ValueError(
            "y_true and y_prob must have equal length."
        )

    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
    }

    try:
        metrics["auc"] = float(
            roc_auc_score(y_true, y_prob)
        )
    except ValueError:
        metrics["auc"] = float("nan")

    return metrics


def compute_all_metrics(
    y_true_reg: np.ndarray,
    y_pred_reg: np.ndarray,
    y_true_cls: Optional[np.ndarray] = None,
    y_prob_cls: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Calculate progression and, optionally, classification metrics.
    """
    metrics = compute_regression_metrics(
        y_true_reg,
        y_pred_reg,
    )

    if y_true_cls is not None and y_prob_cls is not None:
        metrics.update(
            compute_classification_metrics(
                y_true_cls,
                y_prob_cls,
            )
        )

    return metrics