# -*- coding: utf-8 -*-
"""
Statistical rigor utilities for Fed-PhenoGraft results.

- bootstrap_metric_ci: nonparametric bootstrap 95% CI for any metric.
- paired_bootstrap_test: paired bootstrap significance test comparing two
  models' predictions on the SAME test subjects (resamples subjects jointly,
  so the comparison respects pairing).
- summarize_seed_runs: mean ± std across independent training seeds.

All resampling uses a fixed seed so results are reproducible.
"""

import numpy as np

from src.evaluation.metrics import (
    concordance_correlation_coefficient, mae, r2_score, rmse,
)

METRIC_FNS = {
    "ccc": concordance_correlation_coefficient,
    "rmse": rmse,
    "mae": mae,
    "r2": r2_score,
}


def bootstrap_metric_ci(y_true, y_pred, metric="ccc", n_boot=1000, seed=42,
                        ci=0.95):
    """
    Percentile bootstrap CI for a metric on held-out predictions.

    Returns dict: {"point", "ci_low", "ci_high", "boot_std"}.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    fn = METRIC_FNS[metric] if isinstance(metric, str) else metric
    rng = np.random.default_rng(seed)
    n = len(y_true)

    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[b] = fn(y_true[idx], y_pred[idx])

    alpha = (1.0 - ci) / 2.0
    return {
        "point": float(fn(y_true, y_pred)),
        "ci_low": float(np.quantile(stats, alpha)),
        "ci_high": float(np.quantile(stats, 1.0 - alpha)),
        "boot_std": float(np.std(stats)),
    }


def paired_bootstrap_test(y_true, pred_a, pred_b, metric="ccc", n_boot=1000,
                          seed=42):
    """
    Paired bootstrap test of H0: metric(A) <= metric(B) for higher-is-better
    metrics (CCC/R2), or metric(A) >= metric(B) for lower-is-better (RMSE/MAE).

    Both models are evaluated on the SAME resampled subjects each draw, which
    preserves the pairing. The reported p-value is the fraction of draws in
    which A fails to beat B.

    Returns dict: {"delta", "ci_low", "ci_high", "p_value", "significant"}.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    pred_a = np.asarray(pred_a, dtype=np.float64).ravel()
    pred_b = np.asarray(pred_b, dtype=np.float64).ravel()
    fn = METRIC_FNS[metric] if isinstance(metric, str) else metric
    higher_is_better = metric in ("ccc", "r2") if isinstance(metric, str) else True
    rng = np.random.default_rng(seed)
    n = len(y_true)

    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        d = fn(y_true[idx], pred_a[idx]) - fn(y_true[idx], pred_b[idx])
        deltas[b] = d if higher_is_better else -d

    point = fn(y_true, pred_a) - fn(y_true, pred_b)
    if not higher_is_better:
        point = -point
    p_value = float(np.mean(deltas <= 0))
    return {
        "delta": float(point),
        "ci_low": float(np.quantile(deltas, 0.025)),
        "ci_high": float(np.quantile(deltas, 0.975)),
        "p_value": p_value,
        "significant_at_0.05": bool(p_value < 0.05),
    }


def summarize_seed_runs(per_seed_metrics):
    """
    per_seed_metrics: list of metric dicts (one per training seed).
    Returns {metric: {"mean", "std", "values"}} over the shared keys.
    """
    if not per_seed_metrics:
        return {}
    keys = set(per_seed_metrics[0])
    for m in per_seed_metrics[1:]:
        keys &= set(m)
    out = {}
    for k in sorted(keys):
        vals = [float(m[k]) for m in per_seed_metrics]
        out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                  "values": vals}
    return out
