# -*- coding: utf-8 -*-
"""
Statistical rigor utilities for Fed-PhenoGraft results.

- bootstrap_metric_ci:
    Nonparametric bootstrap 95% CI for any metric.

- paired_bootstrap_test:
    Paired bootstrap confidence interval for the difference between two
    models evaluated on the SAME test subjects, combined with a paired
    permutation/randomization test for the p-value.

- summarize_seed_runs:
    Mean ± std across independent training seeds.

All resampling uses a fixed seed so results are reproducible.
"""

import numpy as np

from src.evaluation.metrics import (
    concordance_correlation_coefficient,
    mae,
    r2_score,
    rmse,
)


METRIC_FNS = {
    "ccc": concordance_correlation_coefficient,
    "rmse": rmse,
    "mae": mae,
    "r2": r2_score,
}


HIGHER_IS_BETTER = {
    "ccc": True,
    "r2": True,
    "rmse": False,
    "mae": False,
}


def bootstrap_metric_ci(
    y_true,
    y_pred,
    metric="ccc",
    n_boot=1000,
    seed=42,
    ci=0.95,
):
    """
    Percentile bootstrap confidence interval for a metric.

    Parameters
    ----------
    y_true : array-like
        Ground-truth target values.

    y_pred : array-like
        Model predictions.

    metric : str or callable
        Metric name from METRIC_FNS or a callable.

    n_boot : int
        Number of bootstrap resamples.

    seed : int
        Random seed.

    ci : float
        Confidence level. Default = 0.95.

    Returns
    -------
    dict
        {
            "point": point estimate,
            "ci_low": lower percentile bound,
            "ci_high": upper percentile bound,
            "boot_std": standard deviation of bootstrap estimates
        }
    """

    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have the same length: "
            f"{len(y_true)} != {len(y_pred)}"
        )

    if len(y_true) < 2:
        raise ValueError("At least two observations are required.")

    if not 0 < ci < 1:
        raise ValueError("ci must be between 0 and 1.")

    if n_boot <= 0:
        raise ValueError("n_boot must be positive.")

    fn = METRIC_FNS[metric] if isinstance(metric, str) else metric

    rng = np.random.default_rng(seed)
    n = len(y_true)

    point = float(fn(y_true, y_pred))

    boot_stats = np.empty(n_boot, dtype=np.float64)

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)

        boot_stats[b] = fn(
            y_true[idx],
            y_pred[idx],
        )

    alpha = (1.0 - ci) / 2.0

    return {
        "point": point,
        "ci_low": float(np.quantile(boot_stats, alpha)),
        "ci_high": float(np.quantile(boot_stats, 1.0 - alpha)),
        "boot_std": float(np.std(boot_stats, ddof=1)),
    }


def paired_bootstrap_test(
    y_true,
    pred_a,
    pred_b,
    metric="ccc",
    n_boot=1000,
    seed=42,
    ci=0.95,
):
    """
    Compare two models evaluated on the SAME test subjects.

    The function performs two related analyses:

    1. Paired bootstrap:
       Estimates a confidence interval for the observed metric difference.

    2. Paired permutation/randomization test:
       Estimates a two-sided p-value under the null hypothesis that the two
       models are exchangeable for the same subjects.

    For higher-is-better metrics (CCC, R2):

        delta = metric(A) - metric(B)

    For lower-is-better metrics (RMSE, MAE):

        delta = metric(B) - metric(A)

    Therefore a POSITIVE delta always means that model A performs better
    according to the direction of the metric.

    Returns
    -------
    dict
        {
            "delta": observed directional difference,
            "ci_low": lower bootstrap CI,
            "ci_high": upper bootstrap CI,
            "p_value": two-sided paired permutation p-value,
            "significant_at_0.05": whether p < 0.05
        }

    Notes
    -----
    The test predictions must correspond to exactly the same subjects in
    exactly the same order.
    """

    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    pred_a = np.asarray(pred_a, dtype=np.float64).ravel()
    pred_b = np.asarray(pred_b, dtype=np.float64).ravel()

    if not (
        len(y_true) == len(pred_a) == len(pred_b)
    ):
        raise ValueError(
            "y_true, pred_a, and pred_b must have the same length: "
            f"{len(y_true)}, {len(pred_a)}, {len(pred_b)}"
        )

    if len(y_true) < 2:
        raise ValueError("At least two observations are required.")

    if n_boot <= 0:
        raise ValueError("n_boot must be positive.")

    fn = METRIC_FNS[metric] if isinstance(metric, str) else metric

    if isinstance(metric, str):
        higher_is_better = HIGHER_IS_BETTER[metric]
    else:
        # Custom metrics are assumed to be higher-is-better.
        higher_is_better = True

    rng = np.random.default_rng(seed)
    n = len(y_true)

    # ---------------------------------------------------------------
    # 1. Observed difference
    # ---------------------------------------------------------------

    score_a = float(fn(y_true, pred_a))
    score_b = float(fn(y_true, pred_b))

    raw_delta = score_a - score_b

    # Normalize direction so positive always means "A is better".
    observed_delta = (
        raw_delta
        if higher_is_better
        else -raw_delta
    )

    # ---------------------------------------------------------------
    # 2. Paired bootstrap confidence interval
    # ---------------------------------------------------------------

    bootstrap_deltas = np.empty(
        n_boot,
        dtype=np.float64,
    )

    for b in range(n_boot):
        idx = rng.integers(
            0,
            n,
            size=n,
        )

        boot_a = fn(
            y_true[idx],
            pred_a[idx],
        )

        boot_b = fn(
            y_true[idx],
            pred_b[idx],
        )

        delta = boot_a - boot_b

        if not higher_is_better:
            delta = -delta

        bootstrap_deltas[b] = delta

    alpha = (1.0 - ci) / 2.0

    ci_low = float(
        np.quantile(
            bootstrap_deltas,
            alpha,
        )
    )

    ci_high = float(
        np.quantile(
            bootstrap_deltas,
            1.0 - alpha,
        )
    )

    # ---------------------------------------------------------------
    # 3. Paired permutation/randomization test
    # ---------------------------------------------------------------
    #
    # Under H0, predictions A and B are exchangeable for each subject.
    #
    # For every permutation:
    #   - randomly swap A/B for each subject
    #   - calculate the resulting metric difference
    #
    # This preserves the subject-level pairing.
    # ---------------------------------------------------------------

    permutation_deltas = np.empty(
        n_boot,
        dtype=np.float64,
    )

    for b in range(n_boot):
        swap = rng.random(n) < 0.5

        perm_a = np.where(
            swap,
            pred_b,
            pred_a,
        )

        perm_b = np.where(
            swap,
            pred_a,
            pred_b,
        )

        perm_a_score = fn(
            y_true,
            perm_a,
        )

        perm_b_score = fn(
            y_true,
            perm_b,
        )

        delta = perm_a_score - perm_b_score

        if not higher_is_better:
            delta = -delta

        permutation_deltas[b] = delta

    # Two-sided randomization p-value.
    #
    # +1 correction prevents p=0 and gives a conservative finite-sample
    # estimate.
    extreme = np.sum(
        np.abs(permutation_deltas)
        >= abs(observed_delta)
    )

    p_value = float(
        (extreme + 1.0)
        / (n_boot + 1.0)
    )

    return {
        "delta": float(observed_delta),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
        "significant_at_0.05": bool(
            p_value < 0.05
        ),
    }


def summarize_seed_runs(per_seed_metrics):
    """
    Summarize metrics across independent training seeds.

    Parameters
    ----------
    per_seed_metrics : list[dict]
        One metric dictionary per independent seed.

    Returns
    -------
    dict
        {
            metric_name: {
                "mean": ...,
                "std": ...,
                "values": [...]
            }
        }

    Notes
    -----
    Standard deviation uses population std (ddof=0), matching the previous
    implementation and preserving compatibility with existing result files.
    """

    if not per_seed_metrics:
        return {}

    keys = set(per_seed_metrics[0].keys())

    for metrics in per_seed_metrics[1:]:
        keys &= set(metrics.keys())

    out = {}

    for key in sorted(keys):
        values = [
            float(metrics[key])
            for metrics in per_seed_metrics
        ]

        out[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "values": values,
        }

    return out