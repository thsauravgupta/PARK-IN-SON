# src/evaluation/statistics.py

from __future__ import annotations

from typing import Callable

import numpy as np

from src.evaluation.metrics import (
    concordance_correlation_coefficient,
)


def _prepare_paired_arrays(
    y_true,
    pred_a,
    pred_b=None,
):
    """
    Convert inputs to 1-D float arrays and remove observations
    containing non-finite values.

    If pred_b is supplied, all three arrays are filtered using
    exactly the same mask.
    """
    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    ).reshape(-1)

    pred_a = np.asarray(
        pred_a,
        dtype=np.float64,
    ).reshape(-1)

    if len(y_true) != len(pred_a):
        raise ValueError(
            "y_true and pred_a must have equal length."
        )

    if pred_b is not None:

        pred_b = np.asarray(
            pred_b,
            dtype=np.float64,
        ).reshape(-1)

        if len(y_true) != len(pred_b):
            raise ValueError(
                "y_true and pred_b must have equal length."
            )

        mask = (
            np.isfinite(y_true)
            & np.isfinite(pred_a)
            & np.isfinite(pred_b)
        )

        return (
            y_true[mask],
            pred_a[mask],
            pred_b[mask],
        )

    mask = (
        np.isfinite(y_true)
        & np.isfinite(pred_a)
    )

    return (
        y_true[mask],
        pred_a[mask],
    )


def bootstrap_metric_ci(
    y_true,
    y_pred,
    metric_fn: Callable,
    n_bootstrap=5000,
    confidence=0.95,
    seed=42,
):
    """
    Non-parametric paired bootstrap confidence interval.

    Each bootstrap sample resamples subjects with replacement.
    """
    if not 0 < confidence < 1:
        raise ValueError(
            "confidence must be between 0 and 1."
        )

    if n_bootstrap < 100:
        raise ValueError(
            "Use at least 100 bootstrap samples."
        )

    y_true, y_pred = _prepare_paired_arrays(
        y_true,
        y_pred,
    )

    n = len(y_true)

    if n < 2:
        raise ValueError(
            "At least two valid observations are required."
        )

    rng = np.random.default_rng(
        seed
    )

    bootstrap_values = np.empty(
        n_bootstrap,
        dtype=np.float64,
    )

    for i in range(n_bootstrap):

        indices = rng.integers(
            low=0,
            high=n,
            size=n,
        )

        bootstrap_values[i] = metric_fn(
            y_true[indices],
            y_pred[indices],
        )

    alpha = 1.0 - confidence

    lower = float(
        np.percentile(
            bootstrap_values,
            100.0 * alpha / 2.0,
        )
    )

    upper = float(
        np.percentile(
            bootstrap_values,
            100.0 * (1.0 - alpha / 2.0),
        )
    )

    observed = float(
        metric_fn(
            y_true,
            y_pred,
        )
    )

    return {
        "estimate": observed,
        "ci_lower": lower,
        "ci_upper": upper,
        "confidence": confidence,
        "n_bootstrap": n_bootstrap,
        "n_samples": n,
    }


def paired_bootstrap_difference(
    y_true,
    pred_a,
    pred_b,
    metric_fn=concordance_correlation_coefficient,
    n_bootstrap=5000,
    confidence=0.95,
    seed=42,
):
    """
    Paired bootstrap comparison of two models.

    Difference is defined as:

        metric(model A) - metric(model B)

    The SAME subjects are resampled for both models.

    Returns a two-sided bootstrap p-value.

    IMPORTANT:
        This is a paired subject-level comparison.
        The statistical unit is the test subject, not the random seed.
    """
    if not 0 < confidence < 1:
        raise ValueError(
            "confidence must be between 0 and 1."
        )

    if n_bootstrap < 100:
        raise ValueError(
            "Use at least 100 bootstrap samples."
        )

    (
        y_true,
        pred_a,
        pred_b,
    ) = _prepare_paired_arrays(
        y_true,
        pred_a,
        pred_b,
    )

    n = len(y_true)

    if n < 2:
        raise ValueError(
            "At least two valid paired observations are required."
        )

    observed_a = float(
        metric_fn(
            y_true,
            pred_a,
        )
    )

    observed_b = float(
        metric_fn(
            y_true,
            pred_b,
        )
    )

    observed_difference = (
        observed_a - observed_b
    )

    rng = np.random.default_rng(
        seed
    )

    differences = np.empty(
        n_bootstrap,
        dtype=np.float64,
    )

    for i in range(n_bootstrap):

        indices = rng.integers(
            low=0,
            high=n,
            size=n,
        )

        metric_a = metric_fn(
            y_true[indices],
            pred_a[indices],
        )

        metric_b = metric_fn(
            y_true[indices],
            pred_b[indices],
        )

        differences[i] = (
            metric_a - metric_b
        )

    alpha = 1.0 - confidence

    ci_lower = float(
        np.percentile(
            differences,
            100.0 * alpha / 2.0,
        )
    )

    ci_upper = float(
        np.percentile(
            differences,
            100.0 * (1.0 - alpha / 2.0),
        )
    )

    # ------------------------------------------------------------
    # Two-sided bootstrap sign probability.
    #
    # This replaces the suspicious current p=0.9540 calculation.
    #
    # The bootstrap distribution is used to estimate the two tails:
    #
    #     P(D <= 0)
    #     P(D >= 0)
    #
    # and the smaller tail is doubled.
    # ------------------------------------------------------------

    p_left = float(
        np.mean(
            differences <= 0.0
        )
    )

    p_right = float(
        np.mean(
            differences >= 0.0
        )
    )

    p_value = min(
        1.0,
        2.0 * min(
            p_left,
            p_right,
        ),
    )

    return {
        "metric_a": observed_a,
        "metric_b": observed_b,
        "observed_difference": float(
            observed_difference
        ),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": float(
            p_value
        ),
        "p_left": p_left,
        "p_right": p_right,
        "n_bootstrap": n_bootstrap,
        "n_samples": n,
    }


def paired_bootstrap_ccc_test(
    y_true,
    pred_model,
    pred_baseline,
    n_bootstrap=5000,
    confidence=0.95,
    seed=42,
):
    """
    Convenience wrapper for:

        CCC(model) - CCC(baseline)
    """
    return paired_bootstrap_difference(
        y_true=y_true,
        pred_a=pred_model,
        pred_b=pred_baseline,
        metric_fn=concordance_correlation_coefficient,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
    )


if __name__ == "__main__":
    # Small sanity test.
    rng = np.random.default_rng(42)

    y = rng.normal(
        0,
        1,
        100,
    )

    pred_a = (
        y
        + rng.normal(
            0,
            0.5,
            100,
        )
    )

    pred_b = (
        y
        + rng.normal(
            0,
            0.8,
            100,
        )
    )

    ci = bootstrap_metric_ci(
        y_true=y,
        y_pred=pred_a,
        metric_fn=concordance_correlation_coefficient,
        n_bootstrap=1000,
    )

    comparison = paired_bootstrap_ccc_test(
        y_true=y,
        pred_model=pred_a,
        pred_baseline=pred_b,
        n_bootstrap=1000,
    )

    print(
        "Model A CCC:",
        ci,
    )

    print(
        "Paired comparison:",
        comparison,
    )