import numpy as np


def concordance_correlation_coefficient(y_true, y_pred):
    """
    Lin's Concordance Correlation Coefficient.
    CCC = 2 * cov(x, y) / (var(x) + var(y) + (mean(x) - mean(y))^2)

    Returns 0.0 when either input has zero variance (e.g., a model that
    predicts a constant) instead of propagating NaN — a constant predictor
    has no agreement with the trajectory, and NaNs would silently corrupt
    round-by-round averages and early-stopping comparisons.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    if y_true.size < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0

    cor = np.corrcoef(y_true, y_pred)[0][1]
    if np.isnan(cor):
        return 0.0

    mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
    var_true, var_pred = np.var(y_true), np.var(y_pred)
    sd_true, sd_pred = np.std(y_true), np.std(y_pred)

    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    """Mean Absolute Error."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true, y_pred):
    """Coefficient of determination R². Returns 0.0 when the target has zero
    variance (undefined) instead of dividing by zero."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    ss_res = np.sum((y_true - y_pred) ** 2)
    return float(1.0 - ss_res / ss_tot)


def pearson_r(y_true, y_pred):
    """Pearson correlation coefficient; 0.0 on zero variance or NaN."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if y_true.size < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    r = np.corrcoef(y_true, y_pred)[0][1]
    return 0.0 if np.isnan(r) else float(r)
