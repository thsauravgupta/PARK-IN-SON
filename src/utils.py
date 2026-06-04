# -*- coding: utf-8 -*-
"""
Shared utilities for the PPMI Multimodal Pipeline.

Provides reproducibility helpers, logging configuration, custom metrics,
and config loading. Every script and module in the project imports from here
rather than duplicating these concerns.
"""

import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import yaml


# ===========================================================================
# Reproducibility
# ===========================================================================

def seed_everything(seed: int = 42) -> None:
    """
    Set random seeds across all libraries for deterministic execution.

    WHY: Medical ML results must be reproducible. A single unseeded RNG
    can silently change model performance between runs, making ablation
    studies and peer review impossible.

    Args:
        seed: Random seed value. Default 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # torch not required for non-DL components


# ===========================================================================
# Logging
# ===========================================================================

def setup_logging(
    name: str,
    level: str = "INFO",
    log_dir: Optional[Union[str, Path]] = None,
) -> logging.Logger:
    """
    Configure a logger with both console and file handlers.

    WHY: print() is invisible in production. Structured logging with
    timestamps enables post-hoc debugging of multi-hour pipeline runs
    and is expected in any research-grade codebase.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).
        log_dir: Directory for log files. If None, logs to console only.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Prevent duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            log_path / f"{name.replace('.', '_')}.log",
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ===========================================================================
# Configuration
# ===========================================================================

def load_config(path: Union[str, Path] = "config.yaml") -> Dict[str, Any]:
    """
    Load the project YAML configuration file.

    WHY: A single config.yaml is the contract between all pipeline stages.
    Hardcoded paths and magic numbers scattered across files are the #1
    cause of irreproducible results in ML research.

    Args:
        path: Path to the YAML config file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    config_path = Path(path)
    if not config_path.exists():
        # Try relative to project root
        project_root = Path(__file__).resolve().parent.parent
        config_path = project_root / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found at {path} or {config_path}. "
            "Run from project root or pass explicit path."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def get_project_root() -> Path:
    """
    Return the project root directory.

    Assumes this file lives at ``<root>/src/utils.py``.

    Returns:
        Absolute path to the project root.
    """
    return Path(__file__).resolve().parent.parent


def ensure_dirs(config: Dict[str, Any]) -> None:
    """
    Create all directories specified in the config paths section.

    Args:
        config: Parsed configuration dictionary.
    """
    root = get_project_root()
    for key, rel_path in config.get("paths", {}).items():
        dir_path = root / rel_path
        dir_path.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# Custom Metrics
# ===========================================================================

def concordance_correlation_coefficient(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Compute Lin's Concordance Correlation Coefficient (CCC).

    WHY CCC over R²: R² measures how well predictions correlate with truth
    but ignores systematic bias. A model that predicts y_pred = y_true + 10
    gets perfect R² but is clinically useless for UPDRS scoring. CCC
    penalises both poor correlation AND systematic offset, making it the
    gold standard metric for clinical score prediction tasks.

    Formula:
        CCC = 2 * cov(y, ŷ) / (var(y) + var(ŷ) + (mean(y) - mean(ŷ))²)

    Args:
        y_true: Ground-truth values, shape ``(n_samples,)``.
        y_pred: Predicted values, shape ``(n_samples,)``.

    Returns:
        CCC value in ``[-1, 1]``. Perfect agreement = 1.

    References:
        Lin, L.I. (1989). A concordance correlation coefficient to evaluate
        reproducibility. Biometrics, 45(1), 255-268.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))

    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2

    if denominator < 1e-12:
        return 0.0

    return float(2.0 * covariance / denominator)


def safe_log_transform(
    x: np.ndarray,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """
    Apply log(x + epsilon) transform, handling non-positive values.

    WHY: DaTscan SBR values are right-skewed. Log-transform normalises
    the distribution for downstream linear models and PCA. The epsilon
    prevents log(0) = -inf from corrupting the pipeline.

    Args:
        x: Input array.
        epsilon: Small constant added before log. Default 1e-8.

    Returns:
        Log-transformed array.
    """
    return np.log(np.maximum(np.asarray(x, dtype=np.float64), epsilon))
