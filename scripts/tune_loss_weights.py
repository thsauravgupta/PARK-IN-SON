# -*- coding: utf-8 -*-
"""
Fed-PhenoGraft - Loss Weight Tuning
====================================

Purpose
-------
Quickly tune:
    - cls_weight
    - hsic_weight

without running the full main.py pipeline.

This script performs ONLY the development-stage experiment:

    Raw data
        ↓
    Fixed subject-level train/val/test split
        ↓
    Preprocessing fitted on TRAIN only
        ↓
    Federated training on TRAIN
        ↓
    Validation-based early stopping
        ↓
    Record best validation CCC

IMPORTANT
---------
- The TEST set is NEVER used.
- Final TRAIN+VALIDATION fitting is NOT performed.
- Baselines are NOT run.
- Ablations are NOT run.
- Explainability is NOT run.
- Only seed 42 is used by default.
- The same train/validation split is reused for every trial.
- The same preprocessing is reused for every trial.
- Only cls_weight / hsic_weight changes between trials.

Usage
-----
From the project root:

    python scripts/tune_loss_weights.py

By default the script performs:

1. CLS sweep:
       hsic_weight = 0.10
       cls_weight = [0.00, 0.05, 0.10, 0.20, 0.30, 0.50]

2. HSIC sweep:
       best cls_weight from step 1
       hsic_weight = [0.00, 0.01, 0.05, 0.10, 0.20]

3. Saves results to:
       outputs/loss_weight_tuning.csv

4. Saves the best configuration to:
       outputs/best_loss_weights.yaml
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

# ---------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

from src.utils import seed_everything

from src.data.data_builder import build_real_dataset

from src.data.dataset import (
    FederatedPPMIDataset,
    create_federated_splits,
    load_site_labels,
)

from src.data.preprocessing import (
    ModalityPreprocessor,
    create_subject_splits,
)

from src.models.fed_phenograft import FedPhenoGraft

from src.federated.fedavg_orchestrator import (
    simulate_federated_training,
    evaluate_model,
)


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(message)s",
)

logger = logging.getLogger("FedPhenoGraftLossTuning")


# =====================================================================
# CONFIG
# =====================================================================

CONFIG_PATH = PROJECT_ROOT / "config.yaml"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

RESULTS_PATH = OUTPUT_DIR / "loss_weight_tuning.csv"

BEST_CONFIG_PATH = OUTPUT_DIR / "best_loss_weights.yaml"


# ---------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------

CLS_VALUES = [
    0.0,
    0.01,
    0.025,
    0.05,
    0.1,
    0.2,
]

HSIC_VALUES = [
    0.0,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
    50.0,
    100.0,
]

CCC_WEIGHTS = [
    2000,
    2500,
    3000,
    4000,
    5000,
]


# ---------------------------------------------------------------------
# Quick tuning settings
# ---------------------------------------------------------------------

# Only one seed during hyperparameter search.
TUNING_SEED = 42

# Keep this equal to your normal training setup unless you deliberately
# want a faster coarse search.
#
# Your current pipeline uses 30 rounds and early stopping.
# Keeping 30 here makes the validation comparison consistent with the
# actual development protocol.
NUM_ROUNDS = 30

EARLY_STOPPING_PATIENCE = 5


# =====================================================================
# CONFIG LOADING
# =====================================================================

def load_config():
    """
    Load the project's existing config.yaml.
    """

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Could not find config file:\n{CONFIG_PATH}"
        )

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(
            f"{CONFIG_PATH} is empty or invalid."
        )

    return config


# =====================================================================
# DATASET HELPER
# =====================================================================

def build_dataset_from_indices(
    prep,
    clin,
    mri,
    pet,
    gen,
    y,
    diag,
    indices,
):
    """
    Transform one subset using an already-fitted preprocessor.

    The preprocessor must have been fitted only on the training subjects.
    """

    clinical, mri_scaled, pet_scaled, genetic = prep.transform(
        clin.iloc[indices],
        mri.iloc[indices],
        pet.iloc[indices],
        gen.iloc[indices],
    )

    dataset = FederatedPPMIDataset(
        clinical,
        mri_scaled,
        pet_scaled,
        genetic,
        pd.Series(
            y[indices],
            index=clinical.index,
        ),
        diagnosis=diag[indices],
    )

    return dataset


# =====================================================================
# DATA PREPARATION
# =====================================================================

def prepare_development_data(config):
    """
    Load the raw PPMI data and create the exact development regime:

        TRAIN
          ↓
        fit preprocessing

        VALIDATION
          ↓
        transform only

        TEST
          ↓
        transform only

    The test set is created so that the subject split is identical to
    the main pipeline, but it is never used by this tuning script.
    """

    seed = int(
        config.get(
            "seed",
            42,
        )
    )

    split_cfg = config.get(
        "split",
        {},
    )

    logger.info("=" * 72)
    logger.info("Loading raw PPMI data")
    logger.info("=" * 72)

    (
        clin,
        mri,
        pet,
        gen,
        targets,
        diagnosis,
    ) = build_real_dataset(config)

    y = np.asarray(
        targets.values
        if hasattr(targets, "values")
        else targets,
        dtype=np.float64,
    ).ravel()

    diag = np.nan_to_num(
        np.asarray(
            diagnosis.values
            if hasattr(diagnosis, "values")
            else diagnosis,
            dtype=np.float64,
        ).ravel()
    )

    logger.info(
        "Subjects available: %d",
        len(y),
    )

    # --------------------------------------------------------------
    # Subject-level split
    # --------------------------------------------------------------

    (
        train_idx,
        val_idx,
        test_idx,
    ) = create_subject_splits(
        diagnosis,
        val_fraction=split_cfg.get(
            "val_fraction",
            0.15,
        ),
        test_fraction=split_cfg.get(
            "test_fraction",
            0.15,
        ),
        seed=seed,
        stratify=split_cfg.get(
            "stratify",
            True,
        ),
    )

    logger.info(
        "Fixed subject split:"
    )

    logger.info(
        "  Train: %d",
        len(train_idx),
    )

    logger.info(
        "  Validation: %d",
        len(val_idx),
    )

    logger.info(
        "  Test: %d (UNTOUCHED)",
        len(test_idx),
    )

    # --------------------------------------------------------------
    # Fit preprocessing ONLY on training subjects
    # --------------------------------------------------------------

    logger.info(
        "Fitting preprocessing on TRAIN only..."
    )

    prep = ModalityPreprocessor().fit(
        clin.iloc[train_idx],
        mri.iloc[train_idx],
        pet.iloc[train_idx],
        gen.iloc[train_idx],
    )

    # --------------------------------------------------------------
    # Build train / validation datasets
    # --------------------------------------------------------------

    train_ds = build_dataset_from_indices(
        prep=prep,
        clin=clin,
        mri=mri,
        pet=pet,
        gen=gen,
        y=y,
        diag=diag,
        indices=train_idx,
    )

    val_ds = build_dataset_from_indices(
        prep=prep,
        clin=clin,
        mri=mri,
        pet=pet,
        gen=gen,
        y=y,
        diag=diag,
        indices=val_idx,
    )

    logger.info(
        "Development datasets ready:"
    )

    logger.info(
        "  Train samples: %d",
        len(train_ds),
    )

    logger.info(
        "  Validation samples: %d",
        len(val_ds),
    )

    # --------------------------------------------------------------
    # Input dimensions
    # --------------------------------------------------------------

    input_dims = {
        "clinical": train_ds.clinical.shape[1],
        "mri": train_ds.mri.shape[1],
        "pet": train_ds.pet.shape[1],
        "genetic": train_ds.genetic.shape[1],
    }

    logger.info(
        "Input dimensions: %s",
        input_dims,
    )

    # --------------------------------------------------------------
    # MRI feature names
    # --------------------------------------------------------------

    mri_feature_names = list(
        mri.columns
    )

    return {
        "clin": clin,
        "mri": mri,
        "pet": pet,
        "gen": gen,
        "y": y,
        "diag": diag,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "train_ds": train_ds,
        "val_ds": val_ds,
        "input_dims": input_dims,
        "mri_feature_names": mri_feature_names,
    }


# =====================================================================
# MODEL FACTORY
# =====================================================================

def make_model(
    input_dims,
    model_cfg,
    config,
    mri_feature_names,
):
    """
    Create a completely fresh Fed-PhenoGraft model.

    This mirrors the model construction used by the main Option-B
    pipeline.
    """

    mri_cfg = config.get(
        "mri",
        {},
    )

    model = FedPhenoGraft(
        input_dims,

        embed_dim=model_cfg.get(
            "embed_dim",
            32,
        ),

        num_heads=model_cfg.get(
            "num_heads",
            4,
        ),

        dropout=model_cfg.get(
            "dropout",
            0.35,
        ),

        mri_feature_names=list(
            mri_feature_names
        ),

        mri_graph_features=mri_cfg.get(
            "graph_features",
            [],
        ),

        mri_gnn_hidden=int(
            mri_cfg.get(
                "gnn_hidden_dim",
                64,
            )
        ),
    )

    return model


# =====================================================================
# SITE LABELS
# =====================================================================

def get_train_site_labels(
    config,
    clin,
    train_idx,
):
    """
    Load real acquisition-site labels if available.

    This mirrors the behavior of the main pipeline.

    If no site labels are available, None is returned.

    The actual partition type is still taken from config.yaml.
    """

    paths_cfg = config.get(
        "paths",
        {},
    )

    raw_path = paths_cfg.get(
        "raw",
        "data/raw",
    )

    site_series = load_site_labels(
        PROJECT_ROOT / raw_path
    )

    if site_series is None:
        return None

    site_labels = (
        site_series
        .reindex(clin.index)
        .values[train_idx]
    )

    return site_labels


# =====================================================================
# SINGLE TUNING TRIAL
# =====================================================================

def run_trial(
    cls_weight,
    hsic_weight,
    ccc_weight,
    data,
    config,
):
    """
    Run exactly ONE development-stage federated training trial.

    No test evaluation is performed.

    Returns a dictionary containing:
        cls_weight
        hsic_weight
        ccc_weight
        best_round
        best_val_ccc
        best_val_rmse
        best_val_mae
        best_val_r2
        best_val_pearson
        best_val_auc
        best_val_accuracy
        best_val_f1
    """

    train_cfg = config.get(
        "training",
        {}
    )

    model_cfg = config.get(
        "model",
        {}
    )

    # --------------------------------------------------------------
    # Fixed seed
    # --------------------------------------------------------------

    seed_everything(
        TUNING_SEED
    )

    # --------------------------------------------------------------
    # Configuration
    # --------------------------------------------------------------

    partition = train_cfg.get(
        "partition",
        "iid",
    )

    num_clients = int(
        train_cfg.get(
            "num_clients",
            4,
        )
    )

    dirichlet_alpha = float(
        train_cfg.get(
            "dirichlet_alpha",
            0.5,
        )
    )

    local_epochs = int(
        train_cfg.get(
            "local_epochs",
            2,
        )
    )

    lr = float(
        train_cfg.get(
            "lr",
            1e-3,
        )
    )

    weight_decay = float(
        train_cfg.get(
            "weight_decay",
            1e-4,
        )
    )

    grad_clip = float(
        train_cfg.get(
            "grad_clip",
            1.0,
        )
    )

    batch_size = int(
        train_cfg.get(
            "batch_size",
            32,
        )
    )

    device = train_cfg.get(
        "device",
        "cpu",
    )

    # --------------------------------------------------------------
    # Client partition
    # --------------------------------------------------------------

    train_site_labels = get_train_site_labels(
        config=config,
        clin=data["clin"],
        train_idx=data["train_idx"],
    )

    client_datasets = create_federated_splits(
        data["train_ds"],
        num_clients=num_clients,
        seed=TUNING_SEED,
        partition=partition,
        dirichlet_alpha=dirichlet_alpha,
        site_labels=train_site_labels,
    )

    logger.info(
        "Client sizes: %s",
        [
            len(ds)
            for ds in client_datasets
        ],
    )

    # --------------------------------------------------------------
    # Fresh model
    # --------------------------------------------------------------

    model = make_model(
        input_dims=data["input_dims"],
        model_cfg=model_cfg,
        config=config,
        mri_feature_names=data["mri_feature_names"],
    )

    # --------------------------------------------------------------
    # Federated development training
    # --------------------------------------------------------------

    logger.info(
        "Starting trial: "
        "cls_weight=%.3f | "
        "hsic_weight=%.3f | "
        "ccc_weight=%.3f",
        cls_weight,
        hsic_weight,
        ccc_weight,
    )

    (
        model,
        history,
        best_round,
    ) = simulate_federated_training(
        model,
        client_datasets,
        data["val_ds"],

        num_rounds=NUM_ROUNDS,

        local_epochs=local_epochs,

        lr=lr,

        weight_decay=weight_decay,

        hsic_weight=hsic_weight,

        cls_weight=cls_weight,

        ccc_weight=ccc_weight,

        grad_clip=grad_clip,

        batch_size=batch_size,

        early_stopping_patience=EARLY_STOPPING_PATIENCE,

        train_eval_dataset=data["train_ds"],

        device=device,
    )

    # --------------------------------------------------------------
    # Extract the metrics corresponding to the best round
    # --------------------------------------------------------------

    best_history_entry = None

    for entry in history:
        if int(entry["round"]) == int(best_round):
            best_history_entry = entry
            break

    if best_history_entry is None:
        raise RuntimeError(
            "Could not find best round in training history. "
            f"best_round={best_round}"
        )

    val_metrics = best_history_entry["val"]

    result = {
        "cls_weight": float(cls_weight),
        "hsic_weight": float(hsic_weight),
        "ccc_weight": float(ccc_weight),

        "seed": int(TUNING_SEED),

        "best_round": int(best_round),

        "val_ccc": float(
            val_metrics["ccc"]
        ),

        "val_rmse": float(
            val_metrics["rmse"]
        ),

        "val_mae": float(
            val_metrics["mae"]
        ),

        "val_r2": float(
            val_metrics["r2"]
        ),

        "val_pearson": float(
            val_metrics["pearson"]
        ),

        "val_pred_mean": float(
            val_metrics.get(
                "pred_mean",
                np.nan,
            )
        ),

        "val_target_mean": float(
            val_metrics.get(
                "target_mean",
                np.nan,
            )
        ),

        "val_pred_std": float(
            val_metrics.get(
                "pred_std",
                np.nan,
            )
        ),

        "val_target_std": float(
            val_metrics.get(
                "target_std",
                np.nan,
            )
        ),

        "val_mean_error": float(
            val_metrics.get(
                "mean_error",
                np.nan,
            )
        ),

        "val_std_ratio": float(
            val_metrics.get(
                "std_ratio",
                np.nan,
            )
        ),

        "val_auc": float(
            val_metrics.get(
                "auc",
                np.nan,
            )
        ),

        "val_accuracy": float(
            val_metrics.get(
                "accuracy",
                np.nan,
            )
        ),

        "val_f1": float(
            val_metrics.get(
                "f1",
                np.nan,
            )
        ),
    }

    logger.info(
        "RESULT | "
        "cls=%.3f | "
        "hsic=%.3f | "
        "ccc=%.3f | "
        "best_round=%d | "
        "val_CCC=%.4f | "
        "val_RMSE=%.4f | "
        "val_MAE=%.4f",
        result["cls_weight"],
        result["hsic_weight"],
        result["ccc_weight"],
        result["best_round"],
        result["val_ccc"],
        result["val_rmse"],
        result["val_mae"],
    )

    return result


# =====================================================================
# SAVE RESULTS
# =====================================================================

def save_results(results):
    """
    Save all tuning results as CSV.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.DataFrame(
        results
    )

    # Sort by validation CCC.
    df = df.sort_values(
        by="val_ccc",
        ascending=False,
    ).reset_index(
        drop=True
    )

    df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    logger.info(
        "Saved tuning results to:\n%s",
        RESULTS_PATH,
    )

    return df


# =====================================================================
# SAVE BEST CONFIGURATION
# =====================================================================

def save_best_config(
    config,
    best_result,
):
    """
    Save a small YAML file containing the selected loss weights.

    The original config.yaml is NOT modified automatically.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_config = {
        "training": {
            "cls_weight": float(
                best_result["cls_weight"]
            ),
            "hsic_weight": float(
                best_result["hsic_weight"]
            ),
            "ccc_weight": float(
                best_result["ccc_weight"]
            ),
        },

        "selection": {
            "seed": int(
                best_result["seed"]
            ),
            "best_round": int(
                best_result["best_round"]
            ),
            "validation_ccc": float(
                best_result["val_ccc"]
            ),
            "validation_rmse": float(
                best_result["val_rmse"]
            ),
            "validation_mae": float(
                best_result["val_mae"]
            ),
            "validation_r2": float(
                best_result["val_r2"]
            ),
        },
    }

    with open(
        BEST_CONFIG_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        yaml.safe_dump(
            best_config,
            f,
            sort_keys=False,
        )

    logger.info(
        "Saved best configuration to:\n%s",
        BEST_CONFIG_PATH,
    )


# =====================================================================
# PRINT RESULTS TABLE
# =====================================================================

def print_results_table(
    title,
    results,
):
    """
    Print a compact ranking table.
    """

    if not results:
        return

    df = pd.DataFrame(
        results
    )

    df = df.sort_values(
        by="val_ccc",
        ascending=False,
    )

    print()
    print("=" * 90)
    print(title)
    print("=" * 90)

    columns = [
        "cls_weight",
        "hsic_weight",
        "best_round",
        "val_ccc",
        "val_rmse",
        "val_mae",
        "val_r2",
    ]

    display_df = df[
        columns
    ].copy()

    print(
        display_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("=" * 90)


# =====================================================================
# MAIN TUNING PROCEDURE
# =====================================================================

def main():
    """
    Execute the two-stage tuning procedure.

    Stage 1:
        Sweep cls_weight while keeping HSIC fixed at 0.10.

    Stage 2:
        Fix the best cls_weight and sweep hsic_weight.
    """

    logger.info("=" * 72)
    logger.info("Fed-PhenoGraft LOSS WEIGHT TUNING")
    logger.info("=" * 72)

    logger.info(
        "This script performs DEVELOPMENT-ONLY training."
    )

    logger.info(
        "The TEST SET WILL NOT BE USED."
    )

    logger.info(
        "Tuning seed: %d",
        TUNING_SEED,
    )

    logger.info(
        "Maximum rounds per trial: %d",
        NUM_ROUNDS,
    )

    # --------------------------------------------------------------
    # Load configuration
    # --------------------------------------------------------------

    config = load_config()

    # Make sure the tuning split uses the same primary seed.
    #
    # We intentionally do not modify config.yaml.
    config["seed"] = TUNING_SEED

    # --------------------------------------------------------------
    # Prepare data ONCE
    # --------------------------------------------------------------

    data = prepare_development_data(
        config
    )

    all_results = []

    # ==============================================================
    # STAGE 1
    # ==============================================================

    # fixed_hsic = 0.10

    # logger.info("")
    # logger.info("=" * 72)
    # logger.info(
    #     "STAGE 1: TUNING cls_weight"
    # )
    # logger.info("=" * 72)

    # logger.info(
    #     "Fixed hsic_weight = %.3f",
    #     fixed_hsic,
    # )

    # logger.info(
    #     "Testing cls_weight values: %s",
    #     CLS_VALUES,
    # )

    # cls_results = []

    # for index, cls_weight in enumerate(
    #     CLS_VALUES,
    #     start=1,
    # ):

    #     logger.info("")
    #     logger.info(
    #         "-" * 72
    #     )

    #     logger.info(
    #         "CLS TRIAL %d/%d",
    #         index,
    #         len(CLS_VALUES),
    #     )

    #     logger.info(
    #         "cls_weight=%.3f | hsic_weight=%.3f",
    #         cls_weight,
    #         fixed_hsic,
    #     )

    #     result = run_trial(
    #         cls_weight=cls_weight,
    #         hsic_weight=fixed_hsic,
    #         ccc_weight=0.0,
    #         data=data,
    #         config=config,
    #     )

    #     cls_results.append(
    #         result
    #     )

    #     all_results.append(
    #         result
    #     )

    # print_results_table(
    #     "STAGE 1 RESULTS - cls_weight",
    #     cls_results,
    # )

    # # --------------------------------------------------------------
    # # Best CLS value
    # # --------------------------------------------------------------

    # best_cls_result = max(
    #     cls_results,
    #     key=lambda r: r["val_ccc"],
    # )

    # best_cls_weight = float(
    #     best_cls_result["cls_weight"]
    # )

    # logger.info(
    #     ""
    # )

    # logger.info(
    #     "=" * 72
    # )

    # logger.info(
    #     "BEST cls_weight = %.3f",
    #     best_cls_weight,
    # )

    # logger.info(
    #     "Validation CCC = %.4f",
    #     best_cls_result["val_ccc"],
    # )

    # logger.info(
    #     "Best round = %d",
    #     best_cls_result["best_round"],
    # )

    # logger.info(
    #     "=" * 72
    # )

    # # ==============================================================
    # # STAGE 2
    # # ==============================================================

    # logger.info("")
    # logger.info("=" * 72)
    # logger.info(
    #     "STAGE 2: TUNING hsic_weight"
    # )
    # logger.info("=" * 72)

    # logger.info(
    #     "Fixed cls_weight = %.3f",
    #     best_cls_weight,
    # )

    # logger.info(
    #     "Testing hsic_weight values: %s",
    #     HSIC_VALUES,
    # )

    # hsic_results = []

    # for index, hsic_weight in enumerate(
    #     HSIC_VALUES,
    #     start=1,
    # ):

    #     logger.info("")
    #     logger.info(
    #         "-" * 72
    #     )

    #     logger.info(
    #         "HSIC TRIAL %d/%d",
    #         index,
    #         len(HSIC_VALUES),
    #     )

    #     logger.info(
    #         "cls_weight=%.3f | hsic_weight=%.3f",
    #         best_cls_weight,
    #         hsic_weight,
    #     )

    #     result = run_trial(
    #         cls_weight=best_cls_weight,
    #         hsic_weight=hsic_weight,
    #         ccc_weight=0.0,
    #         data=data,
    #         config=config,
    #     )

    #     hsic_results.append(
    #         result
    #     )

    #     all_results.append(
    #         result
    #     )

    # print_results_table(
    #     "STAGE 2 RESULTS - hsic_weight",
    #     hsic_results,
    # )

    # # --------------------------------------------------------------
    # # Best HSIC value
    # # --------------------------------------------------------------

    # best_hsic_result = max(
    #     hsic_results,
    #     key=lambda r: r["val_ccc"],
    # )

    # best_hsic_weight = float(
    #     best_hsic_result["hsic_weight"]
    # )

    # logger.info("")
    # logger.info(
    #     "=" * 72
    # )

    # logger.info(
    #     "BEST hsic_weight = %.3f",
    #     best_hsic_weight,
    # )

    # logger.info(
    #     "Validation CCC = %.4f",
    #     best_hsic_result["val_ccc"],
    # )

    # logger.info(
    #     "Best round = %d",
    #     best_hsic_result["best_round"],
    # )

    # logger.info(
    #     "=" * 72
    # )

    # ==============================================================
    # STAGE 3: TUNING ccc_weight
    # ==============================================================

    logger.info("")
    logger.info("=" * 72)
    logger.info("STAGE 3: TUNING ccc_weight")
    logger.info("=" * 72)

    best_cls = 0.0
    best_hsic = 0.0

    logger.info(
        "Fixed cls_weight = %.3f",
        best_cls,
    )

    logger.info(
        "Fixed hsic_weight = %.3f",
        best_hsic,
    )

    logger.info(
        "Testing ccc_weight values: %s",
        CCC_WEIGHTS,
    )

    ccc_results = []

    for trial_idx, ccc_weight in enumerate(
        CCC_WEIGHTS,
        start=1,
    ):

        logger.info("")
        logger.info("-" * 72)
        logger.info(
            "CCC TRIAL %d/%d",
            trial_idx,
            len(CCC_WEIGHTS),
        )

        logger.info(
            "cls_weight=%.3f | "
            "hsic_weight=%.3f | "
            "ccc_weight=%.3f",
            best_cls,
            best_hsic,
            ccc_weight,
        )

        result = run_trial(
            cls_weight=best_cls,
            hsic_weight=best_hsic,
            ccc_weight=ccc_weight,
            data=data,
            config=config,
        )

        ccc_results.append(result)

    # --------------------------------------------------------------
    # CCC results table
    # --------------------------------------------------------------

    ccc_results_df = pd.DataFrame(
        ccc_results
    )

    ccc_results_df = ccc_results_df.sort_values(
        by="val_ccc",
        ascending=False,
    )

    logger.info("")
    logger.info("=" * 90)
    logger.info("STAGE 3 RESULTS - ccc_weight")
    logger.info("=" * 90)

    print(
        ccc_results_df[
            [
                "ccc_weight",
                "best_round",
                "val_ccc",
                "val_pearson",
                "val_rmse",
                "val_mae",
                "val_r2",
                "val_pred_mean",
                "val_target_mean",
                "val_pred_std",
                "val_target_std",
                "val_mean_error",
                "val_std_ratio",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # --------------------------------------------------------------
    # Select best CCC weight
    # --------------------------------------------------------------

    best_ccc_result = ccc_results_df.iloc[0]

    best_ccc = float(
        best_ccc_result["ccc_weight"]
    )

    best_round = int(
        best_ccc_result["best_round"]
    )

    logger.info("")
    logger.info("=" * 72)
    logger.info(
        "BEST ccc_weight = %.3f",
        best_ccc,
    )
    logger.info(
        "Validation CCC = %.4f",
        best_ccc_result["val_ccc"],
    )
    logger.info(
        "Best round = %d",
        best_round,
    )
    logger.info("=" * 72)

    # ==============================================================
    # FINAL BEST CONFIGURATION
    # ==============================================================

    #
    # IMPORTANT:
    #
    # We select the final pair from the HSIC sweep because CLS was
    # already fixed at the best value found during Stage 1.
    #

    # best_result = best_hsic_result
    best_result = best_ccc_result

    # For CCC-only tuning, save the CCC sweep results.
    all_results = ccc_results

    # --------------------------------------------------------------
    # Save all results
    # --------------------------------------------------------------

    results_df = save_results(
        all_results
    )

    save_best_config(
        config=config,
        best_result=best_result,
    )

    # --------------------------------------------------------------
    # Final summary
    # --------------------------------------------------------------

    print()
    print()
    print("=" * 90)
    print("FINAL TUNING RESULT")
    print("=" * 90)

    print(
        f"Best cls_weight : "
        f"{best_result['cls_weight']:.3f}"
    )

    print(
        f"Best hsic_weight: "
        f"{best_result['hsic_weight']:.3f}"
    )

    print(
        f"Best ccc_weight : "
        f"{best_result['ccc_weight']:.3f}"
    )

    print(
        f"Best round     : "
        f"{best_result['best_round']}"
    )

    print(
        f"Validation CCC : "
        f"{best_result['val_ccc']:.4f}"
    )

    print(
        f"Validation RMSE: "
        f"{best_result['val_rmse']:.4f}"
    )

    print(
        f"Validation MAE : "
        f"{best_result['val_mae']:.4f}"
    )

    print(
        f"Validation R²  : "
        f"{best_result['val_r2']:.4f}"
    )

    print()
    print(
        f"All results saved to:"
    )

    print(
        f"  {RESULTS_PATH}"
    )

    print(
        f"Best configuration saved to:"
    )

    print(
        f"  {BEST_CONFIG_PATH}"
    )

    print("=" * 90)

    # --------------------------------------------------------------
    # Show top configurations overall
    # --------------------------------------------------------------

    print()
    print(
        "TOP CONFIGURATIONS ACROSS BOTH SWEEPS"
    )

    print(
        results_df[
    [
        "cls_weight",
        "hsic_weight",
        "ccc_weight",
        "best_round",
        "val_ccc",
        "val_pearson",
        "val_rmse",
        "val_mae",
        "val_r2",
        "val_pred_mean",
        "val_target_mean",
        "val_pred_std",
        "val_target_std",
        "val_mean_error",
        "val_std_ratio",
    ]
    ].head(10).to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print(
        "IMPORTANT: Test data was not used during tuning."
    )

    print(
        "Run the full Option-B pipeline only after selecting"
        " the final weights."
    )


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()