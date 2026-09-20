# -*- coding: utf-8 -*-
"""
Fed-PhenoGraft: End-to-End Pipeline.

Final evaluation protocol: Option B
------------------------------------

1. Load raw data.
2. Create subject-level train/validation/test split.
3. DEVELOPMENT STAGE:
      - preprocessing fitted on TRAIN only
      - FedAvg trained on TRAIN
      - VALIDATION used for early stopping / round selection
      - TEST is untouched
4. FINAL STAGE:
      - validation is promoted into the training population
      - preprocessing is refitted on TRAIN + VALIDATION
      - a FRESH model is initialized
      - final FedAvg is trained on TRAIN + VALIDATION
      - number of rounds is fixed from development-stage selection
      - no validation / early stopping is used
5. FINAL TEST:
      - TEST is evaluated once
      - test predictions are saved
      - bootstrap CIs are calculated
      - paired bootstrap comparison against the strongest baseline
6. Ablations and explainability are run after the primary evaluation.
"""

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------
# Project path
# ---------------------------------------------------------------------

project_root = Path(__file__).resolve().parent.parent

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


# ---------------------------------------------------------------------
# Matplotlib
# ---------------------------------------------------------------------

import matplotlib

matplotlib.use("Agg")


# ---------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------

import logging

import numpy as np
import pandas as pd
import torch
import yaml

from torch.utils.data import DataLoader


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
    fit_final_on_train_val,
)

from src.evaluation.xai import (
    extract_attention_weights,
    visualize_attention,
    plot_pred_vs_actual,
    plot_confusion_matrix,
    stress_test_missing_modalities,
    visualize_feature_importance,
    counterfactual_gene_analysis,
)

from src.evaluation.stats import (
    bootstrap_metric_ci,
    paired_bootstrap_test,
    summarize_seed_runs,
)

from src.evaluation.ablation import (
    run_ablation_suite,
)

from src.evaluation.results_report import (
    generate_report,
)

from src.baselines.runner import (
    run_baselines,
)


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(message)s",
)

logger = logging.getLogger(
    "FedPhenoGraft"
)


# =====================================================================
# CONFIG
# =====================================================================

def load_config():
    """
    Load config.yaml from the project root.
    """
    config_path = (
        project_root / "config.yaml"
    )

    with open(
        config_path,
        "r",
    ) as f:
        return yaml.safe_load(f)


# =====================================================================
# DATASET CONSTRUCTION
# =====================================================================

def _build_dataset_from_indices(
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
    Transform a subset using an already-fitted ModalityPreprocessor.

    The returned dataset preserves the subject ordering represented
    by `indices`.
    """
    c, m, p, g = prep.transform(
        clin.iloc[indices],
        mri.iloc[indices],
        pet.iloc[indices],
        gen.iloc[indices],
    )

    return FederatedPPMIDataset(
        c,
        m,
        p,
        g,
        pd.Series(
            y[indices],
            index=c.index,
        ),
        diagnosis=diag[indices],
    )


def _build_development_and_final_datasets(
    clin,
    mri,
    pet,
    gen,
    y,
    diag,
    train_idx,
    val_idx,
    test_idx,
):
    """
    Build the two preprocessing regimes required by Option B.

    DEVELOPMENT preprocessing:
        fitted only on TRAIN.

    FINAL preprocessing:
        fitted on TRAIN + VALIDATION.

    TEST is transformed using the corresponding preprocessing object,
    but never contributes to fitting either preprocessor.

    Returns:

        development_train_ds
        development_val_ds
        development_test_ds
        final_train_ds
        final_val_ds
        final_test_ds
        development_prep
        final_prep
    """

    # ==============================================================
    # DEVELOPMENT PREPROCESSING
    # ==============================================================

    logger.info(
        "Fitting DEVELOPMENT preprocessing on TRAIN only..."
    )

    development_prep = (
        ModalityPreprocessor().fit(
            clin.iloc[train_idx],
            mri.iloc[train_idx],
            pet.iloc[train_idx],
            gen.iloc[train_idx],
        )
    )

    development_train_ds = (
        _build_dataset_from_indices(
            development_prep,
            clin,
            mri,
            pet,
            gen,
            y,
            diag,
            train_idx,
        )
    )

    development_val_ds = (
        _build_dataset_from_indices(
            development_prep,
            clin,
            mri,
            pet,
            gen,
            y,
            diag,
            val_idx,
        )
    )

    development_test_ds = (
        _build_dataset_from_indices(
            development_prep,
            clin,
            mri,
            pet,
            gen,
            y,
            diag,
            test_idx,
        )
    )

    # ==============================================================
    # FINAL PREPROCESSING
    # ==============================================================

    development_idx = np.sort(
        np.concatenate(
            [
                train_idx,
                val_idx,
            ]
        )
    )

    logger.info(
        "Fitting FINAL preprocessing on TRAIN + VALIDATION..."
    )

    final_prep = (
        ModalityPreprocessor().fit(
            clin.iloc[development_idx],
            mri.iloc[development_idx],
            pet.iloc[development_idx],
            gen.iloc[development_idx],
        )
    )

    final_train_ds = (
        _build_dataset_from_indices(
            final_prep,
            clin,
            mri,
            pet,
            gen,
            y,
            diag,
            train_idx,
        )
    )

    final_val_ds = (
        _build_dataset_from_indices(
            final_prep,
            clin,
            mri,
            pet,
            gen,
            y,
            diag,
            val_idx,
        )
    )

    final_test_ds = (
        _build_dataset_from_indices(
            final_prep,
            clin,
            mri,
            pet,
            gen,
            y,
            diag,
            test_idx,
        )
    )

    return (
        development_train_ds,
        development_val_ds,
        development_test_ds,
        final_train_ds,
        final_val_ds,
        final_test_ds,
        development_prep,
        final_prep,
    )


# =====================================================================
# MODEL FACTORY
# =====================================================================

def _make_model(
    input_dims,
    model_cfg,
    config,
    mri_feature_names,
):
    """
    Construct a completely fresh Fed-PhenoGraft model.
    """

    mri_cfg = config.get(
        "mri",
        {},
    )

    return FedPhenoGraft(
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


# =====================================================================
# DEVELOPMENT FEDERATED RUN
# =====================================================================

def _run_development_seed(
    run_seed,
    train_ds,
    val_ds,
    input_dims,
    model_cfg,
    config,
    train_cfg,
    loss_cfg,
    partition,
    train_site_labels,
    mri_feature_names,
):
    """
    Stage A.

    Train only on the original training subjects.

    Validation is used for:
        - early stopping
        - selecting the best round

    Test is never touched.
    """

    seed_everything(
        run_seed
    )

    client_datasets = (
        create_federated_splits(
            train_ds,
            num_clients=train_cfg.get(
                "num_clients",
                4,
            ),
            seed=run_seed,
            partition=partition,
            dirichlet_alpha=train_cfg.get(
                "dirichlet_alpha",
                0.5,
            ),
            site_labels=train_site_labels,
        )
    )

    logger.info(
        f"  Development client sizes "
        f"(seed {run_seed}): "
        f"{[len(ds) for ds in client_datasets]}"
    )

    model = _make_model(
        input_dims=input_dims,
        model_cfg=model_cfg,
        config=config,
        mri_feature_names=mri_feature_names,
    )

    model, history, best_round = (
        simulate_federated_training(
            model,
            client_datasets,
            val_ds,

            num_rounds=train_cfg.get(
                "num_rounds",
                30,
            ),

            local_epochs=train_cfg.get(
                "local_epochs",
                2,
            ),

            lr=train_cfg.get(
                "lr",
                1e-3,
            ),

            weight_decay=train_cfg.get(
                "weight_decay",
                1e-4,
            ),

            hsic_weight=loss_cfg.get(
                "hsic_weight",
                0.0,
            ),

            cls_weight=loss_cfg.get(
                "cls_weight",
                0.0,
            ),

            ccc_weight=loss_cfg.get(
                "ccc_weight",
                0.0,
            ),

            grad_clip=train_cfg.get(
                "grad_clip",
                1.0,
            ),

            batch_size=train_cfg.get(
                "batch_size",
                32,
            ),

            early_stopping_patience=train_cfg.get(
                "early_stopping_patience",
                5,
            ),

            train_eval_dataset=train_ds,

            device=train_cfg.get(
                "device",
                "cpu",
            ),
        )
    )

    return (
        model,
        history,
        best_round,
        client_datasets,
    )


# =====================================================================
# FINAL FEDERATED RUN
# =====================================================================

def _run_final_seed(
    run_seed,
    best_round,
    final_train_ds,
    final_val_ds,
    input_dims,
    model_cfg,
    config,
    train_cfg,
    loss_cfg,
    partition,
    train_site_labels,
    mri_feature_names,
):
    """
    Stage B.

    Train on TRAIN + VALIDATION.

    The model is freshly initialized.

    `best_round` was selected during Stage A using validation.

    There is no validation monitoring here.
    """

    seed_everything(
        run_seed
    )

    # --------------------------------------------------------------
    # Create the original training clients from the original
    # training subjects.
    #
    # fit_final_on_train_val() then distributes the former
    # validation subjects across those clients.
    # --------------------------------------------------------------

    train_client_datasets = (
        create_federated_splits(
            final_train_ds,
            num_clients=train_cfg.get(
                "num_clients",
                4,
            ),
            seed=run_seed,
            partition=partition,
            dirichlet_alpha=train_cfg.get(
                "dirichlet_alpha",
                0.5,
            ),
            site_labels=train_site_labels,
        )
    )

    logger.info(
        f"  Final-stage original client sizes "
        f"(seed {run_seed}): "
        f"{[len(ds) for ds in train_client_datasets]}"
    )

    # --------------------------------------------------------------
    # Fresh model.
    #
    # DO NOT reuse the development model.
    # --------------------------------------------------------------

    final_model = _make_model(
        input_dims=input_dims,
        model_cfg=model_cfg,
        config=config,
        mri_feature_names=mri_feature_names,
    )

    final_model, final_history = (
        fit_final_on_train_val(
            global_model=final_model,

            train_client_datasets=train_client_datasets,

            val_dataset=final_val_ds,

            num_rounds=best_round,

            local_epochs=train_cfg.get(
                "local_epochs",
                2,
            ),

            lr=train_cfg.get(
                "lr",
                1e-3,
            ),

            weight_decay=train_cfg.get(
                "weight_decay",
                1e-4,
            ),

            hsic_weight=loss_cfg.get(
                "hsic_weight",
                0.0,
            ),

            cls_weight=loss_cfg.get(
                "cls_weight",
                0.0,
            ),

            ccc_weight=loss_cfg.get(
                "ccc_weight",
                0.0,
            ),

            grad_clip=train_cfg.get(
                "grad_clip",
                1.0,
            ),

            batch_size=train_cfg.get(
                "batch_size",
                32,
            ),

            seed=run_seed,

            device=train_cfg.get(
                "device",
                "cpu",
            ),
        )
    )

    return (
        final_model,
        final_history,
    )


# =====================================================================
# MAIN PIPELINE
# =====================================================================

def run_pipeline():

    logger.info(
        "=" * 70
    )

    logger.info(
        "  Fed-PhenoGraft: "
        "Phenotype-Guided Multimodal PD Prediction"
    )

    logger.info(
        "=" * 70
    )

    config = load_config()

    seed = int(
        config.get(
            "seed",
            42,
        )
    )

    seed_everything(
        seed
    )

    split_cfg = config.get(
        "split",
        {}
    )

    train_cfg = config.get(
        "training",
        {}
    )

    loss_cfg = config.get(
        "loss",
        {}
    )

    model_cfg = config.get(
        "model",
        {}
    )

    eval_cfg = config.get(
        "evaluation",
        {}
    )

    target_mode = (
        config.get(
            "target",
            {}
        ).get(
            "mode",
            "absolute",
        )
    )

    target_label = (
        "Δ UPDRS-III (BL → Year 2)"
        if target_mode == "delta"
        else "UPDRS-III @ Year 2"
    )

    logger.info(
        f"Regression target: "
        f"{target_label} "
        f"(mode='{target_mode}')"
    )

    # ==============================================================
    # PHASE 1 — LOAD DATA
    # ==============================================================

    logger.info(
        "\n[Phase 1] Loading Raw Data "
        "(no global scaling)..."
    )

    (
        clin,
        mri,
        pet,
        gen,
        targets,
        diagnosis,
    ) = build_real_dataset(
        config
    )

    y = np.asarray(
        targets.values
        if hasattr(
            targets,
            "values"
        )
        else targets,
        dtype=np.float64,
    ).ravel()

    diag = np.nan_to_num(
        np.asarray(
            diagnosis.values
            if hasattr(
                diagnosis,
                "values"
            )
            else diagnosis,
            dtype=np.float64,
        ).ravel()
    )

    # ==============================================================
    # PHASE 2 — SUBJECT-LEVEL SPLIT
    # ==============================================================

    logger.info(
        "\n[Phase 2] Subject-Level "
        "Train/Val/Test Split..."
    )

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
        f"Split sizes: "
        f"train={len(train_idx)}, "
        f"val={len(val_idx)}, "
        f"test={len(test_idx)}"
    )

    # ==============================================================
    # PHASE 3 — DEVELOPMENT + FINAL DATASETS
    # ==============================================================

    logger.info(
        "\n[Phase 3] Building "
        "development and final preprocessing regimes..."
    )

    (
        train_ds,
        val_ds,
        test_ds_development,
        final_train_ds,
        final_val_ds,
        test_ds,
        development_prep,
        final_prep,
    ) = _build_development_and_final_datasets(
        clin=clin,
        mri=mri,
        pet=pet,
        gen=gen,
        y=y,
        diag=diag,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )

    # ==============================================================
    # PHASE 4 — BASELINES
    # ==============================================================

    logger.info(
        "\n[Phase 4] Baseline Models "
        "(CV + final fit on TRAIN + VALIDATION)..."
    )

    raw_features = np.hstack(
        [
            clin.values,
            mri.values,
            pet.values,
            gen.values,
        ]
    )

    trainval_idx = np.sort(
        np.concatenate(
            [
                train_idx,
                val_idx,
            ]
        )
    )

    baseline_results = run_baselines(
        raw_features[
            trainval_idx
        ],

        y[
            trainval_idx
        ],

        raw_features[
            test_idx
        ],

        y[
            test_idx
        ],

        n_splits=config.get(
            "cv",
            {}
        ).get(
            "n_splits",
            5,
        ),

        seed=seed,
    )

    # ==============================================================
    # PHASE 5 — FEDERATED DEVELOPMENT + FINAL TRAINING
    # ==============================================================

    partition = train_cfg.get(
        "partition",
        "iid",
    )

    num_seeds = max(
        1,
        int(
            eval_cfg.get(
                "num_seeds",
                3,
            )
        ),
    )

    device = train_cfg.get(
        "device",
        "cpu",
    )

    logger.info(
        "\n[Phase 5] Federated Training "
        "with Option-B evaluation protocol"
    )

    logger.info(
        f"Partition: {partition}"
    )

    logger.info(
        f"Number of seeds: {num_seeds}"
    )

    # --------------------------------------------------------------
    # Federated site labels
    # --------------------------------------------------------------

    site_series = load_site_labels(
        project_root
        / config["paths"]["raw"]
    )

    train_site_labels = None

    if site_series is not None:

        train_site_labels = (
            site_series
            .reindex(
                clin.index
            )
            .values[
                train_idx
            ]
        )

        logger.info(
            "Site labels found — "
            "'site' partition available."
        )

    elif partition == "site":

        logger.warning(
            "partition='site' requested but "
            "no Center-Subject list CSV was found. "
            "Falling back to Dirichlet label-skew."
        )

    # --------------------------------------------------------------
    # Model input dimensions
    # --------------------------------------------------------------

    input_dims = {
        "clinical": train_ds.clinical.shape[1],
        "mri": train_ds.mri.shape[1],
        "pet": train_ds.pet.shape[1],
        "genetic": train_ds.genetic.shape[1],
    }

    mri_feature_names = list(
        mri.columns
    )

    # --------------------------------------------------------------
    # Evaluation loaders
    # --------------------------------------------------------------

    development_train_loader = DataLoader(
        train_ds,
        batch_size=64,
        shuffle=False,
    )

    development_val_loader = DataLoader(
        val_ds,
        batch_size=64,
        shuffle=False,
    )

    final_train_loader = DataLoader(
        final_train_ds,
        batch_size=64,
        shuffle=False,
    )

    final_val_loader = DataLoader(
        final_val_ds,
        batch_size=64,
        shuffle=False,
    )

    final_test_loader = DataLoader(
        test_ds,
        batch_size=64,
        shuffle=False,
    )

    # ==============================================================
    # STAGE A — DEVELOPMENT
    # ==============================================================

    logger.info(
        "\n" + "=" * 70
    )

    logger.info(
        "STAGE A — DEVELOPMENT "
        "(TRAIN → VALIDATION)"
    )

    logger.info(
        "=" * 70
    )

    development_runs = []

    for seed_number in range(
        num_seeds
    ):

        run_seed = (
            seed
            + seed_number * 101
        )

        logger.info(
            f"\n--- DEVELOPMENT SEED "
            f"{seed_number + 1}/{num_seeds} "
            f"(seed={run_seed}) ---"
        )

        (
            development_model,
            development_history,
            best_round,
            development_clients,
        ) = _run_development_seed(
            run_seed=run_seed,

            train_ds=train_ds,

            val_ds=val_ds,

            input_dims=input_dims,

            model_cfg=model_cfg,

            config=config,

            train_cfg=train_cfg,

            loss_cfg=loss_cfg,

            partition=partition,

            train_site_labels=train_site_labels,

            mri_feature_names=mri_feature_names,
        )

        development_val_metrics = evaluate_model(
            development_model,
            development_val_loader,
            device=device,
        )

        logger.info(
            f"Development seed {run_seed}: "
            f"best round={best_round}, "
            f"final selected Val CCC="
            f"{development_val_metrics['ccc']:.4f}"
        )

        development_runs.append(
            {
                "seed": run_seed,
                "best_round": int(
                    best_round
                ),
                "val_metrics": development_val_metrics,
                "history": development_history,
                "client_sizes": [
                    len(ds)
                    for ds in development_clients
                ],
            }
        )

    # ==============================================================
    # Determine primary seed
    # ==============================================================

    # IMPORTANT:
    #
    # We do NOT choose the primary seed using TEST performance.
    #
    # Seed 42 remains the primary reproducibility seed when it exists.
    # Other seeds are robustness runs.
    #
    # If the configured seed is not represented for some reason,
    # fall back to the first development seed.

    primary_seed = seed

    primary_development_run = next(
        (
            run
            for run in development_runs
            if run["seed"] == primary_seed
        ),
        development_runs[0],
    )

    primary_best_round = int(
        primary_development_run[
            "best_round"
        ]
    )

    logger.info(
        "\nDevelopment-stage round selection:"
    )

    for run in development_runs:

        logger.info(
            f"  Seed {run['seed']}: "
            f"best round={run['best_round']} | "
            f"Val CCC="
            f"{run['val_metrics']['ccc']:.4f}"
        )

    logger.info(
        f"Primary seed={primary_seed}; "
        f"primary fixed final round="
        f"{primary_best_round}"
    )

    # ==============================================================
    # STAGE B — FINAL FIT
    # ==============================================================

    logger.info(
        "\n" + "=" * 70
    )

    logger.info(
        "STAGE B — FINAL FIT "
        "(TRAIN + VALIDATION → FIXED ROUNDS)"
    )

    logger.info(
        "=" * 70
    )

    final_runs = []

    for development_run in development_runs:

        run_seed = development_run[
            "seed"
        ]

        fixed_rounds = int(
            development_run[
                "best_round"
            ]
        )

        logger.info(
            f"\n--- FINAL SEED "
            f"{run_seed} "
            f"({fixed_rounds} fixed rounds) ---"
        )

        (
            final_model,
            final_history,
        ) = _run_final_seed(
            run_seed=run_seed,

            best_round=fixed_rounds,

            final_train_ds=final_train_ds,

            final_val_ds=final_val_ds,

            input_dims=input_dims,

            model_cfg=model_cfg,

            config=config,

            train_cfg=train_cfg,

            loss_cfg=loss_cfg,

            partition=partition,

            train_site_labels=train_site_labels,

            mri_feature_names=mri_feature_names,
        )

        # ----------------------------------------------------------
        # TEST EVALUATION
        #
        # This is the first time the final model sees TEST.
        # ----------------------------------------------------------

        test_metrics, test_preds, test_targets = (
            evaluate_model(
                final_model,
                final_test_loader,
                device=device,
                return_predictions=True,
            )
        )

        final_runs.append(
            {
                "seed": run_seed,
                "fixed_rounds": fixed_rounds,
                "model": final_model,
                "history": final_history,
                "test_metrics": test_metrics,
                "test_predictions": np.asarray(
                    test_preds,
                    dtype=np.float64,
                ),
                "test_targets": np.asarray(
                    test_targets,
                    dtype=np.float64,
                ),
            }
        )

        logger.info(
            f"Final seed {run_seed}: "
            f"test CCC="
            f"{test_metrics['ccc']:.4f} | "
            f"RMSE="
            f"{test_metrics['rmse']:.3f}"
        )

    # ==============================================================
    # FINAL SEED SUMMARY
    # ==============================================================

    per_seed_test_metrics = [
        run["test_metrics"]
        for run in final_runs
    ]

    seed_summary = {
        "num_seeds": num_seeds,
        "primary_seed": primary_seed,
        "development_runs": [
            {
                "seed": run["seed"],
                "best_round": run["best_round"],
                "val_metrics": run["val_metrics"],
                "client_sizes": run["client_sizes"],
            }
            for run in development_runs
        ],
        "final_test": summarize_seed_runs(
            per_seed_test_metrics
        ),
    }

    if num_seeds > 1:

        test_ccc_summary = (
            seed_summary[
                "final_test"
            ][
                "ccc"
            ]
        )

        logger.info(
            f"\nAcross {num_seeds} FINAL seeds — "
            f"test CCC "
            f"{test_ccc_summary['mean']:.4f} "
            f"± "
            f"{test_ccc_summary['std']:.4f}"
        )

    # ==============================================================
    # PRIMARY MODEL
    # ==============================================================

    primary_final_run = next(
        (
            run
            for run in final_runs
            if run["seed"] == primary_seed
        ),
        final_runs[0],
    )

    model = primary_final_run[
        "model"
    ]

    # history = primary_final_run[
    #     "history"
    # ]
    history = primary_development_run[
        "history"
    ]

    test_metrics = primary_final_run[
        "test_metrics"
    ]

    fed_test_preds = primary_final_run[
        "test_predictions"
    ]

    test_targets = primary_final_run[
        "test_targets"
    ]

    # ==============================================================
    # FINAL PRIMARY TEST METRICS
    # ==============================================================

    logger.info(
        "\n[Phase 6] Final Held-Out Test "
        "Evaluation + Statistics..."
    )

    # --------------------------------------------------------------
    # Test metrics
    # --------------------------------------------------------------

    train_metrics = evaluate_model(
        model,
        final_train_loader,
        device=device,
    )

    # NOTE:
    #
    # We do NOT use final validation as a model-selection metric.
    # It is now part of the training population.
    #
    # We can still report its predictions descriptively if desired,
    # but it should not be called "validation performance" in the
    # final evaluation table.

    final_development_metrics = evaluate_model(
        model,
        DataLoader(
            torch.utils.data.ConcatDataset(
                [
                    final_train_ds,
                    final_val_ds,
                ]
            ),
            batch_size=64,
            shuffle=False,
        ),
        device=device,
    )

    logger.info(
        f"Primary final model:"
    )

    logger.info(
        f"  Final development CCC: "
        f"{final_development_metrics['ccc']:.4f}"
    )

    logger.info(
        f"  TEST CCC: "
        f"{test_metrics['ccc']:.4f}"
    )

    logger.info(
        f"  TEST RMSE: "
        f"{test_metrics['rmse']:.3f}"
    )

    logger.info(
        f"  TEST MAE: "
        f"{test_metrics['mae']:.3f}"
    )

    logger.info(
        f"  TEST R²: "
        f"{test_metrics['r2']:.4f}"
    )

    logger.info(
        f"  TEST Pearson: "
        f"{test_metrics['pearson']:.4f}"
    )

    if "auc" in test_metrics:

        logger.info(
            f"  TEST PD vs HC AUC: "
            f"{test_metrics['auc']:.4f}"
        )

        logger.info(
            f"  TEST Accuracy: "
            f"{test_metrics['accuracy']:.4f}"
        )

        logger.info(
            f"  TEST F1: "
            f"{test_metrics['f1']:.4f}"
        )

    # ==============================================================
    # BOOTSTRAP CONFIDENCE INTERVAL
    # ==============================================================

    n_boot = int(
        eval_cfg.get(
            "n_bootstrap",
            5000,
        )
    )

    logger.info(
        f"Calculating bootstrap CIs "
        f"with n={n_boot}..."
    )

    fed_test_ci = {}

    for metric_name in (
        "ccc",
        "rmse",
        "mae",
        "r2",
    ):

        fed_test_ci[
            metric_name
        ] = bootstrap_metric_ci(
            test_targets,
            fed_test_preds,
            metric=metric_name,
            n_boot=n_boot,
            seed=seed,
        )

    ccc_ci = fed_test_ci[
        "ccc"
    ]

    logger.info(
        f"  Fed-PhenoGraft TEST CCC "
        f"{ccc_ci['point']:.4f} "
        f"[95% CI "
        f"{ccc_ci['ci_low']:.4f}, "
        f"{ccc_ci['ci_high']:.4f}]"
    )

    # ==============================================================
    # PAIRED COMPARISON AGAINST STRONGEST BASELINE
    # ==============================================================

    significance = None

    baselines_with_preds = {
        name: result
        for name, result
        in baseline_results.items()
        if result.get(
            "test_predictions"
        ) is not None
    }

    if baselines_with_preds:

        best_bl = max(
            baselines_with_preds,
            key=lambda name:
                baselines_with_preds[
                    name
                ][
                    "test_ccc"
                ],
        )

        baseline_predictions = np.asarray(
            baselines_with_preds[
                best_bl
            ][
                "test_predictions"
            ],
            dtype=np.float64,
        )

        significance = (
            paired_bootstrap_test(
                test_targets,
                fed_test_preds,
                baseline_predictions,
                metric="ccc",
                n_boot=n_boot,
                seed=seed,
            )
        )

        significance[
            "compared_against"
        ] = best_bl

        logger.info(
            f"  Paired bootstrap vs "
            f"strongest baseline "
            f"({best_bl}): "
            f"ΔCCC "
            f"{significance['delta']:+.4f}"
        )

        logger.info(
            f"  95% CI: "
            f"["
            f"{significance.get('ci_low', np.nan):.4f}, "
            f"{significance.get('ci_high', np.nan):.4f}"
            f"]"
        )

        logger.info(
            f"  p = "
            f"{significance['p_value']:.4f}"
        )

    # ==============================================================
    # SAVE TEST PREDICTIONS
    # ==============================================================

    results_dir = (
        project_root
        / config["paths"]["results"]
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez(
        results_dir
        / "test_predictions.npz",

        y_true=test_targets,

        fed_phenograft=fed_test_preds,

        **{
            f"baseline_{name}":
                np.asarray(
                    result[
                        "test_predictions"
                    ],
                    dtype=np.float64,
                )
            for name, result
            in baselines_with_preds.items()
        },
    )

    logger.info(
        f"Saved test predictions → "
        f"{results_dir / 'test_predictions.npz'}"
    )

    # ==============================================================
    # TRAINING DIAGNOSTIC
    # ==============================================================

    gap = (
        train_metrics["ccc"]
        - final_development_metrics["ccc"]
    )

    logger.info(
        f"Final-model development/train "
        f"diagnostic gap: "
        f"{gap:+.4f}"
    )

    # NOTE:
    #
    # This is no longer a train-vs-validation gap because validation
    # has been absorbed into final training.
    #
    # Therefore do not label this as a train-val generalization gap.
    #
    # The original early-stopping stage already provides that
    # development-stage diagnostic.

    # ==============================================================
    # PHASE 7 — ABLATIONS
    # ==============================================================

    ablation_results = None

    if config.get(
        "ablation",
        {}
    ).get(
        "enabled",
        False,
    ):

        logger.info(
            "\n[Phase 7] Ablation Suite "
            "(each variant retrained)..."
        )

        ablation_results = run_ablation_suite(
            train_ds,
            val_ds,
            test_ds,
            input_dims,
            config,
            mri_feature_names=mri_feature_names,
            site_labels=train_site_labels,
        )

        ablation_results[
            "full"
        ] = {
            "test": test_metrics,
            "final_protocol": "option_b",
        }

    # ==============================================================
    # PHASE 8 — EXPLAINABILITY
    # ==============================================================

    logger.info(
        "\n[Phase 8] Generating "
        "Explainability & Presentation Figures..."
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=32,
        shuffle=False,
    )

    fig_dir = (
        project_root
        / config["paths"]["figures"]
    )

    fig_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    attn_weights = (
        extract_attention_weights(
            model,
            test_loader,
        )
    )

    visualize_attention(
        attn_weights,
        str(
            fig_dir
            / "attention_maps.png"
        ),
    )

    plot_pred_vs_actual(
        model,
        test_loader,
        save_path=str(
            fig_dir
            / "pred_vs_actual.png"
        ),
        target_label=target_label,
    )

    plot_confusion_matrix(
        model,
        test_loader,
        save_path=str(
            fig_dir
            / "confusion_matrix.png"
        ),
    )

    stress_test_missing_modalities(
        model,
        test_loader,
        save_path=str(
            fig_dir
            / "modality_robustness.png"
        ),
    )

    visualize_feature_importance(
        model,
        test_loader,
        save_path=str(
            fig_dir
            / "global_feature_importance.png"
        ),
    )

    counterfactual_deltas = (
        counterfactual_gene_analysis(
            model,
            final_prep,
            clin.iloc[test_idx],
            mri.iloc[test_idx],
            pet.iloc[test_idx],
            gen.iloc[test_idx],
            save_path=str(
                fig_dir
                / "counterfactual_genes.png"
            ),
            target_label=target_label,
        )
    )

    if counterfactual_deltas:

        logger.info(
            f"  Counterfactual gene shifts "
            f"({target_label}): "
            f"{counterfactual_deltas}"
        )

    # ==============================================================
    # SAVE MODEL
    # ==============================================================

    model_dir = (
        project_root
        / config["paths"]["models"]
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        model.state_dict(),
        str(
            model_dir
            / "fed_phenograft_best.pt"
        ),
    )

    # ==============================================================
    # FINAL RESULTS JSON
    # ==============================================================

    summary = {

        "evaluation_protocol": {
            "name": "Option B",

            "development_train_subjects":
                int(len(train_idx)),

            "development_validation_subjects":
                int(len(val_idx)),

            "final_fit_subjects":
                int(
                    len(train_idx)
                    + len(val_idx)
                ),

            "test_subjects":
                int(len(test_idx)),

            "test_used_for_selection":
                False,

            "final_preprocessing_fit":
                "train + validation",

            "baseline_final_fit":
                "train + validation",

            "fed_phenograft_final_fit":
                "train + validation",

            "test_evaluated_once":
                True,
        },

        "target": {
            "mode": target_mode,
            "label": target_label,
        },

        "split_sizes": {
            "train": len(train_idx),
            "val": len(val_idx),
            "test": len(test_idx),
        },

        "federated_partition": partition,

        "fed_phenograft": {

            "primary_seed":
                primary_seed,

            "selected_development_rounds":
                primary_best_round,

            "train":
                train_metrics,

            "final_development":
                final_development_metrics,

            "test":
                test_metrics,

            "rounds_history":
                history,

            "final_test_predictions_file":
                "test_predictions.npz",
        },

        "statistics": {

            "n_bootstrap":
                n_boot,

            "test_ci_95":
                fed_test_ci,

            "significance_vs_strongest_baseline":
                significance,

            "seed_runs":
                seed_summary,
        },

        "ablations":
            ablation_results,

        "baselines":
            baseline_results,

        "counterfactual_gene_deltas":
            counterfactual_deltas,

        "config": {
            "training":
                train_cfg,

            "model":
                model_cfg,

            "split":
                split_cfg,

            "evaluation":
                eval_cfg,

            "seed":
                seed,
        },
    }

    with open(
        results_dir
        / "final_metrics.json",
        "w",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            default=float,
        )

    # ==============================================================
    # REPORT
    # ==============================================================

    generate_report(
        results_dir,
        fig_dir,
    )

    # ==============================================================
    # DONE
    # ==============================================================

    logger.info(
        "\n" + "=" * 70
    )

    logger.info(
        "  Pipeline Complete!"
    )

    logger.info(
        f"  Metrics summary → "
        f"{results_dir / 'final_metrics.json'}"
    )

    logger.info(
        f"  Test predictions → "
        f"{results_dir / 'test_predictions.npz'}"
    )

    logger.info(
        f"  Results report → "
        f"{results_dir / 'RESULTS.md'}"
    )

    logger.info(
        f"  Figures → "
        f"{fig_dir}"
    )

    logger.info(
        f"  Model weights → "
        f"{model_dir / 'fed_phenograft_best.pt'}"
    )

    logger.info(
        "=" * 70
    )


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    run_pipeline()