# src/federated/fedavg_orchestrator.py

import copy
import logging
import random

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Subset

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
)

from src.evaluation.metrics import (
    concordance_correlation_coefficient,
    mae,
    pearson_r,
    r2_score,
    rmse,
)

from src.models.ccc_loss import CCCLoss

logger = logging.getLogger(__name__)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int) -> None:
    """
    Set Python, NumPy and PyTorch random seeds.

    This is used before both the development and final
    federated training stages.
    """
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Reproducibility settings.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# FEDAVG
# ============================================================

def fedavg_aggregate(
    global_model,
    client_models,
    client_sizes,
):
    """
    Sample-size-weighted FedAvg (McMahan et al.).

    Client updates are weighted by the number of samples at each
    client so that clients with larger cohorts contribute
    proportionally more to the global model.

    Floating-point parameters/buffers are averaged.

    Non-floating-point tensors, such as integer counters, are
    copied from the first client.
    """
    if len(client_models) == 0:
        raise ValueError(
            "fedavg_aggregate received no client models."
        )

    if len(client_models) != len(client_sizes):
        raise ValueError(
            "Number of client models and client sizes must match."
        )

    weights = np.asarray(
        client_sizes,
        dtype=np.float64,
    )

    if np.any(weights <= 0):
        raise ValueError(
            "All client sizes must be positive."
        )

    weights = weights / weights.sum()

    global_dict = global_model.state_dict()

    client_dicts = [
        model.state_dict()
        for model in client_models
    ]

    for key in global_dict.keys():

        if global_dict[key].dtype.is_floating_point:

            stacked = torch.stack(
                [
                    client_dicts[i][key].float()
                    * float(weights[i])
                    for i in range(len(client_models))
                ],
                dim=0,
            )

            global_dict[key] = (
                stacked.sum(dim=0)
                .to(global_dict[key].dtype)
            )

        else:
            # Non-floating buffers are not averaged.
            global_dict[key] = client_dicts[0][key]

    global_model.load_state_dict(
        global_dict
    )

    return global_model


# ============================================================
# LOCAL CLIENT TRAINING
# ============================================================

def client_update(
    client_model,
    dataloader,
    epochs,
    lr=1e-3,
    weight_decay=1e-4,
    hsic_weight=0.0,
    cls_weight=0.0,
    ccc_weight=0.0,
    grad_clip=1.0,
    device="cpu",
):
    """
    Perform one client's local training.

    Objective:

        total_loss =
            regression_mse
            + ccc_weight * ccc_loss
            + hsic_weight * hsic_loss
            + cls_weight * classification_loss

    The primary regression target is ΔUPDRS-III.

    CCC is included as an optional auxiliary regression objective
    because CCC is the primary evaluation metric.
    """

    client_model.train()

    optimizer = optim.AdamW(
        client_model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    mse_criterion = torch.nn.MSELoss()

    bce_criterion = torch.nn.BCEWithLogitsLoss()

    ccc_criterion = CCCLoss()

    diagnostic_printed = False

    for _ in range(epochs):

        for batch in dataloader:

            batch = {
                key: (
                    value.to(device)
                    if isinstance(value, torch.Tensor)
                    else value
                )
                for key, value in batch.items()
            }

            optimizer.zero_grad()

            out = client_model(batch)

            # -------------------------------------------------
            # MSE regression loss
            # -------------------------------------------------

            regression_loss = mse_criterion(
                out["pred"],
                batch["target"],
            )

            # -------------------------------------------------
            # CCC regression loss
            # -------------------------------------------------

            raw_ccc_loss = ccc_criterion(
                out["pred"],
                batch["target"],
            )

            weighted_ccc_loss = (
                ccc_weight * raw_ccc_loss
            )

            # -------------------------------------------------
            # HSIC loss
            # -------------------------------------------------

            raw_hsic_loss = out["loss_hsic"]

            weighted_hsic_loss = (
                hsic_weight * raw_hsic_loss
            )

            # -------------------------------------------------
            # Classification loss
            # -------------------------------------------------

            raw_classification_loss = (
                bce_criterion(
                    out["cls_logit"],
                    batch["diagnosis"],
                )
            )

            weighted_classification_loss = (
                cls_weight
                * raw_classification_loss
            )

            # -------------------------------------------------
            # Total objective
            # -------------------------------------------------

            loss = (
                regression_loss
                + weighted_ccc_loss
                + weighted_hsic_loss
                + weighted_classification_loss
            )

            # -------------------------------------------------
            # Diagnostic
            # -------------------------------------------------

            if not diagnostic_printed:

                print(
                    "\n[Loss Diagnostic]"
                )

                print(
                    f"  Regression MSE        : "
                    f"{regression_loss.detach().item():.8f}"
                )

                print(
                    f"  Raw CCC loss          : "
                    f"{raw_ccc_loss.detach().item():.8f}"
                )

                print(
                    f"  CCC weight            : "
                    f"{ccc_weight:.6f}"
                )

                print(
                    f"  Weighted CCC loss     : "
                    f"{weighted_ccc_loss.detach().item():.8f}"
                )

                print(
                    f"  Raw HSIC              : "
                    f"{raw_hsic_loss.detach().item():.8f}"
                )

                print(
                    f"  HSIC weight           : "
                    f"{hsic_weight:.6f}"
                )

                print(
                    f"  Weighted HSIC         : "
                    f"{weighted_hsic_loss.detach().item():.8f}"
                )

                print(
                    f"  Raw classification    : "
                    f"{raw_classification_loss.detach().item():.8f}"
                )

                print(
                    f"  Classification weight : "
                    f"{cls_weight:.6f}"
                )

                print(
                    f"  Weighted classification: "
                    f"{weighted_classification_loss.detach().item():.8f}"
                )

                print(
                    f"  Total loss            : "
                    f"{loss.detach().item():.8f}"
                )

                total_value = (
                    loss.detach().item()
                )

                if abs(total_value) > 1e-12:

                    print(
                        f"  MSE fraction          : "
                        f"{regression_loss.detach().item() / total_value:.6%}"
                    )

                    print(
                        f"  CCC fraction          : "
                        f"{weighted_ccc_loss.detach().item() / total_value:.6%}"
                    )

                    print(
                        f"  HSIC fraction         : "
                        f"{weighted_hsic_loss.detach().item() / total_value:.6%}"
                    )

                    print(
                        f"  Classification fraction: "
                        f"{weighted_classification_loss.detach().item() / total_value:.6%}"
                    )

                print()

                diagnostic_printed = True

            # -------------------------------------------------
            # Backpropagation
            # -------------------------------------------------

            loss.backward()

            # -------------------------------------------------
            # Gradient clipping
            # -------------------------------------------------

            if (
                grad_clip is not None
                and grad_clip > 0
            ):

                torch.nn.utils.clip_grad_norm_(
                    client_model.parameters(),
                    grad_clip,
                )

            optimizer.step()

    return client_model


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    model,
    dataloader,
    device="cpu",
    return_predictions=False,
):
    """
    Evaluate FedPhenoGraft on a complete dataset.

    Regression metrics:
        - CCC
        - RMSE
        - MAE
        - R²
        - Pearson correlation

    Classification metrics:
        - AUC
        - Accuracy
        - F1

    CCC/calibration diagnostics:
        - Prediction mean/std
        - Target mean/std
        - Mean error
        - Prediction/target std ratio
        - CCC numerator/denominator

    IMPORTANT:
        CCC is calculated globally across all predictions and targets,
        NOT independently per batch.
    """

    model.eval()

    all_preds = []
    all_targets = []
    all_cls_logits = []
    all_diagnosis = []

    with torch.no_grad():

        for batch in dataloader:

            # ---------------------------------------------------------
            # Move batch to device
            # ---------------------------------------------------------
            batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }

            # ---------------------------------------------------------
            # Forward pass
            # ---------------------------------------------------------
            output = model(batch)

            preds = output["pred"]
            cls_logits = output["cls_logit"]

            # ---------------------------------------------------------
            # Store regression outputs
            # ---------------------------------------------------------
            all_preds.append(
                preds.detach().cpu().reshape(-1)
            )

            all_targets.append(
                batch["target"].detach().cpu().reshape(-1)
            )

            # ---------------------------------------------------------
            # Store classification outputs
            # ---------------------------------------------------------
            all_cls_logits.append(
                cls_logits.detach().cpu().reshape(-1)
            )

            all_diagnosis.append(
                batch["diagnosis"].detach().cpu().reshape(-1)
            )

    # =================================================================
    # Combine ALL batches
    # =================================================================

    preds_all = torch.cat(all_preds).numpy()
    targets_all = torch.cat(all_targets).numpy()

    cls_logits_all = torch.cat(all_cls_logits).numpy()
    diagnosis_all = torch.cat(all_diagnosis).numpy()

    # =================================================================
    # Remove invalid regression values
    # =================================================================

    valid_reg = (
        np.isfinite(preds_all)
        & np.isfinite(targets_all)
    )

    preds_all = preds_all[valid_reg]
    targets_all = targets_all[valid_reg]

    # =================================================================
    # Remove invalid classification values
    # =================================================================

    valid_cls = (
        np.isfinite(cls_logits_all)
        & np.isfinite(diagnosis_all)
    )

    cls_logits_all = cls_logits_all[valid_cls]
    diagnosis_all = diagnosis_all[valid_cls]

    # =================================================================
    # Regression metrics
    # =================================================================

    if len(preds_all) < 2:
        raise ValueError(
            "Not enough valid regression samples to calculate metrics."
        )

    # ---------------------------------------------------------
    # CCC
    #
    # IMPORTANT:
    # This is calculated across the COMPLETE evaluation set.
    # ---------------------------------------------------------

    ccc = concordance_correlation_coefficient(
        targets_all,
        preds_all
    )

    # ---------------------------------------------------------
    # RMSE
    # ---------------------------------------------------------

    rmse = float(
        np.sqrt(
            np.mean(
                (targets_all - preds_all) ** 2
            )
        )
    )

    # ---------------------------------------------------------
    # MAE
    # ---------------------------------------------------------

    mae = float(
        np.mean(
            np.abs(
                targets_all - preds_all
            )
        )
    )

    # ---------------------------------------------------------
    # R²
    # ---------------------------------------------------------

    ss_res = np.sum(
        (targets_all - preds_all) ** 2
    )

    ss_tot = np.sum(
        (targets_all - np.mean(targets_all)) ** 2
    )

    if ss_tot > 1e-12:
        r2 = float(
            1.0 - (ss_res / ss_tot)
        )
    else:
        r2 = float("nan")

    # ---------------------------------------------------------
    # Pearson correlation
    # ---------------------------------------------------------

    target_std = np.std(targets_all)
    pred_std = np.std(preds_all)

    if (
        target_std > 1e-12
        and pred_std > 1e-12
    ):
        pearson = float(
            np.corrcoef(
                targets_all,
                preds_all
            )[0, 1]
        )
    else:
        pearson = float("nan")

    # =================================================================
    # Prediction distribution / CCC diagnostics
    # =================================================================

    target_mean = float(
        np.mean(targets_all)
    )

    pred_mean = float(
        np.mean(preds_all)
    )

    target_std = float(
        np.std(targets_all)
    )

    pred_std = float(
        np.std(preds_all)
    )

    # Difference between prediction and target means.
    mean_error = float(
        pred_mean - target_mean
    )

    # How much of the target variability is reproduced by the model.
    #
    # Example:
    #     0.66 means prediction std is 66% of target std.
    #
    if target_std > 1e-12:
        std_ratio = float(
            pred_std / target_std
        )
    else:
        std_ratio = float("nan")

    # =================================================================
    # Explicit CCC decomposition
    #
    # CCC =
    #
    #       2 * covariance
    # -----------------------------------
    # target_var + pred_var + mean_error²
    #
    # Using Pearson:
    #
    # covariance = Pearson * target_std * pred_std
    # =================================================================

    if (
        np.isfinite(pearson)
        and target_std > 1e-12
        and pred_std > 1e-12
    ):

        ccc_numerator = float(
            2.0
            * pearson
            * target_std
            * pred_std
        )

    else:

        ccc_numerator = float("nan")

    ccc_denominator = float(
        target_std ** 2
        + pred_std ** 2
        + mean_error ** 2
    )

    # =================================================================
    # Classification metrics
    # =================================================================

    cls_auc = float("nan")
    cls_accuracy = float("nan")
    cls_f1 = float("nan")

    if len(cls_logits_all) > 0:

        # Convert logits to probabilities
        cls_probs = 1.0 / (
            1.0 + np.exp(
                -np.clip(
                    cls_logits_all,
                    -50,
                    50
                )
            )
        )

        # Binary predictions
        cls_preds = (
            cls_probs >= 0.5
        ).astype(int)

        # ---------------------------------------------------------
        # AUC
        # ---------------------------------------------------------

        unique_labels = np.unique(
            diagnosis_all
        )

        if len(unique_labels) >= 2:

            cls_auc = float(
                roc_auc_score(
                    diagnosis_all,
                    cls_probs
                )
            )

        # ---------------------------------------------------------
        # Accuracy
        # ---------------------------------------------------------

        cls_accuracy = float(
            accuracy_score(
                diagnosis_all,
                cls_preds
            )
        )

        # ---------------------------------------------------------
        # F1
        # ---------------------------------------------------------

        cls_f1 = float(
            f1_score(
                diagnosis_all,
                cls_preds,
                zero_division=0
            )
        )

    # =================================================================
    # Return metrics
    # =================================================================

    metrics = {

        # ---------------------------------------------------------
        # Primary regression metric
        # ---------------------------------------------------------
        "ccc": float(ccc),

        # ---------------------------------------------------------
        # Standard regression metrics
        # ---------------------------------------------------------
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson": pearson,

        # ---------------------------------------------------------
        # Prediction distribution diagnostics
        # ---------------------------------------------------------
        "pred_mean": pred_mean,
        "target_mean": target_mean,

        "pred_std": pred_std,
        "target_std": target_std,

        "mean_error": mean_error,
        "std_ratio": std_ratio,

        # ---------------------------------------------------------
        # CCC decomposition
        # ---------------------------------------------------------
        "ccc_numerator": ccc_numerator,
        "ccc_denominator": ccc_denominator,

        # ---------------------------------------------------------
        # Classification metrics
        # ---------------------------------------------------------
        "cls_auc": cls_auc,
        "cls_accuracy": cls_accuracy,
        "cls_f1": cls_f1,

        # ---------------------------------------------------------
        # Number of evaluated samples
        # ---------------------------------------------------------
        "n_regression": int(len(preds_all)),
        "n_classification": int(len(diagnosis_all)),
    }

    if return_predictions:
        return metrics, preds_all, targets_all

    return metrics


# ============================================================
# DEVELOPMENT FEDERATED TRAINING
# ============================================================

def simulate_federated_training(
    global_model,
    train_client_datasets,
    val_dataset,
    num_rounds=30,
    local_epochs=2,
    lr=1e-3,
    weight_decay=1e-4,
    hsic_weight=0.0,
    cls_weight=0.0,
    ccc_weight=2000,
    grad_clip=1.0,
    batch_size=32,
    early_stopping_patience=5,
    train_eval_dataset=None,
    device="cpu",
):
    """
    Stage A: development FedAvg training.

    Training data:
        train_client_datasets

    Validation data:
        val_dataset

    The validation set is used ONLY for:

        - round-by-round monitoring
        - early stopping
        - selecting the best development round

    The held-out TEST set must never be passed here.

    Returns:
        best_model
        history
        best_round

    history contains the validation metrics for every completed
    round and, when requested, training metrics.
    """

    global_model = global_model.to(device)

    num_clients = len(train_client_datasets)

    if num_clients == 0:
        raise ValueError(
            "No federated clients were provided."
        )

    client_sizes = [
        len(dataset)
        for dataset in train_client_datasets
    ]

    train_dataloaders = [
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        for dataset in train_client_datasets
    ]

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
    )

    train_eval_loader = (
        DataLoader(
            train_eval_dataset,
            batch_size=64,
            shuffle=False,
        )
        if train_eval_dataset is not None
        else None
    )

    history = []

    best_val_ccc = -np.inf

    best_state = copy.deepcopy(
        global_model.state_dict()
    )

    best_round = 0

    rounds_without_improvement = 0

    for round_idx in range(num_rounds):

        client_models = [
            copy.deepcopy(
                global_model
            ).to(device)
            for _ in range(num_clients)
        ]

        for i in range(num_clients):

            client_models[i] = client_update(
                client_models[i],
                train_dataloaders[i],
                local_epochs,
                lr=lr,
                weight_decay=weight_decay,
                hsic_weight=hsic_weight,
                cls_weight=cls_weight,
                ccc_weight=ccc_weight,
                grad_clip=grad_clip,
                device=device,
            )

        global_model = fedavg_aggregate(
            global_model,
            client_models,
            client_sizes,
        )

        val_metrics = evaluate_model(
            global_model,
            val_dataloader,
            device=device,
        )

        entry = {
            "round": round_idx + 1,
            "val": val_metrics,
        }

        if train_eval_loader is not None:

            train_metrics = evaluate_model(
                global_model,
                train_eval_loader,
                device=device,
            )

            entry["train"] = train_metrics

            entry["ccc_gap"] = (
                train_metrics["ccc"]
                - val_metrics["ccc"]
            )

            gap_str = (
                f" | Train CCC: "
                f"{train_metrics['ccc']:.4f}"
                f" | Gap: "
                f"{entry['ccc_gap']:+.4f}"
            )

        else:
            gap_str = ""

        auc_str = (
            f" | Val AUC: "
            f"{val_metrics['auc']:.4f}"
            if "auc" in val_metrics
            else ""
        )

        logger.info(
            f"Round {round_idx + 1}/"
            f"{num_rounds} | "
            f"Val CCC: "
            f"{val_metrics['ccc']:.4f} | "
            f"Val RMSE: "
            f"{val_metrics['rmse']:.3f}"
            f"{auc_str}"
            f"{gap_str}"
        )

        history.append(entry)

        if (
            val_metrics["ccc"]
            > best_val_ccc
        ):

            best_val_ccc = (
                val_metrics["ccc"]
            )

            best_state = copy.deepcopy(
                global_model.state_dict()
            )

            best_round = (
                round_idx + 1
            )

            rounds_without_improvement = 0

        else:

            rounds_without_improvement += 1

            if (
                rounds_without_improvement
                >= early_stopping_patience
            ):

                logger.info(
                    f"Early stopping at round "
                    f"{round_idx + 1} "
                    f"(no val-CCC improvement "
                    f"for "
                    f"{early_stopping_patience} "
                    f"rounds). "
                    f"Restoring best weights "
                    f"from round "
                    f"{best_round}."
                )

                break

    global_model.load_state_dict(
        best_state
    )

    logger.info(
        f"Federated development training done. "
        f"Best Val CCC: "
        f"{best_val_ccc:.4f} "
        f"(round {best_round})."
    )

    return global_model, history, best_round    


# ============================================================
# FINAL OPTION-B DATA PARTITION
# ============================================================

def _build_final_client_datasets(
    train_client_datasets,
    val_dataset,
    seed=42,
):
    """
    Construct the final federated client datasets.

    The development-stage client datasets contain only the original
    training subjects.

    For the final fit, validation subjects must also become training
    subjects.

    We therefore distribute validation subjects across the EXISTING
    clients rather than creating an additional validation client.

    This preserves the number of federated clients.

    Validation samples are shuffled using the supplied seed before
    assignment.

    Each final client receives:

        original training data
        +
        a subset of the former validation data
    """
    if len(train_client_datasets) == 0:
        raise ValueError(
            "train_client_datasets cannot be empty."
        )

    num_clients = len(
        train_client_datasets
    )

    val_size = len(
        val_dataset
    )

    if val_size == 0:
        raise ValueError(
            "val_dataset is empty."
        )

    rng = np.random.default_rng(
        seed
    )

    shuffled_indices = rng.permutation(
        val_size
    )

    # Split validation indices as evenly as possible.
    index_chunks = np.array_split(
        shuffled_indices,
        num_clients,
    )

    final_client_datasets = []

    for client_idx in range(
        num_clients
    ):

        val_indices = [
            int(index)
            for index in index_chunks[
                client_idx
            ]
        ]

        val_subset = Subset(
            val_dataset,
            val_indices,
        )

        combined_dataset = ConcatDataset(
            [
                train_client_datasets[
                    client_idx
                ],
                val_subset,
            ]
        )

        final_client_datasets.append(
            combined_dataset
        )

    logger.info(
        "Final Option-B federated partition:"
    )

    for idx, dataset in enumerate(
        final_client_datasets
    ):

        logger.info(
            f"  Client {idx + 1}: "
            f"{len(dataset)} samples"
        )

    logger.info(
        f"Total final development samples: "
        f"{sum(len(ds) for ds in final_client_datasets)} "
        f"(original train + validation)"
    )

    return final_client_datasets


# ============================================================
# FINAL OPTION-B FEDERATED TRAINING
# ============================================================

def fit_final_on_train_val(
    global_model,
    train_client_datasets,
    val_dataset,
    num_rounds,
    local_epochs=2,
    lr=1e-3,
    weight_decay=1e-4,
    hsic_weight=0.0,
    cls_weight=0.0,
    ccc_weight=0.0,
    grad_clip=1.0,
    batch_size=32,
    seed=42,
    device="cpu",
):
    """
    Stage B: final Option-B FedAvg training.

    IMPORTANT:

    This function MUST receive a FRESHLY INITIALIZED global_model.

    It does NOT continue from the development-stage model.

    Input data:

        train_client_datasets
            Original training subjects.

        val_dataset
            Original validation subjects.

    Both are merged into the final development population.

    The model is then trained for exactly `num_rounds`, where
    `num_rounds` was selected during Stage A using validation CCC.

    There is:

        - NO validation
        - NO early stopping
        - NO test evaluation
        - NO model selection

    The returned model is the final model to be evaluated on the
    untouched test set.

    This is the Option-B protocol:

        Stage A:
            train -> validation
            choose best round

        Stage B:
            train + validation
            fixed number of rounds

        Stage C:
            test once
    """
    if num_rounds < 1:
        raise ValueError(
            "num_rounds must be >= 1."
        )

    set_seed(
        seed
    )

    global_model = global_model.to(
        device
    )

    # --------------------------------------------------------
    # Merge validation subjects into the existing clients.
    # --------------------------------------------------------

    final_client_datasets = (
        _build_final_client_datasets(
            train_client_datasets=train_client_datasets,
            val_dataset=val_dataset,
            seed=seed,
        )
    )

    num_clients = len(
        final_client_datasets
    )

    client_sizes = [
        len(dataset)
        for dataset in final_client_datasets
    ]

    final_dataloaders = [
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        for dataset in final_client_datasets
    ]

    logger.info(
        "Starting final Option-B FedAvg fit."
    )

    logger.info(
        f"Final development subjects: "
        f"{sum(client_sizes)}"
    )

    logger.info(
        f"Final federated clients: "
        f"{num_clients}"
    )

    logger.info(
        f"Fixed training rounds: "
        f"{num_rounds}"
    )

    logger.info(
        "Validation is now part of the training data. "
        "No validation metrics or early stopping will be used."
    )

    # --------------------------------------------------------
    # Fixed-round FedAvg.
    # --------------------------------------------------------

    history = []

    for round_idx in range(
        num_rounds
    ):

        client_models = [
            copy.deepcopy(
                global_model
            ).to(device)
            for _ in range(num_clients)
        ]

        for client_idx in range(
            num_clients
        ):

            client_models[client_idx] = client_update(
                                            client_models[client_idx],
                                            final_dataloaders[client_idx],
                                            local_epochs,
                                            lr=lr,
                                            weight_decay=weight_decay,
                                            hsic_weight=hsic_weight,
                                            cls_weight=cls_weight,
                                            ccc_weight=ccc_weight,
                                            grad_clip=grad_clip,
                                            device=device,
                                        )

        global_model = fedavg_aggregate(
            global_model,
            client_models,
            client_sizes,
        )

        history.append(
            {
                "round": round_idx + 1,
            }
        )

        logger.info(
            f"Final-fit round "
            f"{round_idx + 1}/"
            f"{num_rounds} complete."
        )

    logger.info(
        "Final Option-B FedAvg training complete. "
        f"Model trained on "
        f"{sum(client_sizes)} "
        f"development subjects for "
        f"{num_rounds} rounds."
    )

    return global_model, history