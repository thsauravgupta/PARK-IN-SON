import copy
import logging

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.evaluation.metrics import (
    concordance_correlation_coefficient, mae, pearson_r, r2_score, rmse,
)

logger = logging.getLogger(__name__)


def fedavg_aggregate(global_model, client_models, client_sizes):
    """
    Sample-size-weighted FedAvg (McMahan et al.). Weighting by client size
    keeps the aggregate unbiased when sites hold unequal cohorts.
    Non-floating-point buffers (e.g. integer counters) are copied, not averaged.
    """
    weights = np.asarray(client_sizes, dtype=np.float64)
    weights = weights / weights.sum()

    global_dict = global_model.state_dict()
    client_dicts = [m.state_dict() for m in client_models]

    for k in global_dict.keys():
        if global_dict[k].dtype.is_floating_point:
            stacked = torch.stack(
                [client_dicts[i][k].float() * weights[i] for i in range(len(client_models))], 0
            )
            global_dict[k] = stacked.sum(0).to(global_dict[k].dtype)
        else:
            global_dict[k] = client_dicts[0][k]

    global_model.load_state_dict(global_dict)
    return global_model


def client_update(client_model, dataloader, epochs, lr=1e-3, weight_decay=1e-4,
                  hsic_weight=0.1, cls_weight=0.3, grad_clip=1.0, device='cpu'):
    """One client's local training pass (multitask: UPDRS regression + PD/HC
    classification). Weight decay and gradient clipping regularize each local
    update so no single site overfits its shard."""
    client_model.train()
    optimizer = optim.AdamW(client_model.parameters(), lr=lr, weight_decay=weight_decay)
    mse_criterion = torch.nn.MSELoss()
    bce_criterion = torch.nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            optimizer.zero_grad()
            out = client_model(batch)

            loss = (mse_criterion(out['pred'], batch['target'])
                    + out['loss_hsic'] * hsic_weight
                    + bce_criterion(out['cls_logit'], batch['diagnosis']) * cls_weight)
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(client_model.parameters(), grad_clip)
            optimizer.step()

    return client_model


def evaluate_model(model, dataloader, device='cpu', return_predictions=False,
                   hsic_weight=0.1, cls_weight=0.3,
                   return_classification=False):
    """
    Returns regression metrics {'ccc', 'rmse', 'mae', 'r2', 'pearson'} plus
    classification metrics {'auc', 'accuracy', 'f1'} when both PD and HC
    labels are present. With return_predictions=True, returns
    (metrics, preds, targets) so callers can bootstrap CIs / paired tests.
    """
    model.eval()
    preds_all, targets_all, logits_all, diag_all = [], [], [], []
    loss_total, sample_total = 0.0, 0
    mse_criterion = torch.nn.MSELoss()
    bce_criterion = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            out = model(batch)
            preds_all.append(out['pred'].cpu().numpy())
            targets_all.append(batch['target'].cpu().numpy())
            logits_all.append(out['cls_logit'].cpu().numpy())
            diag_all.append(batch['diagnosis'].cpu().numpy())
            batch_size = len(batch['target'])
            batch_loss = (mse_criterion(out['pred'], batch['target'])
                          + cls_weight * bce_criterion(out['cls_logit'], batch['diagnosis'])
                          + hsic_weight * out['loss_hsic'])
            loss_total += float(batch_loss.item()) * batch_size
            sample_total += batch_size

    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)
    logits_all = np.concatenate(logits_all)
    diag_all = np.concatenate(diag_all)

    metrics = {
        'loss': loss_total / max(sample_total, 1),
        'ccc': concordance_correlation_coefficient(targets_all, preds_all),
        'rmse': rmse(targets_all, preds_all),
        'mae': mae(targets_all, preds_all),
        'r2': r2_score(targets_all, preds_all),
        'pearson': pearson_r(targets_all, preds_all),
    }

    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 confusion_matrix, f1_score, precision_score,
                                 recall_score, roc_auc_score)
    probs = 1.0 / (1.0 + np.exp(-logits_all))
    class_preds = (probs >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(diag_all.astype(int), class_preds,
                                      labels=[0, 1]).ravel()
    metrics.update({
        'accuracy': float(accuracy_score(diag_all, class_preds)),
        'balanced_accuracy': float(balanced_accuracy_score(diag_all, class_preds)),
        'precision': float(precision_score(diag_all, class_preds, zero_division=0)),
        'recall': float(recall_score(diag_all, class_preds, zero_division=0)),
        'sensitivity': float(tp / max(tp + fn, 1)),
        'specificity': float(tn / max(tn + fp, 1)),
        'f1': float(f1_score(diag_all, class_preds, zero_division=0)),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'n_hc': int(np.sum(diag_all == 0)), 'n_pd': int(np.sum(diag_all == 1)),
    })
    if len(np.unique(diag_all)) > 1:
        metrics['auc'] = float(roc_auc_score(diag_all, probs))

    if return_predictions:
        if return_classification:
            probabilities = 1.0 / (1.0 + np.exp(-logits_all))
            return metrics, preds_all, targets_all, diag_all, probabilities
        return metrics, preds_all, targets_all
    return metrics


def simulate_federated_training(global_model, train_client_datasets, val_dataset,
                                num_rounds=30, local_epochs=2, lr=1e-3,
                                weight_decay=1e-4, hsic_weight=0.1, cls_weight=0.3,
                                grad_clip=1.0, batch_size=32, early_stopping_patience=5,
                                train_eval_dataset=None, device='cpu'):
    """
    FedAvg simulation with validation-based early stopping.

    IMPORTANT: the held-out TEST set must never be passed here. Round-by-round
    monitoring and model selection use `val_dataset` only; evaluate the returned
    model on the test set exactly once, after this function returns.

    Returns:
        (best_model, history) where history is a list of per-round dicts with
        train/val metrics and the train-val gap (over/underfitting diagnostic).
    """
    global_model.to(device)
    num_clients = len(train_client_datasets)
    client_sizes = [len(ds) for ds in train_client_datasets]
    train_dataloaders = [DataLoader(ds, batch_size=batch_size, shuffle=True)
                         for ds in train_client_datasets]
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    train_eval_loader = (DataLoader(train_eval_dataset, batch_size=64, shuffle=False)
                         if train_eval_dataset is not None else None)

    history = []
    best_val_ccc = -np.inf
    best_state = copy.deepcopy(global_model.state_dict())
    best_round = 0
    rounds_without_improvement = 0

    for round_idx in range(num_rounds):
        client_models = [copy.deepcopy(global_model).to(device) for _ in range(num_clients)]

        for i in range(num_clients):
            client_models[i] = client_update(
                client_models[i], train_dataloaders[i], local_epochs,
                lr=lr, weight_decay=weight_decay, hsic_weight=hsic_weight,
                cls_weight=cls_weight, grad_clip=grad_clip, device=device,
            )

        global_model = fedavg_aggregate(global_model, client_models, client_sizes)

        val_metrics = evaluate_model(global_model, val_dataloader, device,
                                     hsic_weight=hsic_weight, cls_weight=cls_weight)
        entry = {'round': round_idx + 1, 'val': val_metrics}

        # Client-local metrics are evaluated only on each client's own shard.
        # No client data or predictions leave the client; only scalar summaries
        # are retained in the diagnostic history.
        client_metrics = []
        for client_id, (client_model, loader, n_train) in enumerate(
                zip(client_models, train_dataloaders, client_sizes), start=1):
            local = evaluate_model(client_model, loader, device,
                                   hsic_weight=hsic_weight, cls_weight=cls_weight)
            client_metrics.append({
                'client_id': client_id, 'n_train': n_train,
                **{f'train_{key}': value for key, value in local.items()},
            })
        entry['clients'] = client_metrics

        if train_eval_loader is not None:
            train_metrics = evaluate_model(global_model, train_eval_loader, device,
                                           hsic_weight=hsic_weight, cls_weight=cls_weight)
            entry['train'] = train_metrics
            entry['ccc_gap'] = train_metrics['ccc'] - val_metrics['ccc']
            gap_str = (f" | Train CCC: {train_metrics['ccc']:.4f}"
                       f" | Gap: {entry['ccc_gap']:+.4f}")
        else:
            gap_str = ""

        auc_str = (f" | Val AUC: {val_metrics['auc']:.4f}"
                   if 'auc' in val_metrics else "")
        logger.info(f"Round {round_idx + 1}/{num_rounds} | "
                    f"Val CCC: {val_metrics['ccc']:.4f} | "
                    f"Val RMSE: {val_metrics['rmse']:.3f}{auc_str}{gap_str}")
        history.append(entry)

        if val_metrics['ccc'] > best_val_ccc:
            best_val_ccc = val_metrics['ccc']
            best_state = copy.deepcopy(global_model.state_dict())
            best_round = round_idx + 1
            rounds_without_improvement = 0
        else:
            rounds_without_improvement += 1
            if rounds_without_improvement >= early_stopping_patience:
                logger.info(f"Early stopping at round {round_idx + 1} "
                            f"(no val-CCC improvement for {early_stopping_patience} rounds). "
                            f"Restoring best weights from round {best_round}.")
                break

    global_model.load_state_dict(best_state)
    logger.info(f"Federated training done. Best Val CCC: {best_val_ccc:.4f} "
                f"(round {best_round}).")
    return global_model, history
