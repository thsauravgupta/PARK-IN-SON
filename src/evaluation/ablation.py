# -*- coding: utf-8 -*-
"""
Ablation suite for Fed-PhenoGraft.

Each variant retrains the model from scratch under the SAME leak-free
protocol (train clients → validation early stopping → one-shot test), so the
resulting table shows what every architectural component and modality
contributes:

  - no_mri / no_pet / no_genetic : the modality is unavailable everywhere
    (all-zero features → learned mask tokens engage), isolating its value.
  - clinical_only                : all three auxiliary modalities removed.
  - no_attention                 : asymmetric cross-attention replaced by
    plain concatenation of shared embeddings.
  - no_hsic                      : shared-private orthogonality loss off.
  - centralized                  : single client — upper bound showing the
    cost of federation.
"""

import copy
import logging

import numpy as np
from torch.utils.data import DataLoader

from src.data.dataset import FederatedPPMIDataset, create_federated_splits
from src.federated.fedavg_orchestrator import evaluate_model, simulate_federated_training
from src.models.fed_phenograft import FedPhenoGraft
from src.utils import seed_everything

logger = logging.getLogger(__name__)

MODALITIES = ("mri", "pet", "genetic")

VARIANTS = {
    "no_mri": {"drop": ["mri"]},
    "no_pet": {"drop": ["pet"]},
    "no_genetic": {"drop": ["genetic"]},
    "clinical_only": {"drop": ["mri", "pet", "genetic"]},
    "no_attention": {"use_attention": False},
    "no_hsic": {"hsic_weight": 0.0},
    "centralized": {"num_clients": 1},
}


def _zero_modality(ds: FederatedPPMIDataset, drop_names) -> FederatedPPMIDataset:
    """Copy of the dataset with the named modalities zeroed out. All-zero rows
    are detected by FederatedPPMIDataset and served with mask=1, so the model's
    learned mask tokens take over — i.e., 'this modality was never collected'."""
    arrays = {name: getattr(ds, name).copy() for name in
              ("clinical", "mri", "pet", "genetic")}
    for name in drop_names:
        arrays[name] = np.zeros_like(arrays[name])
    return FederatedPPMIDataset(
        arrays["clinical"], arrays["mri"], arrays["pet"], arrays["genetic"],
        ds.targets.copy(), diagnosis=ds.diagnosis.copy(), client_id=ds.client_id,
    )


def run_ablation_suite(train_ds, val_ds, test_ds, input_dims, config,
                       site_labels=None):
    """
    Trains every ablation variant and returns
    {variant: {"val": metrics, "test": metrics}}.

    Reuses the pipeline's training hyperparameters but with the (shorter)
    round budget from config['ablation'].
    """
    train_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    abl_cfg = config.get("ablation", {})
    seed = config.get("seed", 42)

    num_rounds = abl_cfg.get("num_rounds", 20)
    patience = abl_cfg.get("early_stopping_patience", 3)

    results = {}
    for name, spec in VARIANTS.items():
        logger.info(f"[Ablation] {name} ...")
        seed_everything(seed)

        drop = spec.get("drop", [])
        tr = _zero_modality(train_ds, drop) if drop else train_ds
        va = _zero_modality(val_ds, drop) if drop else val_ds
        te = _zero_modality(test_ds, drop) if drop else test_ds

        num_clients = spec.get("num_clients", train_cfg.get("num_clients", 4))
        partition = "iid" if num_clients == 1 else train_cfg.get("partition", "iid")
        clients = create_federated_splits(
            tr, num_clients=num_clients, seed=seed, partition=partition,
            dirichlet_alpha=train_cfg.get("dirichlet_alpha", 0.5),
            site_labels=site_labels,
        )

        model = FedPhenoGraft(
            input_dims,
            embed_dim=model_cfg.get("embed_dim", 32),
            num_heads=model_cfg.get("num_heads", 4),
            dropout=model_cfg.get("dropout", 0.3),
            use_attention=spec.get("use_attention", True),
        )

        model, _ = simulate_federated_training(
            model, clients, va,
            num_rounds=num_rounds,
            local_epochs=train_cfg.get("local_epochs", 2),
            lr=train_cfg.get("lr", 1e-3),
            weight_decay=train_cfg.get("weight_decay", 1e-4),
            hsic_weight=spec.get("hsic_weight", train_cfg.get("hsic_weight", 0.1)),
            cls_weight=train_cfg.get("cls_weight", 0.3),
            grad_clip=train_cfg.get("grad_clip", 1.0),
            batch_size=train_cfg.get("batch_size", 32),
            early_stopping_patience=patience,
        )

        val_metrics = evaluate_model(model, DataLoader(va, batch_size=64, shuffle=False))
        test_metrics = evaluate_model(model, DataLoader(te, batch_size=64, shuffle=False))
        results[name] = {"val": val_metrics, "test": test_metrics}
        logger.info(f"[Ablation] {name}: val CCC {val_metrics['ccc']:.4f} | "
                    f"test CCC {test_metrics['ccc']:.4f}")

    return results
