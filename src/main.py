# -*- coding: utf-8 -*-
"""
Fed-PhenoGraft: End-to-End Pipeline
Orchestrates data loading, baseline comparison, federated training, and XAI analysis.
"""
import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import yaml
import torch
import numpy as np
import pandas as pd
import logging

from src.data.data_builder import build_real_dataset
from src.data.dataset import FederatedPPMIDataset, create_federated_splits
from src.models.fed_phenograft import FedPhenoGraft
from src.federated.fedavg_orchestrator import simulate_federated_training
from src.evaluation.xai import extract_attention_weights, visualize_attention
from src.baselines.runner import run_baselines

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")
logger = logging.getLogger("FedPhenoGraft")


def load_config():
    config_path = project_root / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_pipeline():
    logger.info("=" * 70)
    logger.info("  Fed-PhenoGraft: Phenotype-Guided Multimodal PD Prediction")
    logger.info("=" * 70)

    config = load_config()
    seed = config.get("seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ── Phase 1: Data Loading ────────────────────────────────────────
    logger.info("\n[Phase 1] Loading & Preprocessing Data...")
    clin, mri, pet, gen, targets, diagnosis = build_real_dataset(config)

    # ── Phase 2: Baseline Evaluation ────────────────────────────────
    logger.info("\n[Phase 2] Running Baseline Models (Symmetric Fusion)...")
    flat_features = np.hstack([
        clin.values, mri.values, pet.values, gen.values
    ])
    target_array = targets.values if hasattr(targets, 'values') else targets
    run_baselines(flat_features, target_array, n_splits=config.get("cv", {}).get("n_splits", 5))

    # ── Phase 3: Build PyTorch Datasets ─────────────────────────────
    logger.info("\n[Phase 3] Constructing PyTorch Datasets...")
    dataset = FederatedPPMIDataset(clin, mri, pet, gen, targets)

    # 80/20 train/test split
    indices = np.random.permutation(len(dataset))
    split_point = int(0.8 * len(dataset))
    train_idx, test_idx = indices[:split_point], indices[split_point:]

    def get_subset(ds, idx):
        return FederatedPPMIDataset(
            ds.clinical[idx],
            ds.mri[idx],
            ds.pet[idx],
            ds.genetic[idx],
            ds.targets[idx]
        )

    train_ds = get_subset(dataset, train_idx)
    test_ds = get_subset(dataset, test_idx)

    # ── Phase 4: Federated Training ─────────────────────────────────
    logger.info("\n[Phase 4] Federated Training (FedAvg)...")
    client_datasets = create_federated_splits(train_ds, num_clients=4, seed=seed)

    input_dims = {
        'clinical': clin.shape[1],
        'mri': mri.shape[1],
        'pet': pet.shape[1],
        'genetic': gen.shape[1]
    }

    model = FedPhenoGraft(input_dims, embed_dim=32, num_heads=4, dropout=0.2)
    model, history = simulate_federated_training(
        model, client_datasets, test_ds,
        num_rounds=config.get("cv", {}).get("n_splits", 5),
        local_epochs=3
    )

    # ── Phase 5: Explainability ─────────────────────────────────────
    logger.info("\n[Phase 5] Generating Explainability Outputs...")
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    attn_weights = extract_attention_weights(model, test_loader)

    fig_dir = project_root / config["paths"]["figures"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    visualize_attention(attn_weights, str(fig_dir / "attention_maps.png"))

    # Save model
    model_dir = project_root / config["paths"]["models"]
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(model_dir / "fed_phenograft_final.pt"))

    logger.info("\n" + "=" * 70)
    logger.info("  Pipeline Complete!")
    logger.info(f"  Attention maps → {fig_dir / 'attention_maps.png'}")
    logger.info(f"  Model weights  → {model_dir / 'fed_phenograft_final.pt'}")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_pipeline()
