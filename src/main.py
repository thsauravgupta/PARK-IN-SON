# -*- coding: utf-8 -*-
"""
Fed-PhenoGraft: End-to-End Pipeline (leak-free evaluation protocol).

Protocol:
  1. Load raw data (no global scaling — see src/data/preprocessing.py).
  2. Subject-level stratified train/val/test split (70/15/15 by default).
  3. Fit imputer/scalers on TRAIN subjects only; transform val/test with them.
  4. Baselines: per-fold pipelines (CV on train+val), scored once on test.
  5. Federated training: val set drives early stopping + best-model selection.
  6. The TEST set is evaluated exactly once, after training completes.
"""
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import yaml
import torch
import numpy as np
import pandas as pd
import logging

from src.utils import seed_everything
from src.data.data_builder import build_real_dataset
from src.data.dataset import (
    FederatedPPMIDataset, create_federated_splits, load_site_labels,
)
from src.data.preprocessing import ModalityPreprocessor, create_subject_splits
from src.models.fed_phenograft import FedPhenoGraft
from src.federated.fedavg_orchestrator import simulate_federated_training, evaluate_model
from src.evaluation.xai import (
    extract_attention_weights, visualize_attention, plot_pred_vs_actual,
    plot_confusion_matrix, stress_test_missing_modalities,
    visualize_feature_importance, counterfactual_gene_analysis,
)
from src.evaluation.stats import (
    bootstrap_metric_ci, paired_bootstrap_test, summarize_seed_runs,
)
from src.evaluation.ablation import run_ablation_suite
from src.evaluation.results_report import generate_report
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
    seed_everything(seed)

    split_cfg = config.get("split", {})
    train_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    eval_cfg = config.get("evaluation", {})
    target_mode = config.get("target", {}).get("mode", "absolute")
    target_label = ("Δ UPDRS-III (BL → Year 2)" if target_mode == "delta"
                    else "UPDRS-III @ Year 2")
    logger.info(f"Regression target: {target_label} (mode='{target_mode}')")

    # ── Phase 1: Load raw data (unscaled) ───────────────────────────
    logger.info("\n[Phase 1] Loading Raw Data (no global scaling)...")
    clin, mri, pet, gen, targets, diagnosis = build_real_dataset(config)
    y = np.asarray(targets.values if hasattr(targets, 'values') else targets,
                   dtype=np.float64).ravel()
    diag = np.nan_to_num(np.asarray(
        diagnosis.values if hasattr(diagnosis, 'values') else diagnosis,
        dtype=np.float64
    ).ravel())

    # ── Phase 2: Subject-level stratified split ─────────────────────
    logger.info("\n[Phase 2] Subject-Level Train/Val/Test Split...")
    train_idx, val_idx, test_idx = create_subject_splits(
        diagnosis,
        val_fraction=split_cfg.get("val_fraction", 0.15),
        test_fraction=split_cfg.get("test_fraction", 0.15),
        seed=seed,
        stratify=split_cfg.get("stratify", True),
    )

    # ── Phase 3: Fit preprocessing on TRAIN only ────────────────────
    logger.info("\n[Phase 3] Fitting Preprocessing on Training Subjects Only...")
    prep = ModalityPreprocessor().fit(
        clin.iloc[train_idx], mri.iloc[train_idx],
        pet.iloc[train_idx], gen.iloc[train_idx],
    )

    def transform_split(idx):
        c, m, p, g = prep.transform(
            clin.iloc[idx], mri.iloc[idx], pet.iloc[idx], gen.iloc[idx]
        )
        return FederatedPPMIDataset(c, m, p, g, pd.Series(y[idx], index=c.index),
                                    diagnosis=diag[idx])

    train_ds = transform_split(train_idx)
    val_ds = transform_split(val_idx)
    test_ds = transform_split(test_idx)

    # ── Phase 4: Baselines (leak-free CV + one-shot test) ───────────
    logger.info("\n[Phase 4] Baseline Models (per-fold pipelines)...")
    raw_features = np.hstack([clin.values, mri.values, pet.values, gen.values])
    trainval_idx = np.sort(np.concatenate([train_idx, val_idx]))
    baseline_results = run_baselines(
        raw_features[trainval_idx], y[trainval_idx],
        raw_features[test_idx], y[test_idx],
        n_splits=config.get("cv", {}).get("n_splits", 5),
        seed=seed,
    )

    # ── Phase 5: Federated training (multi-seed, non-IID clients) ───
    partition = train_cfg.get("partition", "iid")
    num_seeds = max(1, int(eval_cfg.get("num_seeds", 3)))
    logger.info(f"\n[Phase 5] Federated Training (FedAvg, partition='{partition}', "
                f"{num_seeds} seeds)...")

    from torch.utils.data import DataLoader

    # Real acquisition sites when a Center-Subject list CSV is available;
    # otherwise create_federated_splits falls back to Dirichlet label-skew.
    site_series = load_site_labels(project_root / config["paths"]["raw"])
    train_site_labels = None
    if site_series is not None:
        train_site_labels = site_series.reindex(clin.index).values[train_idx]
        logger.info("Site labels found — 'site' partition available.")
    elif partition == "site":
        logger.warning("partition='site' requested but no Center-Subject list "
                       "CSV found in data/raw — falling back to Dirichlet "
                       "label-skew partitioning.")

    input_dims = {
        'clinical': train_ds.clinical.shape[1],
        'mri': train_ds.mri.shape[1],
        'pet': train_ds.pet.shape[1],
        'genetic': train_ds.genetic.shape[1],
    }

    def _new_model():
        return FedPhenoGraft(
            input_dims,
            embed_dim=model_cfg.get("embed_dim", 32),
            num_heads=model_cfg.get("num_heads", 4),
            dropout=model_cfg.get("dropout", 0.3),
        )

    def _train_once(run_seed):
        seed_everything(run_seed)
        client_datasets = create_federated_splits(
            train_ds, num_clients=train_cfg.get("num_clients", 4),
            seed=run_seed, partition=partition,
            dirichlet_alpha=train_cfg.get("dirichlet_alpha", 0.5),
            site_labels=train_site_labels,
        )
        logger.info(f"  Client sizes (seed {run_seed}): "
                    f"{[len(ds) for ds in client_datasets]}")
        return simulate_federated_training(
            _new_model(), client_datasets, val_ds,
            num_rounds=train_cfg.get("num_rounds", 30),
            local_epochs=train_cfg.get("local_epochs", 2),
            lr=train_cfg.get("lr", 1e-3),
            weight_decay=train_cfg.get("weight_decay", 1e-4),
            hsic_weight=train_cfg.get("hsic_weight", 0.1),
            cls_weight=train_cfg.get("cls_weight", 0.3),
            grad_clip=train_cfg.get("grad_clip", 1.0),
            batch_size=train_cfg.get("batch_size", 32),
            early_stopping_patience=train_cfg.get("early_stopping_patience", 5),
            train_eval_dataset=train_ds,
        )

    # Model selection across seeds uses the VALIDATION set only; the test set
    # is scored per finished seed solely for the mean ± std report and never
    # influences training or selection.
    test_loader_eval = DataLoader(test_ds, batch_size=64, shuffle=False)
    val_loader_eval = DataLoader(val_ds, batch_size=64, shuffle=False)
    train_loader_eval = DataLoader(train_ds, batch_size=64, shuffle=False)

    per_seed_test, per_seed_val = [], []
    model, history, best_seed_val_ccc, best_seed = None, None, -np.inf, seed
    for s in range(num_seeds):
        run_seed = seed + s * 101
        logger.info(f"--- Seed run {s + 1}/{num_seeds} (seed={run_seed}) ---")
        seed_model, seed_history = _train_once(run_seed)
        v = evaluate_model(seed_model, val_loader_eval)
        t = evaluate_model(seed_model, test_loader_eval)
        per_seed_val.append(v)
        per_seed_test.append(t)
        if v['ccc'] > best_seed_val_ccc:
            best_seed_val_ccc = v['ccc']
            model, history, best_seed = seed_model, seed_history, run_seed

    seed_summary = {
        "num_seeds": num_seeds,
        "selected_seed": best_seed,
        "val": summarize_seed_runs(per_seed_val),
        "test": summarize_seed_runs(per_seed_test),
    }
    if num_seeds > 1:
        tccc = seed_summary["test"]["ccc"]
        logger.info(f"  Across {num_seeds} seeds — test CCC "
                    f"{tccc['mean']:.4f} ± {tccc['std']:.4f} "
                    f"(primary model = best-val seed {best_seed})")

    # ── Phase 6: One-shot test evaluation + statistics ──────────────
    logger.info("\n[Phase 6] Final Held-Out Test Evaluation + Statistics...")
    test_metrics, fed_test_preds, test_targets = evaluate_model(
        model, test_loader_eval, return_predictions=True)
    train_metrics = evaluate_model(model, train_loader_eval)
    val_metrics = evaluate_model(model, val_loader_eval)

    n_boot = int(eval_cfg.get("n_bootstrap", 1000))
    fed_test_ci = {m: bootstrap_metric_ci(test_targets, fed_test_preds,
                                          metric=m, n_boot=n_boot, seed=seed)
                   for m in ("ccc", "rmse", "mae", "r2")}
    ci = fed_test_ci["ccc"]
    logger.info(f"  Fed-PhenoGraft test CCC {ci['point']:.4f} "
                f"[95% CI {ci['ci_low']:.4f}, {ci['ci_high']:.4f}]")

    # Paired bootstrap significance vs the strongest baseline
    significance = None
    baselines_with_preds = {k: v for k, v in baseline_results.items()
                            if v.get("test_predictions")}
    if baselines_with_preds:
        best_bl = max(baselines_with_preds,
                      key=lambda k: baselines_with_preds[k]["test_ccc"])
        significance = paired_bootstrap_test(
            test_targets, fed_test_preds,
            np.asarray(baselines_with_preds[best_bl]["test_predictions"]),
            metric="ccc", n_boot=n_boot, seed=seed)
        significance["compared_against"] = best_bl
        logger.info(f"  Paired bootstrap vs best baseline ({best_bl}): "
                    f"ΔCCC {significance['delta']:+.4f}, "
                    f"p = {significance['p_value']:.4f} "
                    f"({'significant' if significance['significant_at_0.05'] else 'not significant'} at α=0.05)")

    gap = train_metrics['ccc'] - val_metrics['ccc']
    logger.info(f"  Train CCC: {train_metrics['ccc']:.4f} | "
                f"Val CCC: {val_metrics['ccc']:.4f} | "
                f"Test CCC: {test_metrics['ccc']:.4f}")
    logger.info(f"  Test RMSE: {test_metrics['rmse']:.3f} | Test MAE: {test_metrics['mae']:.3f}")
    if 'auc' in test_metrics:
        logger.info(f"  PD vs HC — Test AUC: {test_metrics['auc']:.4f} | "
                    f"Accuracy: {test_metrics['accuracy']:.4f}")
    if gap > 0.15:
        logger.warning(f"  Train-val CCC gap = {gap:+.3f} → OVERFITTING signal. "
                       f"Increase dropout/weight_decay or reduce local_epochs.")
    elif train_metrics['ccc'] < 0.2:
        logger.warning(f"  Train CCC = {train_metrics['ccc']:.3f} → UNDERFITTING signal. "
                       f"Increase capacity (embed_dim), rounds, or lr.")
    else:
        logger.info(f"  Train-val gap = {gap:+.3f} → no over/underfitting red flags.")

    # ── Phase 7: Ablation suite ─────────────────────────────────────
    ablation_results = None
    if config.get("ablation", {}).get("enabled", False):
        logger.info("\n[Phase 7] Ablation Suite (each variant retrained)...")
        ablation_results = run_ablation_suite(
            train_ds, val_ds, test_ds, input_dims, config,
            site_labels=train_site_labels,
        )
        ablation_results["full"] = {"val": val_metrics, "test": test_metrics}

    # ── Phase 8: Explainability & presentation figures ──────────────
    logger.info("\n[Phase 8] Generating Explainability & Presentation Figures...")
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    fig_dir = project_root / config["paths"]["figures"]
    fig_dir.mkdir(parents=True, exist_ok=True)

    attn_weights = extract_attention_weights(model, test_loader)
    visualize_attention(attn_weights, str(fig_dir / "attention_maps.png"))
    plot_pred_vs_actual(model, test_loader,
                        save_path=str(fig_dir / "pred_vs_actual.png"),
                        target_label=target_label)
    plot_confusion_matrix(model, test_loader,
                          save_path=str(fig_dir / "confusion_matrix.png"))
    stress_test_missing_modalities(model, test_loader,
                                   save_path=str(fig_dir / "modality_robustness.png"))
    visualize_feature_importance(model, test_loader,
                                 save_path=str(fig_dir / "global_feature_importance.png"))
    counterfactual_deltas = counterfactual_gene_analysis(
        model, prep, clin.iloc[test_idx], mri.iloc[test_idx],
        pet.iloc[test_idx], gen.iloc[test_idx],
        save_path=str(fig_dir / "counterfactual_genes.png"),
        target_label=target_label,
    )
    if counterfactual_deltas:
        logger.info(f"  Counterfactual gene shifts ({target_label}): {counterfactual_deltas}")

    # ── Save artifacts ──────────────────────────────────────────────
    model_dir = project_root / config["paths"]["models"]
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(model_dir / "fed_phenograft_best.pt"))

    results_dir = project_root / config["paths"]["results"]
    results_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "target": {"mode": target_mode, "label": target_label},
        "split_sizes": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "federated_partition": partition,
        "fed_phenograft": {
            "train": train_metrics, "val": val_metrics, "test": test_metrics,
            "train_val_ccc_gap": gap,
            "rounds_history": history,
        },
        "statistics": {
            "n_bootstrap": n_boot,
            "test_ci_95": fed_test_ci,
            "significance_vs_best_baseline": significance,
            "seed_runs": seed_summary,
        },
        "ablations": ablation_results,
        "baselines": baseline_results,
        "counterfactual_gene_deltas": counterfactual_deltas,
        "config": {"training": train_cfg, "model": model_cfg, "split": split_cfg,
                   "evaluation": eval_cfg, "seed": seed},
    }
    with open(results_dir / "final_metrics.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    # Presentation report: comparison chart, training curve, RESULTS.md
    generate_report(results_dir, fig_dir)

    logger.info("\n" + "=" * 70)
    logger.info("  Pipeline Complete!")
    logger.info(f"  Metrics summary → {results_dir / 'final_metrics.json'}")
    logger.info(f"  Results report  → {results_dir / 'RESULTS.md'}")
    logger.info(f"  Figures         → {fig_dir}")
    logger.info(f"  Model weights   → {model_dir / 'fed_phenograft_best.pt'}")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_pipeline()
