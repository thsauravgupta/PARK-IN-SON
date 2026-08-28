# -*- coding: utf-8 -*-
"""
Presentation-ready results generation for Fed-PhenoGraft.

Turns outputs/results/final_metrics.json into the figures and tables promised
in the project presentation: model comparison chart, federated training curve,
and a Markdown results summary. Model-dependent figures (attention maps,
pred-vs-actual, confusion matrix, robustness, counterfactuals) are produced by
src/main.py during the pipeline run; this module covers everything derivable
from the metrics JSON alone so results can be regenerated without retraining.
"""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def plot_model_comparison(summary: dict, save_path: Path):
    """Bar chart: held-out test CCC of every baseline vs Fed-PhenoGraft."""
    names, values = [], []
    for name, r in summary.get("baselines", {}).items():
        if r.get("test_ccc") is not None:
            names.append(name)
            values.append(r["test_ccc"])
    fed_test = summary["fed_phenograft"]["test"]["ccc"]
    names.append("Fed-PhenoGraft")
    values.append(fed_test)

    target_label = summary.get("target", {}).get("label", "UPDRS-III")
    colors = ['#95a5a6'] * (len(names) - 1) + ['#8e44ad']
    plt.figure(figsize=(max(9, 1.1 * len(names)), 5.5))
    bars = plt.bar(names, values, color=colors)
    plt.ylabel("Held-out Test CCC")
    plt.title(f"{target_label} Prediction — Test CCC "
              f"({len(names) - 1} baselines vs Fed-PhenoGraft)")
    plt.ylim(0, max(values) * 1.2)
    plt.xticks(rotation=30, ha='right')
    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2.0, v + 0.01, f'{v:.3f}',
                 va='bottom', ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_training_curve(summary: dict, save_path: Path):
    """Train/val CCC per federated round, with the best (restored) round marked."""
    history = summary["fed_phenograft"].get("rounds_history", [])
    if not history:
        return
    rounds = [h["round"] for h in history]
    val_ccc = [h["val"]["ccc"] for h in history]
    train_ccc = [h["train"]["ccc"] for h in history if "train" in h]

    best_i = max(range(len(val_ccc)), key=lambda i: val_ccc[i])

    plt.figure(figsize=(9, 5))
    plt.plot(rounds, val_ccc, marker='o', label="Validation CCC", color='#2980b9')
    if len(train_ccc) == len(rounds):
        plt.plot(rounds, train_ccc, marker='s', label="Train CCC", color='#7f8c8d',
                 linestyle='--')
    plt.axvline(rounds[best_i], color='#27ae60', linestyle=':',
                label=f"Best round ({rounds[best_i]}) — weights restored")
    plt.xlabel("Federated Round (FedAvg)")
    plt.ylabel("CCC")
    plt.title("Federated Training with Validation-Based Early Stopping")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_ablation_chart(summary: dict, save_path: Path):
    """Bar chart: held-out test CCC of the full model vs every ablation."""
    ablations = summary.get("ablations")
    if not ablations:
        return
    order = ["full", "no_attention", "no_hsic", "centralized",
             "no_mri", "no_pet", "no_genetic", "clinical_only"]
    names = [n for n in order if n in ablations] + \
            [n for n in ablations if n not in order]
    values = [ablations[n]["test"]["ccc"] for n in names]

    colors = ['#8e44ad' if n == "full" else '#95a5a6' for n in names]
    plt.figure(figsize=(max(9, 1.1 * len(names)), 5.5))
    bars = plt.bar(names, values, color=colors)
    plt.ylabel("Held-out Test CCC")
    plt.title("Ablation Study — Contribution of Each Component / Modality")
    plt.ylim(min(0, min(values) * 1.2), max(values) * 1.25)
    plt.xticks(rotation=30, ha='right')
    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2.0, v + 0.005, f'{v:.3f}',
                 va='bottom', ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def _fmt(v, digits=4):
    return f"{v:.{digits}f}" if isinstance(v, (int, float)) else "—"


def write_results_markdown(summary: dict, save_path: Path):
    """Markdown results summary suitable for pasting into the report/slides."""
    fed = summary["fed_phenograft"]
    test, val, train = fed["test"], fed["val"], fed["train"]
    sizes = summary["split_sizes"]
    target_label = summary.get("target", {}).get("label", "UPDRS-III @ Year 2")
    partition = summary.get("federated_partition", "iid")
    stats = summary.get("statistics") or {}
    ci = (stats.get("test_ci_95") or {}).get("ccc")

    ccc_str = f"**{_fmt(test['ccc'])}**"
    if ci:
        ccc_str += f" [95% CI {_fmt(ci['ci_low'])}, {_fmt(ci['ci_high'])}]"

    lines = [
        "# Fed-PhenoGraft — Results Summary",
        "",
        f"Regression target: **{target_label}**. Federated client partition: "
        f"**{partition}**.",
        "",
        f"Subject-level split: **{sizes['train']} train / {sizes['val']} val / "
        f"{sizes['test']} test** (stratified on diagnosis; test evaluated once).",
        "",
        "## Fed-PhenoGraft (held-out test)",
        "",
        "| Task | Metric | Value |",
        "|------|--------|-------|",
        f"| Progression regression | CCC | {ccc_str} |",
        f"| Progression regression | RMSE (UPDRS-III pts) | {_fmt(test['rmse'], 2)} |",
        f"| Progression regression | MAE (UPDRS-III pts) | {_fmt(test['mae'], 2)} |",
        f"| Progression regression | R² | {_fmt(test.get('r2'))} |",
        f"| Progression regression | Pearson r | {_fmt(test.get('pearson'))} |",
    ]
    if "auc" in test:
        lines += [
            f"| PD vs HC classification | ROC-AUC | **{_fmt(test['auc'])}** |",
            f"| PD vs HC classification | Accuracy | {_fmt(test['accuracy'])} |",
            f"| PD vs HC classification | F1 | {_fmt(test.get('f1'))} |",
        ]
    lines += [
        "",
        f"Generalization check: train CCC {_fmt(train['ccc'])} vs val CCC "
        f"{_fmt(val['ccc'])} (gap {fed['train_val_ccc_gap']:+.3f}) — "
        + ("no overfitting signal." if fed['train_val_ccc_gap'] < 0.15
           else "**overfitting signal — retune.**"),
        "",
        f"## Baseline comparison — {len(summary.get('baselines', {}))} models "
        "(same held-out test set)",
        "",
        "| Model | CV CCC (mean ± std) | Test CCC | Test RMSE | Test MAE | Test R² | Test Pearson |",
        "|-------|---------------------|----------|-----------|----------|---------|--------------|",
    ]
    for name, r in summary.get("baselines", {}).items():
        lines.append(
            f"| {name} | {_fmt(r['cv_mean'])} ± {_fmt(r['cv_std'])} "
            f"| {_fmt(r.get('test_ccc'))} | {_fmt(r.get('test_rmse'), 2)} "
            f"| {_fmt(r.get('test_mae'), 2)} | {_fmt(r.get('test_r2'))} "
            f"| {_fmt(r.get('test_pearson'))} |"
        )
    lines += [
        f"| **Fed-PhenoGraft** | val {_fmt(val['ccc'])} (early-stopped) "
        f"| **{_fmt(test['ccc'])}** | {_fmt(test['rmse'], 2)} "
        f"| {_fmt(test['mae'], 2)} | {_fmt(test.get('r2'))} "
        f"| {_fmt(test.get('pearson'))} |",
        "",
    ]

    # ── Statistical rigor ───────────────────────────────────────────
    sig = stats.get("significance_vs_best_baseline")
    seed_runs = stats.get("seed_runs") or {}
    if ci or sig or seed_runs.get("num_seeds", 1) > 1:
        lines += ["## Statistical analysis", ""]
    if ci:
        lines.append(
            f"- Bootstrap 95% CI (n={stats.get('n_bootstrap', 1000)} resamples) "
            f"on test CCC: **[{_fmt(ci['ci_low'])}, {_fmt(ci['ci_high'])}]**.")
    if sig:
        verdict = ("**statistically significant**" if sig.get("significant_at_0.05")
                   else "not statistically significant")
        lines.append(
            f"- Paired bootstrap vs the strongest baseline "
            f"(**{sig.get('compared_against')}**): ΔCCC {sig['delta']:+.4f} "
            f"[95% CI {_fmt(sig['ci_low'])}, {_fmt(sig['ci_high'])}], "
            f"p = {_fmt(sig['p_value'])} — {verdict} at α = 0.05.")
    if seed_runs.get("num_seeds", 1) > 1:
        t = seed_runs.get("test", {}).get("ccc")
        if t:
            lines.append(
                f"- Across **{seed_runs['num_seeds']} independent training seeds**: "
                f"test CCC {_fmt(t['mean'])} ± {_fmt(t['std'])} "
                f"(primary model = best-validation seed; test never used for selection).")
    if ci or sig or seed_runs.get("num_seeds", 1) > 1:
        lines.append("")

    # ── Ablations ───────────────────────────────────────────────────
    ablations = summary.get("ablations")
    if ablations:
        pretty = {
            "full": "Full Fed-PhenoGraft", "no_attention": "− Asymmetric attention",
            "no_hsic": "− HSIC shared-private loss", "centralized": "Centralized (1 client)",
            "no_mri": "− MRI modality", "no_pet": "− PET/DaTScan modality",
            "no_genetic": "− Genetics modality", "clinical_only": "Clinical only",
        }
        order = ["full", "no_attention", "no_hsic", "centralized",
                 "no_mri", "no_pet", "no_genetic", "clinical_only"]
        lines += [
            "## Ablation study (each variant retrained, same protocol)",
            "",
            "| Variant | Val CCC | Test CCC | Test RMSE | Test MAE |",
            "|---------|---------|----------|-----------|----------|",
        ]
        for key in [k for k in order if k in ablations] + \
                   [k for k in ablations if k not in order]:
            a = ablations[key]
            name = pretty.get(key, key)
            bold = "**" if key == "full" else ""
            lines.append(
                f"| {bold}{name}{bold} | {_fmt(a['val']['ccc'])} "
                f"| {bold}{_fmt(a['test']['ccc'])}{bold} "
                f"| {_fmt(a['test']['rmse'], 2)} | {_fmt(a['test']['mae'], 2)} |")
        lines.append("")

    lines += [
        "## Figures",
        "",
        "| Figure | File |",
        "|--------|------|",
        "| Model comparison | `outputs/figures/model_comparison.png` |",
        "| Training curve | `outputs/figures/training_curve.png` |",
        "| Predicted vs actual | `outputs/figures/pred_vs_actual.png` |",
        "| Confusion matrix (PD vs HC) | `outputs/figures/confusion_matrix.png` |",
        "| Attention maps | `outputs/figures/attention_maps.png` |",
        "| Missing-modality robustness | `outputs/figures/modality_robustness.png` |",
        "| Feature attribution (IG) | `outputs/figures/global_feature_importance.png` |",
        "| Counterfactual gene analysis | `outputs/figures/counterfactual_genes.png` |",
    ]
    if summary.get("ablations"):
        lines.append("| Ablation study | `outputs/figures/ablation_study.png` |")
    lines.append("")
    save_path.write_text("\n".join(lines), encoding="utf-8")


def generate_report(results_dir: Path, figures_dir: Path):
    """Regenerates all JSON-derived figures + the Markdown summary."""
    metrics_path = Path(results_dir) / "final_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"{metrics_path} not found — run `python src/main.py` first."
        )
    with open(metrics_path) as f:
        summary = json.load(f)

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_model_comparison(summary, figures_dir / "model_comparison.png")
    plot_training_curve(summary, figures_dir / "training_curve.png")
    plot_ablation_chart(summary, figures_dir / "ablation_study.png")
    write_results_markdown(summary, Path(results_dir) / "RESULTS.md")
    logger.info(f"Results report written: {Path(results_dir) / 'RESULTS.md'}")
    return summary
