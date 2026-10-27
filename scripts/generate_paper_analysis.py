#!/usr/bin/env python3
"""Build paper-ready tables, plots, and an evidence/limitations index.

This is a post-processing script: it reads saved results and never trains a
model. Optional CSV inputs enable client-level, class-wise, FL-algorithm, and
activation/optimizer analyses that are not present in the current results JSON.

Usage (from repository root):
    python scripts/generate_paper_analysis.py
    python scripts/generate_paper_analysis.py --client-history path.csv \
        --predictions heldout_predictions.csv --fl-comparison fl_runs.csv

CSV schemas:
  predictions: y_true, y_pred, y_true_class, y_pred_class (or y_prob)
  client-history: round, client_id, accuracy, loss
  fl-comparison: method, metric, value (one row per method/metric)
  activation-study: variant, metric, value (include activation/optimizer in variant)
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    columns = list(dict.fromkeys(k for row in rows for k in row))
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def metric_table(summary: dict) -> list[dict]:
    rows = []
    fed = summary.get("fed_phenograft", {})
    for metric in ("loss", "ccc", "rmse", "mae", "r2", "pearson", "auc",
                   "accuracy", "balanced_accuracy", "precision", "recall",
                   "sensitivity", "specificity", "f1"):
        if metric in fed.get("test", {}):
            rows.append({"model": "Fed-PhenoGraft", "metric": metric,
                         "test": fed["test"][metric], "cv_mean": None, "cv_std": None})
    for name, result in summary.get("baselines", {}).items():
        for metric in ("ccc", "rmse", "mae", "r2", "pearson", "auc", "accuracy", "f1"):
            key = "test_" + metric
            if key in result:
                rows.append({"model": name, "metric": metric, "test": result[key],
                             "cv_mean": result.get("cv_mean") if metric == "ccc" else None,
                             "cv_std": result.get("cv_std") if metric == "ccc" else None})
    return rows


def ablation_rows(summary: dict) -> list[dict]:
    rows = []
    for variant, scores in summary.get("ablations", {}).items():
        for metric, value in scores.get("test", {}).items():
            rows.append({"variant": variant, "metric": metric, "test": value,
                         "validation": scores.get("val", {}).get(metric)})
    return rows


def plot_ablation(summary: dict, out: Path) -> str | None:
    ablations = summary.get("ablations", {})
    pairs = [(name, vals.get("test", {}).get("ccc"))
             for name, vals in ablations.items()]
    pairs = [(name, value) for name, value in pairs if isinstance(value, (int, float))]
    if not pairs:
        return None
    order = ["full", "no_mri", "no_pet", "no_genetic", "clinical_only",
             "no_attention", "no_hsic", "centralized"]
    pairs.sort(key=lambda p: order.index(p[0]) if p[0] in order else len(order))
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    names, values = zip(*pairs)
    bars = ax.bar(range(len(names)), values,
                  color=["#654ea3" if name == "full" else "#7da0c4" for name in names])
    ax.set_xticks(range(len(names)), [n.replace("_", "\n") for n in names])
    ax.set_ylabel("Held-out test CCC")
    ax.set_title("Retrained ablation variants")
    ax.grid(axis="y", alpha=.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, value, f"{value:.3f}",
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    name = "ablation_test_ccc.png"
    fig.savefig(out / name, dpi=180); plt.close(fig)
    return name


def plot_model_ccc(summary: dict, out: Path) -> str | None:
    values = [(name, result.get("test_ccc"))
              for name, result in summary.get("baselines", {}).items()]
    fed = summary.get("fed_phenograft", {}).get("test", {}).get("ccc")
    if isinstance(fed, (int, float)):
        values.append(("Fed-PhenoGraft", fed))
    values = [(name, value) for name, value in values if isinstance(value, (int, float))]
    if not values:
        return None
    values.sort(key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    names, scores = zip(*values)
    bars = ax.barh(range(len(names)), scores,
                   color=["#654ea3" if n == "Fed-PhenoGraft" else "#8ba7bd" for n in names])
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Held-out test CCC")
    ax.set_title("Progression prediction: baseline comparison")
    ax.grid(axis="x", alpha=.25)
    for bar, value in zip(bars, scores):
        ax.text(value, bar.get_y()+bar.get_height()/2, f" {value:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    name = "model_test_ccc.png"
    fig.savefig(out / name, dpi=180); plt.close(fig)
    return name


def plot_architecture(out: Path) -> str:
    fig, ax = plt.subplots(figsize=(12, 5.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5.4); ax.axis("off")
    colors = ["#dceaf7", "#e7e0f2", "#dcefe7", "#f7ead7", "#eee7f7"]

    def box(x, y, w, h, label, color, size=10):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.12",
                               facecolor=color, edgecolor="#425466", linewidth=1.2)
        ax.add_patch(patch)
        ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=size, wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=14, linewidth=1.2, color="#526579"))

    inputs = [(0.25, "Clinical\nphenotypes"), (2.15, "Structural\nMRI"),
              (4.05, "DaTScan PET\nSBR"), (5.95, "Genetic\ncarriers")]
    for i, (x, label) in enumerate(inputs):
        box(x, 4.15, 1.55, .75, label, colors[i])
        arrow(x+.775, 4.12, x+.775, 3.58)
        box(x, 2.75, 1.55, .8, "Modality\nencoder", colors[i], 9)
    box(7.95, 3.7, 1.7, 1.0, "Shared/private\nlatents + HSIC", colors[4])
    for x, _ in inputs[1:]:
        arrow(x+1.58, 3.15, 7.9, 4.0)
    box(7.95, 2.15, 1.7, .9, "Clinical-query\ncross-attention", "#d9e8f5")
    arrow(1.82, 2.72, 7.9, 2.6)
    arrow(8.8, 3.65, 8.8, 3.1)
    box(10.2, 2.6, 1.5, .95, "Multitask\nprediction heads", "#e4eadb")
    arrow(9.7, 2.6, 10.15, 3.05)
    box(9.95, .75, 1.9, .8, "ΔUPDRS-III\nregression", "#e1edf8", 9)
    box(7.65, .75, 1.9, .8, "PD vs HC\nclassification", "#f6e5dd", 9)
    arrow(10.7, 2.55, 10.85, 1.6)
    arrow(10.3, 2.55, 8.65, 1.6)
    ax.text(6.0, .25, "Federated optimization: local client updates aggregated by sample-weighted FedAvg",
            ha="center", va="center", fontsize=10, color="#34495e",
            bbox=dict(boxstyle="round,pad=.4", facecolor="#f4f6f8", edgecolor="#aab7c4"))
    ax.set_title("Fed-PhenoGraft model and federated learning workflow", fontsize=14, pad=12)
    fig.tight_layout()
    name = "fed_phenograft_architecture.png"
    fig.savefig(out / name, dpi=200, bbox_inches="tight"); plt.close(fig)
    return name


def plot_global_history(history: list, out: Path) -> list[str]:
    made = []
    for metric in ("loss", "ccc", "rmse", "mae", "r2", "pearson", "auc",
                   "accuracy", "balanced_accuracy", "precision", "recall",
                   "sensitivity", "specificity", "f1"):
        train_by_round, val_by_round = {}, {}
        for item in history:
            if not isinstance(item, dict) or not isinstance(item.get("round"), (int, float)):
                continue
            round_no = item["round"]
            train_value = item.get("train", {}).get(metric)
            val_value = item.get("val", {}).get(metric)
            if isinstance(train_value, (int, float)):
                train_by_round.setdefault(round_no, []).append(train_value)
            if isinstance(val_value, (int, float)):
                val_by_round.setdefault(round_no, []).append(val_value)
        if not train_by_round and not val_by_round:
            continue
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        for values_by_round, label, marker, color in (
                (train_by_round, "Train", "o", "#617a99"),
                (val_by_round, "Validation", "s", "#d17c42")):
            if not values_by_round:
                continue
            xs = sorted(values_by_round)
            means = [np.mean(values_by_round[x]) for x in xs]
            stds = [np.std(values_by_round[x]) for x in xs]
            ax.plot(xs, means, marker=marker, label=label, color=color)
            ax.fill_between(xs, np.asarray(means)-stds, np.asarray(means)+stds,
                            color=color, alpha=.16, linewidth=0)
        ax.set(xlabel="Federated round", ylabel=metric.upper(),
               title=f"Global model {metric.upper()} by round")
        ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
        name = f"global_{metric}_by_round.png"
        fig.savefig(out / name, dpi=180); plt.close(fig); made.append(name)
    return made


def plot_client_history(path: Path, out: Path) -> list[str]:
    if path is None or not path.exists():
        return []
    df = pd.read_csv(path)
    required = {"round", "client_id"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path}: expected columns round, client_id")
    made = []
    for metric in ("loss", "accuracy", "balanced_accuracy", "precision",
                   "recall", "sensitivity", "specificity", "f1", "auc",
                   "ccc", "rmse", "mae", "r2", "pearson"):
        col = f"train_{metric}" if f"train_{metric}" in df else metric
        if col not in df:
            continue
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        group_keys = [key for key in ("seed", "client_id") if key in df]
        for client, part in df.groupby(group_keys):
            part = part.sort_values("round")
            seed_id, client_id = client if isinstance(client, tuple) else (None, client)
            title = f"Client {client_id}" + (f", seed {seed_id}" if seed_id is not None else "")
            ax.plot(part["round"], part[col], marker="o", linewidth=1.0, label=title)
        ax.set(xlabel="Federated round", ylabel=metric.title(),
               title=f"Client-wise training {metric}")
        ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
        name = f"client_{metric}_by_round.png"
        fig.savefig(out / name, dpi=180); plt.close(fig); made.append(name)
    return made


def class_analysis(path: Path, out: Path) -> tuple[list[dict], str | None]:
    if path is None or not path.exists():
        return [], None
    from sklearn.metrics import (classification_report, confusion_matrix,
                                 ConfusionMatrixDisplay)
    df = pd.read_csv(path)
    ytrue_key = "y_true_class" if "y_true_class" in df else "y_true"
    ypred_key = "y_pred_class" if "y_pred_class" in df else None
    if ypred_key is None and "y_prob" in df:
        ypred = (df["y_prob"].to_numpy() >= .5).astype(int)
    elif ypred_key:
        ypred = df[ypred_key].to_numpy()
    else:
        raise ValueError("predictions CSV needs y_true_class and y_pred_class, or y_prob")
    ytrue = df[ytrue_key].to_numpy()
    labels = sorted(set(ytrue.tolist()) | set(ypred.tolist()))
    cm = confusion_matrix(ytrue, ypred, labels=labels)
    ConfusionMatrixDisplay(cm, display_labels=[str(x) for x in labels]).plot(
        cmap="Blues", values_format="d", colorbar=False)
    plt.title("Held-out test confusion matrix")
    plt.tight_layout(); name = "class_confusion_matrix.png"
    plt.savefig(out / name, dpi=180); plt.close()
    report = classification_report(ytrue, ypred, labels=labels,
                                  output_dict=True, zero_division=0)
    rows = [{"class": str(k), **v} for k, v in report.items() if isinstance(v, dict)]
    return rows, name


def summarize_round_csv(path: Path, out: Path, client: bool) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    keys = (["seed", "client_id"] if client else ["seed", "split"])
    keys = [key for key in keys if key in df.columns]
    ids = [key for key in ("n_train",) if key in df.columns]
    metrics = [col for col in df.select_dtypes(include="number").columns
               if col not in set(keys + ids + ["round"])]
    if not keys or not metrics:
        return None
    means = df.groupby(keys, dropna=False)[metrics].mean().add_suffix("_mean")
    stds = df.groupby(keys, dropna=False)[metrics].std(ddof=0).add_suffix("_std")
    sizes = df.groupby(keys, dropna=False)[ids].first() if ids else None
    result = means.join(stds)
    if sizes is not None:
        result = sizes.join(result)
    result = result.reset_index()
    result.to_csv(out, index=False)
    return result


def flat_markdown_table(rows: list[dict], columns: list[str]) -> list[str]:
    if not rows:
        return ["_No data recorded for this analysis._", ""]
    lines = ["| " + " | ".join(columns) + " |",
             "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        vals = []
        for col in columns:
            value = row.get(col)
            vals.append("—" if value is None else (f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value)))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=root / "outputs/results/final_metrics.json")
    parser.add_argument("--out-dir", type=Path, default=root / "outputs/paper_analysis")
    parser.add_argument("--client-history", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--fl-comparison", type=Path)
    parser.add_argument("--activation-study", type=Path)
    args = parser.parse_args()
    if not args.metrics.exists():
        raise SystemExit(f"Metrics not found: {args.metrics}; run src/main.py first.")
    summary = json.loads(args.metrics.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    performance = metric_table(summary)
    ablations = ablation_rows(summary)
    write_csv(performance, args.out_dir / "model_performance.csv")
    write_csv(ablations, args.out_dir / "ablation_metrics.csv")
    write_csv([{"split": "heldout_test", **summary.get("fed_phenograft", {}).get("test", {})}],
              args.out_dir / "heldout_server_metrics.csv")
    global_figs = plot_global_history(
        summary.get("fed_phenograft", {}).get("all_seed_rounds_history")
        or summary.get("fed_phenograft", {}).get("rounds_history", []), args.out_dir)
    ablation_fig = plot_ablation(summary, args.out_dir)
    if ablation_fig:
        global_figs.append(ablation_fig)
    model_fig = plot_model_ccc(summary, args.out_dir)
    if model_fig:
        global_figs.append(model_fig)
    global_figs.append(plot_architecture(args.out_dir))
    client_history_path = args.client_history or (root / "outputs/federated_diagnostics/client_round_metrics.csv")
    client_figs = plot_client_history(client_history_path, args.out_dir)
    client_summary = summarize_round_csv(
        client_history_path, args.out_dir / "client_metrics_summary.csv", client=True)
    server_summary = summarize_round_csv(
        root / "outputs/federated_diagnostics/server_round_metrics.csv",
        args.out_dir / "server_metrics_summary.csv", client=False)
    predictions_path = args.predictions or (root / "outputs/federated_diagnostics/heldout_classification_predictions.csv")
    class_rows, class_fig = class_analysis(predictions_path, args.out_dir)
    write_csv(class_rows, args.out_dir / "class_wise_metrics.csv")
    figure_root = root / "outputs/figures"
    confusion_figs = [name for name in (
        "client_confusion_matrices.png", "server_confusion_matrices.png",
        "confusion_matrix.png") if (figure_root / name).exists()]
    confusion_counts = figure_root / "federated_confusion_matrix_counts.csv"

    # Optional externally run experiments are summarized without claiming they
    # were executed by this post-processing script.
    optional = {}
    for label, path in (("federated_algorithm_comparison", args.fl_comparison),
                        ("activation_optimizer_study", args.activation_study)):
        optional[label] = pd.read_csv(path).to_dict(orient="records") if path and path.exists() else []
        if optional[label]:
            write_csv(optional[label], args.out_dir / f"{label}.csv")

    fed = summary.get("fed_phenograft", {})
    cfg = summary.get("config", {})
    train_cfg = cfg.get("training", {})
    split = summary.get("split_sizes", {})
    data_files = sorted((root / "data/raw").glob("*.csv"))
    data_rows = [{"file": p.name, "bytes": p.stat().st_size} for p in data_files]
    write_csv(data_rows, args.out_dir / "input_data_inventory.csv")

    missing = []
    if not client_figs: missing.append("Client-wise accuracy/loss: no per-client round history CSV provided.")
    if not any("loss" in h.get("train", {}) for h in fed.get("rounds_history", []) if isinstance(h, dict)):
        missing.append("Global train/validation loss: loss was not saved in round_history; loss curves cannot be recovered from accuracy or regression metrics.")
    if not any("test" in h for h in fed.get("rounds_history", []) if isinstance(h, dict)):
        missing.append("Global test accuracy/loss by round: per-round test metrics are not saved (and repeated test-set monitoring would compromise the held-out evaluation); report the one-shot test metrics instead.")
    if not class_fig: missing.append("Confusion matrix and class-wise metrics: supply held-out predictions with true and predicted class labels (or probabilities). Existing PNG, if any, is a plot only and does not support a reproducible table.")
    if not optional["federated_algorithm_comparison"]:
        missing.append("FedProx/SCAFFOLD comparison: actual runs and a fl-comparison CSV are required; these values cannot be inferred from FedAvg.")
    if not optional["activation_optimizer_study"]:
        missing.append("Activation/optimizer ablation: controlled retraining runs and an activation-study CSV are required. Adam/AdamW is an optimizer, not an activation function; vary these in separate experiments.")
    if not ablations:
        missing.append("Retrained ablation results are absent from this metrics file; run the configured ablation pipeline to populate them.")

    lines = ["# Paper results and analysis artifacts", "",
             "Generated from saved results; this script does not retrain models or synthesize missing measurements.", "",
             "## Dataset description", "",
             f"Aligned cohort/split counts recorded by the run: {split.get('train', 'NA')} train, {split.get('val', 'NA')} validation, {split.get('test', 'NA')} held-out test (total {sum(v for v in split.values() if isinstance(v, int))}).",
             f"Regression target: {summary.get('target', {}).get('label', 'not recorded')}. Raw CSV inventory is in `input_data_inventory.csv`; byte sizes do not establish subject counts or clinical inclusion criteria.", "",
             "## Experimental setup", "",
             f"Federated method: {summary.get('federated_partition', 'not recorded')} partition; {train_cfg.get('num_clients', 'NA')} clients; {train_cfg.get('num_rounds', 'NA')} maximum rounds; {train_cfg.get('local_epochs', 'NA')} local epochs; learning rate {train_cfg.get('lr', 'NA')}; weight decay {train_cfg.get('weight_decay', 'NA')}; batch size {train_cfg.get('batch_size', 'NA')}; seed {cfg.get('seed', 'NA')}.",
             f"Split sizes: {split}. Target mode: {summary.get('target', {}).get('mode', 'not recorded')}. The results JSON is the source of these setup values.", "",
             "## Traditional ML and multimodal model comparison", "",
             *flat_markdown_table(performance, ["model", "metric", "cv_mean", "cv_std", "test"]), "",
             "## Ablation experiments", "",
             "Available modalities and fusion ablations below come from saved, retrained runs. Interpret metric changes as observed comparisons; statistical significance requires paired predictions or repeated-run uncertainty.", "",
             *flat_markdown_table(ablations, ["variant", "metric", "validation", "test"]), "",
             "## Federated learning results", "",
             f"Saved global train/validation history produced: {', '.join(global_figs) if global_figs else 'no round plots'}. These are global aggregate metrics, not per-client curves.",
             f"Client-wise figures produced: {', '.join(client_figs) if client_figs else 'none'}.",
             f"FedAvg training configuration: {train_cfg.get('num_clients', 'NA')} clients, {train_cfg.get('partition', 'NA')} partition.", "",
             "## Class-wise analysis", "",
             f"Confusion matrix figure: {class_fig or 'not generated from this run'}.", "",
             (f"Federated confusion panels: {', '.join(confusion_figs)}. These show each simulated client at its final logged round, the global server on train/validation splits, and the selected model on held-out test data. Exact cell counts are in `outputs/figures/federated_confusion_matrix_counts.csv`." if confusion_figs else "Federated client/server confusion panels have not been generated."), "",
             *flat_markdown_table(class_rows, ["class", "precision", "recall", "f1-score", "support"]), "",
             "## Explainability", "",
             "Existing model-dependent explainability outputs in `outputs/figures/`: " + (", ".join(p.name for p in sorted((root / "outputs/figures").glob("*.png"))) or "none found") + ". These are surfaced as previously generated artifacts; this script does not rerun attribution or certify causal interpretation.", "",
             "## Analyses requiring additional experimental records", "",
             *([f"- {item}" for item in missing] if missing else ["- All requested inputs were supplied."]), "",
             "## Important scope note", "",
             "A high diagnostic classification score does not by itself validate progression prediction. Report diagnosis and ΔUPDRS-III regression separately, with their own test metrics and uncertainty. Do not describe observational counterfactual feature changes as causal treatment effects.", ""]
    (args.out_dir / "PAPER_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")
    report = {"metrics": str(args.metrics), "out_dir": str(args.out_dir),
              "figures": global_figs + client_figs + ([class_fig] if class_fig else []) + confusion_figs,
              "federated_confusion_matrix_counts": str(confusion_counts) if confusion_counts.exists() else None,
              "tables": ["model_performance.csv", "ablation_metrics.csv", "input_data_inventory.csv"] + (["class_wise_metrics.csv"] if class_rows else []),
              "client_server_metric_summaries": {
                  "client": client_summary is not None,
                  "server": server_summary is not None,
              },
              "limitations": missing}
    (args.out_dir / "analysis_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
