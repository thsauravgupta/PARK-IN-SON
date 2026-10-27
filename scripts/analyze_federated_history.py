#!/usr/bin/env python3
"""
Inspect Fed-PhenoGraft's saved metrics and produce training-history diagnostics.

Run from the repository root:
    python scripts/analyze_federated_history.py
or:
    python scripts/analyze_federated_history.py --metrics outputs/results/final_metrics.json

This script does not retrain models or touch the held-out test set.
It reports which federated signals are actually present in the saved history.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def get_path(obj: dict, path: str, default=None):
    cur: Any = obj
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def find_history(payload: dict) -> tuple[list, str]:
    candidates = [
        ("fed_phenograft.all_seed_rounds_history", get_path(payload, "fed_phenograft.all_seed_rounds_history")),
        ("fed_phenograft.rounds_history", get_path(payload, "fed_phenograft.rounds_history")),
        ("fed_phenograft.development_history", get_path(payload, "fed_phenograft.development_history")),
        ("rounds_history", payload.get("rounds_history")),
        ("history", payload.get("history")),
    ]
    for name, value in candidates:
        if isinstance(value, list) and value:
            return value, name
    return [], "not found"


def flatten_history(history: list[dict]) -> list[dict]:
    rows = []
    for item in history:
        if not isinstance(item, dict) or "_selection" in item:
            continue
        row = {"seed": item.get("seed"), "round": item.get("round")}
        for split in ("train", "val"):
            metrics = item.get(split)
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)):
                        row[f"{split}_{metric}"] = value
        if isinstance(item.get("ccc_gap"), (int, float)):
            row["ccc_gap"] = item["ccc_gap"]
        # Existing source does not log client-level performance. Preserve
        # client metrics if a future implementation adds them.
        clients = item.get("clients")
        if isinstance(clients, list):
            row["client_count_logged"] = len(clients)
            for idx, client in enumerate(clients):
                if isinstance(client, dict):
                    for metric, value in client.items():
                        if isinstance(value, (int, float)):
                            row[f"client_{idx}_{metric}"] = value
        rows.append(row)
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    keys = sorted({key for row in rows for key in row})
    if not keys:
        path.write_text("No round history found.\\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(rows: list[dict], metric: str, output: Path) -> bool:
    train_by_round, val_by_round = {}, {}
    for row in rows:
        round_no = row.get("round")
        if not isinstance(round_no, (int, float)):
            continue
        if isinstance(row.get(f"train_{metric}"), (int, float)):
            train_by_round.setdefault(round_no, []).append(row[f"train_{metric}"])
        if isinstance(row.get(f"val_{metric}"), (int, float)):
            val_by_round.setdefault(round_no, []).append(row[f"val_{metric}"])
    if not train_by_round and not val_by_round:
        return False

    plt.figure(figsize=(8, 4.8))
    for grouped, label, color in ((train_by_round, "Train", "#617a99"),
                                  (val_by_round, "Validation", "#d17c42")):
        if not grouped:
            continue
        xs = sorted(grouped)
        means = [sum(grouped[x]) / len(grouped[x]) for x in xs]
        sds = [float(np.std(grouped[x])) for x in xs]
        plt.plot(xs, means, marker="o", label=f"{label} {metric.upper()}", color=color)
        plt.fill_between(xs, [m-s for m, s in zip(means, sds)],
                         [m+s for m, s in zip(means, sds)], color=color, alpha=.16)
    plt.xlabel("Federated round")
    plt.ylabel(metric.upper())
    plt.title(f"Fed-PhenoGraft training history: {metric.upper()}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="outputs/results/final_metrics.json")
    parser.add_argument("--out-dir", default="outputs/federated_diagnostics")
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not metrics_path.exists():
        raise SystemExit(
            f"Metrics file not found: {metrics_path}\\n"
            "Run the existing project pipeline first, or pass --metrics PATH."
        )

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    history, history_source = find_history(payload)
    rows = flatten_history(history)

    csv_path = out_dir / "round_history.csv"
    write_csv(rows, csv_path)

    generated = []
    for metric in ("loss", "ccc", "rmse", "mae", "r2", "auc", "accuracy",
                   "balanced_accuracy", "precision", "recall", "sensitivity",
                   "specificity", "f1"):
        path = out_dir / f"{metric}_curve.png"
        if plot_metric(rows, metric, path):
            generated.append(path.name)

    # Check what is and is not recorded in this run.
    has_client_metrics = any(
        isinstance(item, dict) and isinstance(item.get("clients"), list)
        for item in history
    )
    algorithm_keys = set()
    for key in ("federated_algorithms", "algorithm_comparison", "fl_comparison"):
        value = payload.get(key)
        if isinstance(value, (dict, list)):
            algorithm_keys.add(key)

    report = {
        "metrics_file": str(metrics_path),
        "history_source": history_source,
        "rounds_found": len(rows),
        "round_level_train_metrics": any("train_ccc" in row for row in rows),
        "round_level_validation_metrics": any("val_ccc" in row for row in rows),
        "client_level_metrics_logged": has_client_metrics,
        "algorithm_comparison_present": bool(algorithm_keys),
        "algorithm_comparison_keys": sorted(algorithm_keys),
        "generated_figures": generated,
        "csv": str(csv_path),
        "interpretation": (
            ("Per-client scalar metrics are logged each round in "
             "outputs/federated_diagnostics/client_round_metrics.csv. "
             if has_client_metrics else
             "Per-client curves require per-client round logging. ")
            + "FedProx/SCAFFOLD comparisons require actual retraining with those "
              "algorithms; they cannot be inferred from FedAvg results."
        ),
    }
    report_path = out_dir / "diagnostic_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\\nWrote: {csv_path}")
    print(f"Wrote: {report_path}")


if __name__ == "__main__":
    main()
