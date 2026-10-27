"""Create final-round per-client and global-server confusion matrices.

Client matrices use each simulated client's own training shard. Server matrices
use the globally aggregated model on train/validation data. The held-out test
matrix uses the one-shot final predictions saved by the training pipeline.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DIAG = ROOT / "outputs/federated_diagnostics"
OUT = ROOT / "outputs/figures"
LABELS = ["HC (0)", "PD (1)"]


def matrix_from_counts(row) -> np.ndarray:
    prefix = "train_" if hasattr(row, "train_tn") else ""
    return np.array([[int(getattr(row, prefix + "tn")),
                      int(getattr(row, prefix + "fp"))],
                     [int(getattr(row, prefix + "fn")),
                      int(getattr(row, prefix + "tp"))]], dtype=int)


def draw_matrix(ax, matrix, title: str):
    ax.imshow(matrix, cmap="Blues")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([0, 1], LABELS)
    ax.set_yticks([0, 1], LABELS)
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("Actual", fontsize=8)
    threshold = matrix.max() / 2 if matrix.size else 0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center",
                    color="white" if matrix[i, j] > threshold else "black",
                    fontsize=9)


def save_panel(rows, title: str, path: Path, columns: int, prefix: str,
               matrix_getter):
    count = len(rows)
    nrows = int(np.ceil(count / columns))
    fig, axes = plt.subplots(nrows, columns,
                             figsize=(4.0 * columns, 3.55 * nrows),
                             squeeze=False)
    for ax, row in zip(axes.flat, rows):
        matrix = matrix_getter(row)
        title_text = prefix(row, matrix)
        draw_matrix(ax, matrix, title_text)
    for ax in axes.flat[count:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=15, y=1.005)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    clients = pd.read_csv(DIAG / "client_round_metrics.csv")
    server = pd.read_csv(DIAG / "server_round_metrics.csv")
    clients["round"] = pd.to_numeric(clients["round"], errors="raise")
    server["round"] = pd.to_numeric(server["round"], errors="raise")

    # The final recorded round can differ by seed because early stopping is used.
    final_client_rows = (clients.sort_values("round")
                         .groupby(["seed", "client_id"], as_index=False)
                         .tail(1).sort_values(["seed", "client_id"]))
    final_server_rows = (server.sort_values("round")
                         .groupby(["seed", "split"], as_index=False)
                         .tail(1).sort_values(["seed", "split"]))

    summaries = []
    for row in final_client_rows.itertuples(index=False):
        matrix = matrix_from_counts(row)
        summaries.append({"source": "client", "seed": int(row.seed),
                          "client_id": int(row.client_id), "split": "local_train",
                          "round": int(row.round), "n": int(matrix.sum()),
                          "tn": matrix[0, 0], "fp": matrix[0, 1],
                          "fn": matrix[1, 0], "tp": matrix[1, 1]})
        fig, ax = plt.subplots(figsize=(5.0, 4.5))
        draw_matrix(ax, matrix,
                    f"Client {int(row.client_id)} | seed {int(row.seed)} | round {int(row.round)}")
        fig.tight_layout()
        fig.savefig(OUT / f"client_confusion_seed{int(row.seed)}_client{int(row.client_id)}.png",
                    dpi=220, bbox_inches="tight")
        plt.close(fig)

    save_panel(list(final_client_rows.itertuples(index=False)),
               "Final-Round Client Confusion Matrices (Local Training Shards)",
               OUT / "client_confusion_matrices.png", 4,
               lambda r, m: f"Client {int(r.client_id)} | seed {int(r.seed)} | n={m.sum():,}",
               matrix_from_counts)

    for row in final_server_rows.itertuples(index=False):
        matrix = matrix_from_counts(row)
        summaries.append({"source": "server_global", "seed": int(row.seed),
                          "client_id": "", "split": str(row.split),
                          "round": int(row.round), "n": int(matrix.sum()),
                          "tn": matrix[0, 0], "fp": matrix[0, 1],
                          "fn": matrix[1, 0], "tp": matrix[1, 1]})
        fig, ax = plt.subplots(figsize=(5.0, 4.5))
        draw_matrix(ax, matrix,
                    f"Server {row.split} | seed {int(row.seed)} | round {int(row.round)}")
        fig.tight_layout()
        fig.savefig(OUT / f"server_confusion_seed{int(row.seed)}_{row.split}.png",
                    dpi=220, bbox_inches="tight")
        plt.close(fig)

    save_panel(list(final_server_rows.itertuples(index=False)),
               "Final-Round Global Server Confusion Matrices (Train / Validation)",
               OUT / "server_confusion_matrices.png", 3,
               lambda r, m: f"{str(r.split).title()} | seed {int(r.seed)} | n={m.sum():,}",
               matrix_from_counts)

    predictions = pd.read_csv(DIAG / "heldout_classification_predictions.csv")
    test = np.zeros((2, 2), dtype=int)
    for row in predictions.itertuples(index=False):
        test[int(row.y_true_class), int(row.y_pred_class)] += 1
    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    draw_matrix(ax, test, f"Global Model | held-out test | n={test.sum():,}")
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrix.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    summaries.append({"source": "global_heldout_test", "seed": "selected",
                      "client_id": "", "split": "test", "round": "final",
                      "n": int(test.sum()), "tn": test[0, 0], "fp": test[0, 1],
                      "fn": test[1, 0], "tp": test[1, 1]})

    pd.DataFrame(summaries).to_csv(OUT / "federated_confusion_matrix_counts.csv", index=False)
    print({"client_matrices": len(final_client_rows),
           "server_matrices": len(final_server_rows),
           "global_heldout_n": int(test.sum()),
           "counts_csv": str(OUT / "federated_confusion_matrix_counts.csv")})


if __name__ == "__main__":
    main()
