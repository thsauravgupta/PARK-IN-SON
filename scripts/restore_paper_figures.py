"""Restore existing paper figures and add client/server trace figures in the revised TeX."""
from pathlib import Path
import re

path = Path(__file__).resolve().parents[1] / "outputs/paper_analysis/FedPhenoGraft_updated.tex"
text = path.read_text(encoding="utf-8")
text = text.replace(r"\graphicspath{{outputs/figures/}{outputs/paper_analysis/}}" + "\n", "")

history_end = r"\label{fig:results_history}" + "\n" + r"\end{figure*}"
history_extra = r"""

\begin{figure*}[!t]
\centering
\includegraphics[width=0.48\textwidth]{pred_vs_actual.png}
\hfill
\includegraphics[width=0.48\textwidth]{training_curve.png}
\caption{Held-out regression fit and FedAvg convergence. The scatter compares predicted and observed two-year UPDRS-III change; the curve shows train and validation CCC.}
\label{fig:fit_convergence}
\end{figure*}

\subsection{Federated Client and Server Metrics}
For each communication round, we record global-model objective loss, diagnostic accuracy and classification metrics, and progression metrics on the global training and validation partitions. Each client records the same metric family on its own local training shard after its local update. Client identifiers refer to simulated partitions and must not be interpreted as independent hospitals unless actual site assignments were used. Held-out test metrics are computed after training and are not plotted round by round. Complete round-level values are exported to \texttt{client\_round\_metrics.csv} and \texttt{server\_round\_metrics.csv}.

\begin{figure*}[!t]
\centering
\includegraphics[width=0.48\textwidth]{global_loss_by_round.png}
\hfill
\includegraphics[width=0.48\textwidth]{global_accuracy_by_round.png}\\[2mm]
\includegraphics[width=0.48\textwidth]{client_loss_by_round.png}
\hfill
\includegraphics[width=0.48\textwidth]{client_accuracy_by_round.png}
\caption{Federated training dynamics across communication rounds. Top: global model train/validation objective loss and diagnostic accuracy. Bottom: each client's local training loss and accuracy. Shaded regions on global curves show between-seed standard deviation when multiple seeds are available.}
\label{fig:client_server_dynamics}
\end{figure*}
"""
if history_end in text and "fig:client_server_dynamics" not in text:
    text = text.replace(history_end, history_end + history_extra, 1)

# Add complete metric families alongside the compact loss/accuracy overview.
# Every PNG is deliberately referenced by basename so Overleaf root uploads work.
def metric_figure(label, caption, names):
    lines = [r"\begin{figure*}[!t]", r"\centering"]
    for i, name in enumerate(names):
        lines.append(r"\includegraphics[width=0.48\textwidth]{" + name + "}")
        if i % 2 == 0 and i + 1 < len(names):
            lines.append(r"\hfill")
        elif i + 1 < len(names):
            lines.append(r"\\[2mm]")
    lines.extend([r"\caption{" + caption + "}", r"\label{" + label + "}", r"\end{figure*}"])
    return "\n".join(lines)

metric_groups = [
    ("fig:server_progression_metrics", "Global server progression metrics by communication round: concordance correlation coefficient (CCC), root mean-square error (RMSE), mean absolute error (MAE), coefficient of determination ($R^2$), and Pearson correlation. Curves show train and validation splits.",
     ["global_ccc_by_round.png", "global_rmse_by_round.png", "global_mae_by_round.png", "global_r2_by_round.png", "global_pearson_by_round.png"]),
    ("fig:server_classification_metrics", "Global server diagnostic classification metrics by communication round on train and validation splits.",
     ["global_balanced_accuracy_by_round.png", "global_precision_by_round.png", "global_recall_by_round.png", "global_sensitivity_by_round.png", "global_specificity_by_round.png", "global_f1_by_round.png", "global_auc_by_round.png"]),
    ("fig:client_progression_metrics", "Client-local progression metrics by communication round, evaluated on each client's local training shard.",
     ["client_ccc_by_round.png", "client_rmse_by_round.png", "client_mae_by_round.png", "client_r2_by_round.png", "client_pearson_by_round.png"]),
    ("fig:client_classification_metrics", "Client-local diagnostic classification metrics by communication round, evaluated on each client's local training shard.",
     ["client_balanced_accuracy_by_round.png", "client_precision_by_round.png", "client_recall_by_round.png", "client_sensitivity_by_round.png", "client_specificity_by_round.png", "client_f1_by_round.png", "client_auc_by_round.png"]),
]
if "fig:server_progression_metrics" not in text:
    anchor = r"\label{fig:client_server_dynamics}" + "\n" + r"\end{figure*}"
    if anchor not in text:
        raise RuntimeError("Could not find client/server dynamics figure anchor")
    extra = "\n\n" + "\n\n".join(metric_figure(*group) for group in metric_groups)
    text = text.replace(anchor, anchor + extra, 1)

# Add final-round confusion matrices for every simulated client and the merged
# global server model. The global held-out test matrix remains a separate plot.
if "fig:client_server_confusion_matrices" not in text:
    matrix_anchor = r"\label{fig:client_classification_metrics}" + "\n" + r"\end{figure*}"
    if matrix_anchor not in text:
        raise RuntimeError("Could not find client classification metrics figure anchor")
    matrix_block = r"""

\begin{figure*}[!t]
\centering
\includegraphics[width=0.49\textwidth]{client_confusion_matrices.png}
\hfill
\includegraphics[width=0.49\textwidth]{server_confusion_matrices.png}
\caption{Confusion matrices at each seed's final recorded round. Left: all 12 simulated clients, evaluated on each client's local training shard. Right: the globally aggregated server model evaluated on train and validation splits for each seed. Counts are shown in each cell. The selected global model's one-shot held-out test confusion matrix is shown separately in Fig.~\ref{fig:classification_counterfactual}.}
\label{fig:client_server_confusion_matrices}
\end{figure*}
"""
    text = text.replace(matrix_anchor, matrix_anchor + matrix_block, 1)

# The original caption predated the client-level metric logging.
text = text.replace(
    "The held-out test set is evaluated once after training; no per-round test curve or client-level trace is available in the saved records.",
    "The held-out test set is evaluated once after training. Client-level loss and accuracy traces are shown in Fig.~\\ref{fig:client_server_dynamics}.")
text = text.replace(
    "Complete round-level values are exported to \\texttt{client\\_round\\_metrics.csv} and \\texttt{server\\_round\\_metrics.csv}.",
    "Complete round-level values are exported to \\texttt{client\\_round\\_metrics.csv} and \\texttt{server\\_round\\_metrics.csv}. Final-round client and merged-server confusion matrices are shown in Fig.~\\ref{fig:client_server_confusion_matrices}; the selected global model's held-out test confusion matrix is shown in Fig.~\\ref{fig:classification_counterfactual}.")

# Replace the new ablation/baseline panel with ablation plus the original
# missing-modality figure, then keep a separate baseline comparison panel.
text = text.replace(r"\includegraphics[width=0.48\textwidth]{model_test_ccc.png}",
                    r"\includegraphics[width=0.48\textwidth]{modality_robustness.png}", 1)
text = text.replace("Held-out test CCC from the saved experiments. (Left) Retrained modality, fusion, and centralization ablations. (Right) Fed-PhenoGraft and twelve traditional machine-learning baselines. Higher CCC indicates better agreement; the federated model does not exceed the best baseline in this run.",
                    "(Left) Held-out CCC across retrained modality and fusion ablations. (Right) Test RMSE under missing-modality stress scenarios.", 1)
abl_end = r"\label{fig:ablation_comparison}" + "\n" + r"\end{figure*}"
base_panel = r"""

\begin{figure*}[!t]
\centering
\includegraphics[width=0.72\textwidth]{model_test_ccc.png}
\caption{Held-out progression CCC for Fed-PhenoGraft and twelve traditional machine-learning baselines. The federated model does not exceed the best baseline in this run.}
\label{fig:baseline_comparison}
\end{figure*}
"""
if abl_end in text and "fig:baseline_comparison" not in text:
    text = text.replace(abl_end, abl_end + base_panel, 1)

# Reinsert the original confusion-matrix, counterfactual, attention, and IG
# figures, with captions that avoid unsupported numeric or causal claims.
section_pattern = r"\\subsection\{Explainability and In Silico Genetic Perturbations\}.*?(?=\\section\{Roadmap for High-Impact Clinical Translation\})"
explainability = r"""\subsection{Explainability and In Silico Genetic Perturbations}
The pipeline generates a held-out confusion matrix, a missing-modality stress test, attention summaries, Integrated Gradients feature attributions, and genetic counterfactual shifts. The counterfactual results are model-based sensitivity analyses and are not causal treatment effects.

\begin{figure*}[!t]
\centering
\includegraphics[width=0.48\textwidth]{confusion_matrix.png}
\hfill
\includegraphics[width=0.48\textwidth]{counterfactual_genes.png}
\caption{(Left) Held-out binary diagnostic confusion matrix. (Right) Model-predicted progression changes under genetic carrier-status counterfactuals; these are sensitivity analyses, not causal estimates.}
\label{fig:classification_counterfactual}
\end{figure*}

\begin{figure*}[!t]
\centering
\includegraphics[width=0.48\textwidth]{attention_maps.png}
\hfill
\includegraphics[width=0.48\textwidth]{global_feature_importance.png}
\caption{Model explainability outputs. (Left) Attention-weight summary across auxiliary modalities. (Right) Integrated Gradients feature attribution summary. These visualizations are descriptive and should be interpreted alongside quantitative validation.}
\label{fig:xai_suite}
\end{figure*}
"""
text, n = re.subn(section_pattern, lambda _match: explainability, text,
                  count=1, flags=re.S)
if n != 1:
    raise RuntimeError("Could not restore the explainability subsection")

# Clean up paragraph/float joins introduced by prior update passes.
text = text.replace("not saved.\\begin{figure*}", "not saved.\n\n\\begin{figure*}")
text = text.replace("not reported.\\begin{figure*}", "not reported.\n\n\\begin{figure*}")
text = text.replace("variants.% No current-run", "variants.\n\n% No current-run")
text = text.replace("before inclusion as evidence.\\section{Roadmap", "before inclusion as evidence.\n\n\\section{Roadmap")
text = text.replace("\\label{tab:ablation_study} and Fig.~\\ref{fig:ablation_comparison} (left)",
                    "\\label{tab:ablation_study} and Fig.~\\ref{fig:ablation_comparison}")
text = text.replace("Fig.~\\ref{fig:ablation_comparison} (left)", "Fig.~\\ref{fig:ablation_comparison}")

path.write_text(text, encoding="utf-8")
print(path)
