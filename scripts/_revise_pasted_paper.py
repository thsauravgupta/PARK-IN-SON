from pathlib import Path
import re

src = Path(r"C:\Users\Amit\.codex\attachments\7431a8e0-dcfa-4a77-9ca6-40b2ffdee108\Pasted text.txt")
out = Path(__file__).resolve().parents[1] / "outputs" / "paper_analysis" / "FedPhenoGraft_updated.tex"
text = src.read_text(encoding="utf-8")
text = re.sub(r"(?ms)^% \\begin\{table\}\[!t\].*?^% \\end\{table\}\s*", "", text)

def replace_once(pattern, replacement, flags=re.S):
    global text
    text, n = re.subn(pattern, lambda _match: replacement, text, count=1, flags=flags)
    if n != 1:
        raise RuntimeError(f"Expected one replacement for: {pattern!r}; found {n}")

text = text.replace(r"\usepackage{graphicx}", r"\usepackage{graphicx}" + "\n" + r"\graphicspath{{outputs/figures/}{outputs/paper_analysis/}}", 1)

replace_once(r"\\begin\{abstract\}.*?\\end\{abstract\}", r"""\begin{abstract}
Parkinson's disease (PD) progression prediction is vulnerable to baseline-score autocorrelation, while clinical data fragmentation and incomplete modalities complicate deployment across sites. We present Fed-PhenoGraft, a federated multimodal framework for predicting two-year change in MDS-UPDRS Part III ($\Delta$UPDRS-III) and concurrently classifying diagnosis. The method combines phenotype-guided asymmetric cross-modal attention, shared-private latent decomposition with HSIC regularization, and learnable missing-modality tokens. On the current 3,472-subject PPMI run using a four-client Dirichlet ($\alpha=0.5$) partition, held-out progression performance is CCC $0.1894$ (bootstrap 95\% CI $[0.1076,0.2655]$), RMSE $7.30$, and MAE $4.91$. The paired bootstrap comparison against LightGBM is not statistically significant ($\Delta$CCC $=-0.0406$, $p=0.857$). Diagnostic classification yields ROC-AUC $0.9759$ and accuracy $90.60\%$; this secondary classification result does not establish strong progression prediction. Results include twelve traditional baselines and eight retrained architecture/modality ablations. Alternative federated algorithms, client-level traces, and optimizer/activation ablations remain to be evaluated.
\end{abstract}""")

# The supplied source used a comparison chart as if it were an architecture diagram.
text = text.replace(r"\includegraphics[width=0.92\textwidth]{model_comparison.png}", r"\includegraphics[width=0.92\textwidth]{model_test_ccc.png}", 1)
text = text.replace("{parkinson_arch_diag.png}", "{fed_phenograft_architecture.png}", 1)
text = text.replace("% PLACEHOLDER: Fig. 1 Architecture Diagram & Benchmark Comparison\n% File location: outputs/figures/model_comparison.png", "% Figure 1: current-run baseline comparison (outputs/paper_analysis/model_test_ccc.png)", 1)
text = text.replace("Overall framework of \\textbf{Fed-PhenoGraft}. Clinical phenotype vectors serve as attention queries probing shared-private decomposed structural MRI, $^{123}\\text{I}$-FP-CIT DaTScan PET SBR, and genetic mutation profiles. Privacy-preserving optimization is conducted across multi-institutional nodes under Dirichlet ($\\alpha=0.5$) non-IID distribution, simultaneously outputting honest 2-year motor rate-of-change ($\\Delta\\text{UPDRS-III}$) and diagnostic subtyping.", "Held-out test CCC for Fed-PhenoGraft and twelve machine-learning baselines on the current saved run. The federated model's CCC is 0.1894; LightGBM has the highest baseline CCC (0.2300).", 1)
text = text.replace("\\label{fig:architecture}", "\\label{fig:baseline_overview}", 1)
replace_once(r"\\begin\{figure\*\}\[!t\]\s*\\centering\s*% Figure 1: current-run baseline comparison.*?\\end\{figure\*\}", "")
text = text.replace("Fed-PhenoGraft (Fig.~\\ref{fig:architecture})", "Fed-PhenoGraft", 1)
text = text.replace("\\textbf{Exhaustive Leak-Free Evaluation}: We validate Fed-PhenoGraft on $3,472$ PPMI subjects against 12 machine learning baselines across 8 retrained ablation variants, accompanied by bootstrap 95\\% confidence intervals, paired hypothesis testing, and mechanistic explainability (Integrated Gradients, counterfactual genetic analysis).", "\\textbf{Leak-Free Evaluation}: We evaluate Fed-PhenoGraft on $3,472$ PPMI subjects against 12 machine-learning baselines and report bootstrap confidence intervals, a paired bootstrap comparison, and eight retrained ablation variants.", 1)
text = text.replace("\\textbf{AUC: 0.982, Acc: 93.7\\%}", "\\textbf{AUC: 0.976, Acc: 90.6\\%}", 1)

# Replace the earlier result figure, which quoted an obsolete CCC gap.
replace_once(r"\\begin\{figure\*\}\[!t\]\s*\\centering\s*% PLACEHOLDER: Fig\. 2.*?\\end\{figure\*\}", r"""\begin{figure*}[!t]
\centering
\includegraphics[width=0.48\textwidth]{global_ccc_by_round.png}
\hfill
\includegraphics[width=0.48\textwidth]{global_accuracy_by_round.png}
\caption{Saved aggregate global training history. (Left) Train and validation CCC across communication rounds. (Right) Train and validation diagnostic accuracy. The held-out test set is evaluated once after training; no per-round test curve or client-level trace is available in the saved records.}
\label{fig:results_history}
\end{figure*}""")

# Replace the active progression table, leaving the historical commented drafts intact.
section_start = text.index(r"\subsection{Benchmark Performance Against Baselines}")
start = section_start + re.search(r"(?m)^\\begin\{table\}\[!t\]", text[section_start:]).start()
end = text.index(r"\end{table}", start) + len(r"\end{table}")
table = r"""\begin{table}[!t]
\caption{Progression Performance on the Held-Out Test Set}
\label{tab:baseline_results}
\centering
\scriptsize
\setlength{\tabcolsep}{2.5pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
\textbf{Model} & \textbf{CV CCC} & \textbf{Test CCC} & \textbf{RMSE} & \textbf{MAE} & $\mathbf{R^2}$ & $\mathbf{r}$ \\
\midrule
Linear & $0.237\pm0.025$ & 0.2037 & 6.96 & 4.78 & 0.0893 & 0.3105 \\
Ridge & $0.237\pm0.025$ & 0.2036 & 6.96 & 4.78 & 0.0893 & 0.3105 \\
Lasso & $0.220\pm0.023$ & 0.1833 & 6.91 & 4.70 & 0.1000 & 0.3208 \\
ElasticNet & $0.210\pm0.022$ & 0.1776 & 6.92 & 4.71 & 0.0974 & 0.3165 \\
SVM (RBF) & $0.060\pm0.007$ & 0.0552 & 7.16 & 4.75 & 0.0342 & 0.2533 \\
$k$-NN & $0.078\pm0.021$ & 0.0464 & 7.40 & 4.98 & $-0.0296$ & 0.0950 \\
Random Forest & $0.099\pm0.015$ & 0.0821 & 7.07 & 4.73 & 0.0603 & 0.2949 \\
Extra Trees & $0.064\pm0.011$ & 0.0511 & 7.14 & 4.76 & 0.0410 & 0.2883 \\
GBDT & $0.232\pm0.026$ & 0.1911 & 7.01 & 4.74 & 0.0758 & 0.2921 \\
MLP & $0.140\pm0.075$ & 0.1924 & 7.10 & 4.93 & 0.0518 & 0.2729 \\
XGBoost & $0.242\pm0.023$ & 0.2002 & 7.01 & 4.75 & 0.0751 & 0.2960 \\
LightGBM & $0.225\pm0.021$ & \textbf{0.2300} & 6.94 & 4.67 & 0.0931 & \textbf{0.3233} \\
\midrule
\textbf{Fed-PhenoGraft} & -- & 0.1894 & 7.30 & 4.91 & $-0.0041$ & 0.2431 \\
\bottomrule
\end{tabular}%
}
\end{table}"""
text = text[:start] + table + text[end:]

# Replace stale summary claims with metrics from final_metrics.json.
replace_once(r"In progression prediction, Fed-PhenoGraft achieves.*?(?=\\begin\{figure\*\}\[!t\]\s*\\centering\s*% PLACEHOLDER: Fig\. 3)", r"""On the held-out cohort, the current saved run obtains a progression CCC of $0.1894$ (bootstrap 95\% CI $[0.1076,0.2655]$), RMSE $7.30$ points (95\% CI $[6.62,8.05]$), MAE $4.91$ points (95\% CI $[4.48,5.41]$), $R^2=-0.0041$, and Pearson $r=0.2431$. These results are lower than the strongest baseline LightGBM (CCC $0.2300$). The paired bootstrap difference is $\Delta$CCC $=-0.0406$ (95\% CI $[-0.1233,0.0400]$, $p=0.857$), which does not show a statistically significant difference at $\alpha=0.05$. Across three seeds, test CCC is $0.1594\pm0.0220$; the primary model is selected by validation CCC. Training history is shown in Fig.~\ref{fig:results_history}.

For diagnostic classification, the same saved run reports test ROC-AUC $0.9759$, accuracy $90.60\%$, and F1 $0.8537$. These classification scores are a separate secondary task and should not be interpreted as evidence of accurate progression prediction. Per-class metrics and a reproducible confusion matrix are not reported because held-out class predictions were not saved.""")

# Replace stale ablation/missingness figure with newly generated ablation and model comparison figures.
replace_once(r"\\begin\{figure\*\}\[!t\]\s*\\centering\s*% PLACEHOLDER: Fig\. 3.*?\\end\{figure\*\}", r"""\begin{figure*}[!t]
\centering
\includegraphics[width=0.48\textwidth]{ablation_test_ccc.png}
\hfill
\includegraphics[width=0.48\textwidth]{model_test_ccc.png}
\caption{Held-out test CCC from the saved experiments. (Left) Retrained modality, fusion, and centralization ablations. (Right) Fed-PhenoGraft and twelve traditional machine-learning baselines. Higher CCC indicates better agreement; the federated model does not exceed the best baseline in this run.}
\label{fig:ablation_comparison}
\end{figure*}""")

# Replace active ablation table.
section_start = text.index(r"\subsection{Ablation Study Analysis}")
start = section_start + re.search(r"(?m)^\\begin\{table\}\[!t\]", text[section_start:]).start()
end = text.index(r"\end{table}", start) + len(r"\end{table}")
table = r"""\begin{table}[!t]
\caption{Retrained Ablation Results on the Held-Out Test Set}
\label{tab:ablation_study}
\centering
\scriptsize
\begin{tabular}{lccc}
\toprule
\textbf{Variant} & \textbf{Test CCC} & \textbf{RMSE} & \textbf{MAE} \\
\midrule
Full Fed-PhenoGraft & 0.1894 & 7.30 & 4.91 \\
Without MRI & 0.2098 & 6.92 & 4.64 \\
Without PET/DaTScan & 0.1219 & 7.60 & 5.08 \\
Without genetics & 0.1546 & 7.49 & 5.00 \\
Clinical only & 0.1994 & 6.88 & 4.65 \\
Without cross-attention & 0.1694 & 7.42 & 4.95 \\
Without HSIC loss & 0.1547 & 7.49 & 5.00 \\
Centralized (one client) & 0.1235 & 7.67 & 5.06 \\
\bottomrule
\end{tabular}
\end{table}"""
text = text[:start] + table + text[end:]

replace_once(r"Key findings from the ablation study include:.*?(?=\\begin\{figure\*\}\[!t\]\s*\\centering\s*% PLACEHOLDER: Fig\. 4)", r"""The modality comparison shows the largest observed CCC decrease when PET/DaTScan features are removed (from $0.1894$ to $0.1219$). Removing MRI modestly increases CCC to $0.2098$, while the clinical-only model reaches $0.1994$. The fusion ablations yield CCC $0.1694$ without cross-attention and $0.1547$ without HSIC. These are single-run comparisons; the available file does not include paired predictions or confidence intervals for each ablation, so differences should be treated as descriptive rather than statistically established. The current ablation suite does not include activation-function or optimizer variants.""")
text = text.replace(r"(Fig.~\ref{fig:architecture}), ", "")
text = text.replace(r"Fig.~\ref{fig:ablation_robustness}A", r"Fig.~\ref{fig:ablation_comparison} (left)")
text = text.replace("Directly solves autocorrelation, privacy, missingness", "Uses a delta target with simulated federated learning")
text = text.replace("State-of-the-art leak-free multi-site prognostic benchmark", "Evaluated against fixed baselines; not superior on CCC")
text = text.replace("AUC: 0.982, Acc: 93.7\\%", "AUC: 0.976, Acc: 90.6\\%")

# Remove stale confusion matrix and counterfactual figure, whose reported numbers do not match this run.
replace_once(r"\\begin\{figure\*\}\[!t\]\s*\\centering\s*% PLACEHOLDER: Fig\. 4.*?\\end\{figure\*\}", r"""% No current-run class-wise confusion matrix is included: held-out class predictions
% were not stored with final_metrics.json, so the old figure is not cited here.""")

# Replace XAI plot/caption and unsupported numerical statements.
replace_once(r"\\begin\{figure\*\}\[!t\]\s*\\centering\s*% PLACEHOLDER: Fig\. 5.*?\\end\{figure\*\}", r"""% Existing attention/attribution images in outputs/figures are not regenerated
% from the current saved results; no quantitative XAI claim is made for this run.""")
replace_once(r"\\subsection\{Explainability \\& In Silico Genetic Perturbations\}.*?(?=\\section\{Roadmap for High-Impact Clinical Translation\})", r"""\subsection{Explainability and In Silico Genetic Perturbations}
The repository contains previously generated attention, Integrated Gradients, and genetic counterfactual images. The current saved metrics do not include attribution values, held-out counterfactual outputs, or a model-checkpoint provenance record sufficient to tie those figures to this result set. We therefore do not make quantitative feature-importance or counterfactual claims here. These analyses should be regenerated from the selected checkpoint and accompanied by uncertainty and sensitivity analyses before inclusion as evidence.""")

# Correct roadmap statements that framed single-run ablation differences as conclusive.
text = text.replace("As demonstrated in Table~\\ref{tab:ablation_study}, substituting synthetic MRI fallback features with volumetric FreeSurfer/Schaefer parcels from real T1-weighted MPRAGE scans will resolve synthetic noise and unlock positive cross-modal synergy.", "The saved modality ablation shows a small increase in CCC when MRI is removed; this is descriptive and does not establish the effect of real MRI. A controlled evaluation with verified real T1-weighted MRI features remains future work.")
text = text.replace("Advanced Federated Algorithms (FedProx \\& Ditto)", "Advanced Federated Algorithms (FedProx and SCAFFOLD)")

replace_once(r"\\section\{Conclusion\}.*?(?=\\begin\{thebibliography\})", r"""\section{Conclusion}
We presented Fed-PhenoGraft, a federated multimodal framework for predicting two-year change in UPDRS-III while performing diagnostic classification. In the current saved 3,472-subject PPMI run, the held-out progression CCC is $0.1894$ (95\% bootstrap CI $[0.1076,0.2655]$), with RMSE $7.30$ and MAE $4.91$ UPDRS-III points. The result is statistically indistinguishable from the strongest evaluated baseline under the paired bootstrap test ($p=0.857$). Diagnostic ROC-AUC is $0.9759$, but classification performance does not imply accurate progression estimation. Modality and fusion ablations provide descriptive comparisons; client-level learning curves, loss curves, class-wise test metrics, alternative federated algorithms, and activation/optimizer ablations remain unreported because the corresponding run records are unavailable. Future work should add those controlled experiments and external validation before clinical claims are made.

""")

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(text, encoding="utf-8")
print(out)
