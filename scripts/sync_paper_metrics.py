"""Refresh the revised manuscript tables/text from the latest saved run."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "outputs/paper_analysis/FedPhenoGraft_updated.tex"
SUMMARY = json.loads((ROOT / "outputs/results/final_metrics.json").read_text(encoding="utf-8"))
PAPER_DIR = ROOT / "outputs/paper_analysis"


def fnum(value, digits=3):
    return "--" if value is None else f"{float(value):.{digits}f}"


def replace_table(text: str, label: str, replacement: str) -> str:
    pattern = rf"\\begin\{{table\*?\}}(?:(?!\\begin\{{table\*?\}}).)*?\\label\{{{re.escape(label)}\}}.*?\\end\{{table\*?\}}"
    updated, n = re.subn(pattern, lambda _: replacement, text, count=1, flags=re.S)
    if n != 1:
        raise RuntimeError(f"Could not find table {label}")
    return updated


def baseline_table() -> str:
    names = {
        "linear": "Linear", "ridge": "Ridge", "lasso": "Lasso",
        "elastic_net": "ElasticNet", "svm": "SVM (RBF)", "knn": "$k$-NN",
        "random_forest": "Random Forest", "extra_trees": "Extra Trees",
        "gradient_boosting": "GBDT", "mlp": "MLP", "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
    }
    lines = []
    for key, res in SUMMARY.get("baselines", {}).items():
        lines.append(f"{names.get(key, key)} & ${fnum(res.get('cv_mean'))}\\pm{fnum(res.get('cv_std'))}$ & {fnum(res.get('test_ccc'),4)} & {fnum(res.get('test_rmse'),2)} & {fnum(res.get('test_mae'),2)} & {fnum(res.get('test_r2'),4)} & {fnum(res.get('test_pearson'),4)} \\\\")
    fed = SUMMARY["fed_phenograft"]["test"]
    lines += [r"\midrule",
              f"\\textbf{{Fed-PhenoGraft}} & -- & {fnum(fed['ccc'],4)} & {fnum(fed['rmse'],2)} & {fnum(fed['mae'],2)} & {fnum(fed['r2'],4)} & {fnum(fed['pearson'],4)} \\\\"]
    return "\n".join([
        r"\begin{table}[!t]",
        r"\caption{Progression Performance on the Held-Out Test Set}",
        r"\label{tab:baseline_results}", r"\centering", r"\scriptsize",
        r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{CV CCC} & \textbf{Test CCC} & \textbf{RMSE} & \textbf{MAE} & $\mathbf{R^2}$ & $\mathbf{r}$ \\",
        r"\midrule", *lines, r"\bottomrule", r"\end{tabular}%", "}", r"\end{table}",
    ])


def ablation_table() -> str:
    labels = {
        "full": "Full Fed-PhenoGraft", "no_mri": "Without MRI",
        "no_pet": "Without PET/DaTScan", "no_genetic": "Without genetics",
        "clinical_only": "Clinical only", "no_attention": "Without cross-attention",
        "no_hsic": "Without HSIC loss", "centralized": "Centralized (one client)",
    }
    order = ["full", "no_mri", "no_pet", "no_genetic", "clinical_only",
             "no_attention", "no_hsic", "centralized"]
    rows = []
    for key in order:
        result = SUMMARY.get("ablations", {}).get(key, {}).get("test")
        if result:
            rows.append(f"{labels[key]} & {fnum(result.get('ccc'),4)} & {fnum(result.get('rmse'),2)} & {fnum(result.get('mae'),2)} \\\\")
    return "\n".join([
        r"\begin{table}[!t]", r"\caption{Retrained Ablation Results on the Held-Out Test Set}",
        r"\label{tab:ablation_study}", r"\centering", r"\scriptsize",
        r"\begin{tabular}{lccc}", r"\toprule",
        r"\textbf{Variant} & \textbf{Test CCC} & \textbf{RMSE} & \textbf{MAE} \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ])


def metric_cell(row, prefix, metric):
    mean = row.get(f"{prefix}{metric}_mean")
    std = row.get(f"{prefix}{metric}_std")
    if pd.isna(mean):
        return "--"
    return f"{float(mean):.3f} $\\pm$ {float(std or 0):.3f}"


def round_summary_table(path: Path, label: str, caption: str, is_client: bool,
                        classification: bool) -> str:
    df = pd.read_csv(path)
    id_name = "Client (seed)" if is_client else "Server split (seed)"
    first_col = "client_id" if is_client else "split"
    prefix = "train_" if is_client else ""
    rows = []
    for _, row in df.iterrows():
        ident = f"{int(row['seed'])}:{int(row['client_id'])}" if is_client else f"{row['split']} ({int(row['seed'])})"
        ntrain = str(int(row["n_train"])) if is_client else str(SUMMARY["split_sizes"].get(row["split"], "--"))
        fields = [ident, ntrain]
        metrics = (("loss", "accuracy", "balanced_accuracy", "precision",
                    "sensitivity", "specificity", "f1", "auc") if classification
                   else ("loss", "ccc", "rmse", "mae", "r2", "pearson"))
        for metric in metrics:
            fields.append(metric_cell(row, prefix, metric))
        rows.append(" & ".join(fields) + r" \\")
    colspec = "l" + "r" * (len(metrics) + 1)
    metric_headers = (["Loss", "Acc.", "Bal. Acc.", "Prec.", "Sens.", "Spec.", "F1", "AUC"]
                      if classification else ["Loss", "CCC", "RMSE", "MAE", "$R^2$", "$r$"])
    header = [id_name, "N"] + metric_headers
    return "\n".join([
        r"\begin{table*}[!t]", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        r"\centering", r"\scriptsize", r"\resizebox{\textwidth}{!}{%",
        f"\\begin{{tabular}}{{{colspec}}}", r"\toprule",
        " & ".join(header) + r" \\", r"\midrule", *rows,
        r"\bottomrule", r"\end{tabular}%", "}", r"\end{table*}",
    ])


def classwise_table() -> str:
    df = pd.read_csv(PAPER_DIR / "class_wise_metrics.csv")
    labels = {"0": "HC", "1": "PD", "macro avg": "Macro avg.", "weighted avg": "Weighted avg."}
    rows = []
    for _, row in df.iterrows():
        cls = str(row["class"])
        if cls not in labels:
            continue
        rows.append(f"{labels[cls]} & {float(row['precision']):.3f} & {float(row['recall']):.3f} & {float(row['f1-score']):.3f} & {int(row['support'])} \\\\")
    return "\n".join([
        r"\begin{table}[!t]", r"\caption{Held-Out Diagnostic Class-Wise Metrics}",
        r"\label{tab:classwise}", r"\centering", r"\scriptsize",
        r"\begin{tabular}{lcccc}", r"\toprule",
        r"\textbf{Class} & \textbf{Precision} & \textbf{Recall} & \textbf{F1} & \textbf{Support} \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ])


def main():
    text = TEX.read_text(encoding="utf-8")
    text = replace_table(text, "tab:baseline_results", baseline_table())
    text = replace_table(text, "tab:ablation_study", ablation_table())

    fed = SUMMARY["fed_phenograft"]["test"]
    ci = SUMMARY["statistics"]["test_ci_95"]
    sig = SUMMARY["statistics"].get("significance_vs_best_baseline", {})
    seeds = SUMMARY["statistics"].get("seed_runs", {})
    tccc = seeds.get("test", {}).get("ccc", {})
    ptxt = (f"On the held-out cohort, this run obtains progression CCC {fnum(fed['ccc'],4)} "
            f"(bootstrap 95\\% CI [{fnum(ci['ccc']['ci_low'],4)},{fnum(ci['ccc']['ci_high'],4)}]), "
            f"RMSE {fnum(fed['rmse'],2)} points (95\\% CI [{fnum(ci['rmse']['ci_low'],2)},{fnum(ci['rmse']['ci_high'],2)}]), "
            f"MAE {fnum(fed['mae'],2)} points (95\\% CI [{fnum(ci['mae']['ci_low'],2)},{fnum(ci['mae']['ci_high'],2)}]), "
            f"$R^2={fnum(fed['r2'],4)}$, and Pearson $r={fnum(fed['pearson'],4)}$. "
            f"The paired bootstrap versus {sig.get('compared_against','the strongest baseline')} gives "
            f"$\\Delta$CCC={fnum(sig.get('delta'),4)} (95\\% CI [{fnum(sig.get('ci_low'),4)},{fnum(sig.get('ci_high'),4)}], "
            f"$p={fnum(sig.get('p_value'),3)}$). Across {seeds.get('num_seeds','--')} seeds, test CCC is "
            f"{fnum(tccc.get('mean'),4)}$\\pm${fnum(tccc.get('std'),4)}. The primary model is selected by validation CCC.")
    cls = (f"The selected model's held-out diagnostic metrics are ROC-AUC {fnum(fed.get('auc'),4)}, "
           f"accuracy {100*fed.get('accuracy',0):.2f}\\%, balanced accuracy {fnum(fed.get('balanced_accuracy'),4)}, "
           f"sensitivity {fnum(fed.get('sensitivity'),4)}, specificity {fnum(fed.get('specificity'),4)}, "
           f"precision {fnum(fed.get('precision'),4)}, and F1 {fnum(fed.get('f1'),4)}. "
           "These diagnostic results are separate from progression prediction.")
    # The manuscript may already cite the restored legacy ablation figure name.
    pattern = r"On the held-out cohort,.*?(?=\\begin\{figure\*\}\[!t\]\s*\\centering\s*\\includegraphics\[width=0\.48\\textwidth\]\{(?:ablation_test_ccc|ablation_study)\.png\})"
    results_insert = ptxt + "\n\n" + cls + "\n\n" + classwise_table() + "\n\n"
    text, n = re.subn(pattern, lambda _: results_insert, text, count=1, flags=re.S)
    if n != 1:
        raise RuntimeError("Could not replace the results paragraph/class-wise table")

    abl = SUMMARY.get("ablations", {})
    full_ccc = abl.get("full", {}).get("test", {}).get("ccc", fed["ccc"])
    no_pet = abl.get("no_pet", {}).get("test", {}).get("ccc")
    no_mri = abl.get("no_mri", {}).get("test", {}).get("ccc")
    clinical = abl.get("clinical_only", {}).get("test", {}).get("ccc")
    no_attn = abl.get("no_attention", {}).get("test", {}).get("ccc")
    no_hsic = abl.get("no_hsic", {}).get("test", {}).get("ccc")
    ablation_text = (f"The modality comparison shows the largest observed CCC decrease when PET/DaTScan features are removed "
                     f"(full model {fnum(full_ccc,4)} vs. without PET {fnum(no_pet,4)}). Removing MRI yields CCC {fnum(no_mri,4)}, "
                     f"while clinical-only reaches {fnum(clinical,4)}. The fusion ablations yield CCC {fnum(no_attn,4)} without cross-attention "
                     f"and {fnum(no_hsic,4)} without HSIC. These are single-run descriptive comparisons; confidence intervals for each ablation were not computed. "
                     "The current ablation suite does not include activation-function or optimizer variants.")
    text, _ = re.subn(r"The modality comparison shows.*?The current ablation suite does not include activation-function or optimizer variants\.",
                      lambda _: ablation_text, text, count=1, flags=re.S)

    client_csv = PAPER_DIR / "client_metrics_summary.csv"
    server_csv = PAPER_DIR / "server_metrics_summary.csv"
    tables = round_summary_table(client_csv, "tab:client_metrics_reg",
                                 "Client-Local Progression Metrics Averaged Across Rounds (Mean $\\pm$ SD)", True, False)
    tables += "\n\n" + round_summary_table(client_csv, "tab:client_metrics_cls",
                                 "Client-Local Classification Metrics Averaged Across Rounds (Mean $\\pm$ SD)", True, True)
    tables += "\n\n" + round_summary_table(server_csv, "tab:server_metrics_reg",
                                 "Global Server Progression Metrics on Train and Validation Splits (Mean $\\pm$ SD)", False, False)
    tables += "\n\n" + round_summary_table(server_csv, "tab:server_metrics_cls",
                                 "Global Server Classification Metrics on Train and Validation Splits (Mean $\\pm$ SD)", False, True)
    for old_label in ("tab:client_metrics", "tab:server_metrics"):
        text = re.sub(rf"\\begin\{{table\*?\}}(?:(?!\\begin\{{table\*?\}}).)*?\\label\{{{old_label}}}.*?\\end\{{table\*?\}}\s*",
                      "", text, count=1, flags=re.S)
    for table_label in ("tab:client_metrics_reg", "tab:client_metrics_cls",
                        "tab:server_metrics_reg", "tab:server_metrics_cls"):
        text = re.sub(rf"\\begin\{{table\*?\}}(?:(?!\\begin\{{table\*?\}}).)*?\\label\{{{table_label}}}.*?\\end\{{table\*?\}}\s*",
                      "", text, count=1, flags=re.S)
    marker = r"\begin{figure*}[!t]"  # first one after the client/server subsection text
    section = text.index(r"\subsection{Federated Client and Server Metrics}")
    pos = text.index(marker, section)
    text = text[:pos] + tables + "\n\n" + text[pos:]

    # Update abstract and conclusion claims from this actual rerun.
    abstract_update = (f"On the current {sum(SUMMARY['split_sizes'].values()):,}-subject PPMI run using a four-client Dirichlet ($\\alpha=0.5$) partition, held-out progression performance is CCC {fnum(fed['ccc'],4)} "
                   f"(bootstrap 95\\% CI $[{fnum(ci['ccc']['ci_low'],4)},{fnum(ci['ccc']['ci_high'],4)}]$), RMSE {fnum(fed['rmse'],2)}, and MAE {fnum(fed['mae'],2)}. "
                   f"The paired bootstrap against LightGBM is not statistically significant ($p={fnum(sig.get('p_value'),3)}$). Diagnostic classification yields ROC-AUC {fnum(fed.get('auc'),4)} and accuracy {100*fed.get('accuracy',0):.2f}\\%. "
                   "Results include twelve baselines, eight retrained ablations, client-local and global round metrics, class-wise test results, and saved explainability figures. FedProx/SCAFFOLD and activation/optimizer ablations remain future experiments.")
    text, n = re.subn(r"On the current [\d,]+-subject PPMI run.*?Alternative federated algorithms, client-level traces, and optimizer/activation ablations remain to be evaluated\.",
                      lambda _: abstract_update, text, count=1, flags=re.S)
    abstract_block = (r"\begin{abstract}" + "\n" +
        "Parkinson's disease (PD) progression prediction is vulnerable to baseline-score autocorrelation, while data fragmentation and incomplete modalities complicate multi-site deployment. We present Fed-PhenoGraft, a federated multimodal framework for predicting two-year change in MDS-UPDRS Part III ($\\Delta$UPDRS-III) and concurrently classifying diagnosis. The method combines phenotype-guided asymmetric cross-modal attention, shared-private latent decomposition with HSIC regularization, and learnable missing-modality tokens. " +
        f"On {sum(SUMMARY['split_sizes'].values()):,} PPMI subjects under a four-client Dirichlet ($\\alpha=0.5$) partition, held-out progression CCC is {fnum(fed['ccc'],4)} (bootstrap 95\\% CI $[{fnum(ci['ccc']['ci_low'],4)},{fnum(ci['ccc']['ci_high'],4)}]$), RMSE {fnum(fed['rmse'],2)}, and MAE {fnum(fed['mae'],2)}. The paired bootstrap against LightGBM is not significant ($p={fnum(sig.get('p_value'),3)}$). Diagnostic ROC-AUC is {fnum(fed.get('auc'),4)} and accuracy is {100*fed.get('accuracy',0):.2f}\\%. We compare twelve traditional baselines, eight retrained ablations, client-level and global training dynamics, class-wise test performance, and model explainability. FedProx/SCAFFOLD and activation/optimizer ablations remain future work.\n" +
        r"\end{abstract}")
    text = re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}",
                  lambda _: abstract_block, text, count=1, flags=re.S)
    conclusion = (r"\section{Conclusion}" + "\n" + ptxt + " " + cls +
                  " The new training logs also quantify client heterogeneity across rounds. FedProx/SCAFFOLD and activation/optimizer ablations remain future experiments.\n\n")
    text = re.sub(r"\\section\{Conclusion\}.*?(?=\\begin\{thebibliography\})",
                  lambda _: conclusion, text, count=1, flags=re.S)

    text = text.replace("\\alpha=0.5", "\\alpha=0.5")
    text = text.replace(r"\includegraphics[width=0.48\textwidth]{class_confusion_matrix.png}",
                        r"\includegraphics[width=0.48\textwidth]{confusion_matrix.png}")
    text = text.replace("ablation_test_ccc.png", "ablation_study.png")
    text = text.replace("% No current-run class-wise confusion matrix is included: held-out class predictions\n% were not stored with final_metrics.json, so the old figure is not cited here.\n\n", "")
    text = text.replace("% Existing attention/attribution images in outputs/figures are not regenerated\n% from the current saved results; no quantitative XAI claim is made for this run.\n\n", "")
    text = text.replace(r"\end{figure*}\section{Roadmap for High-Impact Clinical Translation}",
                        r"\end{figure*}" + "\n\n" + r"\section{Roadmap for High-Impact Clinical Translation}")

    TEX.write_text(text, encoding="utf-8")
    print(TEX)


if __name__ == "__main__":
    main()
