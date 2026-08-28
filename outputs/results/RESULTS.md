# Fed-PhenoGraft — Results Summary

Regression target: **Δ UPDRS-III (BL → Year 2)**. Federated client partition: **dirichlet**.

Subject-level split: **2430 train / 521 val / 521 test** (stratified on diagnosis; test evaluated once).

## Fed-PhenoGraft (held-out test)

| Task | Metric | Value |
|------|--------|-------|
| Progression regression | CCC | **0.1894** [95% CI 0.1076, 0.2655] |
| Progression regression | RMSE (UPDRS-III pts) | 7.30 |
| Progression regression | MAE (UPDRS-III pts) | 4.91 |
| Progression regression | R² | -0.0041 |
| Progression regression | Pearson r | 0.2431 |
| PD vs HC classification | ROC-AUC | **0.9759** |
| PD vs HC classification | Accuracy | 0.9060 |
| PD vs HC classification | F1 | 0.8537 |

Generalization check: train CCC 0.4192 vs val CCC 0.3588 (gap +0.060) — no overfitting signal.

## Baseline comparison — 12 models (same held-out test set)

| Model | CV CCC (mean ± std) | Test CCC | Test RMSE | Test MAE | Test R² | Test Pearson |
|-------|---------------------|----------|-----------|----------|---------|--------------|
| linear | 0.2369 ± 0.0246 | 0.2037 | 6.96 | 4.78 | 0.0893 | 0.3105 |
| ridge | 0.2368 ± 0.0245 | 0.2036 | 6.96 | 4.78 | 0.0893 | 0.3105 |
| lasso | 0.2195 ± 0.0231 | 0.1833 | 6.91 | 4.70 | 0.1000 | 0.3208 |
| elastic_net | 0.2104 ± 0.0224 | 0.1776 | 6.92 | 4.71 | 0.0974 | 0.3165 |
| svm | 0.0603 ± 0.0074 | 0.0552 | 7.16 | 4.75 | 0.0342 | 0.2533 |
| knn | 0.0783 ± 0.0206 | 0.0464 | 7.40 | 4.98 | -0.0296 | 0.0950 |
| random_forest | 0.0993 ± 0.0151 | 0.0821 | 7.07 | 4.73 | 0.0603 | 0.2949 |
| extra_trees | 0.0635 ± 0.0105 | 0.0511 | 7.14 | 4.76 | 0.0410 | 0.2883 |
| gradient_boosting | 0.2322 ± 0.0262 | 0.1911 | 7.01 | 4.74 | 0.0758 | 0.2921 |
| mlp | 0.1400 ± 0.0751 | 0.1924 | 7.10 | 4.93 | 0.0518 | 0.2729 |
| xgboost | 0.2418 ± 0.0232 | 0.2002 | 7.01 | 4.75 | 0.0751 | 0.2960 |
| lightgbm | 0.2253 ± 0.0213 | 0.2300 | 6.94 | 4.67 | 0.0931 | 0.3233 |
| **Fed-PhenoGraft** | val 0.3588 (early-stopped) | **0.1894** | 7.30 | 4.91 | -0.0041 | 0.2431 |

## Statistical analysis

- Bootstrap 95% CI (n=1000 resamples) on test CCC: **[0.1076, 0.2655]**.
- Paired bootstrap vs the strongest baseline (**lightgbm**): ΔCCC -0.0406 [95% CI -0.1233, 0.0400], p = 0.8570 — not statistically significant at α = 0.05.
- Across **3 independent training seeds**: test CCC 0.1594 ± 0.0220 (primary model = best-validation seed; test never used for selection).

## Ablation study (each variant retrained, same protocol)

| Variant | Val CCC | Test CCC | Test RMSE | Test MAE |
|---------|---------|----------|-----------|----------|
| **Full Fed-PhenoGraft** | 0.3588 | **0.1894** | 7.30 | 4.91 |
| − Asymmetric attention | 0.2831 | 0.1694 | 7.42 | 4.95 |
| − HSIC shared-private loss | 0.3285 | 0.1547 | 7.49 | 5.00 |
| Centralized (1 client) | 0.3318 | 0.1235 | 7.67 | 5.06 |
| − MRI modality | 0.3230 | 0.2098 | 6.92 | 4.64 |
| − PET/DaTScan modality | 0.3055 | 0.1219 | 7.60 | 5.08 |
| − Genetics modality | 0.3275 | 0.1546 | 7.49 | 5.00 |
| Clinical only | 0.2937 | 0.1994 | 6.88 | 4.65 |

## Figures

| Figure | File |
|--------|------|
| Model comparison | `outputs/figures/model_comparison.png` |
| Training curve | `outputs/figures/training_curve.png` |
| Predicted vs actual | `outputs/figures/pred_vs_actual.png` |
| Confusion matrix (PD vs HC) | `outputs/figures/confusion_matrix.png` |
| Attention maps | `outputs/figures/attention_maps.png` |
| Missing-modality robustness | `outputs/figures/modality_robustness.png` |
| Feature attribution (IG) | `outputs/figures/global_feature_importance.png` |
| Counterfactual gene analysis | `outputs/figures/counterfactual_genes.png` |
| Ablation study | `outputs/figures/ablation_study.png` |
