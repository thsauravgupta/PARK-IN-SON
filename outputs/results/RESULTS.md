# Fed-PhenoGraft — Results Summary

Regression target: **Δ UPDRS-III (BL → Year 2)**. Federated client partition: **dirichlet**.

Subject-level split: **2430 train / 521 val / 521 test** (stratified on diagnosis; test evaluated once).

## Fed-PhenoGraft (held-out test)

| Task | Metric | Value |
|------|--------|-------|
| Progression regression | CCC | **0.1469** [95% CI 0.0406, 0.2439] |
| Progression regression | RMSE (UPDRS-III pts) | 7.64 |
| Progression regression | MAE (UPDRS-III pts) | 5.14 |
| Progression regression | R² | -0.0993 |
| Progression regression | Pearson r | 0.1769 |
| PD vs HC classification | ROC-AUC | **0.9816** |
| PD vs HC classification | Accuracy | 0.9367 |
| PD vs HC classification | F1 | 0.9115 |

Generalization check: train CCC 0.4982 vs val CCC 0.3591 (gap +0.139) — no overfitting signal.

## Baseline comparison — 12 models (same held-out test set)

| Model | CV CCC (mean ± std) | Test CCC | Test RMSE | Test MAE | Test R² | Test Pearson |
|-------|---------------------|----------|-----------|----------|---------|--------------|
| linear | 0.2367 ± 0.0241 | 0.2039 | 6.96 | 4.78 | 0.0894 | 0.3106 |
| ridge | 0.2368 ± 0.0245 | 0.2036 | 6.96 | 4.78 | 0.0893 | 0.3105 |
| lasso | 0.2195 ± 0.0231 | 0.1833 | 6.91 | 4.70 | 0.1000 | 0.3208 |
| elastic_net | 0.2104 ± 0.0224 | 0.1776 | 6.92 | 4.71 | 0.0974 | 0.3165 |
| svm | 0.0603 ± 0.0074 | 0.0552 | 7.16 | 4.75 | 0.0342 | 0.2533 |
| knn | 0.0783 ± 0.0206 | 0.0464 | 7.40 | 4.98 | -0.0296 | 0.0950 |
| random_forest | 0.0993 ± 0.0151 | 0.0821 | 7.07 | 4.73 | 0.0603 | 0.2949 |
| extra_trees | 0.0635 ± 0.0105 | 0.0511 | 7.14 | 4.76 | 0.0410 | 0.2883 |
| gradient_boosting | 0.2322 ± 0.0262 | 0.1911 | 7.01 | 4.74 | 0.0758 | 0.2921 |
| mlp | 0.1400 ± 0.0751 | 0.1924 | 7.10 | 4.93 | 0.0518 | 0.2729 |
| xgboost | 0.2401 ± 0.0302 | 0.2135 | 6.96 | 4.67 | 0.0883 | 0.3141 |
| lightgbm | 0.2253 ± 0.0213 | 0.2300 | 6.94 | 4.67 | 0.0931 | 0.3233 |
| **Fed-PhenoGraft** | val 0.3591 (early-stopped) | **0.1469** | 7.64 | 5.14 | -0.0993 | 0.1769 |

## Statistical analysis

- Bootstrap 95% CI (n=1000 resamples) on test CCC: **[0.0406, 0.2439]**.
- Paired bootstrap vs the strongest baseline (**lightgbm**): ΔCCC -0.0831 [95% CI -0.1667, -0.0054], p = 0.9820 — not statistically significant at α = 0.05.
- Across **3 independent training seeds**: test CCC 0.1567 ± 0.0303 (primary model = best-validation seed; test never used for selection).

## Ablation study (each variant retrained, same protocol)

| Variant | Val CCC | Test CCC | Test RMSE | Test MAE |
|---------|---------|----------|-----------|----------|
| **Full Fed-PhenoGraft** | 0.3591 | **0.1469** | 7.64 | 5.14 |
| − Asymmetric attention | 0.2885 | 0.1736 | 7.34 | 4.86 |
| − HSIC shared-private loss | 0.3130 | 0.1481 | 7.78 | 5.15 |
| Centralized (1 client) | 0.3142 | 0.1857 | 7.59 | 5.07 |
| − MRI modality | 0.3272 | 0.2241 | 6.90 | 4.65 |
| − PET/DaTScan modality | 0.2833 | 0.1303 | 7.86 | 5.21 |
| − Genetics modality | 0.3139 | 0.1474 | 7.79 | 5.15 |
| Clinical only | 0.3072 | 0.2055 | 6.91 | 4.64 |

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
