# Fed-PhenoGraft — Results Summary

Regression target: **Δ UPDRS-III (BL → Year 2)**. Federated client partition: **dirichlet**.

Subject-level split: **2430 train / 521 val / 521 test** (stratified on diagnosis; test evaluated once).

## Fed-PhenoGraft (held-out test)

| Task | Metric | Value |
|------|--------|-------|
| Progression regression | CCC | **0.2230** [95% CI 0.1406, 0.2960] |
| Progression regression | RMSE (UPDRS-III pts) | 7.14 |
| Progression regression | MAE (UPDRS-III pts) | 4.88 |
| Progression regression | R² | 0.0405 |
| Progression regression | Pearson r | 0.2854 |
| PD vs HC classification | ROC-AUC | **0.9722** |
| PD vs HC classification | Accuracy | 0.8868 |
| PD vs HC classification | F1 | 0.8173 |

Generalization check: train CCC 0.4111 vs val CCC 0.3648 (gap +0.046) — no overfitting signal.

## Baseline comparison — 12 models (same held-out test set)

| Model | CV CCC (mean ± std) | Test CCC | Test RMSE | Test MAE | Test R² | Test Pearson |
|-------|---------------------|----------|-----------|----------|---------|--------------|
| linear | 0.2374 ± 0.0271 | 0.2036 | 6.96 | 4.78 | 0.0892 | 0.3103 |
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
| **Fed-PhenoGraft** | val 0.3648 (early-stopped) | **0.2230** | 7.14 | 4.88 | 0.0405 | 0.2854 |

## Statistical analysis

- Bootstrap 95% CI (n=1000 resamples) on test CCC: **[0.1406, 0.2960]**.
- Paired bootstrap vs the strongest baseline (**lightgbm**): ΔCCC -0.0070 [95% CI -0.0872, 0.0695], p = 0.5590 — not statistically significant at α = 0.05.
- Across **3 independent training seeds**: test CCC 0.1717 ± 0.0397 (primary model = best-validation seed; test never used for selection).

## Ablation study (each variant retrained, same protocol)

| Variant | Val CCC | Test CCC | Test RMSE | Test MAE |
|---------|---------|----------|-----------|----------|
| **Full Fed-PhenoGraft** | 0.3648 | **0.2230** | 7.14 | 4.88 |
| − Asymmetric attention | 0.2902 | 0.1667 | 7.35 | 4.89 |
| − HSIC shared-private loss | 0.3020 | 0.1356 | 7.52 | 5.02 |
| Centralized (1 client) | 0.3273 | 0.1772 | 7.77 | 5.07 |
| − MRI modality | 0.3246 | 0.2264 | 6.91 | 4.65 |
| − PET/DaTScan modality | 0.2673 | 0.0895 | 7.69 | 5.11 |
| − Genetics modality | 0.3046 | 0.1179 | 7.85 | 5.20 |
| Clinical only | 0.3071 | 0.2056 | 6.91 | 4.67 |

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
