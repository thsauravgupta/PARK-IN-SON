# Fed-PhenoGraft — Results Summary

Regression target: **Δ UPDRS-III (BL → Year 2)**. Federated client partition: **dirichlet**.

Subject-level split: **2430 train / 521 val / 521 test** (stratified on diagnosis; test evaluated once).

## Fed-PhenoGraft (held-out test)

| Task | Metric | Value |
|------|--------|-------|
| Progression regression | CCC | **0.2302** [95% CI 0.1325, 0.3169] |
| Progression regression | RMSE (UPDRS-III pts) | 7.21 |
| Progression regression | MAE (UPDRS-III pts) | 4.77 |
| Progression regression | R² | 0.0224 |
| Progression regression | Pearson r | 0.2856 |
| PD vs HC classification | ROC-AUC | **0.9810** |
| PD vs HC classification | Accuracy | 0.9386 |
| PD vs HC classification | F1 | 0.9126 |

Generalization check: train CCC 0.4927 vs val CCC 0.3112 (gap +0.182) — **overfitting signal — retune.**

## Baseline comparison — 12 models (same held-out test set)

| Model | CV CCC (mean ± std) | Test CCC | Test RMSE | Test MAE | Test R² | Test Pearson |
|-------|---------------------|----------|-----------|----------|---------|--------------|
| linear | 0.1904 ± 0.0972 | 0.2025 | 6.98 | 4.78 | 0.0827 | 0.3024 |
| ridge | 0.2345 ± 0.0221 | 0.2038 | 6.97 | 4.78 | 0.0863 | 0.3068 |
| lasso | 0.2219 ± 0.0167 | 0.1911 | 6.89 | 4.70 | 0.1074 | 0.3322 |
| elastic_net | 0.2106 ± 0.0158 | 0.1822 | 6.90 | 4.70 | 0.1030 | 0.3258 |
| svm | 0.0989 ± 0.0141 | 0.0944 | 7.08 | 4.68 | 0.0561 | 0.3032 |
| knn | 0.0836 ± 0.0125 | 0.0619 | 7.40 | 4.92 | -0.0300 | 0.1093 |
| random_forest | 0.1226 ± 0.0177 | 0.1208 | 6.97 | 4.68 | 0.0845 | 0.3198 |
| extra_trees | 0.0793 ± 0.0111 | 0.0662 | 7.10 | 4.72 | 0.0517 | 0.2926 |
| gradient_boosting | 0.2719 ± 0.0263 | 0.2401 | 6.88 | 4.65 | 0.1078 | 0.3406 |
| mlp | 0.2581 ± 0.0338 | 0.2441 | 6.90 | 4.64 | 0.1040 | 0.3412 |
| xgboost | 0.2708 ± 0.0294 | 0.2430 | 6.93 | 4.68 | 0.0974 | 0.3337 |
| lightgbm | 0.2843 ± 0.0304 | 0.2687 | 6.86 | 4.65 | 0.1135 | 0.3563 |
| **Fed-PhenoGraft** | val 0.3112 (early-stopped) | **0.2302** | 7.21 | 4.77 | 0.0224 | 0.2856 |

## Statistical analysis

- Bootstrap 95% CI (n=1000 resamples) on test CCC: **[0.1325, 0.3169]**.
- Paired bootstrap vs the strongest baseline (**lightgbm**): ΔCCC -0.0386 [95% CI -0.0998, 0.0220], p = 0.9000 — not statistically significant at α = 0.05.
- Across **3 independent training seeds**: test CCC 0.2327 ± 0.0089 (primary model = best-validation seed; test never used for selection).

## Ablation study (each variant retrained, same protocol)

| Variant | Val CCC | Test CCC | Test RMSE | Test MAE |
|---------|---------|----------|-----------|----------|
| **Full Fed-PhenoGraft** | 0.3112 | **0.2302** | 7.21 | 4.77 |
| − Asymmetric attention | 0.3111 | 0.2181 | 7.00 | 4.63 |
| − HSIC shared-private loss | 0.3022 | 0.2046 | 7.18 | 4.70 |
| Centralized (1 client) | 0.3074 | 0.2400 | 6.89 | 4.59 |
| − MRI modality | 0.3135 | 0.2099 | 6.98 | 4.65 |
| − PET/DaTScan modality | 0.2668 | 0.1890 | 7.01 | 4.68 |
| − Genetics modality | 0.3010 | 0.2016 | 7.20 | 4.72 |
| Clinical only | 0.2772 | 0.1962 | 6.93 | 4.68 |

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
