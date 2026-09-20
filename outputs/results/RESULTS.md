# Fed-PhenoGraft — Results Summary

Regression target: **Δ UPDRS-III (BL → Year 2)**. Federated client partition: **dirichlet**.

Subject-level split: **2430 train / 521 validation / 521 test**. The test set was not used for model or round selection.

## Evaluation Protocol

The experiment follows **Option B**:

- Development stage: 2430 training subjects with 521 validation subjects.
- Validation was used only to select the number of federated training rounds.
- Final stage: all 2951 development subjects were used for training.
- A fresh model was initialized for the final fit.
- Final preprocessing was fitted on train + validation.
- The held-out test set contains 521 subjects and was evaluated only after final training.

## Fed-PhenoGraft (held-out test)

| Task | Metric | Value |
|------|--------|-------|
| Progression regression | CCC | **0.3186** [95% CI 0.2204, 0.4014] |
| Progression regression | RMSE (UPDRS-III pts) | 8.08 |
| Progression regression | MAE (UPDRS-III pts) | 5.62 |
| Progression regression | R² | -0.2298 |
| Progression regression | Pearson r | 0.3219 |

### Final development-set diagnostics

The following values describe the final model on the development population after train + validation were combined. They are **not validation metrics** and were not used for model selection.

- Training-subset CCC: 0.4241
- Final development CCC: 0.4297
- Selected final rounds: 28

## Baseline comparison — 12 models

All baselines were fitted on the same train + validation development population and evaluated on the same held-out test set.

| Model | CV CCC (mean ± std) | Test CCC | Test RMSE | Test MAE | Test R² | Test Pearson |
|-------|---------------------|----------|-----------|----------|---------|--------------|
| linear | 0.1907 ± 0.0973 | 0.2026 | 6.98 | 4.78 | 0.0826 | 0.3024 |
| ridge | 0.2345 ± 0.0221 | 0.2038 | 6.97 | 4.78 | 0.0863 | 0.3068 |
| lasso | 0.2219 ± 0.0167 | 0.1911 | 6.89 | 4.70 | 0.1074 | 0.3322 |
| elastic_net | 0.2106 ± 0.0158 | 0.1822 | 6.90 | 4.70 | 0.1030 | 0.3258 |
| svm | 0.0989 ± 0.0141 | 0.0944 | 7.08 | 4.68 | 0.0561 | 0.3032 |
| knn | 0.0836 ± 0.0125 | 0.0619 | 7.40 | 4.92 | -0.0300 | 0.1093 |
| random_forest | 0.1226 ± 0.0177 | 0.1208 | 6.97 | 4.68 | 0.0845 | 0.3198 |
| extra_trees | 0.0793 ± 0.0111 | 0.0662 | 7.10 | 4.72 | 0.0517 | 0.2926 |
| gradient_boosting | 0.2716 ± 0.0262 | 0.2362 | 6.90 | 4.65 | 0.1050 | 0.3369 |
| mlp | 0.2581 ± 0.0338 | 0.2441 | 6.90 | 4.64 | 0.1040 | 0.3412 |
| xgboost | 0.2708 ± 0.0294 | 0.2430 | 6.93 | 4.68 | 0.0974 | 0.3337 |
| lightgbm | 0.2886 ± 0.0279 | 0.2664 | 6.90 | 4.61 | 0.1043 | 0.3493 |
| **Fed-PhenoGraft** | selected round 28 | **0.3186** | 8.08 | 5.62 | -0.2298 | 0.3219 |

## Statistical analysis

- Nonparametric bootstrap 95% CI (n=1000 resamples) for test CCC: **[0.2204, 0.4014]**.
- Paired permutation test against the strongest baseline (**lightgbm**): ΔCCC +0.0521 [95% CI -0.0158, 0.1267], p = 0.1978 — not statistically significant at α = 0.05.
- Across **3 independent training seeds**: final held-out test CCC 0.3275 ± 0.0074. Primary seed = 42; test performance was not used for seed selection.

## Ablation study

| Variant | Val/Development CCC | Test CCC | Test RMSE | Test MAE |
|---------|----------------------|----------|-----------|----------|
| **Full Fed-PhenoGraft** | — | **0.3186** | 8.08 | 5.62 |
| − Asymmetric attention | 0.4137 | 0.2903 | 7.67 | 5.33 |
| − HSIC shared-private loss | 0.4144 | 0.3009 | 8.11 | 5.71 |
| Centralized (1 client) | 0.4336 | 0.3288 | 7.94 | 5.45 |
| − MRI modality | 0.4088 | 0.2939 | 8.15 | 5.70 |
| − PET/DaTScan modality | 0.3848 | 0.2770 | 8.32 | 5.81 |
| − Genetics modality | 0.4123 | 0.2951 | 8.15 | 5.71 |
| Clinical only | 0.3767 | 0.2785 | 8.24 | 5.78 |

## Figures

| Figure | File |
|--------|------|
| Model comparison | `outputs/figures/model_comparison.png` |
| Development training curve | `outputs/figures/training_curve.png` |
| Predicted vs actual | `outputs/figures/pred_vs_actual.png` |
| Confusion matrix (PD vs HC) | `outputs/figures/confusion_matrix.png` |
| Attention maps | `outputs/figures/attention_maps.png` |
| Missing-modality robustness | `outputs/figures/modality_robustness.png` |
| Feature attribution (IG) | `outputs/figures/global_feature_importance.png` |
| Counterfactual gene analysis | `outputs/figures/counterfactual_genes.png` |
| Ablation study | `outputs/figures/ablation_study.png` |
