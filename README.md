# Fed-PhenoGraft: Federated Phenotype-Guided Multimodal Parkinson's Disease Prediction

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

A federated learning framework combining **Phenotype-Guided Asymmetric Cross-Modal Attention** with **Shared-Private Latent Decomposition** for predicting Parkinson's Disease progression (UPDRS-III at Year 2) using multimodal PPMI data.

**Status: execution-ready.** The pipeline runs end-to-end on real PPMI tabular data with a leak-free evaluation protocol (see [Anti-Leakage & Generalization Safeguards](#anti-leakage--generalization-safeguards)).

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Verified Results](#verified-results)
- [Anti-Leakage & Generalization Safeguards](#anti-leakage--generalization-safeguards)
- [Project Structure](#project-structure)
- [Step-by-Step Execution Guide](#-step-by-step-execution-guide)
- [Dataset Files to Download](#dataset-files-to-download)
- [What Needs to Be Done](#what-needs-to-be-done)
- [Technical Details](#technical-details)

---

## Architecture Overview

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Clinical   │     │     MRI     │     │  PET/DaTScan│     │   Genetic   │
│   (CSV)     │     │  (NIfTI)    │     │    (CSV)    │     │    (CSV)    │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
  ┌─────────┐      ┌──────────────┐     ┌─────────────┐     ┌─────────────┐
  │ Clinical │     │  Schaefer    │     │    SBR +    │     │  Carrier    │
  │ Loader   │     │  Parcellation│     │  Asymmetry  │     │  Encoding   │
  └────┬─────┘     └──────┬───────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
  ┌─────────┐     ┌────────┴────────────────────┴────────────────┐
  │  Query  │────▶│     Phenotype-Guided Asymmetric Attention    │
  │(Phenotype)    │     (Clinical queries MRI, PET, Genetic)     │
  └─────────┘     └───────────────────┬──────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │  Shared-Private Latent             │
                    │  Decomposition (HSIC Orthogonality)│
                    └─────────────────┬─────────────────┘
                                      │
                              ┌───────▼───────┐
                              │  FedAvg Loop  │     ← Simulates N clinical sites
                              │  (Privacy)    │
                              └───────┬───────┘
                                      │
                              ┌───────▼───────┐
                              │  Prediction   │
                              │  (UPDRS-III)  │
                              └───────────────┘
```

---

## Verified Results

**The pipeline now targets true progression: ΔUPDRS-III (Year 2 − baseline)**, set by `target.mode: delta` in `config.yaml`. The earlier headline numbers (test CCC ≈ 0.82) were for the *absolute* Year-2 score, which is dominated by baseline-score autocorrelation — a reviewer trap. Predicting the *change* is the honest, clinically meaningful task, and all numbers below are on that harder target. Set `target.mode: absolute` to reproduce the old setup.

Latest full run on real PPMI data — clinical, **real DaTScan/PET SBR**, and genetics (3,472 aligned subjects with valid baseline AND Year-2 UPDRS-III; MRI on synthetic fallback until NIfTIs are downloaded). Protocol: subject-level 70/15/15 split, **Dirichlet(α=0.5) non-IID client partition**, **3 independent training seeds**, bootstrap 95% CIs, paired significance test vs the strongest baseline, and a **retrained 8-variant ablation study**. Runtime ≈ 8 min on CPU.

Fed-PhenoGraft held-out test (multitask, primary = best-validation seed):
- **Progression regression:** CCC 0.189 [95% CI 0.108, 0.266] · RMSE 7.30 · MAE 4.91 (ΔUPDRS-III points); across 3 seeds: CCC 0.159 ± 0.022
- **PD vs HC classification:** ROC-AUC 0.976 · Accuracy 0.906
- Train-val CCC gap +0.060 → no overfitting signal.
- Paired bootstrap vs the strongest baseline (LightGBM, test CCC 0.230): ΔCCC −0.041, p = 0.857 → **currently statistically indistinguishable from the best baseline** on progression. This is the honest starting point; hyperparameter tuning (against the validation set only) is the open lever.

Ablation study (each variant retrained under the same protocol, held-out test CCC):

| Variant | Test CCC | Reading |
|---------|----------|---------|
| **Full Fed-PhenoGraft** | **0.189** | reference |
| − Asymmetric attention | 0.169 | attention contributes |
| − HSIC shared-private loss | 0.155 | HSIC contributes |
| Centralized (1 client) | 0.124 | federation is not hurting |
| − PET/DaTScan | 0.122 | **PET is the most valuable modality** |
| − Genetics | 0.155 | genetics contributes |
| − MRI (synthetic) | 0.210 | **removing the synthetic-MRI branch helps → real MRI data is needed** |
| Clinical only | 0.199 | multimodal gain must come from real imaging |

The full statistical detail (CIs for all metrics, per-seed values, significance test) is in `outputs/results/final_metrics.json` and rendered in `outputs/results/RESULTS.md`. Every run also writes the full figure suite in `outputs/figures/` (now including `ablation_study.png`). Regenerate figures without retraining via `python scripts/generate_results.py`.

---

## Anti-Leakage & Generalization Safeguards

These are engineered into the pipeline — not conventions you have to remember:

1. **Subject-level splits.** Every PATNO lands in exactly one of train/val/test (70/15/15, stratified on diagnosis). No subject identity crosses partitions ([src/data/preprocessing.py](src/data/preprocessing.py)).
2. **Train-only statistics.** Imputers and scalers are fit on the *training* subjects only, then applied unchanged to val/test. Clinical columns unusable in the training split are dropped based on train data alone.
3. **No target imputation.** Subjects without a real Year-2 UPDRS-III score are dropped, never median-filled (imputed labels are label noise and their statistics leak across splits).
4. **Validation-driven early stopping.** The federated loop monitors *validation* CCC each round, stops after `early_stopping_patience` rounds without improvement, and restores the best-round weights. The **test set is evaluated exactly once**, after training ends — it never influences training or model selection.
5. **Per-fold baseline pipelines.** Baselines use `sklearn.Pipeline(imputer → scaler → model)` so preprocessing is re-fit inside every CV fold; models are cloned per fold. CV runs on train+val; the test set is scored once with a pipeline fit on train+val.
6. **Explicit over/underfitting diagnostics.** Each round logs train CCC, val CCC, and their gap; the final report warns if the gap exceeds +0.15 (overfitting) or train CCC is below 0.2 (underfitting).
7. **Regularization defaults.** AdamW weight decay (1e-4), dropout 0.3, gradient clipping (1.0), sample-size-weighted FedAvg, and regularized baseline defaults (depth limits, subsampling, L2, MLP early stopping).
8. **Correct target construction.** UPDRS part scores use PPMI's official total columns (`NP3TOT`, etc.); item sums exclude totals (no double counting) and exclude Hoehn & Yahr stage.
9. **Missing-modality semantics preserved.** All-zero (missing) modality rows stay all-zero after scaling, so learned mask tokens keep working.
10. **Full reproducibility.** `seed_everything` seeds Python, NumPy, and PyTorch (deterministic cuDNN); the split seed is in `config.yaml`.

**Resolved caveat (baseline vs. target autocorrelation):** with `target.mode: delta` (the default) the model predicts the *change* in UPDRS-III, so the baseline score can no longer inflate the metric through autocorrelation — baseline UPDRS-III remains a legitimate prognostic *feature*. Metrics on the delta target are much lower than on the absolute target; that is the honest difficulty of progression prediction, not a regression in the pipeline.

**Statistical rigor (built in):** every run reports bootstrap 95% CIs on all test metrics (n=1000 resamples), trains the federated model with 3 independent seeds (mean ± std; the primary model is chosen by *validation* CCC only), and runs a paired bootstrap significance test against the strongest baseline on the same test subjects.

---

## Project Structure

```
PARK-IN-SON/
├── config.yaml                   # All hyperparameters, split fractions, and paths
├── requirements.txt              # Python dependencies (incl. captum)
├── DATASET_DOWNLOAD.md           # Full PPMI download guide (filenames, IDA locations)
├── .env.example                  # Credential template
│
├── scripts/
│   ├── download_ppmi_data.py     # PPMI data download script
│   └── generate_results.py       # Regenerate figures + RESULTS.md from saved metrics
│
├── src/
│   ├── main.py                   # End-to-end pipeline entry point
│   │
│   ├── data/
│   │   ├── clinical_loader.py    # UPDRS + Demographics + Age_at_visit + MoCA
│   │   ├── mri_pipeline.py       # NIfTI → nilearn Schaefer ROI extraction
│   │   ├── pet_loader.py         # DaTScan SBR CSV + asymmetry features
│   │   ├── genetic_loader.py     # Mutation carrier status encoding
│   │   ├── data_builder.py       # Unified orchestrator (returns RAW features)
│   │   ├── preprocessing.py      # Leak-free split + train-only imputer/scalers
│   │   └── dataset.py            # PyTorch Dataset with missing-modality masks
│   │
│   ├── models/
│   │   ├── fed_phenograft.py     # Main model (Attention + HSIC + MC Dropout)
│   │   ├── attention.py          # Asymmetric Cross-Attention layer
│   │   └── hsic.py               # HSIC independence loss
│   │
│   ├── federated/
│   │   └── fedavg_orchestrator.py # Weighted FedAvg + early stopping + best-weight restore
│   │
│   ├── baselines/
│   │   ├── models.py             # 12-model baseline suite (regularized defaults)
│   │   └── runner.py             # Per-fold pipeline CV + one-shot test evaluation
│   │
│   └── evaluation/
│       ├── metrics.py            # CCC (NaN-safe), RMSE, MAE, R², Pearson r
│       ├── xai.py                # Attention, IG, robustness, counterfactuals
│       └── results_report.py     # Comparison chart, training curve, RESULTS.md
│
├── data/
│   └── raw/                      # ← Downloaded CSVs go here (see table below)
│       └── mri/                  # ← Optional NIfTI scans: {PATNO}/T1w.nii.gz
│
└── outputs/
    ├── figures/                  # attention_maps.png
    ├── models/                   # fed_phenograft_best.pt (best-val-round weights)
    └── results/                  # final_metrics.json (all metrics + history)
```

---

## 🚀 Step-by-Step Execution Guide

### Step 1: Environment Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
pip install nibabel nilearn   # only needed when enabling real MRI
```

### Step 2: Get the PPMI Data (automatic first, manual fallback below)

The tabular CSVs are already expected under `data/raw/` — see [Dataset Files to Download](#dataset-files-to-download) for the exact filenames, or **[DATASET_DOWNLOAD.md](DATASET_DOWNLOAD.md)** for the full step-by-step guide (access request, renaming rules, MRI search filters, verification). If they are missing, the pipeline logs a warning and falls back to synthetic data so development is never blocked (results are then meaningless for science, but the code path is identical).

**Automatic download:** copy `.env.example` → `.env`, fill in `PPMI_USER`/`PPMI_PASSWORD`, then:

```bash
python scripts/download_ppmi_data.py
```

#### ⚠️ If the Automatic Download Doesn't Work → Manual Download

The automatic script needs programmatic IDA access, which many accounts don't have (it exits cleanly with an authorization error in that case). Downloading manually is fully supported and just as good:

1. **Log in** at [https://ida.loni.usc.edu](https://ida.loni.usc.edu) (register first and apply for PPMI Data Access — approval takes 1–3 business days; use an institutional email if possible).
2. Go to **PPMI → Download → Study Data**.
3. Download these **10 CSV files** (file type: comma-separated `.csv`, the IDA default) from the sections shown, and place them in **`data/raw/`**:

   | # | Save as (in `data/raw/`) | IDA Study Data section | Accepted naming variants |
   |---|--------------------------|------------------------|--------------------------|
   | 1 | `MDS_UPDRS_Part_I.csv` | Motor Assessments → MDS-UPDRS | `MDS-UPDRS_Part_I.csv` |
   | 2 | `MDS_UPDRS_Part_II.csv` | Motor Assessments → MDS-UPDRS | `MDS-UPDRS_Part_II.csv` |
   | 3 | `MDS_UPDRS_Part_III.csv` | Motor Assessments → MDS-UPDRS | `MDS-UPDRS_Part_III.csv` |
   | 4 | `MDS_UPDRS_Part_IV.csv` | Motor Assessments → MDS-UPDRS | `MDS-UPDRS_Part_IV.csv` |
   | 5 | `Demographics.csv` | Subject Characteristics | `Screening___Demographics.csv` |
   | 6 | `Age_at_visit.csv` | Subject Characteristics | — |
   | 7 | `MoCA.csv` | Non-motor Assessments | `Montreal_Cognitive_Assessment__MoCA_.csv` |
   | 8 | `Patient_Status.csv` | Subject Characteristics | `Participant_Status.csv` |
   | 9 | `DATScan_Analysis.csv` | Imaging → DaTScan | `DaTscan_Analysis.csv`, `DATscan_Analysis.csv` |
   | 10 | `Genetic_Testing_Results.csv` | Biospecimen → Genetics | `Genetic_Results.csv`, `Genetics.csv` |

4. **Rename rule:** IDA appends a date suffix on export (e.g. `MDS-UPDRS_Part_III_23Aug2026.csv`) — strip it so the filename matches the table.
5. **MRI scans (optional, file type: NIfTI `.nii`/`.nii.gz`):** IDA → PPMI → Search → Advanced Image Search → Modality = MRI, T1-weighted anatomical (MPRAGE / SAG T1), Visit = Baseline → add to collection → download as **NIfTI** → place as `data/raw/mri/{PATNO}/T1w.nii.gz` → set `mri.use_real_mri: true` in `config.yaml`.
6. **Verify:** run `python src/main.py` and check the "Dataset Summary" log block (~3,600 patients expected). Any "CSV not found, skipping" line means a file is missing or misnamed.

### Step 3: Run Everything — Baselines → Federated Training → Testing

```bash
python src/main.py
```

One command runs the entire experiment. What happens, phase by phase:

| Phase | What runs | Leak protection |
|-------|-----------|-----------------|
| 1. Raw data load | Clinical + MRI + PET (DaTScan) + genetics merged on `PATNO`; subjects without a real Year-2 UPDRS-III score are dropped | No target imputation |
| 2. Split | Subject-level 70/15/15 train/val/test, stratified on PD/HC diagnosis | Each subject in exactly one partition |
| 3. Preprocessing | Median imputer + per-modality scalers **fit on train only**, applied to val/test | No test statistics leak |
| 4. **12 baseline models** | linear, ridge, lasso, elastic_net, svm (RBF), knn, random_forest, extra_trees, gradient_boosting, xgboost, lightgbm, mlp — each in a per-fold `Pipeline(imputer → scaler → model)`, 5-fold CV on train+val, then refit and scored **once** on test | Preprocessing re-fit inside every fold; models cloned per fold |
| 5. Federated training | Fed-PhenoGraft trained via FedAvg over 4 simulated sites, multitask loss (UPDRS-III MSE + PD/HC BCE + HSIC), validation-CCC early stopping, best-round weights restored | Test set never seen during training |
| 6. **Testing** | Held-out test set evaluated **exactly once**: regression CCC / RMSE / MAE / R² / Pearson r + classification ROC-AUC / Accuracy / F1, plus train-vs-val gap check for over/underfitting | One-shot evaluation |
| 7. Explainability + report | Attention maps, Integrated Gradients, missing-modality stress test, counterfactual gene flips, pred-vs-actual, confusion matrix, model-comparison chart, training curve → `RESULTS.md` | Run on test predictions only |

Typical runtime on CPU: ~5–10 minutes for the tabular dataset (~3,600 subjects).

### Step 4: Outputs

- `outputs/results/final_metrics.json` — every metric for all 12 baselines + Fed-PhenoGraft, split sizes, per-round history, config snapshot
- `outputs/results/RESULTS.md` — presentation-ready tables (auto-generated)
- `outputs/models/fed_phenograft_best.pt` — best-validation-round weights
- `outputs/figures/` — 8 figures: `model_comparison.png`, `training_curve.png`, `pred_vs_actual.png`, `confusion_matrix.png`, `attention_maps.png`, `modality_robustness.png`, `global_feature_importance.png`, `counterfactual_genes.png`

Re-generate figures and `RESULTS.md` later without retraining:

```bash
python scripts/generate_results.py
```

---

## Dataset Files to Download

Download each file from the IDA LONI portal (**PPMI → Download → Study Data**) and place it in `data/raw/` under the **expected filename** below. IDA exports often append a date suffix (e.g., `_23Aug2026.csv`) — **rename the file to strip it**. The loaders also accept the common PPMI naming variants listed in parentheses.

| Expected filename in `data/raw/` | IDA Study Data section | Contents / role |
|----------------------------------|------------------------|-----------------|
| `MDS_UPDRS_Part_I.csv` | Motor / MDS-UPDRS | Part I non-motor aspects (feature @ BL) |
| `MDS_UPDRS_Part_II.csv` | Motor / MDS-UPDRS | Part II motor aspects of daily living (feature @ BL) |
| `MDS_UPDRS_Part_III.csv` | Motor / MDS-UPDRS | Part III motor exam — **feature @ BL and regression target @ V04** (uses `NP3TOT`) |
| `MDS_UPDRS_Part_IV.csv` | Motor / MDS-UPDRS | Part IV motor complications (feature @ BL) |
| `Demographics.csv` (or `Screening___Demographics.csv`) | Subject Characteristics | Sex, birth date |
| `Age_at_visit.csv` | Subject Characteristics | Exact age at each visit (preferred age source) |
| `MoCA.csv` (or `Montreal_Cognitive_Assessment__MoCA_.csv`) | Non-motor / Neuropsychological | Cognitive score (feature @ BL) |
| `Patient_Status.csv` (or `Participant_Status.csv`) | Subject Characteristics | PD vs HC enrollment label (stratification + future classification head) |
| `DATScan_Analysis.csv` (or `DaTscan_Analysis.csv`) | Imaging / DaTScan | Striatal binding ratios: caudate/putamen L+R |
| `Genetic_Testing_Results.csv` (or `Genetics.csv`) | Biospecimen / Genetics | LRRK2, GBA, SNCA, PINK1, PRKN, APOE carrier status |

**MRI (optional, for the imaging branch):**
- IDA → PPMI → **Image Collections** → search T1-weighted anatomical (MPRAGE/SPGR), download as NIfTI.
- Place as `data/raw/mri/{PATNO}/T1w.nii.gz` (any `.nii`/`.nii.gz` inside the PATNO folder is picked up).
- Then set `mri.use_real_mri: true` in `config.yaml`. Until then, MRI uses the synthetic fallback and the model relies on its learned mask tokens.

---

## What Needs to Be Done

### Completed ✅

- [x] Real CSV loaders for Clinical (UPDRS I–IV, demographics, age-at-visit, MoCA), PET (DaTScan SBR), Genetics
- [x] Correct UPDRS target construction from official `NP*TOT` columns (no double counting, no H&Y contamination)
- [x] **Leak-free evaluation protocol** — subject-level stratified splits, train-only preprocessing, no target imputation
- [x] **Overfitting controls** — validation-based early stopping with best-weight restore, AdamW weight decay, dropout, gradient clipping, train/val gap diagnostics
- [x] Sample-size-weighted FedAvg across simulated sites
- [x] **12 baseline models** (linear, ridge, lasso, elastic_net, svm, knn, random_forest, extra_trees, gradient_boosting, xgboost, lightgbm, mlp) with per-fold `Pipeline` preprocessing and one-shot test scoring
- [x] Fed-PhenoGraft core architecture (asymmetric cross-attention, HSIC shared-private decomposition, mask tokens, MC dropout)
- [x] **Multitask heads** — UPDRS-III severity regression + PD vs HC classification (Test AUC 0.953)
- [x] **Full XAI suite** — attention maps, Integrated Gradients attribution, missing-modality robustness stress test, and counterfactual gene "what-if" analysis
- [x] **Presentation results generation** — `RESULTS.md` + 8 figures produced automatically each run; `scripts/generate_results.py` regenerates without retraining
- [x] Real DaTScan/PET SBR data wired through the architecture (handles current PPMI `DATSCAN_*` column names and the `SC` screening visit coding)
- [x] Expanded metric suite — regression: CCC, RMSE, MAE, R², Pearson r; classification: ROC-AUC, Accuracy, F1
- [x] Metrics JSON export; full-seed reproducibility
- [x] Verified end-to-end on real PPMI data (3,513 subjects, real clinical + PET + genetics) — Test CCC 0.825 / AUC 0.976, better than all 12 baselines

### Newly Completed (publication-readiness pass) ✅

- [x] **ΔUPDRS-III progression target** (`target.mode: delta`) — removes baseline autocorrelation; the model is now judged on true progression signal
- [x] **Non-IID federated partition** — Dirichlet(α) label-skew clients by default (`training.partition: dirichlet`); real-site partitioning auto-activates when a PPMI Center-Subject list CSV is placed in `data/raw/` and `training.partition: site` is set
- [x] **Ablation suite** — 8 retrained variants (modality drops, attention off, HSIC off, centralized) with table + `ablation_study.png` each run
- [x] **Statistical rigor** — bootstrap 95% CIs, 3-seed mean ± std (validation-only model selection), paired bootstrap significance test vs strongest baseline
- [x] **XAI relabeled for the progression target** (pred-vs-actual, counterfactual gene analysis)

### Pending Work 🔧

| # | Task | Priority | Details |
|---|------|----------|---------|
| 1 | **Download MRI NIfTIs** | 🔴 Critical | The ablation study now *proves* this matters: removing the synthetic-MRI branch improves test CCC (0.210 vs 0.189), so real T1w data is the main open lever. See [DATASET_DOWNLOAD.md](DATASET_DOWNLOAD.md), then set `mri.use_real_mri: true` |
| 2 | **Hyperparameter Tuning** | 🔴 High | On the delta target Fed-PhenoGraft (0.189) does not yet beat LightGBM (0.230). Tune `model.embed_dim`, `num_heads`, `training.lr`, `local_epochs`, `hsic_weight`, `cls_weight` — against the **validation** set only; never touch test |
| 3 | **Center-Subject list CSV** | 🟡 Medium | Download it from IDA (Subject Characteristics) into `data/raw/` and set `training.partition: site` to replace simulated heterogeneity with real acquisition sites |
| 4 | **Differential Privacy** | 🟢 Low | DP-SGD noise in client updates for formal privacy guarantees |
| 5 | **Longitudinal Modeling** | 🟢 Low | Multi-visit trajectories (BL→V04→V06→V08) instead of cross-sectional |
| 6 | **Unit Tests** | 🟢 Low | Refresh `tests/` (they reference an older `ModelFactory` API) + add leakage regression tests |

---

## Technical Details

### Key Metric: CCC (Concordance Correlation Coefficient)

We use CCC instead of R² because R² ignores systematic bias — a model that consistently over-predicts by 10 points could still score R²=0.95. CCC penalizes deviations from the line of perfect agreement. Our implementation is NaN-safe: constant predictions score 0, never NaN.

### Federated Learning Strategy

The FedAvg loop simulates `training.num_clients` isolated clinical sites:
1. The global model is broadcast to all clients.
2. Each client trains locally for `local_epochs` (AdamW, weight decay, gradient clipping).
3. Client weights are averaged back **weighted by client sample count**.
4. Validation CCC is computed; training stops early after `early_stopping_patience` stagnant rounds and the best round's weights are restored.
5. The held-out test set is scored once, after training.

### Missing Modality Handling

When a modality is unavailable for a patient, its feature vector is all zeros. `FederatedPPMIDataset` detects this and sets `{modality}_mask = 1`; the model substitutes a **learned mask token** instead of the zero embedding. The train-only scalers preserve all-zero rows so this contract survives preprocessing.

### HSIC Orthogonality

Each non-clinical modality passes through dual encoders: Shared (disease-relevant) and Private (modality-specific noise). The HSIC loss forces statistical independence between these representations using an RBF kernel-based criterion.

---

## Team

- **Saurav Kumar Gupta** (23BCE2336)
- **Amit Adhikari** (23BCE2327)
- **Shreeyam Acharya** (23BCE2330)

**Faculty Guide**: Dhivyaa CR (20701)

---

## References

1. Vamvakas et al., "Prediction of impulse control disorders in Parkinson's disease," npj Parkinson's Disease, 2026.
2. Akram & C. K., "Enhancing Parkinson's Disease Staging: Integrative Deep Learning for Multimodal Feature Selection," J. Molecular Neuroscience, 2026.
3. Awasthi et al., "HyCoSwin-PD: Explainable Hybrid Framework for PD Detection from Neuroimaging," MethodsX, 2026.
