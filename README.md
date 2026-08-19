# Fed-PhenoGraft: Federated Phenotype-Guided Multimodal Parkinson's Disease Prediction

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

A federated learning framework combining **Phenotype-Guided Asymmetric Cross-Modal Attention** with **Shared-Private Latent Decomposition** for predicting Parkinson's Disease progression using multimodal PPMI data.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Data Preparation](#data-preparation)
- [Running the Pipeline](#running-the-pipeline)
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

## Project Structure

```
PARK-IN-SON/
├── config.yaml                   # All hyperparameters and paths
├── requirements.txt              # Python dependencies
├── .env.example                  # Credential template
│
├── scripts/
│   └── download_ppmi_data.py     # PPMI data download script
│
├── src/
│   ├── main.py                   # End-to-end pipeline entry point
│   │
│   ├── data/                     # Data Loading & Preprocessing
│   │   ├── clinical_loader.py    # UPDRS + Demographics + MoCA CSV loader
│   │   ├── mri_pipeline.py       # NIfTI → nilearn Schaefer ROI extraction
│   │   ├── pet_loader.py         # DaTScan SBR CSV + asymmetry features
│   │   ├── genetic_loader.py     # Mutation carrier status encoding
│   │   ├── data_builder.py       # Unified orchestrator (all modalities)
│   │   ├── dataset.py            # PyTorch Dataset with mask tokens
│   │   └── preprocessors.py      # Sklearn-based scalers (legacy/fallback)
│   │
│   ├── models/                   # Core Architecture
│   │   ├── fed_phenograft.py     # Main model (Attention + HSIC + MC Dropout)
│   │   ├── attention.py          # Asymmetric Cross-Attention layer
│   │   └── hsic.py               # HSIC independence loss
│   │
│   ├── federated/                # Federated Learning
│   │   └── fedavg_orchestrator.py # FedAvg training loop
│   │
│   ├── baselines/                # Baseline Comparisons
│   │   ├── models.py             # RF, XGBoost, SVM, Ridge, MLP
│   │   └── runner.py             # Cross-validated baseline evaluation
│   │
│   └── evaluation/               # Evaluation & Explainability
│       ├── metrics.py            # CCC (Concordance Correlation Coefficient)
│       └── xai.py                # Attention maps + Integrated Gradients
│
├── data/
│   ├── raw/                      # ← Place downloaded CSVs here
│   │   ├── MDS_UPDRS_Part_I.csv
│   │   ├── MDS_UPDRS_Part_II.csv
│   │   ├── MDS_UPDRS_Part_III.csv
│   │   ├── MDS_UPDRS_Part_IV.csv
│   │   ├── Demographics.csv
│   │   ├── MoCA.csv
│   │   ├── DATScan_Analysis.csv
│   │   ├── Genetic_Testing_Results.csv
│   │   ├── Patient_Status.csv
│   │   └── mri/                  # ← Place NIfTI scans here
│   │       ├── 3001/             #    Organized by PATNO
│   │       │   └── T1w.nii.gz
│   │       ├── 3002/
│   │       │   └── T1w.nii.gz
│   │       └── ...
│   ├── processed/
│   └── embeddings/
│
└── outputs/
    ├── figures/                  # Attention maps, plots
    ├── models/                   # Saved model weights (.pt)
    └── results/                  # Evaluation CSVs
```

---

## 🚀 Step-by-Step Execution Guide

### Step 1: Environment Setup
First, prepare your Python environment and install the required machine learning dependencies.

```bash
# Create and activate a Virtual Environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# Install standard dependencies
pip install -r requirements.txt

# Install deep learning and neuroimaging packages
pip install nibabel nilearn captum
```

### Step 2: Securing the PPMI Dataset
The full pipeline relies on tabular (CSVs) and imaging (NIfTI) datasets from the Parkinson's Progression Markers Initiative (PPMI).

#### Option A: Automated Download (Requires API Access)
If you have your IDA LONI credentials properly set up for programmatic access:
1. Copy the environment template: `copy .env.example .env` (Windows) or `cp .env.example .env` (Linux/Mac)
2. Add your credentials to the `.env` file (`PPMI_USER` and `PPMI_PASSWORD`).
3. Run the automated fetch script:
   ```bash
   python scripts/download_ppmi_data.py
   ```
   *Note: If your account lacks direct API integration permissions, this script will cleanly exit and recommend Option B.*

#### Option B: Manual Download Fallback (Most Reliable)
If the automation script fails or exits, you must manually download the core files. Navigate to [IDA LONI](https://ida.loni.usc.edu), log in, and download the following files precisely. Place them directly inside the `data/raw/` directory.

**Tabular Data (CSVs)**
| Exact File Name | Description | Where to Find on IDA |
|-----------------|-------------|---------------------|
| `MDS_UPDRS_Part_I.csv` | Motor Symptoms Part I | Study Data → UPDRS |
| `MDS_UPDRS_Part_II.csv` | Motor Symptoms Part II | Study Data → UPDRS |
| `MDS_UPDRS_Part_III.csv` | Motor Examination (**target**) | Study Data → UPDRS |
| `MDS_UPDRS_Part_IV.csv` | Motor Complications | Study Data → UPDRS |
| `Demographics.csv` | Age, gender, education | Study Data → Subject Characteristics |
| `MoCA.csv` | Cognitive Assessment | Study Data → Neuropsychological |
| `DATScan_Analysis.csv` | Striatal Binding Ratios | Study Data → DaTScan Imaging |
| `Genetic_Testing_Results.csv` | Mutation status | Study Data → Biospecimen |
| `Patient_Status.csv` | PD vs HC Labels | Study Data → Subject Characteristics |

**MRI Scans (NIfTI - Optional)**
* Structural T1 `.nii.gz` files from (Image Collections → T1-Anatomical).
* Place them formatted as `data/raw/mri/{PATNO}/xxx.nii.gz`
* *Fallback: If you skip downloading heavy MRIs, ensure `mri.use_real_mri: false` is set in `config.yaml`. The pipeline gracefully handles missing modalities via Learned Mask Tokens.*

### Step 3: Running the Model Training
Once the datasets are available in `data/raw/` (and regardless of whether you have complete modalities), you can seamlessly kick off the federated pipeline.

**Run the Full Pipeline (Fed-PhenoGraft + Baselines + XAI):**
```bash
python src/main.py
```
This execution will logically progress through:
1. **Data Loading:** Binding arrays and resolving missing files natively.
2. **Baselines:** Validating traditional ML splits (RF, XGBoost, etc).
3. **Federated Orchestrator:** Simulating federated averaging instances.
4. **Explainability Extraction:** Unraveling multimodal attention.

**Run ONLY baselines:**
```bash
python src/baselines/runner.py
```

### Step 4: Tracking Outputs
Once `main.py` finishes, all assets are automatically exported to the securely-generated `outputs/` folder:
- **`outputs/models/`**: Final checkpoint files (e.g., `fed_phenograft_final.pt`)
- **`outputs/figures/`**: Visual Interpretability graphs (`attention_maps.png`)

---

## What Needs to Be Done

### Completed ✅

- [x] Data pipeline with real CSV loaders (Clinical, PET, Genetic)
- [x] MRI NIfTI processing pipeline (nilearn Schaefer parcellation)
- [x] Synthetic fallback for each modality when data is unavailable
- [x] Baseline models (RF, XGBoost, SVM, Ridge, MLP) with CCC metric
- [x] Fed-PhenoGraft core architecture:
  - Phenotype-Guided Asymmetric Cross-Attention
  - Shared-Private Latent Decomposition (HSIC loss)
  - Learned Mask Tokens for missing modalities
  - Monte Carlo Dropout for uncertainty
- [x] Federated Learning orchestration (FedAvg)
- [x] XAI: Attention maps + Integrated Gradients (captum)

### Pending Work 🔧

| # | Task | Priority | Details |
|---|------|----------|---------|
| 1 | **Download PPMI Data** | 🔴 Critical | Register at https://ida.loni.usc.edu, download CSVs and MRI NIfTIs listed above |
| 2 | **Enable Real MRI** | 🔴 Critical | Set `mri.use_real_mri: true` in `config.yaml` after placing NIfTI files |
| 3 | **Hyperparameter Tuning** | 🟡 Medium | Tune embed_dim, num_heads, local_epochs, num_rounds on real data |
| 4 | **Counterfactual XAI** | 🟡 Medium | Implement "what-if" analysis (e.g., "what if LRRK2 status were negative?") |
| 5 | **Classification Head** | 🟡 Medium | Add PD vs HC binary classification alongside regression |
| 6 | **Site-Based FL Split** | 🟢 Low | Partition clients by actual PPMI acquisition site instead of random |
| 7 | **Differential Privacy** | 🟢 Low | Add DP-SGD noise to FedAvg for formal privacy guarantees |
| 8 | **Longitudinal Modeling** | 🟢 Low | Track UPDRS progression across multiple visits (BL→V04→V06→V08) |
| 9 | **Unit Tests** | 🟢 Low | Expand pytest coverage for loaders, HSIC stability, attention shapes |

---

## Technical Details

### Key Metric: CCC (Concordance Correlation Coefficient)

We use CCC instead of R² because R² ignores systematic bias — a model that consistently over-predicts by 10 points could still score R²=0.95. CCC penalizes deviations from the line of perfect agreement.

### Federated Learning Strategy

The FedAvg loop simulates `N` isolated clinical sites:
1. Global model is broadcast to all clients
2. Each client trains locally for `local_epochs`
3. Client weights are averaged back into the global model
4. Repeat for `num_rounds`

### Missing Modality Handling

When a modality is unavailable for a patient, the corresponding feature vector is all zeros. The `FederatedPPMIDataset` detects this and sets `{modality}_mask = 1`. The model then substitutes a **learned mask token** (nn.Parameter) instead of the zero embedding, allowing the attention mechanism to learn how to handle incomplete data.

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

