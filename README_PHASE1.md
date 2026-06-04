# PPMI Multimodal Parkinson's Progression Prediction

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange.svg)

Phase 1 of a production-grade machine learning system to predict Parkinson's disease progression using multimodal data from the Parkinson's Progression Markers Initiative (PPMI).

## Quick Start

1. **Environment Setup**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. **Credentials**
   Copy `.env.example` to `.env` and fill in your PPMI portal credentials.

3. **Data Download & Verification**
   ```bash
   python scripts/download_ppmi_data.py
   python scripts/download_mri_data.py
   python scripts/verify_data.py
   ```

4. **Run Pipeline**
   ```bash
   python src/baselines/runner.py
   python src/evaluation/report.py
   ```

## Project Structure
- `scripts/`: Data fetching and validation utilities
- `src/data/`: Modality-specific preprocessors (imputation, engineering)
- `src/embeddings/`: Feature extraction and late-fusion logic
- `src/baselines/`: Cross-validated training loop for 6 model families
- `src/evaluation/`: Results compilation and visualisation

## Evaluation Metrics
- **CCC (Concordance Correlation Coefficient)**: Used as the primary metric instead of R². R² ignores systematic bias, while CCC penalises models that deviate from the line of perfect agreement.

## System Design Notes
- **Data Leakage Prevention**: Standard scalers and missing value imputers are fit strictly *inside* the 5-fold cross-validation loop.
- **MRI Strategy**: Implements a lightweight `nilearn` Schaefer parcellation pipeline, falling back to synthetic standard normal data if NIfTI files cannot be acquired programmatically.

## Interview Talking Points
This project demonstrates:
1. End-to-end multimodal architecture
2. Robust data handling with external, volatile APIs
3. Proper statistical methodology (CCC, nested CV)
4. Production-level code standards (typing, SOLID principles)
