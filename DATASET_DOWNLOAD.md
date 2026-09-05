# PPMI Dataset Download Guide (Fed-PhenoGraft)

Everything the pipeline needs, where to get it, what to name it, and how to verify it. All data comes from the **Parkinson's Progression Markers Initiative (PPMI)** via the **IDA LONI portal**.

> **Current status of this repo:** all 10 required tabular CSVs are already present in `data/raw/` and the pipeline runs end-to-end on them. Only the **MRI NIfTI scans** (optional, for the imaging branch) remain to be downloaded.

---

## 1. Get Access

1. Register at **https://ida.loni.usc.edu** → *PPMI* → apply for Data Access (approval usually takes 1–3 business days; use an institutional email).
2. Once approved, log in → **PPMI** → **Download** → **Study Data**.

## 2. Required Tabular Files (CSVs)

Download each file and place it in **`data/raw/`** under the exact expected filename. IDA exports usually append a date suffix (e.g., `MDS-UPDRS_Part_III_23Aug2026.csv`) — **rename the file to strip the suffix**. The loaders also accept the naming variants listed.

| # | Expected filename in `data/raw/` | IDA Study Data section | Role in the pipeline |
|---|----------------------------------|------------------------|----------------------|
| 1 | `MDS_UPDRS_Part_I.csv` | Motor Assessments → MDS-UPDRS | Non-motor experiences (feature @ baseline) |
| 2 | `MDS_UPDRS_Part_II.csv` | Motor Assessments → MDS-UPDRS | Motor experiences of daily living (feature @ baseline) |
| 3 | `MDS_UPDRS_Part_III.csv` | Motor Assessments → MDS-UPDRS | Motor examination — **baseline feature AND the Year-2 regression target** (`NP3TOT` @ visit `V04`) |
| 4 | `MDS_UPDRS_Part_IV.csv` | Motor Assessments → MDS-UPDRS | Motor complications (feature @ baseline) |
| 5 | `Demographics.csv` | Subject Characteristics | Sex, birth date (accepted variant: `Screening___Demographics.csv`) |
| 6 | `Age_at_visit.csv` | Subject Characteristics | Exact age at each visit — preferred age source |
| 7 | `MoCA.csv` | Non-motor Assessments | Cognition (variants: `Montreal_Cognitive_Assessment__MoCA_.csv`) |
| 8 | `Patient_Status.csv` | Subject Characteristics | PD vs HC enrollment label — classification target + split stratification (variant: `Participant_Status.csv`) |
| 9 | `DATScan_Analysis.csv` | Imaging → DaTScan | Striatal binding ratios: `CAUDATE_R/L`, `PUTAMEN_R/L` (variants: `DaTscan_Analysis.csv`, `DATscan_Analysis.csv`) |
| 10 | `Genetic_Testing_Results.csv` | Biospecimen → Genetics | LRRK2 / GBA / SNCA / PINK1 / PRKN / APOE carrier status (variants: `Genetic_Results.csv`, `Genetics.csv`) |

**Column requirements the loaders rely on** (present in standard PPMI exports):
- All visit-based files need `PATNO` and `EVENT_ID` columns (`BL` = baseline, `V04` = Year 2).
- UPDRS files need either the official totals (`NP1RTOT`/`NP1PTOT`, `NP2PTOT`, `NP3TOT`, `NP4TOT`) or the individual `NP*` item columns.
- DaTScan needs SBR columns for right/left caudate and putamen.

## 3. Optional: MRI Scans (NIfTI)

Needed only for the real structural-MRI branch; without them the pipeline uses learned mask tokens / synthetic fallback.

1. IDA → **PPMI** → **Search** → **Advanced Image Search**.
2. Filter: Modality = **MRI**, Weighting/Description = **T1** anatomical (e.g., *MPRAGE*, *SAG T1*), Visit = **Baseline**.
3. Add results to a collection → download in **NIfTI** format.
4. Place files as:
   ```
   data/raw/mri/{PATNO}/T1w.nii.gz      # any .nii/.nii.gz inside the PATNO folder works
   ```
5. Enable in `config.yaml`:
   ```yaml
   mri:
     use_real_mri: true
   ```
6. Install the imaging dependencies: `pip install nibabel nilearn`.

## 4. Automated Download (alternative)

If your IDA account has programmatic access:

```bash
copy .env.example .env      # then fill in PPMI_USER / PPMI_PASSWORD
python scripts/download_ppmi_data.py
```

If authorization fails, the script exits cleanly — use the manual method above.

## 5. Verify Your Download

```bash
python src/main.py
```

Watch the "Dataset Summary" block in the log. A correct setup on current PPMI data looks approximately like:

```
Patients:  ~3,500  (subjects with a valid Year-2 UPDRS-III target)
Clinical:  7 features     MRI: 100 ROIs     PET: 10 features     Genetic: 9 features
PET missing:  ~195/3,513   ← real DaTScan loaded (if this is ~500+, the DaTScan CSV was not read)
```

Red flags and what they mean:
- **"... CSV not found, skipping"** — a file is missing or misnamed; check the table above.
- **"Generating synthetic ... fallback"** — that whole modality was not found; results will not reflect real signal for it.
- **"Dropping N subjects with missing regression target"** is *normal* — those subjects have no Year-2 UPDRS-III visit and are excluded rather than imputed (prevents label leakage).

## 6. Data Use Reminder

PPMI data is for approved research use only — do **not** commit any file in `data/raw/` to a public repository, and follow the PPMI Data Use Agreement for publications (include the standard PPMI acknowledgment).
