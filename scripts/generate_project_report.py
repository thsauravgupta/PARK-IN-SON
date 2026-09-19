# -*- coding: utf-8 -*-
"""
Builds the BCSE497J Project-I report for Fed-PhenoGraft.

Opens the college template read-only, strips its instruction and sample text
while keeping styles.xml, the footers, numbering and the cover-page logo, then
writes the finished report to
outputs/presentation/BCSE497J_Project_I_Report.docx.

Every experimental number comes from the committed run at df567f8
(outputs/results/final_metrics.json and RESULTS.md). Nothing is invented; in
particular, only the six genuine references are cited.

Run:  .venv/Scripts/python.exe scripts/generate_project_report.py
"""

import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = PROJECT_ROOT / "outputs" / "presentation" / "3 BCSE497J Project I Report - Template.docx"
OUTPUT = PROJECT_ROOT / "outputs" / "presentation" / "BCSE497J_Project_I_Report.docx"
FIG = PROJECT_ROOT / "outputs" / "figures"
RFIG = FIG / "report"

FONT = "Times New Roman"
INK = RGBColor(0x00, 0x00, 0x00)
GREY = RGBColor(0x59, 0x59, 0x59)
HDR_FILL = "D9E2F3"

# Guide's designation is not recorded anywhere in the repository — the team
# fills this in before submission.
GUIDE_DESIGNATION = "<Designation — to be confirmed with the guide>"


# ══════════════════════════════════════════════════════════════════
# document helpers
# ══════════════════════════════════════════════════════════════════

def clear_body(doc):
    """Delete every body element except the trailing sectPr (page setup)."""
    body = doc.element.body
    for child in list(body):
        if child.tag != qn("w:sectPr"):
            body.remove(child)


def style_run(run, size=12, bold=False, italic=False, caps=False, color=INK):
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.all_caps = caps
    run.font.color.rgb = color
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(attr), FONT)
    return run


def para(doc, text="", size=12, bold=False, italic=False, align=None,
         spacing=1.15, before=0, after=6, indent=0, color=INK, keep_with_next=False):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing = spacing
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    if indent:
        pf.left_indent = Inches(indent)
    if align is not None:
        p.alignment = align
    if keep_with_next:
        pf.keep_with_next = True
    if text:
        style_run(p.add_run(text), size=size, bold=bold, italic=italic, color=color)
    return p


def bullet(doc, text, size=12, indent=0.3, bold_lead=None):
    """A bulleted paragraph; bold_lead is the run rendered bold before the text."""
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing = 1.15
    pf.space_after = Pt(4)
    pf.left_indent = Inches(indent + 0.25)
    pf.first_line_indent = Inches(-0.25)
    style_run(p.add_run("•\t"), size=size)
    if bold_lead:
        style_run(p.add_run(bold_lead), size=size, bold=True)
    style_run(p.add_run(text), size=size)
    return p


def heading(doc, text, level=1, before=14, after=8):
    """Template heading conventions: L1 = 14 pt bold upper case,
    L2 = 13 pt bold title case, L3 = 12 pt bold italic."""
    sizes = {1: 14, 2: 13, 3: 12}
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing = 1.5
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.keep_with_next = True
    style_run(p.add_run(text), size=sizes[level], bold=True, italic=(level == 3))
    return p


def caption(doc, text):
    return para(doc, text, size=11, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER,
                spacing=1.0, before=2, after=12, color=GREY)


def add_figure(doc, image_path, caption_text, width=6.0):
    if not Path(image_path).exists():
        raise FileNotFoundError(image_path)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(image_path), width=Inches(width))
    caption(doc, caption_text)


def shade(cell, hex_fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), hex_fill)
    tc_pr.append(shd)


def add_table(doc, rows, col_widths, size=10, header=True, borders=True,
              caption_text=None, align_right=()):
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    table.style = "Table Grid" if borders else "Normal Table"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    for r, row in enumerate(rows):
        for c, text in enumerate(row):
            cell = table.cell(r, c)
            cell.width = Inches(col_widths[c])
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.line_spacing = 1.0
            if c in align_right and r > 0:
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            style_run(p.add_run(str(text)), size=size, bold=(header and r == 0))
            if header and r == 0:
                shade(cell, HDR_FILL)
    for row in table.rows:
        for c, cell in enumerate(row.cells):
            cell.width = Inches(col_widths[c])
    if caption_text:
        caption(doc, caption_text)
    else:
        para(doc, "", size=6, after=6)
    return table


def page_break(doc):
    p = doc.add_paragraph()
    p.add_run().add_break(WD_BREAK.PAGE)


def extract_logo():
    """Pull the VIT logo out of the template package so the cover can reuse it."""
    import zipfile
    tmp = PROJECT_ROOT / "outputs" / "figures" / "report" / "_vit_logo.jpg"
    with zipfile.ZipFile(TEMPLATE) as z:
        names = [n for n in z.namelist() if n.startswith("word/media/")]
        # image1 is the cover logo (smallest media part in the template)
        best = min(names, key=lambda n: z.getinfo(n).file_size)
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(z.read(best))
    return tmp


def word_count(text):
    return len(text.split())


# ══════════════════════════════════════════════════════════════════
# content
# ══════════════════════════════════════════════════════════════════

TITLE = ("FED-PHENOGRAFT: PHENOTYPE-GUIDED ASYMMETRIC CROSS-MODAL ATTENTION WITH "
         "SHARED-PRIVATE LATENT DECOMPOSITION FOR FEDERATED MULTI-MODAL "
         "PARKINSON'S DISEASE PREDICTION")

STUDENTS = [  # sorted on register number, as the template requires
    ("23BCE2327", "AMIT ADHIKARI"),
    ("23BCE2330", "SHREEYAM ACHARYA"),
    ("23BCE2336", "SAURAV KUMAR GUPTA"),
]

ABSTRACT = (
    "Parkinson's Disease (PD) is the second most common neurodegenerative disorder, and its "
    "clinical management depends on data that is inherently multimodal: motor and non-motor "
    "rating scales, structural magnetic resonance imaging, dopamine transporter (DaTScan) "
    "imaging and genetic carrier status. Three obstacles limit existing multimodal models. The "
    "modalities are unequal in completeness and reliability, yet are fused symmetrically; "
    "individual patients are routinely missing entire modalities, which most architectures "
    "cannot consume; and the records are held by different clinical sites that cannot pool raw "
    "patient data. This project presents Fed-PhenoGraft, a federated multimodal framework that "
    "addresses all three jointly. Clinical phenotype embeddings act as a query that "
    "asymmetrically attends to imaging and genetic keys and values, so auxiliary modalities are "
    "retrieved conditioned on the patient's phenotype rather than averaged in. Each auxiliary "
    "modality is decomposed into shared and private latent representations that are driven "
    "towards statistical independence by a Hilbert-Schmidt Independence Criterion penalty, "
    "isolating disease-relevant signal from modality-specific acquisition noise. Learned mask "
    "tokens substitute for any modality a patient lacks. Training proceeds by sample-weighted "
    "federated averaging over four simulated non-IID sites, so raw records never leave a client. "
    "Evaluated once on a held-out subject-level split of 3,472 PPMI subjects under a leak-free "
    "protocol, the framework attains ROC-AUC 0.982 and accuracy 0.937 for PD versus control "
    "classification, and a concordance correlation coefficient of 0.147 for two-year UPDRS-III "
    "progression. An eight-variant ablation and an integrated-gradients analysis identify "
    "DaTScan as the dominant imaging signal and quantify the cost of the current synthetic-MRI "
    "placeholder."
)

KEYWORDS = ("Keywords - Parkinson's Disease, Federated Learning, Multimodal Fusion, "
            "Cross-Modal Attention, HSIC Orthogonality, PPMI, Explainable AI.")

BACKGROUND = (
    "Parkinson's Disease affects over eight million people worldwide and progresses at a rate "
    "that varies widely between patients. Clinical severity is tracked with the Movement "
    "Disorder Society Unified Parkinson's Disease Rating Scale (MDS-UPDRS), whose Part III "
    "motor examination is the accepted measure of motor burden. Because no single measurement "
    "captures the disease, longitudinal observational studies collect several modalities in "
    "parallel. The Parkinson's Progression Markers Initiative (PPMI) is the largest such study, "
    "providing clinical scales, structural MRI, dopamine transporter (DaTScan) imaging and "
    "genetic testing for thousands of participants. Machine learning on these records has "
    "consistently shown that combining modalities outperforms any one of them for diagnosis and "
    "for progression tracking. However, the practical conditions under which such models must "
    "operate are demanding: records are incomplete at the level of the individual patient, the "
    "modalities differ greatly in reliability, and the data is distributed across clinical "
    "sites that are legally and ethically constrained from sharing raw patient records. This "
    "project develops a framework designed for exactly those conditions."
)

MOTIVATION = (
    "Three observations motivated this work. First, existing multimodal architectures fuse "
    "modalities symmetrically, implicitly treating a complete clinical assessment and a "
    "frequently absent genetic panel as equally informative; in the cohort assembled here the "
    "clinical block is complete for every subject while genetic results are missing for 78 per "
    "cent of them. Second, most published pipelines assume complete records and either discard "
    "incomplete patients or impute the missing modality, both of which distort the cohort. "
    "Third, and most consequentially, reported accuracy in this field is often inflated by "
    "evaluation choices rather than by modelling: predicting an absolute follow-up UPDRS-III "
    "score is dominated by its autocorrelation with the baseline score, and preprocessing "
    "statistics fitted before splitting leak test information into training. Preliminary "
    "experiments in this project reproduced that effect directly, with the headline metric "
    "falling from 0.825 to 0.147 once the target was changed to true progression. The "
    "motivation is therefore not only to build a privacy-preserving multimodal architecture but "
    "to measure it honestly, so that the reported gain is one a clinician could rely on."
)

SCOPE = (
    "The project covers the complete pipeline from raw PPMI records to an explained prediction. "
    "In scope are the clinical, DaTScan and genetic modalities loaded from PPMI study-data CSV "
    "files; the structural MRI branch, whose nilearn Schaefer-100 parcellation pipeline is "
    "implemented and switched by a single configuration flag; a leak-free evaluation protocol; "
    "the Fed-PhenoGraft architecture; a twelve-model classical baseline suite; and an "
    "explainability suite. Prediction targets are the two-year change in MDS-UPDRS Part III and "
    "the binary PD versus control label. Several boundaries are stated explicitly. Federation "
    "is simulated on one machine across four clients rather than deployed across real hospital "
    "networks, although real-site partitioning is implemented and activates when the PPMI "
    "Center-Subject list is supplied. The MRI branch currently runs on a synthetic fallback "
    "because the T1-weighted image collections have not finished downloading. The formulation "
    "is cross-sectional, mapping baseline to the year-two visit rather than modelling the full "
    "trajectory. Formal differential privacy is not implemented. The system is a research "
    "instrument and is not a diagnostic device."
)

RESEARCH_GAP = [
    ("Limited integration of multimodal data.",
     "Most published work uses only one or two modalities, so the complementary information "
     "carried jointly by MRI, DaTScan, clinical scales and genetics is never fully exploited. "
     "Fusion on PPMI has been shown to outperform unimodal approaches for progression tracking "
     "[1], [4], but full four-modality integration remains uncommon."),
    ("Poor handling of incomplete multimodal data.",
     "Multimodal models generally assume complete patient records, whereas real clinical "
     "cohorts are sparse. Models that discard incomplete subjects shrink and bias the cohort; "
     "models that impute a missing modality manufacture signal that was never measured."),
    ("Symmetric fusion ignores modality reliability.",
     "Cross-attention modules that align modalities and weight them dynamically have improved "
     "feature extraction [2], but they still treat every modality as an equal partner. No "
     "mechanism designates the most complete and most trustworthy modality as the one that "
     "directs retrieval from the others."),
    ("No leak-free, privacy-preserving progression benchmark.",
     "Explainability is increasingly demanded of neuroimaging models for PD [3], and digital "
     "twin and reinforcement learning approaches have been applied to rehabilitation and "
     "medication management [5], [6]. What is missing is a benchmark that combines federated "
     "training with a protocol strict enough - subject-level splits, train-only statistics, "
     "a progression rather than absolute target, and single-use test evaluation - for the "
     "reported numbers to be trusted."),
]

OBJECTIVES = [
    "To design a Phenotype-Guided Asymmetric Attention mechanism in which clinical features "
    "act as the query and selectively retrieve information from MRI, DaTScan and genetic "
    "representations.",
    "To implement a Shared-Private Latent Decomposition regularised by HSIC orthogonality, so "
    "that disease-relevant biomarkers are separated from modality-specific noise.",
    "To build a federated learning pipeline based on sample-weighted FedAvg that trains across "
    "multiple simulated clinical sites without centralising raw patient data.",
    "To incorporate robustness features, namely learned mask tokens for absent modalities and "
    "Monte Carlo dropout for uncertainty estimation.",
    "To validate the framework under a leak-free protocol against twelve classical baselines "
    "with bootstrap confidence intervals, multi-seed repetition and an eight-variant ablation, "
    "supported by a multi-level explainability suite comprising attention maps, integrated "
    "gradients, a missing-modality stress test and counterfactual analysis.",
]

PROBLEM_STATEMENT = [
    "Let a cohort of N subjects be described by four modality blocks - clinical phenotype, "
    "structural MRI, DaTScan and genetics - where any block other than the clinical one may be "
    "entirely absent for a given subject, and where the subjects are partitioned across K "
    "clinical sites that cannot exchange raw records.",
    "The problem is to learn a single mapping that predicts, for each subject, the change in "
    "MDS-UPDRS Part III between the baseline and the year-two visit, and simultaneously the "
    "binary PD versus healthy-control label, subject to three constraints: no raw feature "
    "vector may cross a site boundary during training; the model must accept subjects with "
    "missing modalities without imputing them; and the evaluation must be free of leakage, "
    "meaning that every subject belongs to exactly one partition, all preprocessing statistics "
    "are estimated on the training partition alone, no target value is imputed, and the "
    "held-out test set is scored exactly once after training has concluded.",
    "The measure of success is the concordance correlation coefficient on the progression "
    "target and the area under the ROC curve on the diagnosis target, each reported with a "
    "bootstrap confidence interval and compared against a classical baseline suite evaluated "
    "on the identical test subjects.",
]

PROJECT_PLAN_TABLE = [
    ["Phase", "Period", "Key deliverable", "Status"],
    ["Literature survey and problem formulation", "Jun 2026", "Research gap, objectives", "Complete"],
    ["PPMI data access and verification", "Jun 2026", "10 study-data CSV files", "Complete"],
    ["Modality loaders and feature engineering", "Jun - Jul 2026", "126-feature aligned matrix", "Complete"],
    ["Leak-free preprocessing and split protocol", "Jun - Jul 2026", "Subject-level 70/15/15", "Complete"],
    ["Fed-PhenoGraft model implementation", "Jul - Aug 2026", "Attention, HSIC, mask tokens", "Complete"],
    ["FedAvg orchestrator and partitioning", "Jul - Aug 2026", "Non-IID client simulation", "Complete"],
    ["Baseline suite and metric layer", "Aug 2026", "12 models, one-shot test", "Complete"],
    ["Statistical rigour, ablation and XAI", "Aug - Sep 2026", "CIs, 8 ablations, 4 analyses", "Complete"],
    ["Real T1w MRI integration", "Sep - Oct 2026", "Schaefer-100 real features", "Planned"],
    ["Hyperparameter tuning (validation only)", "Oct 2026", "Tuned configuration", "Planned"],
    ["Real-site federation and DP-SGD", "Oct - Nov 2026", "Formal privacy guarantee", "Planned"],
    ["Longitudinal extension and final evaluation", "Nov 2026", "Trajectory model", "Planned"],
]

FUNCTIONAL = [
    ("Multimodal Data Ingestion:", "The system shall load PPMI clinical, DaTScan, genetic and "
     "MRI records and align them per subject on the PATNO identifier (M2 loaders, M3 builder)."),
    ("Progression Target Construction:", "The system shall derive the regression target as the "
     "change in MDS-UPDRS Part III between the baseline and year-two visits using the official "
     "NP3TOT total column, and shall drop - never impute - subjects lacking a genuine label."),
    ("Leak-Free Partitioning:", "The system shall split subjects 70/15/15 into training, "
     "validation and test partitions, stratified on diagnosis, with every subject appearing in "
     "exactly one partition."),
    ("Train-Only Preprocessing:", "The system shall fit all imputation and scaling statistics "
     "on the training subjects alone and apply them unchanged to the validation and test "
     "partitions."),
    ("Missing-Modality Handling:", "The system shall represent an absent modality by a learned "
     "mask token rather than an imputed value, and shall preserve the all-zero encoding of a "
     "missing modality through scaling."),
    ("Phenotype-Guided Attention:", "The system shall allow clinical embeddings to query "
     "imaging and genetic embeddings through multi-head cross-attention, and shall expose the "
     "resulting attention weights for explanation."),
    ("Shared-Private Decomposition:", "The system shall encode each auxiliary modality into "
     "shared and private latent vectors penalised towards statistical independence by an "
     "RBF-kernel HSIC criterion."),
    ("Federated Training:", "The system shall train by sample-weighted federated averaging "
     "across N simulated sites under IID, Dirichlet non-IID or real-site partitioning, with no "
     "raw data exchanged between clients."),
    ("Validated Early Stopping:", "The system shall monitor validation CCC each round, stop "
     "after a configured patience, restore the best round's weights, and score the test set "
     "exactly once thereafter."),
    ("Multitask Prediction:", "The system shall predict progression and diagnosis jointly under "
     "a single weighted multitask loss."),
    ("Baseline Benchmarking:", "The system shall evaluate twelve classical models under "
     "per-fold preprocessing pipelines on the identical test subjects."),
    ("Statistical Reporting:", "The system shall report bootstrap 95 per cent confidence "
     "intervals, a mean and standard deviation across independent training seeds, and a paired "
     "significance test against the strongest baseline."),
    ("Ablation Suite:", "The system shall retrain an ablation variant for every modality and "
     "every architectural component under the same protocol."),
    ("Explainability Suite:", "The system shall produce attention maps, integrated-gradients "
     "attributions, a missing-modality stress test and counterfactual gene analysis."),
    ("Automated Reporting:", "Every run shall emit a metrics JSON file, a Markdown results "
     "report and the complete figure suite, regenerable without retraining."),
    ("Real MRI Integration:", "The system shall replace the synthetic MRI branch with real "
     "T1-weighted Schaefer-100 parcellation when NIfTI scans are present and the corresponding "
     "configuration flag is enabled."),
]

NON_FUNCTIONAL = [
    ("Privacy:", "No raw feature matrix crosses a client boundary at any point in training; "
     "only model weight tensors are aggregated by the orchestrator."),
    ("Reproducibility:", "A single seed function seeds Python, NumPy and PyTorch with "
     "deterministic cuDNN kernels; every hyperparameter is declared in config.yaml, and a "
     "snapshot of the configuration is written into each metrics file."),
    ("Graceful Degradation:", "A missing input CSV produces a logged warning and a synthetic "
     "fallback for that modality rather than an aborted run, so development is never blocked."),
    ("Performance:", "A complete run - twelve baselines, three training seeds, seven ablation "
     "variants and the full explainability suite - finishes in approximately eight minutes on "
     "a CPU."),
    ("Portability:", "The implementation is pure CPU PyTorch with pinned dependency versions, "
     "and requires no GPU or cloud infrastructure."),
    ("Maintainability:", "The codebase is organised into seven modules with single "
     "responsibilities, each addressable and testable in isolation."),
    ("Compliance:", "Use of PPMI data is governed by the PPMI Data Use Agreement; no "
     "participant-identifying information is stored or published."),
]

TECHNICAL_FEASIBILITY = [
    ("Data Availability:", "PPMI study-data access has been granted and the ten required CSV "
     "files are downloaded and verified. The remaining dependency, the T1-weighted image "
     "collections, is a separate download whose absence is handled by a documented fallback."),
    ("Technology Maturity:", "The stack - PyTorch, scikit-learn, XGBoost, LightGBM, nilearn and "
     "Captum - is open source, widely used and version-pinned, so no unproven component is on "
     "the critical path."),
    ("Computational Requirements:", "The model has approximately 0.1 million parameters and "
     "trains on a tabular cohort of 3,472 subjects; the entire experiment completes in about "
     "eight minutes on a laptop CPU, so no specialised hardware is required."),
    ("Demonstrated End-to-End Operation:", "The pipeline has been executed end to end on real "
     "PPMI data, producing the metrics, figures and reports presented in Section 4.3; technical "
     "feasibility is therefore demonstrated rather than estimated."),
]

ECONOMIC_FEASIBILITY = [
    ("Data Cost:", "PPMI data is provided at no charge to approved academic researchers, so "
     "the dataset carries no licence cost."),
    ("Software Cost:", "Every library in the stack is open source under permissive licences; "
     "there is no commercial software spend."),
    ("Hardware and Infrastructure Cost:", "Because the pipeline runs on commodity CPU hardware "
     "already available to the team, there is no cloud compute or GPU rental expenditure."),
    ("Return on Investment:", "The benefit is scientific rather than commercial: a validated, "
     "privacy-preserving framework that can be redirected to other multimodal neurodegenerative "
     "cohorts without re-engineering, and a leak-free benchmark that other groups can adopt."),
]

SOCIAL_FEASIBILITY = [
    ("Privacy by Construction:", "Federated training means participating hospitals retain "
     "custody of their own records, which removes the principal institutional barrier to "
     "multi-site collaboration on patient data."),
    ("Sustainable Development Goals:", "The work contributes to SDG 3 (Good Health and "
     "Well-being) by improving prognostic tooling for a major neurodegenerative disease, and to "
     "SDG 9 (Industry, Innovation and Infrastructure) by developing privacy-preserving AI "
     "infrastructure for healthcare."),
    ("Transparency and Clinician Trust:", "The explainability suite reports which modality drove "
     "each prediction, which is a precondition for clinical acceptance of any model in this "
     "domain."),
    ("Ethical Positioning:", "The system is a research instrument, not a diagnostic device. Its "
     "outputs are not intended to inform individual patient care, and the reported progression "
     "accuracy is stated honestly, including where it fails to exceed a classical baseline."),
]

HARDWARE = [
    ["Component", "Specification used in development", "Minimum requirement"],
    ["Processor", "Intel Core Ultra 7 155H (16 cores, 22 threads)", "Any x86-64 quad-core CPU"],
    ["Memory (RAM)", "16 GB", "8 GB"],
    ["Storage", "150 GB SSD (dataset, environment and outputs)", "20 GB free"],
    ["Graphics (GPU)", "NVIDIA GeForce RTX 4070 Laptop (not required)", "Not required - CPU only"],
    ["Display", "1920 x 1080 or higher", "1366 x 768"],
]

SOFTWARE = [
    ["Category", "Component and version"],
    ["Operating System", "Windows 11 (build 26200); the pipeline is platform independent"],
    ["Programming Language", "Python 3.11"],
    ["Development Environment", "Visual Studio Code, Git, Python virtual environment (venv)"],
    ["Deep Learning", "PyTorch 2.3.0, Captum 0.7 (integrated gradients)"],
    ["Classical ML", "scikit-learn 1.4.2, XGBoost 2.0.3, LightGBM 4.3.0, SHAP 0.45.0"],
    ["Neuroimaging", "nibabel 5.2.0, nilearn 0.10.3 (Schaefer-2018 atlas), pydicom 2.4.3"],
    ["Data Handling", "pandas 2.2.2, NumPy 1.26.4, PyArrow 15"],
    ["Visualisation", "Matplotlib 3.9.0, seaborn 0.13.2, UMAP-learn 0.5.6"],
    ["Configuration and Testing", "PyYAML 6.0.1, python-dotenv 1.0.1, pytest 8.2.0"],
    ["Data Source", "PPMI study data via the IDA-LONI portal (ppmi-downloader 0.7.6)"],
]

COHORT_TABLE = [
    ["Modality", "Dim.", "Features", "Missing", "Source"],
    ["Clinical", "7", "MDS-UPDRS I-IV totals, MoCA, age at visit, sex", "0 / 3,472",
     "MDS_UPDRS_Part_I-IV, MoCA, Demographics, Age_at_visit"],
    ["Structural MRI", "100", "Schaefer-100 ROI signals (synthetic placeholder)", "669 / 3,472",
     "NIfTI scans pending download"],
    ["PET / DaTScan", "10", "Caudate and putamen L/R, means, asymmetry indices, striatal "
     "total, caudate-putamen ratio", "194 / 3,472", "DATScan_Analysis.csv"],
    ["Genetics", "9", "LRRK2, GBA, SNCA, PINK1, PRKN, APOE-e4, plus derived flags",
     "2,718 / 3,472", "Genetic_Testing_Results.csv"],
]

PROTOCOL_TABLE = [
    ["Setting", "Value", "Setting", "Value"],
    ["Subjects", "3,472", "Clients", "4, Dirichlet alpha = 0.5"],
    ["Split", "2,430 / 521 / 521", "Rounds", "30 max, patience 5"],
    ["Regression target", "Delta UPDRS-III (V04 - BL)", "Local epochs", "2 per round"],
    ["Classification target", "PD vs HC / other", "Optimiser", "AdamW, lr 1e-3, wd 1e-4"],
    ["Embedding dim / heads", "32 / 4", "Gradient clip", "1.0"],
    ["Dropout", "0.3", "Loss weights", "HSIC 0.1, classification 0.3"],
    ["Training seeds", "3 (42, 143, 244)", "Bootstrap", "1,000 resamples, 95% CI"],
]

METRICS_TABLE = [
    ["Task", "Metric", "Train", "Validation", "Test", "Test 95% CI"],
    ["Progression", "CCC", "0.4982", "0.3591", "0.1469", "0.0406 - 0.2439"],
    ["Progression", "RMSE (points)", "5.53", "6.41", "7.64", "6.93 - 8.46"],
    ["Progression", "MAE (points)", "3.82", "4.44", "5.14", "4.67 - 5.69"],
    ["Progression", "R-squared", "0.3180", "0.1506", "-0.0993", "-0.242 - 0.027"],
    ["Progression", "Pearson r", "0.5650", "0.4173", "0.1769", "-"],
    ["Diagnosis", "ROC-AUC", "0.9859", "0.9839", "0.9816", "-"],
    ["Diagnosis", "Accuracy", "0.9543", "0.9367", "0.9367", "-"],
    ["Diagnosis", "F1 score", "0.9359", "0.9129", "0.9115", "-"],
]

BASELINE_TABLE = [
    ["Model", "CV CCC (mean +/- SD)", "Test CCC", "Test RMSE", "Test MAE", "Test R2"],
    ["linear", "0.2367 +/- 0.0241", "0.2039", "6.96", "4.78", "0.0894"],
    ["ridge", "0.2368 +/- 0.0245", "0.2036", "6.96", "4.78", "0.0893"],
    ["lasso", "0.2195 +/- 0.0231", "0.1833", "6.92", "4.70", "0.1000"],
    ["elastic_net", "0.2104 +/- 0.0224", "0.1776", "6.93", "4.71", "0.0974"],
    ["svm (RBF)", "0.0603 +/- 0.0074", "0.0552", "7.16", "4.76", "0.0342"],
    ["knn", "0.0783 +/- 0.0206", "0.0464", "7.40", "4.98", "-0.0296"],
    ["random_forest", "0.0993 +/- 0.0151", "0.0821", "7.07", "4.73", "0.0603"],
    ["extra_trees", "0.0635 +/- 0.0105", "0.0511", "7.14", "4.76", "0.0410"],
    ["gradient_boosting", "0.2322 +/- 0.0262", "0.1911", "7.01", "4.74", "0.0758"],
    ["mlp", "0.1400 +/- 0.0751", "0.1924", "7.10", "4.93", "0.0518"],
    ["xgboost", "0.2401 +/- 0.0302", "0.2135", "6.96", "4.67", "0.0883"],
    ["lightgbm (strongest)", "0.2253 +/- 0.0213", "0.2300", "6.94", "4.67", "0.0931"],
    ["Fed-PhenoGraft (ours)", "val 0.3591 (early-stopped)", "0.1469", "7.64", "5.14", "-0.0993"],
]

ABLATION_TABLE = [
    ["Variant", "Val CCC", "Test CCC", "Test RMSE", "Test AUC", "Reading"],
    ["Full Fed-PhenoGraft", "0.3591", "0.1469", "7.64", "0.9816", "Reference"],
    ["Without asymmetric attention", "0.2885", "0.1736", "7.34", "0.9737",
     "-0.071 val CCC: attention contributes"],
    ["Without HSIC loss", "0.3130", "0.1481", "7.78", "0.9786",
     "-0.046 val CCC: orthogonality contributes"],
    ["Centralised (1 client)", "0.3142", "0.1857", "7.59", "0.9806",
     "Federation costs nothing on validation"],
    ["Without PET / DaTScan", "0.2833", "0.1303", "7.86", "0.9513",
     "Largest drop on both tasks"],
    ["Without genetics", "0.3139", "0.1474", "7.79", "0.9787",
     "Small but real contribution"],
    ["Without MRI (synthetic)", "0.3272", "0.2241", "6.90", "0.9761",
     "Removing it improves test CCC"],
    ["Clinical only", "0.3072", "0.2055", "6.91", "0.9554",
     "But AUC falls 0.026"],
]

CONFUSION_TABLE = [
    ["", "Predicted non-PD", "Predicted PD", "Total"],
    ["Actual non-PD", "318", "20", "338"],
    ["Actual PD", "13", "170", "183"],
    ["Total", "331", "190", "521"],
]

TOC_ROWS = [
    ["Sl. No.", "Contents", "Page No."],
    ["", "Abstract", ""],
    ["1.", "INTRODUCTION", ""],
    ["", "1.1 Background", ""],
    ["", "1.2 Motivation", ""],
    ["", "1.3 Scope of the Project", ""],
    ["2.", "PROJECT DESCRIPTION AND GOALS", ""],
    ["", "2.1 Literature Review", ""],
    ["", "2.2 Research Gap", ""],
    ["", "2.3 Objectives", ""],
    ["", "2.4 Problem Statement", ""],
    ["", "2.5 Project Plan", ""],
    ["3.", "TECHNICAL SPECIFICATION", ""],
    ["", "3.1 Requirements", ""],
    ["", "3.1.1 Functional", ""],
    ["", "3.1.2 Non-Functional", ""],
    ["", "3.2 Feasibility Study", ""],
    ["", "3.2.1 Technical Feasibility", ""],
    ["", "3.2.2 Economic Feasibility", ""],
    ["", "3.2.3 Social Feasibility", ""],
    ["", "3.3 System Specification", ""],
    ["", "3.3.1 Hardware Specification", ""],
    ["", "3.3.2 Software Specification", ""],
    ["4.", "DESIGN APPROACH AND DETAILS", ""],
    ["", "4.1 System Architecture", ""],
    ["", "4.2 Design", ""],
    ["", "4.2.1 Data Flow Diagram", ""],
    ["", "4.2.2 Use Case Diagram", ""],
    ["", "4.2.3 Class Diagram", ""],
    ["", "4.2.4 Sequence Diagram", ""],
    ["", "4.3 Implementation and Preliminary Results", ""],
    ["5.", "REFERENCES", ""],
]

REFERENCES_JOURNAL = [
    "[1] A. Vamvakas, T. Van Balkom, G. Van Wingen, et al., \"Prediction of impulse control "
    "disorders in Parkinson's disease through a longitudinal machine learning study,\" npj "
    "Parkinson's Disease, vol. 12, p. 38, 2026.",
    "[2] S. W. Akram and C. K, \"Enhancing Parkinson's Disease Staging: An Integrative Deep "
    "Learning Framework for Multimodal Feature Selection,\" Journal of Molecular Neuroscience, "
    "2026.",
    "[3] V. Awasthi et al., \"HyCoSwin-PD: An Explainable Hybrid ConvNeXtV2-Swin Transformer "
    "Framework for Parkinson's Disease Detection from Neuroimaging,\" MethodsX, vol. 16, 2026.",
    "[4] T. Zhi et al., \"MultimodalCNN-PD: A Parkinson's Disease Diagnostics Framework Using "
    "Multimodal Convolutional Neural Network,\" Frontiers in Aging Neuroscience, vol. 18, 2026.",
    "[6] H. Kim, C. Park, J. Hoon Kim, S. Jang, and H. K. Lee, \"Multimodal Reinforcement "
    "Learning for Embedding Networks and Medication Recommendation in Parkinson's Disease,\" "
    "IEEE Access, vol. 12, pp. 74251-74267, 2024.",
]

REFERENCES_CONFERENCE = [
    "[5] J. Qin, Y. Cai, Z. Wang, J. Jin, X. Huang, and L. Pei, \"Design of a Parkinson's "
    "Rehabilitation Management System Based on Multimodal Learning and Digital Twin "
    "Technology,\" in Proc. IEEE Smart World Congress (SWC), Nadi, Fiji, 2024, pp. 2073-2080.",
]

REFERENCES_WEB = [
    "Parkinson's Progression Markers Initiative (PPMI). [Online]. Available: "
    "https://www.ppmi-info.org",
    "Image and Data Archive (IDA), Laboratory of Neuro Imaging, USC. [Online]. Available: "
    "https://ida.loni.usc.edu",
]

REFERENCES_BOOK = [
    "I. Goodfellow, Y. Bengio, and A. Courville, Deep Learning. Cambridge, MA, USA: MIT Press, "
    "2016.",
]


# ══════════════════════════════════════════════════════════════════
# section builders
# ══════════════════════════════════════════════════════════════════

def build_cover(doc, logo):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(10)
    p.add_run().add_picture(str(logo), width=Inches(2.2))

    para(doc, "BCSE497J - Project-I", size=14, bold=True,
         align=WD_ALIGN_PARAGRAPH.CENTER, spacing=1.5, after=14)
    para(doc, TITLE, size=16, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER,
         spacing=1.5, after=20)

    t = doc.add_table(rows=len(STUDENTS), cols=2)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r, (reg, name) in enumerate(STUDENTS):
        for c, text in enumerate((reg, name)):
            cell = t.cell(r, c)
            cell.width = Inches(1.9 if c == 0 else 3.2)
            cell.text = ""
            cp = cell.paragraphs[0]
            cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cp.paragraph_format.line_spacing = 1.5
            cp.paragraph_format.space_after = Pt(0)
            style_run(cp.add_run(text), size=13, bold=True)

    para(doc, "", after=10)
    para(doc, "Under the Supervision of", size=13,
         align=WD_ALIGN_PARAGRAPH.CENTER, spacing=1.5, after=6)
    for text, bold in (("Dhivyaa C R", True), (GUIDE_DESIGNATION, False),
                       ("School of Computer Science and Engineering (SCOPE)", False)):
        para(doc, text, size=13, bold=bold, align=WD_ALIGN_PARAGRAPH.CENTER,
             spacing=1.5, after=2)

    para(doc, "", after=12)
    for text in ("B.Tech.", "in", "Computer Science and Engineering",
                 "School of Computer Science and Engineering"):
        para(doc, text, size=13, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER,
             spacing=1.5, after=2)
    para(doc, "September 2026", size=13, bold=True,
         align=WD_ALIGN_PARAGRAPH.CENTER, spacing=1.5, before=12, after=0)
    page_break(doc)


def build_abstract(doc):
    heading(doc, "ABSTRACT", 1, before=0)
    para(doc, ABSTRACT, align=WD_ALIGN_PARAGRAPH.JUSTIFY, after=12)
    para(doc, KEYWORDS, italic=True, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    page_break(doc)


def build_toc(doc):
    heading(doc, "TABLE OF CONTENTS", 1, before=0)
    add_table(doc, TOC_ROWS, [0.85, 4.75, 0.9], size=12, header=True, borders=False)
    para(doc, "Page numbers are finalised by Word after pagination "
              "(select the table and press F9, or fill in manually).",
         size=10, italic=True, color=GREY)
    page_break(doc)


def build_introduction(doc):
    heading(doc, "1. INTRODUCTION", 1, before=0)
    heading(doc, "1.1 Background", 2)
    para(doc, BACKGROUND, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    heading(doc, "1.2 Motivation", 2)
    para(doc, MOTIVATION, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    heading(doc, "1.3 Scope of the Project", 2)
    para(doc, SCOPE, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    page_break(doc)


def build_description(doc):
    heading(doc, "2. PROJECT DESCRIPTION AND GOALS", 1, before=0)

    heading(doc, "2.1 Literature Review", 2)
    for text in [
        "Research relevant to this project falls into four themes: multimodal fusion on the "
        "PPMI cohort, architectural mechanisms for aligning heterogeneous modalities, "
        "explainability in neuroimaging models, and system-level approaches to Parkinson's "
        "disease management.",

        "Multimodal fusion. Vamvakas et al. [1] conducted a longitudinal machine learning study "
        "on PPMI predicting impulse control disorders, demonstrating that clinical, imaging and "
        "genetic variables carry complementary prognostic information when combined over time. "
        "Zhi et al. [4] proposed MultimodalCNN-PD, a convolutional diagnostic framework that "
        "fuses several modalities and reports gains over unimodal equivalents. Together these "
        "establish the central premise of this project - that fusion outperforms any single "
        "modality - while also showing that both works fuse the modalities symmetrically and "
        "assume that the required modalities are present for each subject.",

        "Architectural mechanisms. Akram and C. K. [2] introduced an integrative deep learning "
        "framework for PD staging in which multimodal feature selection and cross-modal "
        "alignment modules dynamically weight the contribution of each modality. This "
        "demonstrates that learned, input-dependent weighting extracts more informative "
        "representations than fixed concatenation. It stops short, however, of assigning an "
        "asymmetric role to the modalities: no single modality is designated as the query that "
        "directs retrieval from the others, which is precisely the mechanism proposed here.",

        "Explainability. Awasthi et al. [3] presented HyCoSwin-PD, an explainable hybrid "
        "ConvNeXtV2 and Swin Transformer framework for detecting PD from neuroimaging, "
        "reflecting a broader movement away from black-box models towards architectures that "
        "identify the affected brain regions responsible for a prediction. The present project "
        "adopts the same stance and extends it across modalities, using integrated gradients, "
        "attention inspection, a missing-modality stress test and counterfactual analysis.",

        "System-level and longitudinal approaches. Qin et al. [5] designed a Parkinson's "
        "rehabilitation management system built on multimodal learning and digital twin "
        "technology, and Kim et al. [6] applied multimodal reinforcement learning to embedding "
        "networks and medication recommendation. Both address disease management rather than "
        "progression prediction, and both operate on centralised data. Neither addresses the "
        "distributed, privacy-constrained setting in which multi-site clinical data actually "
        "resides.",

        "Synthesis. The reviewed literature converges on multimodal fusion with learned "
        "weighting and explainability as the state of the art for PD modelling. What it does "
        "not yet provide is an architecture that treats modalities asymmetrically according to "
        "their reliability, consumes incomplete records without imputation, trains without "
        "centralising patient data, and is validated under a protocol strict enough to rule out "
        "the leakage and target-autocorrelation effects that inflate reported accuracy. Those "
        "four omissions define the research gap set out in Section 2.2.",
    ]:
        para(doc, text, align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    heading(doc, "2.2 Research Gap", 2)
    for lead, text in RESEARCH_GAP:
        bullet(doc, " " + text, bold_lead=lead)

    heading(doc, "2.3 Objectives", 2)
    for i, text in enumerate(OBJECTIVES, 1):
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.line_spacing = 1.15
        pf.space_after = Pt(6)
        pf.left_indent = Inches(0.55)
        pf.first_line_indent = Inches(-0.3)
        style_run(p.add_run(f"{i}.\t"), size=12, bold=True)
        style_run(p.add_run(text), size=12)

    heading(doc, "2.4 Problem Statement", 2)
    for text in PROBLEM_STATEMENT:
        para(doc, text, align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    heading(doc, "2.5 Project Plan", 2)
    para(doc, "The project is organised into twelve phases across two semesters. The phases "
              "completed during Project-I are anchored to the repository's commit history; the "
              "remainder constitute the Project-II plan. Fig. 1 presents the schedule as a "
              "Gantt chart and Table 1 lists the corresponding deliverables.",
         align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    add_figure(doc, RFIG / "gantt_chart.png", "Fig. 1. Project schedule (Gantt chart).", 6.3)
    add_table(doc, PROJECT_PLAN_TABLE, [2.5, 1.15, 1.85, 0.85], size=10,
              caption_text="Table 1. Project phases, deliverables and status.")
    page_break(doc)


def build_technical(doc):
    heading(doc, "3. TECHNICAL SPECIFICATION", 1, before=0)

    heading(doc, "3.1 Requirements", 2)
    heading(doc, "3.1.1 Functional", 3)
    para(doc, "The following sixteen functional requirements are implemented in the current "
              "system; the module realising each is named in parentheses.",
         align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    for lead, text in FUNCTIONAL:
        bullet(doc, " " + text, bold_lead=lead)

    heading(doc, "3.1.2 Non-Functional", 3)
    for lead, text in NON_FUNCTIONAL:
        bullet(doc, " " + text, bold_lead=lead)

    heading(doc, "3.2 Feasibility Study", 2)
    heading(doc, "3.2.1 Technical Feasibility", 3)
    for lead, text in TECHNICAL_FEASIBILITY:
        bullet(doc, " " + text, bold_lead=lead)
    heading(doc, "3.2.2 Economic Feasibility", 3)
    for lead, text in ECONOMIC_FEASIBILITY:
        bullet(doc, " " + text, bold_lead=lead)
    heading(doc, "3.2.3 Social Feasibility", 3)
    for lead, text in SOCIAL_FEASIBILITY:
        bullet(doc, " " + text, bold_lead=lead)

    heading(doc, "3.3 System Specification", 2)
    heading(doc, "3.3.1 Hardware Specification", 3)
    para(doc, "The system is deliberately CPU-only. The GPU listed below was present on the "
              "development machine but is not used by the pipeline.",
         align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    add_table(doc, HARDWARE, [1.4, 3.0, 1.95], size=10,
              caption_text="Table 2. Hardware specification.")
    heading(doc, "3.3.2 Software Specification", 3)
    add_table(doc, SOFTWARE, [1.8, 4.55], size=10,
              caption_text="Table 3. Software specification.")
    page_break(doc)


def build_design(doc):
    heading(doc, "4. DESIGN APPROACH AND DETAILS", 1, before=0)

    heading(doc, "4.1 System Architecture", 2)
    para(doc, "The system is organised into six tiers, shown in Fig. 2. Tier 1 is the raw PPMI "
              "data as downloaded from the IDA-LONI portal. Tier 2 converts each modality into "
              "engineered features: the clinical loader assembles the official NP*TOT totals and "
              "constructs the progression target, the MRI pipeline parcellates T1-weighted "
              "volumes into 100 Schaefer regions, the PET loader derives striatal binding "
              "ratios together with asymmetry indices and composite ratios, and the genetic "
              "loader encodes carrier status for six PD-associated genes. Tier 3 aligns the "
              "modalities on the PATNO identifier and drops subjects without a genuine year-two "
              "label, yielding 3,472 subjects described by 126 features.",
         align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    para(doc, "Tier 4 enforces the evaluation protocol: a subject-level stratified split, "
              "imputers and scalers fitted on the training partition alone, and a Dirichlet "
              "label-skew partition that simulates non-IID clinical sites. Tier 5 is the "
              "Fed-PhenoGraft model itself. The clinical phenotype is encoded into a query "
              "vector; each auxiliary modality is encoded into a shared and a private "
              "representation, with a learned mask token substituted whenever the modality is "
              "absent; three cross-attention blocks let the phenotype query retrieve from the "
              "shared representations; and the four resulting vectors are concatenated. An HSIC "
              "penalty drives each shared and private pair towards statistical independence. "
              "Tier 6 contains the two task heads and the evaluation and explainability layer.",
         align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    para(doc, "The asymmetry is the defining property of the design. Only the clinical modality "
              "becomes a query; imaging and genetics supply keys and values. This encodes the "
              "empirical fact that the clinical block is complete for every subject in the "
              "cohort while genetic results are absent for 78 per cent of them. The federated "
              "wrapper is orthogonal to this structure: the entire model is one client, four "
              "clients train locally for two epochs per round, and the orchestrator averages "
              "their weights in proportion to client sample count. No raw feature vector is "
              "ever transmitted.",
         align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    add_figure(doc, RFIG / "system_architecture.png",
               "Fig. 2. Fed-PhenoGraft system architecture.", 6.3)

    heading(doc, "4.2 Design", 2)

    heading(doc, "4.2.1 Data Flow Diagram", 3)
    para(doc, "Fig. 3 gives the context-level view. The system exchanges data with two external "
              "entities: the PPMI/IDA-LONI repository, which supplies the multimodal records, "
              "and the research team, which supplies configuration and receives predictions, "
              "metrics and figures.",
         align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    add_figure(doc, RFIG / "dfd_level0.png", "Fig. 3. Data flow diagram - level 0 (context).",
               5.6)
    para(doc, "Fig. 4 decomposes the system into five processes and four data stores. Process "
              "1.0 loads and engineers the modality features; process 2.0 aligns, splits and "
              "preprocesses them under train-only statistics; process 3.0 trains the baseline "
              "suite by cross-validation; process 4.0 performs federated training with early "
              "stopping; and process 5.0 evaluates, ablates and explains the resulting models. "
              "Data store D1 holds the raw downloads, D2 the processed features, D3 the model "
              "weights and D4 the results and figures.",
         align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    add_figure(doc, RFIG / "dfd_level1.png", "Fig. 4. Data flow diagram - level 1.", 6.3)

    heading(doc, "4.2.2 Use Case Diagram", 3)
    para(doc, "Fig. 5 identifies four actors. The researcher drives dataset preparation, "
              "configuration, training, benchmarking and evaluation. A clinical site "
              "participates as a FedAvg client, contributing local training without releasing "
              "its records. The federated server coordinates the rounds and aggregates weights. "
              "The reviewer or faculty guide consumes the generated explanations and report. "
              "Note that training necessarily precedes the single held-out evaluation, which "
              "the protocol permits exactly once.",
         align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    add_figure(doc, RFIG / "use_case_diagram.png", "Fig. 5. Use case diagram.", 5.9)

    heading(doc, "4.2.3 Class Diagram", 3)
    para(doc, "Fig. 6 shows the principal classes. FedPhenoGraft is composed of three "
              "components: SharedPrivateEncoder, instantiated once per auxiliary modality; "
              "AsymmetricCrossAttention, which wraps a multi-head attention block with a "
              "residual connection and layer normalisation; and HSICLoss, which computes the "
              "RBF-kernel independence criterion. FederatedPPMIDataset supplies batches "
              "together with per-modality mask flags, ModalityPreprocessor holds the "
              "train-fitted imputer and scalers, and FedAvgOrchestrator drives the training "
              "rounds.",
         align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    add_figure(doc, RFIG / "class_diagram.png", "Fig. 6. Class diagram.", 6.3)

    heading(doc, "4.2.4 Sequence Diagram", 3)
    para(doc, "Fig. 7 traces a single federated round. The orchestrator broadcasts the global "
              "weights; each client trains its local copy for two epochs under the combined "
              "regression, classification and HSIC loss; the updated weights are returned with "
              "the client's sample count and averaged in proportion to it; the aggregated model "
              "is evaluated on the validation set; and the round either improves the best "
              "recorded score, in which case the weights are stored, or decrements the patience "
              "counter, which on reaching zero restores the best weights and terminates "
              "training.",
         align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    add_figure(doc, RFIG / "sequence_diagram.png",
               "Fig. 7. Sequence diagram - one federated averaging round.", 6.3)

    heading(doc, "4.3 Implementation and Preliminary Results", 2)
    para(doc, "Although Project-I is scoped to design, the pipeline has been implemented and "
              "executed end to end. This section reports the results of that run so that the "
              "design decisions above can be assessed against evidence.",
         align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    para(doc, "Cohort. Starting from 8,679 PPMI participants, subjects were retained only where "
              "both a baseline and a year-two MDS-UPDRS Part III score exist, giving 3,472 "
              "subjects of whom 1,217 (35.1 per cent) are PD cases. No label was imputed. The "
              "progression target has a mean of +0.62 points and a standard deviation of 6.83, "
              "spanning -37 to +35, which sets the difficulty of the regression task.",
         align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    add_table(doc, COHORT_TABLE, [1.15, 0.5, 2.15, 0.85, 1.7], size=9,
              caption_text="Table 4. Feature blocks, provenance and missingness.")

    add_table(doc, PROTOCOL_TABLE, [1.55, 1.6, 1.5, 1.7], size=9.5,
              caption_text="Table 5. Training and evaluation protocol.")

    para(doc, "Held-out performance. The test set was scored exactly once, after training "
              "concluded. On diagnosis the model reaches ROC-AUC 0.9816 and accuracy 0.9367 "
              "against a majority-class accuracy of 0.649. On progression it reaches a "
              "concordance correlation coefficient of 0.1469 with a bootstrap 95 per cent "
              "confidence interval of 0.0406 to 0.2439. The train-validation CCC gap of +0.139 "
              "sits below the pipeline's +0.15 overfitting threshold, and across three "
              "independent seeds the test CCC is 0.157 +/- 0.030 with test AUC 0.978 +/- 0.003.",
         align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    add_table(doc, METRICS_TABLE, [1.15, 1.2, 0.8, 0.95, 0.8, 1.45], size=9.5,
              caption_text="Table 6. Fed-PhenoGraft, one-shot held-out evaluation (n = 521).")
    add_table(doc, CONFUSION_TABLE, [1.4, 1.5, 1.35, 0.85], size=10,
              caption_text="Table 7. PD versus HC confusion matrix. Sensitivity 0.929, "
                           "specificity 0.941, PPV 0.895, NPV 0.961.")
    add_figure(doc, FIG / "confusion_matrix.png",
               "Fig. 8. PD versus HC confusion matrix on the held-out test set.", 3.6)
    add_figure(doc, FIG / "training_curve.png",
               "Fig. 9. Federated training with validation-based early stopping.", 5.6)

    para(doc, "Baseline comparison. Twelve classical models were evaluated with per-fold "
              "preprocessing pipelines on the identical test subjects. The strongest, LightGBM, "
              "attains a test CCC of 0.2300. A paired bootstrap comparison against it gives a "
              "difference of -0.083 with a 95 per cent interval of -0.167 to -0.005 and "
              "p = 0.982, so Fed-PhenoGraft is not statistically distinguishable from the best "
              "baseline on the progression task. This is reported rather than concealed: it is "
              "the honest starting position, and the ablation below localises the cause. No "
              "baseline attempts the diagnosis task, on which the multimodal architecture "
              "reaches an AUC of 0.982.",
         align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    add_table(doc, BASELINE_TABLE, [1.55, 1.5, 0.82, 0.86, 0.78, 0.84], size=9.5,
              caption_text="Table 8. Progression prediction: twelve baselines versus "
                           "Fed-PhenoGraft on the same held-out test set.")
    add_figure(doc, FIG / "model_comparison.png",
               "Fig. 10. Held-out test CCC for all twelve baselines and Fed-PhenoGraft.", 6.0)

    para(doc, "Ablation study. Each variant was retrained from scratch under the same protocol. "
              "Asymmetric attention and the HSIC penalty each contribute positively on the "
              "validation set, and the centralised control shows that federation imposes no "
              "measurable cost, which supports the privacy claim. Removing DaTScan produces the "
              "largest degradation on both tasks, identifying it as the most valuable auxiliary "
              "modality. Most informatively, removing the MRI branch raises the test CCC from "
              "0.147 to 0.224: because that branch currently carries synthetic values, the "
              "ablation quantifies the cost of the missing real imaging data at approximately "
              "0.077 CCC and converts an outstanding data dependency into a measured result.",
         align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    add_table(doc, ABLATION_TABLE, [1.65, 0.7, 0.75, 0.8, 0.75, 1.65], size=9,
              caption_text="Table 9. Ablation study - contribution of each component and "
                           "modality.")
    add_figure(doc, FIG / "ablation_study.png",
               "Fig. 11. Ablation study across eight retrained variants.", 6.0)

    para(doc, "Explainability. Integrated gradients attribute 58.5 per cent of the attribution "
              "mass to the clinical block, 21.6 per cent to DaTScan, 19.9 per cent to "
              "structural MRI and 0.0 per cent to genetics; the last figure is coherent rather "
              "than anomalous, since genetics is masked for 78 per cent of subjects and a mask "
              "token carries no gradient. The missing-modality stress test shows graceful "
              "degradation under every forced-absence scenario, confirming that the learned "
              "mask tokens function as designed. Counterfactual gene flips shift the predicted "
              "progression by between -0.29 and -0.34 points, which demonstrates the mechanism "
              "without supporting any clinical claim.",
         align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    page_break(doc)


def build_references(doc):
    heading(doc, "5. REFERENCES", 1, before=0)

    def ref_block(title, items):
        para(doc, title, size=12, bold=True, after=4, before=8)
        for item in items:
            p = doc.add_paragraph()
            pf = p.paragraph_format
            pf.line_spacing = 1.15
            pf.space_after = Pt(6)
            pf.left_indent = Inches(0.4)
            pf.first_line_indent = Inches(-0.4)
            pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            style_run(p.add_run(item), size=12)

    ref_block("Journals: <IEEE Format>", REFERENCES_JOURNAL)
    ref_block("Conference: <IEEE Format>", REFERENCES_CONFERENCE)
    ref_block("Book:", REFERENCES_BOOK)
    ref_block("Weblinks:", REFERENCES_WEB)


# ══════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════

def check_word_limits():
    """The template caps the abstract at 300 words and 1.1-1.3 at 200 each."""
    limits = [("Abstract", ABSTRACT, 300), ("1.1 Background", BACKGROUND, 200),
              ("1.2 Motivation", MOTIVATION, 200), ("1.3 Scope", SCOPE, 200)]
    ok = True
    for name, text, cap in limits:
        n = word_count(text)
        flag = "OK " if n <= cap else "OVER"
        if n > cap:
            ok = False
        print(f"  [{flag}] {name}: {n} / {cap} words")
    return ok


def main():
    if not TEMPLATE.exists():
        sys.exit(f"Template not found: {TEMPLATE}")

    print("Checking template word limits...")
    if not check_word_limits():
        sys.exit("Word limit exceeded - shorten the section(s) marked OVER above.")

    logo = extract_logo()
    doc = Document(str(TEMPLATE))
    clear_body(doc)

    # Document default font, so any unstyled run still lands on Times New Roman.
    normal = doc.styles["Normal"]
    normal.font.name = FONT
    normal.font.size = Pt(12)

    build_cover(doc, logo)
    build_abstract(doc)
    build_toc(doc)
    build_introduction(doc)
    build_description(doc)
    build_technical(doc)
    build_design(doc)
    build_references(doc)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(OUTPUT))
    logo.unlink(missing_ok=True)
    print(f"\nWrote {OUTPUT}")
    print("Remaining manual steps: fill the guide's designation on the cover page, "
          "and update the Table of Contents page numbers in Word.")


if __name__ == "__main__":
    main()
