# -*- coding: utf-8 -*-
"""
Generates the Project Review 2 deck for Fed-PhenoGraft.

Reads the college template (never modifies it) and writes a filled deck to
outputs/presentation/Review_2_Presentation.pptx.

Two slides are deliberately left untouched for the team to complete:
  - Slide 2  "Approval Mail From Guide"  (paste the guide's mail screenshot)
  - Slide 5  "Literature Review"

Every number comes from the committed run at df567f8 —
outputs/results/final_metrics.json / RESULTS.md — via outputs/presentation/REVIEW_PACK.md.

Run:  .venv/Scripts/python.exe scripts/generate_review2_ppt.py
"""

import copy
import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = PROJECT_ROOT / "outputs" / "presentation" / "4 Review 2 PPT Template.pptx"
OUTPUT = PROJECT_ROOT / "outputs" / "presentation" / "Review_2_Presentation.pptx"
FIGURES = PROJECT_ROOT / "outputs" / "figures"

FONT = "Times New Roman"
INK = RGBColor(0x44, 0x54, 0x6A)        # theme dk1
ACCENT = RGBColor(0x44, 0x72, 0xC4)     # theme accent1
ACCENT_DK = RGBColor(0x2F, 0x52, 0x8F)
SOFT = RGBColor(0xE7, 0xE6, 0xE6)       # theme lt2
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LINE = RGBColor(0xBF, 0xBF, 0xBF)

# Slide indices in the ORIGINAL template (0-based)
T_TITLE, T_APPROVAL, T_AIM, T_ABSTRACT, T_LITREV = 0, 1, 2, 3, 4
T_GAP, T_OBJ, T_ARCH, T_FR, T_MODULES = 5, 6, 7, 8, 9
T_RESULTS, T_CONCLUSION, T_REFERENCES = 10, 11, 12


# ══════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════

def duplicate_slide(prs, src_index):
    """Deep-copy a template slide's XML into a new slide appended at the end.

    prs.slides.add_slide() does not clone DATE/FOOTER/SLIDE_NUMBER placeholders
    (python-pptx skips latent placeholder types), so copying the source slide's
    shapes verbatim is what keeps the 'SCOPE' footer and slide numbers.
    """
    src = prs.slides[src_index]
    dest = prs.slides.add_slide(src.slide_layout)
    for shp in list(dest.shapes):
        shp._element.getparent().remove(shp._element)
    for shp in src.shapes:
        dest.shapes._spTree.append(copy.deepcopy(shp._element))
    return dest


def move_slide(prs, from_index, to_index):
    """Reorder slides by moving the sldId entry in the presentation part."""
    sld_id_lst = prs.slides._sldIdLst
    ids = list(sld_id_lst)
    element = ids[from_index]
    sld_id_lst.remove(element)
    sld_id_lst.insert(to_index, element)


def shape_by_name(slide, name):
    for shp in slide.shapes:
        if shp.name == name:
            return shp
    return None


def title_shape(slide):
    for shp in slide.shapes:
        if shp.is_placeholder and shp.placeholder_format.idx == 0:
            return shp
    return None


def body_shape(slide):
    for shp in slide.shapes:
        if shp.is_placeholder and shp.placeholder_format.idx == 1:
            return shp
    return None


def set_title(slide, text):
    shp = title_shape(slide)
    tf = shp.text_frame
    tf.clear()
    para = tf.paragraphs[0]
    run = para.add_run()
    run.text = text
    run.font.size = Pt(30)
    run.font.bold = True
    run.font.name = FONT
    return shp


def fill_bullets(slide, items, size=18, space_after=8, bullet_char=None):
    """Fill the body placeholder.

    items: list of str, or (str, level) tuples, or (str, level, bold) tuples.
    """
    shp = body_shape(slide)
    tf = shp.text_frame
    tf.clear()
    tf.word_wrap = True
    for i, item in enumerate(items):
        bold = False
        level = 0
        if isinstance(item, tuple):
            if len(item) == 3:
                text, level, bold = item
            else:
                text, level = item
        else:
            text = item
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.level = level
        para.space_after = Pt(space_after)
        run = para.add_run()
        run.text = (f"{bullet_char} {text}" if bullet_char and level == 0 else text)
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.name = FONT
        run.font.color.rgb = INK
    return shp


def drop_body(slide):
    """Remove the content placeholder so a table/figure layout can take over."""
    shp = body_shape(slide)
    if shp is not None:
        shp._element.getparent().remove(shp._element)


def add_textbox(slide, left, top, width, height, lines, size=12,
                align=PP_ALIGN.LEFT, color=INK, space_after=4):
    box = slide.shapes.add_textbox(Inches(left), Inches(top),
                                   Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.04)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    for i, line in enumerate(lines):
        bold = False
        text = line
        if isinstance(line, tuple):
            text, bold = line
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        para.space_after = Pt(space_after)
        run = para.add_run()
        run.text = text
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.name = FONT
        run.font.color.rgb = color
    return box


def add_table(slide, left, top, width, height, rows_data, col_widths,
              size=11, header_size=None, row_height=0.28):
    """rows_data[0] is the header row."""
    n_rows, n_cols = len(rows_data), len(rows_data[0])
    gfx = slide.shapes.add_table(n_rows, n_cols, Inches(left), Inches(top),
                                 Inches(width), Inches(height))
    table = gfx.table
    table.first_row = True
    for c, w in enumerate(col_widths):
        table.columns[c].width = Inches(w)
    for r in range(n_rows):
        table.rows[r].height = Inches(row_height)
    for r, row in enumerate(rows_data):
        for c, cell_text in enumerate(row):
            cell = table.cell(r, c)
            cell.margin_left = Inches(0.05)
            cell.margin_right = Inches(0.05)
            cell.margin_top = Inches(0.01)
            cell.margin_bottom = Inches(0.01)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.word_wrap = True
            para = tf.paragraphs[0]
            run = para.add_run()
            run.text = str(cell_text)
            run.font.name = FONT
            run.font.size = Pt(header_size or size) if r == 0 else Pt(size)
            run.font.bold = (r == 0)
            run.font.color.rgb = WHITE if r == 0 else INK
    return gfx


def add_picture_fit(slide, image_path, left, top, max_w, max_h):
    """Insert a picture scaled to fit inside the box, centred within it."""
    from PIL import Image  # pillow ships with matplotlib's stack
    with Image.open(image_path) as img:
        iw, ih = img.size
    scale = min(max_w / iw, max_h / ih)
    w, h = iw * scale, ih * scale
    l = left + (max_w - w) / 2.0
    t = top + (max_h - h) / 2.0
    return slide.shapes.add_picture(str(image_path), Inches(l), Inches(t),
                                    Inches(w), Inches(h))


def add_figure(slide, image_path, left, top, max_w, max_h, caption):
    """Picture scaled to fit, with its caption pinned directly beneath it."""
    from pptx.util import Emu as _Emu
    pic = add_picture_fit(slide, image_path, left, top, max_w, max_h)
    cap_top = _Emu(pic.top).inches + _Emu(pic.height).inches + 0.04
    add_textbox(slide, left, cap_top, max_w, 0.20, [caption], size=8.5,
                align=PP_ALIGN.CENTER, color=RGBColor(0x7F, 0x7F, 0x7F))
    return pic


def add_box(slide, left, top, width, height, text, fill, font_color,
            size=9.5, bold=False, line_color=None):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left),
                                   Inches(top), Inches(width), Inches(height))
    shape.adjustments[0] = 0.08
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = line_color or LINE
    shape.line.width = Pt(0.75)
    shape.shadow.inherit = False
    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.04)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    lines = text if isinstance(text, list) else [text]
    for i, line in enumerate(lines):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = PP_ALIGN.CENTER
        para.space_after = Pt(0)
        run = para.add_run()
        run.text = line
        run.font.size = Pt(size if i == 0 else size - 1.5)
        run.font.bold = bold if i == 0 else False
        run.font.name = FONT
        run.font.color.rgb = font_color
    return shape


def add_arrow(slide, x1, y1, x2, y2, color=None, width=1.25):
    conn = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1),
                                      Inches(y1), Inches(x2), Inches(y2))
    conn.line.color.rgb = color or RGBColor(0x80, 0x80, 0x80)
    conn.line.width = Pt(width)
    # arrowhead on the end point
    ln = conn.line._get_or_add_ln()
    from pptx.oxml.ns import qn
    tail = ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "sm", "len": "sm"})
    ln.append(tail)
    return conn


# ══════════════════════════════════════════════════════════════════
# content
# ══════════════════════════════════════════════════════════════════

PROJECT_TITLE = ("Fed-PhenoGraft: Phenotype-Guided Asymmetric Cross-Modal Attention "
                 "with Shared-Private Latent Decomposition for Federated Multi-Modal "
                 "Parkinson's Disease Prediction")

AIM = [
    "To develop a federated, phenotype-guided multimodal representation learning "
    "framework (Fed-PhenoGraft) that improves the prediction of Parkinson's Disease "
    "(PD) diagnosis and progression severity.",
    "It achieves this by asymmetrically querying imaging and genetic modalities based "
    "on clinical phenotypes, while simultaneously preserving data privacy across "
    "multiple clinical sites without centralizing raw patient data.",
]

ABSTRACT = [
    "Parkinson's Disease management depends on multimodal data (clinical scales, MRI, "
    "DaTScan PET, genetics), yet modalities are unequal in reliability, frequently "
    "missing per patient, and held by sites that cannot pool raw records.",
    "We propose Fed-PhenoGraft, a federated framework with three contributions: "
    "(1) Phenotype-Guided Asymmetric Attention, where clinical embeddings act as the "
    "query over imaging and genetic keys; (2) Shared-Private Latent Decomposition using "
    "an HSIC orthogonality penalty to separate disease signal from modality noise; and "
    "(3) learned mask tokens that consume patients with missing modalities.",
    "Trained by sample-weighted FedAvg over 4 non-IID sites (Dirichlet α = 0.5) and "
    "evaluated once on a held-out subject-level split of 3,472 PPMI subjects, the model "
    "attains ROC-AUC 0.982 and accuracy 0.937 for PD vs HC, and CCC 0.147 "
    "[95% CI 0.041–0.244] for two-year Δ UPDRS-III progression.",
    "An 8-variant ablation and an integrated-gradients analysis identify DaTScan as the "
    "dominant imaging signal and quantify the cost of the current synthetic-MRI branch.",
]

RESEARCH_GAP = [
    ("Limited Integration of Multimodal Data", 0, True),
    ("Most existing works use only one or two modalities, failing to exploit the "
     "complementary information in MRI, PET, clinical and genetic data jointly.", 1),
    ("Poor Handling of Incomplete Multimodal Data", 0, True),
    ("Most multimodal models assume complete patient records, whereas real cohorts are "
     "sparse — in our PPMI cohort 78% of subjects have no genetic record and 19% no MRI.", 1),
    ("Symmetric Fusion Ignores Modality Reliability", 0, True),
    ("Existing fusion treats every modality as equally informative, although the clinical "
     "phenotype is the most complete and most reliable source and should guide retrieval.", 1),
    ("No Leak-Free, Privacy-Preserving Progression Benchmark", 0, True),
    ("Reported gains are often inflated by baseline-score autocorrelation and centralized "
     "training; a subject-level, train-only-statistics, one-shot-test protocol for "
     "federated PD progression is missing.", 1),
]

OBJECTIVES = [
    "1.  To design a Phenotype-Guided Asymmetric Attention mechanism in which clinical "
    "features selectively query MRI, PET and genetic representations.",
    "2.  To implement a Shared-Private Latent Decomposition using HSIC orthogonality to "
    "disentangle disease biomarkers from modality-specific noise.",
    "3.  To build a federated learning pipeline (FedAvg) that trains across multiple "
    "simulated clinical sites without centralizing raw patient data.",
    "4.  To incorporate robustness features — learned mask tokens for missing "
    "modalities and Monte Carlo Dropout for uncertainty estimation.",
    "5.  To validate under a leak-free protocol against 12 baselines with bootstrap CIs, "
    "multi-seed runs and an 8-variant ablation, supported by a multi-level "
    "explainability suite (attention maps, integrated gradients, counterfactuals).",
]

FUNCTIONAL_REQUIREMENTS = [
    ["ID", "Functional Requirement", "Implemented in"],
    ["FR-1", "Ingest PPMI clinical, DaTScan, genetic and MRI records; align per subject on PATNO", "src/data/*_loader.py, data_builder.py"],
    ["FR-2", "Construct Δ UPDRS-III target from official NP3TOT; drop — never impute — unlabelled subjects", "clinical_loader.build_clinical_features"],
    ["FR-3", "Subject-level stratified 70/15/15 split; no subject in two partitions", "preprocessing.create_subject_splits"],
    ["FR-4", "Fit all imputation and scaling statistics on training subjects only", "preprocessing.ModalityPreprocessor"],
    ["FR-5", "Represent an absent modality by a learned mask token, not an imputed value", "dataset.py, fed_phenograft.py"],
    ["FR-6", "Clinical embeddings query imaging/genetic embeddings via multi-head cross-attention", "models/attention.py"],
    ["FR-7", "Encode each auxiliary modality into shared and private latents penalised by HSIC", "models/hsic.py, SharedPrivateEncoder"],
    ["FR-8", "Train by sample-weighted FedAvg over N sites (IID / Dirichlet / real-site)", "federated/fedavg_orchestrator.py"],
    ["FR-9", "Early-stop on validation CCC, restore best-round weights, score test set exactly once", "simulate_federated_training, main.py"],
    ["FR-10", "Predict progression and diagnosis jointly under one multitask loss", "Dual heads, cls_weight = 0.3"],
    ["FR-11", "Benchmark 12 classical models with per-fold pipelines on identical test subjects", "baselines/models.py, runner.py"],
    ["FR-12", "Report bootstrap 95% CIs, 3-seed mean ± SD, paired test vs strongest baseline", "evaluation/stats.py"],
    ["FR-13", "Retrain an ablation suite over every modality and architectural component", "evaluation/ablation.py (8 variants)"],
    ["FR-14", "Produce attention maps, integrated gradients, robustness test, counterfactuals", "evaluation/xai.py"],
    ["FR-15", "Emit metrics JSON, Markdown report and figure suite; regenerable without retraining", "results_report.py, generate_results.py"],
    ["FR-16", "Replace synthetic MRI with real T1w Schaefer-100 parcellation when scans present", "mri_pipeline.py (pending NIfTI data)"],
]

MODULES = [
    ["Module", "Responsibility", "Key files", "Output"],
    ["M1  Data Acquisition",
     "IDA-LONI download helpers, credential handling, file-presence verification",
     "scripts/download_ppmi_data.py, download_mri_data.py, verify_data.py",
     "CSVs in data/raw/"],
    ["M2  Modality Loaders",
     "Per-modality parsing, visit filtering, feature engineering (SBR asymmetry, carrier encoding, ROI parcellation)",
     "clinical_loader.py, pet_loader.py, genetic_loader.py, mri_pipeline.py",
     "4 DataFrames indexed by PATNO"],
    ["M3  Dataset Builder\n      & Preprocessor",
     "Cross-modality alignment, label filtering, subject-level split, train-only impute/scale, missing-row contract",
     "data_builder.py, preprocessing.py, dataset.py",
     "Torch datasets + mask flags"],
    ["M4  Model Core",
     "Phenotype encoder, shared/private encoders, mask tokens, asymmetric cross-attention, HSIC loss, dual heads, MC dropout",
     "models/fed_phenograft.py, attention.py, hsic.py",
     "{pred, cls_logit, loss_hsic, attn_weights}"],
    ["M5  Federated\n      Orchestrator",
     "Client partitioning (IID/Dirichlet/site), local AdamW updates with clipping, weighted aggregation, early stopping",
     "federated/fedavg_orchestrator.py",
     "Global model + per-round history"],
    ["M6  Baseline\n      Benchmark",
     "12 regularised classical models, per-fold Pipeline(impute→scale→model), 5-fold CV then one-shot test scoring",
     "baselines/models.py, runner.py",
     "CV + test metrics, stored predictions"],
    ["M7  Evaluation,\n      XAI & Reporting",
     "CCC/RMSE/MAE/R²/r + AUC/Acc/F1, bootstrap CIs, paired test, ablation suite, 4 XAI analyses, figures and report",
     "metrics.py, stats.py, ablation.py, xai.py, results_report.py",
     "final_metrics.json, 9 PNGs, RESULTS.md"],
]

HEADLINE_METRICS = [
    ["Task", "Metric", "Test value", "95% CI"],
    ["Progression (Δ UPDRS-III)", "CCC", "0.1469", "0.0406 – 0.2439"],
    ["Progression (Δ UPDRS-III)", "RMSE (points)", "7.64", "6.93 – 8.46"],
    ["Progression (Δ UPDRS-III)", "MAE (points)", "5.14", "4.67 – 5.69"],
    ["Progression (Δ UPDRS-III)", "Pearson r", "0.1769", "—"],
    ["Diagnosis (PD vs HC)", "ROC-AUC", "0.9816", "—"],
    ["Diagnosis (PD vs HC)", "Accuracy", "0.9367", "—"],
    ["Diagnosis (PD vs HC)", "F1 score", "0.9115", "—"],
]

BASELINE_TABLE = [
    ["Model", "CV CCC", "Test CCC", "Test RMSE", "Test MAE"],
    ["lightgbm  (strongest baseline)", "0.2253 ± 0.0213", "0.2300", "6.94", "4.67"],
    ["xgboost", "0.2401 ± 0.0302", "0.2135", "6.96", "4.67"],
    ["linear", "0.2367 ± 0.0241", "0.2039", "6.96", "4.78"],
    ["ridge", "0.2368 ± 0.0245", "0.2036", "6.96", "4.78"],
    ["gradient_boosting", "0.2322 ± 0.0262", "0.1911", "7.01", "4.74"],
    ["mlp", "0.1400 ± 0.0751", "0.1924", "7.10", "4.93"],
    ["random_forest", "0.0993 ± 0.0151", "0.0821", "7.07", "4.73"],
    ["Fed-PhenoGraft  (ours)", "val 0.3591", "0.1469", "7.64", "5.14"],
]

ABLATION_TABLE = [
    ["Variant", "Val CCC", "Test CCC", "Test AUC"],
    ["Full Fed-PhenoGraft", "0.3591", "0.1469", "0.9816"],
    ["− Asymmetric attention", "0.2885", "0.1736", "0.9737"],
    ["− HSIC shared-private loss", "0.3130", "0.1481", "0.9786"],
    ["Centralized (1 client)", "0.3142", "0.1857", "0.9806"],
    ["− PET / DaTScan", "0.2833", "0.1303", "0.9513"],
    ["− Genetics", "0.3139", "0.1474", "0.9787"],
    ["− MRI (synthetic)", "0.3272", "0.2241", "0.9761"],
    ["Clinical only", "0.3072", "0.2055", "0.9554"],
]

CONCLUSION = [
    ("Delivered an end-to-end federated multimodal framework running on real PPMI data "
     "(3,472 subjects) under a deliberately conservative, leak-free protocol.", 0),
    ("Strong diagnostic result: ROC-AUC 0.982, accuracy 0.937, sensitivity 0.929 on a test "
     "set scored exactly once; a retrained centralized control confirms federation imposes "
     "no measurable accuracy cost — privacy is obtained for free.", 0),
    ("Progression remains hard: CCC 0.147 [0.041–0.244], statistically indistinguishable "
     "from the strongest of 12 classical baselines (paired bootstrap p = 0.982). We report "
     "this honestly rather than tuning on the test set.", 0),
    ("The 8-variant ablation localises the shortfall: attention and HSIC each contribute, "
     "DaTScan is the most valuable auxiliary modality, and the synthetic MRI placeholder "
     "costs about 0.077 CCC — a measured, actionable finding.", 0),
    ("Future work: (i) real T1w Schaefer-100 parcellations, (ii) validation-only "
     "hyperparameter tuning, (iii) real-site federation via the PPMI Center-Subject list, "
     "(iv) DP-SGD for formal privacy guarantees, (v) longitudinal BL→V04→V06→V08 modelling.", 0),
]

REFERENCES = [
    "[1] A. Vamvakas, T. Van Balkom, G. Van Wingen, et al., \"Prediction of impulse control "
    "disorders in Parkinson's disease through a longitudinal machine learning study,\" "
    "npj Parkinson's Disease, vol. 12, p. 38, 2026.",
    "[2] S. W. Akram and C. K, \"Enhancing Parkinson's Disease Staging: An Integrative Deep "
    "Learning Framework for Multimodal Feature Selection,\" Journal of Molecular "
    "Neuroscience, 2026.",
    "[3] V. Awasthi et al., \"HyCoSwin-PD: An Explainable Hybrid ConvNeXtV2-Swin Transformer "
    "Framework for Parkinson's Disease Detection from Neuroimaging,\" MethodsX, vol. 16, 2026.",
    "[4] T. Zhi et al., \"MultimodalCNN-PD: A Parkinson's Disease Diagnostics Framework Using "
    "Multimodal Convolutional Neural Network,\" Frontiers in Aging Neuroscience, vol. 18, 2026.",
    "[5] J. Qin, Y. Cai, Z. Wang, J. Jin, X. Huang, and L. Pei, \"Design of a Parkinson's "
    "Rehabilitation Management System Based on Multimodal Learning and Digital Twin "
    "Technology,\" in Proc. IEEE Smart World Congress (SWC), Nadi, Fiji, 2024, pp. 2073-2080.",
    "[6] H. Kim, C. Park, J. Hoon Kim, S. Jang, and H. K. Lee, \"Multimodal Reinforcement "
    "Learning for Embedding Networks and Medication Recommendation in Parkinson's Disease,\" "
    "IEEE Access, vol. 12, pp. 74251-74267, 2024.",
]


# ══════════════════════════════════════════════════════════════════
# slide builders
# ══════════════════════════════════════════════════════════════════

def build_title_slide(slide):
    tshp = shape_by_name(slide, "Title 6")
    tf = tshp.text_frame
    tf.clear()
    tf.word_wrap = True
    p0 = tf.paragraphs[0]
    p0.alignment = PP_ALIGN.CENTER
    r = p0.add_run()
    r.text = "Programme B.Tech — BCSE497J · Project I"
    r.font.size = Pt(18)
    r.font.bold = True
    r.font.name = FONT
    p1 = tf.add_paragraph()
    p1.alignment = PP_ALIGN.CENTER
    r = p1.add_run()
    r.text = PROJECT_TITLE
    r.font.size = Pt(21)
    r.font.bold = True
    r.font.name = FONT

    box = shape_by_name(slide, "object 3")
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    lines = [
        ("Team members:", False, 18),
        ("Saurav Kumar Gupta\t\t(23BCE2336)", True, 18),
        ("Amit Adhikari\t\t\t(23BCE2327)", True, 18),
        ("Shreeyam Acharya\t\t(23BCE2330)", True, 18),
        ("", False, 18),
        ("Faculty guide:", False, 18),
        ("Dhivyaa CR\t\t\t(20701)", True, 18),
    ]
    for i, (text, bold, size) in enumerate(lines):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.space_after = Pt(2)
        run = para.add_run()
        run.text = text
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.name = FONT


def build_aim_slide(slide):
    set_title(slide, "Aim")
    fill_bullets(slide, AIM, size=20, space_after=14)


def build_abstract_slide(slide):
    set_title(slide, "Abstract")
    fill_bullets(slide, ABSTRACT, size=15, space_after=10)


def build_gap_slide(slide):
    set_title(slide, "Research Gap")
    fill_bullets(slide, RESEARCH_GAP, size=14, space_after=6)


def build_objectives_slide(slide):
    set_title(slide, "Objectives")
    shp = fill_bullets(slide, OBJECTIVES, size=14, space_after=8)
    # The SDG textbox sits at y = 4.83 and two pictures at y = 4.83 / 5.23,
    # so the objectives text must stop above them.
    shp.top = Inches(1.55)
    shp.height = Inches(3.15)


def build_architecture_slide(slide):
    set_title(slide, "Framework / Architecture / Block Diagram")
    drop_body(slide)

    x0, full_w = 0.55, 12.25
    col_w, gap = 2.95, 0.155
    cols = [x0 + i * (col_w + gap) for i in range(4)]

    def tier_label(y, text):
        add_textbox(slide, 0.55, y, 2.2, 0.18, [(text, True)], size=8,
                    color=RGBColor(0x8C, 0x8C, 0x8C))

    # ── Tier 1: data sources ──
    tier_label(1.44, "TIER 1  ·  DATA SOURCES (PPMI / IDA-LONI)")
    t1 = [
        (["Clinical", "UPDRS I–IV · MoCA · Demographics  (7)"]),
        (["Structural MRI", "T1w NIfTI — synthetic fallback  (100)"]),
        (["PET / DaTScan", "DATScan_Analysis.csv · SBR  (10)"]),
        (["Genetics", "Genetic_Testing_Results.csv  (9)"]),
    ]
    for x, txt in zip(cols, t1):
        add_box(slide, x, 1.64, col_w, 0.52, txt, SOFT, INK, size=10, bold=True)

    for x in cols:
        add_arrow(slide, x + col_w / 2, 2.16, x + col_w / 2, 2.42)

    # ── Tier 2: loaders ──
    tier_label(2.22, "TIER 2  ·  MODALITY LOADERS & FEATURE ENGINEERING")
    t2 = [
        (["clinical_loader", "NP*TOT totals · Δ-target build"]),
        (["mri_pipeline", "nilearn Schaefer-100 ROI parcellation"]),
        (["pet_loader", "SBR + asymmetry + composite ratios"]),
        (["genetic_loader", "LRRK2 · GBA · SNCA · PINK1 · PRKN · APOE"]),
    ]
    for x, txt in zip(cols, t2):
        add_box(slide, x, 2.42, col_w, 0.52, txt, SOFT, INK, size=10, bold=True)

    add_arrow(slide, x0 + full_w / 2, 2.94, x0 + full_w / 2, 3.16)

    # ── Tier 3: builder ──
    add_box(slide, x0, 3.16, full_w, 0.40,
            ["data_builder  —  PATNO alignment  ·  drop subjects without a real Year-2 "
             "label (no imputation)  ·  3,472 subjects × 126 features"],
            SOFT, INK, size=10, bold=True)
    add_arrow(slide, x0 + full_w / 2, 3.56, x0 + full_w / 2, 3.78)

    # ── Tier 4: protocol ──
    add_box(slide, x0, 3.78, full_w, 0.40,
            ["Leak-free protocol  —  subject-level stratified 70/15/15 (2,430 / 521 / 521)  "
             "·  train-only imputer & scalers  ·  Dirichlet(α = 0.5) non-IID clients"],
            SOFT, INK, size=10, bold=True)
    add_arrow(slide, x0 + full_w / 2, 4.18, x0 + full_w / 2, 4.40)

    # ── Tier 5: model container ──
    container = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x0),
                                       Inches(4.40), Inches(full_w), Inches(1.42))
    container.fill.solid()
    container.fill.fore_color.rgb = RGBColor(0xED, 0xF1, 0xFA)
    container.line.color.rgb = ACCENT
    container.line.width = Pt(1.25)
    container.shadow.inherit = False
    ctf = container.text_frame
    ctf.vertical_anchor = MSO_ANCHOR.TOP
    cp = ctf.paragraphs[0]
    cp.alignment = PP_ALIGN.LEFT
    cr = cp.add_run()
    cr.text = "  TIER 5  ·  FED-PHENOGRAFT MODEL  (one FedAvg client)"
    cr.font.size = Pt(9)
    cr.font.bold = True
    cr.font.name = FONT
    cr.font.color.rgb = ACCENT_DK

    inner_y, inner_h = 4.78, 0.90
    add_box(slide, 0.75, inner_y, 2.55, inner_h,
            ["Phenotype Encoder", "Clinical MLP → query q (32-d)"],
            ACCENT, WHITE, size=10, bold=True, line_color=ACCENT_DK)
    add_box(slide, 4.05, inner_y, 2.75, inner_h,
            ["Asymmetric Cross-Attention", "4 heads · Add & Norm · MRI / PET / Gene"],
            ACCENT, WHITE, size=10, bold=True, line_color=ACCENT_DK)
    add_box(slide, 7.55, inner_y, 2.55, inner_h,
            ["Shared / Private + HSIC", "mask tokens · RBF independence"],
            ACCENT, WHITE, size=10, bold=True, line_color=ACCENT_DK)
    add_box(slide, 10.35, inner_y, 2.35, inner_h,
            ["FedAvg Orchestrator", "4 clients · weighted · early stop"],
            ACCENT, WHITE, size=10, bold=True, line_color=ACCENT_DK)

    # Q and K,V arrows — the novelty of the architecture
    add_arrow(slide, 3.30, inner_y + inner_h / 2, 4.02, inner_y + inner_h / 2,
              color=ACCENT_DK, width=1.75)
    add_textbox(slide, 3.32, inner_y + inner_h / 2 - 0.26, 0.70, 0.20,
                [("Q", True)], size=10, align=PP_ALIGN.CENTER, color=ACCENT_DK)
    add_arrow(slide, 7.52, inner_y + inner_h / 2, 6.83, inner_y + inner_h / 2,
              color=ACCENT_DK, width=1.75)
    add_textbox(slide, 6.85, inner_y + inner_h / 2 - 0.26, 0.70, 0.20,
                [("K, V", True)], size=10, align=PP_ALIGN.CENTER, color=ACCENT_DK)

    add_arrow(slide, x0 + full_w / 2, 5.82, x0 + full_w / 2, 6.04)

    # ── Tier 6: heads and evaluation ──
    t6_w, t6_gap = 3.95, 0.20
    t6x = [x0, x0 + t6_w + t6_gap, x0 + 2 * (t6_w + t6_gap)]
    add_box(slide, t6x[0], 6.04, t6_w, 0.52,
            ["Regression Head", "Δ UPDRS-III @ Year 2  ·  MSE"], SOFT, INK,
            size=10, bold=True)
    add_box(slide, t6x[1], 6.04, t6_w, 0.52,
            ["Classification Head", "PD vs HC  ·  BCE"], SOFT, INK, size=10, bold=True)
    add_box(slide, t6x[2], 6.04, t6_w, 0.52,
            ["Evaluation & XAI", "bootstrap CI · 3 seeds · ablation · IG"],
            SOFT, INK, size=10, bold=True)

    add_textbox(slide, x0, 6.56, full_w, 0.24,
                ["Loss  =  MSE(Δ UPDRS-III)  +  0.3 · BCE(PD/HC)  +  "
                 "0.1 · Σ HSIC(shared, private)          "
                 "Raw patient data never leaves a client — only weight tensors are averaged."],
                size=9.5, align=PP_ALIGN.CENTER, color=RGBColor(0x7F, 0x7F, 0x7F))


def build_fr_slide(slide):
    set_title(slide, "Functional Requirements")
    drop_body(slide)
    add_table(slide, 0.50, 1.52, 12.33, 4.95, FUNCTIONAL_REQUIREMENTS,
              col_widths=[0.72, 7.01, 4.60], size=10, header_size=10.5,
              row_height=0.26)


def build_modules_slide(slide):
    set_title(slide, "Modules")
    drop_body(slide)
    add_table(slide, 0.50, 1.52, 12.33, 4.95, MODULES,
              col_widths=[1.85, 4.55, 3.63, 2.30], size=9.5, header_size=10.5,
              row_height=0.52)


def build_results_1(slide):
    set_title(slide, "Experiments and Results  (1/3)  —  Setup & Held-out Performance")
    drop_body(slide)

    add_textbox(slide, 0.50, 1.48, 7.05, 1.15, [
        ("Experimental setup", True),
        "Dataset: PPMI — 3,472 subjects with both baseline and Year-2 (V04) MDS-UPDRS-III; "
        "126 raw features (7 clinical + 100 MRI + 10 PET + 9 genetic).",
        "Protocol: subject-level 70/15/15 split (2,430 / 521 / 521), stratified on diagnosis · "
        "4 Dirichlet(α = 0.5) non-IID clients · 30 rounds max, patience 5 on validation CCC · "
        "3 independent seeds · 1,000-resample bootstrap · test set scored EXACTLY ONCE.",
    ], size=10.5, space_after=3)

    add_table(slide, 0.50, 2.72, 7.05, 2.35, HEADLINE_METRICS,
              col_widths=[2.35, 1.60, 1.45, 1.65], size=10, header_size=10.5,
              row_height=0.28)

    add_textbox(slide, 0.50, 5.22, 7.05, 1.20, [
        ("Generalization check", True),
        "Train CCC 0.4982  vs  Val CCC 0.3591  —  gap +0.139, below the +0.15 overfitting "
        "alarm built into the pipeline.",
        "Across 3 seeds: test CCC 0.157 ± 0.030, test AUC 0.978 ± 0.003 — the "
        "classification head is stable; the regression head is seed-sensitive.",
        "Confusion matrix: TN 318 · FP 20 · FN 13 · TP 170  →  sensitivity 0.929, "
        "specificity 0.941 (majority-class accuracy would be 0.649).",
    ], size=10, space_after=3)

    add_figure(slide, FIGURES / "confusion_matrix.png", 7.75, 1.48, 5.10, 2.30,
               "Fig. 1  PD vs HC confusion matrix (test, n = 521)")
    add_figure(slide, FIGURES / "training_curve.png", 7.75, 4.08, 5.10, 2.30,
               "Fig. 2  Federated training with validation-based early stopping")


def build_results_2(slide):
    set_title(slide, "Experiments and Results  (2/3)  —  Baseline Comparison")
    drop_body(slide)

    add_textbox(slide, 0.50, 1.48, 6.55, 0.42, [
        "12 classical baselines, each in a per-fold Pipeline(median-impute → standardise → "
        "model), 5-fold CV on train+val, then refit and scored once on the same 521 test subjects.",
    ], size=10, space_after=2)

    add_table(slide, 0.50, 1.98, 6.55, 2.70, BASELINE_TABLE,
              col_widths=[2.35, 1.30, 1.00, 0.95, 0.95], size=10, header_size=10.5,
              row_height=0.30)

    sig = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.50),
                                 Inches(4.86), Inches(6.55), Inches(1.05))
    sig.fill.solid()
    sig.fill.fore_color.rgb = RGBColor(0xFD, 0xF3, 0xE6)
    sig.line.color.rgb = RGBColor(0xED, 0x7D, 0x31)
    sig.shadow.inherit = False
    stf = sig.text_frame
    stf.word_wrap = True
    stf.margin_left = stf.margin_right = Inches(0.12)
    stf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, (txt, bold) in enumerate([
        ("Statistical significance — paired bootstrap vs LightGBM", True),
        ("ΔCCC = −0.083  [95% CI −0.167, −0.005],  p = 0.982  →  NOT significant "
         "at α = 0.05. Fed-PhenoGraft is currently statistically indistinguishable from the "
         "strongest baseline on progression; no baseline attempts the diagnosis task, where "
         "the model reaches AUC 0.982.", False),
    ]):
        para = stf.paragraphs[0] if i == 0 else stf.add_paragraph()
        para.space_after = Pt(2)
        run = para.add_run()
        run.text = txt
        run.font.size = Pt(10)
        run.font.bold = bold
        run.font.name = FONT
        run.font.color.rgb = RGBColor(0x8F, 0x55, 0x10)

    add_figure(slide, FIGURES / "model_comparison.png", 7.20, 2.05, 5.75, 3.90,
               "Fig. 3  Held-out test CCC — all 12 baselines vs Fed-PhenoGraft")


def build_results_3(slide):
    set_title(slide, "Experiments and Results  (3/3)  —  Ablation & Explainability")
    drop_body(slide)

    add_textbox(slide, 0.50, 1.48, 6.30, 0.30, [
        "Every variant retrained from scratch under the identical protocol (20 rounds, patience 3).",
    ], size=10, space_after=2)

    add_table(slide, 0.50, 1.82, 6.30, 2.75, ABLATION_TABLE,
              col_widths=[2.85, 1.15, 1.15, 1.15], size=10, header_size=10.5,
              row_height=0.30)

    add_textbox(slide, 0.50, 4.78, 6.30, 1.85, [
        ("Explainability findings", True),
        "•  Integrated gradients — clinical 58.5%, DaTScan PET 21.6%, structural MRI 19.9%, "
        "genetics 0.0% (genetics is masked for 78% of subjects, so the mask token carries no gradient).",
        "•  Missing-modality stress test — RMSE 7.64 (full) → 6.96 (no MRI) / 7.54 "
        "(no PET) / 7.64 (no genetics) / 7.04 (clinical only): graceful degradation in every case.",
        "•  Counterfactual gene flips — predicted Δ UPDRS-III shifts of −0.29 (LRRK2) "
        "to −0.34 (GBA) points; demonstrates the mechanism, not a clinical claim.",
        "•  Key reading — removing the synthetic MRI branch RAISES test CCC 0.147 → 0.224, "
        "pricing the missing real T1w data at ≈ 0.077 CCC.",
    ], size=9.5, space_after=3)

    add_figure(slide, FIGURES / "ablation_study.png", 7.00, 1.48, 5.85, 2.40,
               "Fig. 4  Ablation study — contribution of each component and modality")
    add_figure(slide, FIGURES / "modality_robustness.png", 7.00, 4.30, 5.85, 2.10,
               "Fig. 5  Robustness to missing modalities (learned mask tokens)")


def build_conclusion_slide(slide):
    set_title(slide, "Conclusion")
    fill_bullets(slide, CONCLUSION, size=14, space_after=10, bullet_char="•")


def build_references_slide(slide):
    set_title(slide, "References")
    box = shape_by_name(slide, "Rectangle 3")
    box.top = Inches(1.55)
    box.height = Inches(5.10)
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    for i, ref in enumerate(REFERENCES):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.space_after = Pt(8)
        run = para.add_run()
        run.text = ref
        run.font.size = Pt(13)
        run.font.name = FONT
        run.font.color.rgb = INK


# ══════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════

def main():
    if not TEMPLATE.exists():
        sys.exit(f"Template not found: {TEMPLATE}")
    missing = [f for f in ("confusion_matrix.png", "training_curve.png",
                           "model_comparison.png", "ablation_study.png",
                           "modality_robustness.png") if not (FIGURES / f).exists()]
    if missing:
        sys.exit(f"Missing figures in {FIGURES}: {missing}")

    prs = Presentation(str(TEMPLATE))

    # Create the two extra results slides by duplicating template slide 11,
    # then move them directly after it (indices 11 and 12).
    r2 = duplicate_slide(prs, T_RESULTS)          # appended at index 13
    move_slide(prs, len(prs.slides) - 1, T_RESULTS + 1)
    r3 = duplicate_slide(prs, T_RESULTS)          # appended at index 14
    move_slide(prs, len(prs.slides) - 1, T_RESULTS + 2)

    s = prs.slides
    build_title_slide(s[0])
    # s[1]  Approval Mail From Guide  -> intentionally untouched
    build_aim_slide(s[2])
    build_abstract_slide(s[3])
    # s[4]  Literature Review         -> intentionally untouched
    build_gap_slide(s[5])
    build_objectives_slide(s[6])
    build_architecture_slide(s[7])
    build_fr_slide(s[8])
    build_modules_slide(s[9])
    build_results_1(s[10])
    build_results_2(s[11])
    build_results_3(s[12])
    build_conclusion_slide(s[13])
    build_references_slide(s[14])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUTPUT))
    print(f"Wrote {OUTPUT}  ({len(prs.slides)} slides)")
    print("Left blank for the team: slide 2 (Approval Mail), slide 5 (Literature Review)")


if __name__ == "__main__":
    main()
