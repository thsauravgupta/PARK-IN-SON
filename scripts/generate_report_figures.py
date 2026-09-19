# -*- coding: utf-8 -*-
"""
Draws the diagrams the BCSE497J Project-I report needs but the pipeline does not
produce: Gantt chart, system architecture, DFD (level 0 and 1), use case, class
and sequence diagrams.

All figures are written to outputs/figures/report/ at 200 dpi on a white ground,
sized for a single-column A4 page with 1 inch margins (6.5 in usable width).

Run:  .venv/Scripts/python.exe scripts/generate_report_figures.py
"""

import datetime as dt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "outputs" / "figures" / "report"

FONT = "Times New Roman"
INK = "#44546A"
ACCENT = "#4472C4"
ACCENT_DK = "#2F528F"
SOFT = "#E7E6E6"
SOFT_BLUE = "#EDF1FA"
GREY = "#8C8C8C"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [FONT, "DejaVu Serif"],
    "axes.edgecolor": INK,
    "text.color": INK,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})


# ══════════════════════════════════════════════════════════════════
# primitives
# ══════════════════════════════════════════════════════════════════

def box(ax, x, y, w, h, text, fc=SOFT, ec=GREY, tc=INK, fs=8, bold=False,
        rounded=True, lw=0.9):
    """Rounded box with centred text. (x, y) is the bottom-left corner."""
    style = "round,pad=0,rounding_size=0.06" if rounded else "square,pad=0"
    patch = FancyBboxPatch((x, y), w, h, boxstyle=style, linewidth=lw,
                           edgecolor=ec, facecolor=fc, mutation_aspect=1)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal", linespacing=1.35,
            zorder=5)
    return patch


def arrow(ax, xy_from, xy_to, color=GREY, lw=1.0, style="-|>", ls="-",
          connectionstyle="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch(xy_from, xy_to, arrowstyle=style,
                                 mutation_scale=9, color=color, linewidth=lw,
                                 linestyle=ls, connectionstyle=connectionstyle,
                                 shrinkA=1, shrinkB=1, zorder=4))


def label(ax, x, y, text, fs=7.5, color=GREY, ha="center", va="center",
          bold=False, style="normal"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=color,
            fontweight="bold" if bold else "normal", fontstyle=style, zorder=6)


def blank_axes(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"  wrote {path.relative_to(PROJECT_ROOT)}")


# ══════════════════════════════════════════════════════════════════
# Fig. 1 — Gantt chart
# ══════════════════════════════════════════════════════════════════

def gantt_chart():
    """Project timeline. Completed phases are anchored to the real commit history
    (8e75b07 04-Jun 2026 to df567f8 08-Sep 2026); the rest is the Project-II plan."""
    tasks = [
        ("Literature survey and problem formulation", "2026-06-04", "2026-06-20", True),
        ("PPMI data access, download and verification", "2026-06-04", "2026-06-25", True),
        ("Modality loaders (clinical, DaTScan, genetics)", "2026-06-10", "2026-07-05", True),
        ("MRI parcellation pipeline (nilearn Schaefer-100)", "2026-06-12", "2026-07-10", True),
        ("Leak-free preprocessing and split protocol", "2026-06-15", "2026-07-20", True),
        ("Fed-PhenoGraft model (attention, HSIC, mask tokens)", "2026-07-01", "2026-08-05", True),
        ("FedAvg orchestrator and client partitioning", "2026-07-20", "2026-08-18", True),
        ("Baseline suite (12 models) and metric layer", "2026-08-01", "2026-08-25", True),
        ("Statistical rigour, ablation and XAI suite", "2026-08-19", "2026-09-08", True),
        ("Review-2 documentation and project report", "2026-08-28", "2026-09-20", True),
        ("Real T1w MRI acquisition and integration", "2026-09-20", "2026-10-20", False),
        ("Hyperparameter tuning (validation only)", "2026-10-01", "2026-10-31", False),
        ("Real-site federation and differential privacy", "2026-10-15", "2026-11-20", False),
        ("Longitudinal extension and final evaluation", "2026-11-01", "2026-11-30", False),
    ]

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for i, (name, start, end, done) in enumerate(reversed(tasks)):
        s = dt.datetime.strptime(start, "%Y-%m-%d")
        e = dt.datetime.strptime(end, "%Y-%m-%d")
        ax.barh(i, (e - s).days, left=s, height=0.58,
                color=ACCENT if done else SOFT,
                edgecolor=ACCENT_DK if done else GREY, linewidth=0.8, zorder=2)

    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([t[0] for t in reversed(tasks)], fontsize=8.5)
    ax.set_ylim(-1.6, len(tasks) - 0.3)

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.set_xlim(dt.datetime(2026, 6, 1), dt.datetime(2026, 12, 5))
    ax.tick_params(axis="x", labelsize=8.5)
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    review2 = dt.datetime(2026, 9, 9)
    ax.axvline(review2, color="#C0504D", linewidth=1.3, linestyle="--", zorder=3)
    ax.text(review2, len(tasks) - 0.35, " Review 2", fontsize=8.5, color="#C0504D",
            va="bottom", ha="left", fontweight="bold")

    done_patch = Rectangle((0, 0), 1, 1, facecolor=ACCENT, edgecolor=ACCENT_DK)
    plan_patch = Rectangle((0, 0), 1, 1, facecolor=SOFT, edgecolor=GREY)
    ax.legend([done_patch, plan_patch], ["Completed (Project-I)", "Planned (Project-II)"],
              loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2, fontsize=8.5,
              frameon=False)
    fig.tight_layout()
    save(fig, "gantt_chart.png")


# ══════════════════════════════════════════════════════════════════
# Fig. 2 — System architecture
# ══════════════════════════════════════════════════════════════════

def system_architecture():
    fig, ax = blank_axes(9.2, 7.4)
    cols = [2, 26.5, 51, 75.5]
    cw = 22.5

    def tier(y, text):
        label(ax, 2, y, text, fs=7.5, color=GREY, ha="left", bold=True)

    tier(98, "TIER 1  ·  DATA SOURCES (PPMI / IDA-LONI)")
    t1 = ["Clinical\nUPDRS I–IV, MoCA,\nDemographics  (7)",
          "Structural MRI\nT1w NIfTI —\nsynthetic fallback  (100)",
          "PET / DaTScan\nDATScan_Analysis.csv\nSBR  (10)",
          "Genetics\nGenetic_Testing_\nResults.csv  (9)"]
    for x, t in zip(cols, t1):
        box(ax, x, 88, cw, 8.5, t, fs=7.5)

    tier(85.2, "TIER 2  ·  MODALITY LOADERS AND FEATURE ENGINEERING")
    for x in cols:
        arrow(ax, (x + cw / 2, 83.4), (x + cw / 2, 80.3))

    t2 = ["clinical_loader\nNP*TOT totals,\nΔ-target construction",
          "mri_pipeline\nnilearn Schaefer-100\nROI parcellation",
          "pet_loader\nSBR, asymmetry indices,\ncomposite ratios",
          "genetic_loader\nLRRK2, GBA, SNCA,\nPINK1, PRKN, APOE"]
    for x, t in zip(cols, t2):
        box(ax, x, 71.5, cw, 8.5, t, fs=7.5)
    arrow(ax, (50, 71.3), (50, 67.2))

    box(ax, 2, 61.5, 96, 5.5,
        "data_builder  —  PATNO alignment  ·  subjects without a real Year-2 label are "
        "dropped, never imputed  ·  3,472 subjects × 126 features", fs=8)
    arrow(ax, (50, 61.3), (50, 57.2))

    box(ax, 2, 51.5, 96, 5.5,
        "Leak-free protocol  —  subject-level stratified split 70/15/15 "
        "(2,430 / 521 / 521)  ·  train-only imputer and scalers  ·  "
        "Dirichlet(α = 0.5) non-IID clients", fs=8)
    arrow(ax, (50, 51.3), (50, 47.2))

    box(ax, 2, 24.5, 96, 22.5, "", fc=SOFT_BLUE, ec=ACCENT, lw=1.2)
    label(ax, 4, 44.8, "TIER 5  ·  FED-PHENOGRAFT MODEL  (one FedAvg client)",
          fs=7.5, color=ACCENT_DK, ha="left", bold=True)

    box(ax, 5, 31.5, 20, 10, "Phenotype Encoder\n\nClinical MLP\n→ query q (32-d)",
        fc=ACCENT, ec=ACCENT_DK, tc="white", fs=7.5)
    box(ax, 32, 31.5, 22, 10,
        "Asymmetric\nCross-Attention\n4 heads · Add & Norm\nMRI / PET / Gene",
        fc=ACCENT, ec=ACCENT_DK, tc="white", fs=7.5)
    box(ax, 60, 31.5, 20, 10,
        "Shared / Private\n+ HSIC\nmask tokens ·\nRBF independence",
        fc=ACCENT, ec=ACCENT_DK, tc="white", fs=7.5)
    box(ax, 84, 31.5, 12.5, 10,
        "FedAvg\nOrchestrator\n4 clients ·\nweighted ·\nearly stop",
        fc=ACCENT, ec=ACCENT_DK, tc="white", fs=7)

    arrow(ax, (25.3, 36.5), (31.7, 36.5), color=ACCENT_DK, lw=1.6)
    label(ax, 28.5, 38.6, "Q", fs=8.5, color=ACCENT_DK, bold=True)
    arrow(ax, (59.7, 36.5), (54.3, 36.5), color=ACCENT_DK, lw=1.6)
    label(ax, 57, 38.6, "K, V", fs=8.5, color=ACCENT_DK, bold=True)
    label(ax, 50, 27.5,
          "Loss  =  MSE(Δ UPDRS-III)  +  0.3 · BCE(PD/HC)  +  0.1 · Σ HSIC(shared, private)",
          fs=7.5, color=ACCENT_DK)

    arrow(ax, (50, 24.3), (50, 21.2))

    t6 = ["Regression Head\nΔ UPDRS-III @ Year 2\nMSE",
          "Classification Head\nPD vs HC\nBCE",
          "Evaluation and XAI\nbootstrap CI · 3 seeds\nablation · IG"]
    for x, t in zip([2, 35, 68], t6):
        box(ax, x, 12.5, 30, 8.5, t, fs=7.5)
    label(ax, 50, 9.0,
          "Raw patient data never leaves a client — only model weight tensors are averaged.",
          fs=7.5, color=GREY, style="italic")
    ax.set_ylim(6.5, 100)
    save(fig, "system_architecture.png")


# ══════════════════════════════════════════════════════════════════
# Fig. 3 — DFD level 0 (context)
# ══════════════════════════════════════════════════════════════════

def dfd_level0():
    fig, ax = blank_axes(8.6, 3.4)
    box(ax, 2, 38, 20, 24, "PPMI /\nIDA-LONI\nRepository", rounded=False, fs=8.5,
        bold=True)
    box(ax, 78, 38, 20, 24, "Researcher /\nProject Team", rounded=False, fs=8.5,
        bold=True)

    ax.add_patch(Ellipse((50, 50), 34, 30, facecolor=SOFT_BLUE, edgecolor=ACCENT,
                         linewidth=1.3))
    ax.text(50, 53, "0", ha="center", va="center", fontsize=9, color=ACCENT_DK,
            fontweight="bold")
    ax.text(50, 46, "Fed-PhenoGraft\nPrediction System", ha="center", va="center",
            fontsize=8.5, color=INK, linespacing=1.4)

    arrow(ax, (22.3, 55), (32.5, 55), color=INK)
    label(ax, 27, 60.5, "Multimodal\nCSV / NIfTI records", fs=7.5)
    arrow(ax, (32.5, 44), (22.3, 44), color=INK)
    label(ax, 27, 36.5, "Download and\nverification requests", fs=7.5)

    arrow(ax, (67.5, 55), (77.7, 55), color=INK)
    label(ax, 72.6, 62.5, "Predictions, metrics,\nfigures, report", fs=7.5)
    arrow(ax, (77.7, 44), (67.5, 44), color=INK)
    label(ax, 72.6, 36.5, "Configuration\n(config.yaml)", fs=7.5)

    label(ax, 50, 24, "Context-level data flow diagram", fs=8, color=GREY,
          style="italic")
    ax.set_ylim(20, 78)
    save(fig, "dfd_level0.png")


# ══════════════════════════════════════════════════════════════════
# Fig. 4 — DFD level 1
# ══════════════════════════════════════════════════════════════════

def dfd_level1():
    fig, ax = blank_axes(9.2, 5.6)

    def store(x, y, w, h, text):
        ax.add_patch(Rectangle((x, y), w, h, facecolor="#F4F4F6", edgecolor=INK,
                               linewidth=0.9))
        ax.plot([x, x + w], [y + h - 2.8, y + h - 2.8], color=INK, linewidth=0.9)
        ax.text(x + w / 2, y + h / 2 - 1.3, text, ha="center", va="center",
                fontsize=7.5, color=INK, linespacing=1.35)

    def process(x, y, w, h, num, text):
        box(ax, x, y, w, h, "", fc=SOFT_BLUE, ec=ACCENT, lw=1.1)
        ax.text(x + w / 2, y + h - 3.4, num, ha="center", va="center", fontsize=8,
                color=ACCENT_DK, fontweight="bold", zorder=6)
        ax.text(x + w / 2, y + h / 2 - 2.0, text, ha="center", va="center",
                fontsize=7.5, color=INK, linespacing=1.35, zorder=6)

    box(ax, 1, 78, 17, 12, "PPMI /\nIDA-LONI", rounded=False, fs=8, bold=True)
    store(1, 55, 17, 12, "D1  data/raw\nCSV + NIfTI")
    box(ax, 1, 20, 17, 12, "Researcher /\nProject Team", rounded=False, fs=8, bold=True)

    process(26, 76, 22, 15, "1.0", "Load and engineer\nmodality features\n(M2 loaders)")
    process(26, 50, 22, 15, "2.0", "Align, split and\npreprocess\n(M3, train-only stats)")
    process(56, 76, 22, 15, "3.0", "Train baselines\n5-fold CV\n(M6)")
    process(56, 50, 22, 15, "4.0", "Federated training\nFedAvg + early stop\n(M5)")
    process(38, 18, 24, 15, "5.0", "Evaluate, ablate\nand explain\n(M7)")

    store(84, 79, 15, 12, "D2  processed\nfeatures")
    store(84, 55, 15, 12, "D3  model\nweights")
    store(84, 21, 15, 12, "D4  results\nJSON + figures")

    arrow(ax, (18.3, 84), (25.7, 84), color=INK)
    label(ax, 22, 91, "raw\nrecords", fs=7)
    arrow(ax, (18.3, 61), (25.7, 61), color=INK)
    arrow(ax, (37, 75.8), (37, 65.3), color=INK)
    label(ax, 39, 70.5, "aligned\nfeatures", fs=7, ha="left")
    arrow(ax, (48.3, 83), (55.7, 83), color=INK)
    label(ax, 52, 89.5, "raw feature\nmatrix", fs=7)
    arrow(ax, (48.3, 57), (55.7, 57), color=INK)
    # placed below-right of the arrow so it clears the vertical channel at x = 52
    label(ax, 56.5, 45.5, "train /\nval / test", fs=7, ha="left")
    arrow(ax, (78.3, 83), (83.7, 83), color=INK)
    arrow(ax, (78.3, 57), (83.7, 57), color=INK)

    # 3.0 to 5.0, routed down the clear channel at x = 52 (between processes 2.0 and 4.0)
    ax.plot([56, 52, 52], [80, 80, 35], color=INK, linewidth=1.0, zorder=3)
    arrow(ax, (52, 36), (52, 33.3), color=INK)
    label(ax, 50.5, 43, "baseline\npredictions", fs=7, ha="right")

    # 4.0 to 5.0, routed down the right-hand channel at x = 67
    ax.plot([67, 67, 64], [49.8, 25.5, 25.5], color=INK, linewidth=1.0, zorder=3)
    arrow(ax, (64.5, 25.5), (62.1, 25.5), color=INK)
    label(ax, 68.5, 38, "global\nmodel", fs=7, ha="left")

    arrow(ax, (62.3, 21), (83.7, 21), color=INK)
    arrow(ax, (37.7, 26), (18.3, 26), color=INK)
    label(ax, 28, 29.5, "metrics,\nfigures", fs=7)
    ax.set_ylim(14, 96)
    save(fig, "dfd_level1.png")


# ══════════════════════════════════════════════════════════════════
# Fig. 5 — Use case diagram
# ══════════════════════════════════════════════════════════════════

def use_case_diagram():
    fig, ax = blank_axes(9.0, 6.6)

    def actor(x, y, name):
        ax.add_patch(Ellipse((x, y + 9), 3.2, 4.2, facecolor="white", edgecolor=INK,
                             linewidth=1.1, zorder=5))
        ax.plot([x, x], [y + 6.9, y + 1.5], color=INK, linewidth=1.1, zorder=5)
        ax.plot([x - 3.2, x + 3.2], [y + 5.4, y + 5.4], color=INK, linewidth=1.1,
                zorder=5)
        ax.plot([x, x - 2.8], [y + 1.5, y - 3.2], color=INK, linewidth=1.1, zorder=5)
        ax.plot([x, x + 2.8], [y + 1.5, y - 3.2], color=INK, linewidth=1.1, zorder=5)
        ax.text(x, y - 6.2, name, ha="center", va="center", fontsize=8, color=INK,
                linespacing=1.3)

    def usecase(x, y, text, w=25, h=8.5):
        ax.add_patch(Ellipse((x, y), w, h, facecolor=SOFT_BLUE, edgecolor=ACCENT,
                             linewidth=1.0, zorder=3))
        ax.text(x, y, text, ha="center", va="center", fontsize=7.5, color=INK,
                linespacing=1.3, zorder=5)

    ax.add_patch(Rectangle((22, 6), 56, 88, facecolor="none", edgecolor=GREY,
                           linewidth=1.0, linestyle="--"))
    label(ax, 50, 96.5, "Fed-PhenoGraft System", fs=9, color=INK, bold=True)

    ys = [86, 74, 62, 50, 38, 26, 14]
    texts = ["Download and verify\nPPMI dataset",
             "Build aligned\nmultimodal dataset",
             "Configure experiment\n(config.yaml)",
             "Train model via\nfederated averaging",
             "Benchmark 12\nbaseline models",
             "Evaluate on held-out\ntest set (once)",
             "Generate explanations\nand report"]
    for y, t in zip(ys, texts):
        usecase(50, y, t)

    actor(8, 58, "Researcher /\nProject Team")
    actor(92, 74, "Clinical Site\n(FedAvg client)")
    actor(92, 30, "Federated\nServer")
    actor(8, 20, "Reviewer /\nFaculty Guide")

    for y in (86, 74, 62, 50, 38, 26):
        ax.plot([11.5, 37.5], [58, y], color=INK, linewidth=0.8, zorder=1)
    ax.plot([88.5, 62.5], [74, 74], color=INK, linewidth=0.8, zorder=1)
    ax.plot([88.5, 62.5], [74, 50], color=INK, linewidth=0.8, zorder=1)
    ax.plot([88.5, 62.5], [30, 50], color=INK, linewidth=0.8, zorder=1)
    ax.plot([88.5, 62.5], [30, 26], color=INK, linewidth=0.8, zorder=1)
    ax.plot([11.5, 37.5], [20, 14], color=INK, linewidth=0.8, zorder=1)

    # «precedes» routed outside the use-case ellipses, which end at x = 62.5
    ax.plot([62.5, 68, 68], [50, 47, 29], color=GREY, linewidth=0.9, linestyle="--",
            zorder=1)
    arrow(ax, (68, 29.5), (63.2, 26.8), color=GREY, ls="--")
    label(ax, 69.5, 38, "«precedes»", fs=6.5, color=GREY, ha="left", style="italic")
    save(fig, "use_case_diagram.png")


# ══════════════════════════════════════════════════════════════════
# Fig. 6 — Class diagram
# ══════════════════════════════════════════════════════════════════

def class_diagram():
    fig, ax = blank_axes(9.2, 5.8)
    H_NAME, LINE_H, PAD = 6.0, 3.6, 2.6

    def cls(x, y_top, w, name, attrs, methods):
        """UML class with (x, y_top) as its top-left. Returns its measured box."""
        h = H_NAME + LINE_H * (len(attrs) + len(methods)) + PAD
        ax.add_patch(Rectangle((x, y_top - h), w, h, facecolor="white", edgecolor=INK,
                               linewidth=1.0, zorder=3))
        ax.add_patch(Rectangle((x, y_top - H_NAME), w, H_NAME, facecolor=SOFT_BLUE,
                               edgecolor=INK, linewidth=1.0, zorder=4))
        ax.text(x + w / 2, y_top - H_NAME / 2, name, ha="center", va="center",
                fontsize=8, color=ACCENT_DK, fontweight="bold", zorder=5)
        yy = y_top - H_NAME - 2.4
        for a in attrs:
            ax.text(x + 1.4, yy, a, ha="left", va="center", fontsize=6.8, color=INK,
                    zorder=5)
            yy -= LINE_H
        sep = yy + LINE_H / 2 + 0.3
        ax.plot([x, x + w], [sep, sep], color=INK, linewidth=0.9, zorder=5)
        yy -= 0.6
        for m in methods:
            ax.text(x + 1.4, yy, m, ha="left", va="center", fontsize=6.8, color=INK,
                    zorder=5)
            yy -= LINE_H
        return {"x": x, "w": w, "top": y_top, "bottom": y_top - h, "cx": x + w / 2}

    root = cls(30, 100, 40, "FedPhenoGraft",
               ["- clin_encoder : Sequential",
                "- mri_enc / pet_enc / gen_enc",
                "- *_mask_token : Parameter",
                "- prediction_head, classification_head"],
               ["+ forward(batch) : dict", "+ enable_mc_dropout()"])

    mid_top = 56
    a = cls(1, mid_top, 30, "SharedPrivateEncoder",
            ["- shared_mlp : Sequential", "- private_mlp : Sequential"],
            ["+ forward(x) : (shared, private)"])
    b = cls(35, mid_top, 30, "AsymmetricCrossAttention",
            ["- mha : MultiheadAttention", "- norm : LayerNorm"],
            ["+ forward(query, kv)"])
    c = cls(69, mid_top, 30, "HSICLoss",
            ["- sigma : float"],
            ["+ rbf_kernel(X, Y)", "+ forward(z_shared, z_private)"])

    low_top = 26
    d = cls(1, low_top, 30, "FederatedPPMIDataset",
            ["- clinical / mri / pet / genetic", "- targets, diagnosis, client_id"],
            ["+ __getitem__(idx) : dict"])
    e = cls(35, low_top, 30, "ModalityPreprocessor",
            ["- clinical_imputer, clinical_scaler", "- modality_scalers : dict"],
            ["+ fit(...)   + transform(...)"])
    f = cls(69, low_top, 30, "FedAvgOrchestrator",
            ["- num_clients, num_rounds", "- early_stopping_patience"],
            ["+ client_update(...)", "+ fedavg_aggregate(...)"])

    # Composition: FedPhenoGraft is built from the three model components.
    bus = (root["bottom"] + mid_top) / 2
    ax.plot([root["cx"], root["cx"]], [root["bottom"] - 2.4, bus], color=INK,
            linewidth=0.9)
    ax.add_patch(Rectangle((root["cx"] - 1.2, root["bottom"] - 3.6), 2.4, 2.4,
                           facecolor="white", edgecolor=INK, linewidth=0.9, angle=45,
                           zorder=6))
    ax.plot([a["cx"], c["cx"]], [bus, bus], color=INK, linewidth=0.9)
    for child in (a, b, c):
        ax.plot([child["cx"], child["cx"]], [bus, child["top"]], color=INK,
                linewidth=0.9)
    label(ax, root["cx"] + 3.5, bus + 1.8, "composed of", fs=6.8, color=GREY,
          ha="left", style="italic")

    # Dependencies between the middle and lower rows.
    for src, dst, text, upward in ((a, d, "consumes", False), (e, b, "feeds", True),
                                   (f, c, "trains", True)):
        x = src["cx"]
        if upward:
            y0, y1 = src["top"] + 0.3, dst["bottom"] - 0.5
        else:
            y0, y1 = src["bottom"] - 0.3, dst["top"] + 0.5
        arrow(ax, (x, y0), (x, y1), color=INK, style="-|>", lw=0.9, ls="--")
        label(ax, x + 1.8, (y0 + y1) / 2, text, fs=6.8, color=GREY, ha="left",
              style="italic")

    ax.set_ylim(min(d["bottom"], e["bottom"], f["bottom"]) - 3, 102)
    save(fig, "class_diagram.png")


# ══════════════════════════════════════════════════════════════════
# Fig. 7 — Sequence diagram (one FedAvg round)
# ══════════════════════════════════════════════════════════════════

def sequence_diagram():
    fig, ax = blank_axes(9.2, 5.8)
    lifelines = [(9, "Orchestrator"), (30, "Client 1..4\n(sites)"),
                 (52, "FedPhenoGraft\n(local copy)"), (74, "Validation\nset"),
                 (93, "Best-weight\nstore")]
    for x, name in lifelines:
        box(ax, x - 8, 90, 16, 8, name, fc=SOFT_BLUE, ec=ACCENT, fs=7.5, bold=True)
        ax.plot([x, x], [89.5, 8], color=GREY, linewidth=0.8, linestyle="--")

    msgs = [
        (9, 30, 83, "1  broadcast global weights θ(t)"),
        (30, 52, 76, "2  local training, 2 epochs (AdamW, clip 1.0)"),
        (52, 52, 69, "3  loss = MSE + 0.3·BCE + 0.1·HSIC"),
        (52, 30, 62, "4  return updated weights θᵢ"),
        (30, 9, 55, "5  send θᵢ and client sample count nᵢ"),
        (9, 9, 48, "6  θ(t+1) = Σ (nᵢ / n) · θᵢ   [weighted FedAvg]"),
        (9, 74, 41, "7  evaluate global model"),
        (74, 9, 34, "8  validation CCC, RMSE, AUC"),
        (9, 93, 27, "9  if CCC improved: store weights, reset patience"),
        (93, 9, 20, "10  else patience decrements; at 0 restore best round and stop"),
    ]
    for x1, x2, y, text in msgs:
        if x1 == x2:
            ax.plot([x1, x1 + 6, x1 + 6, x1], [y + 1.6, y + 1.6, y - 1.6, y - 1.6],
                    color=INK, linewidth=0.9)
            arrow(ax, (x1 + 6, y - 1.6), (x1 + 0.4, y - 1.6), color=INK, lw=0.9)
            label(ax, x1 + 8.5, y + 3.2, text, fs=7, color=INK, ha="left")
        else:
            arrow(ax, (x1, y), (x2, y), color=INK, lw=0.9)
            label(ax, (x1 + x2) / 2, y + 2.6, text, fs=7, color=INK)

    ax.add_patch(Rectangle((3, 12), 94, 76, facecolor="none", edgecolor=GREY,
                           linewidth=0.9, linestyle=":"))
    label(ax, 50, 9.5, "loop  [round = 1 .. 30, until early stopping]", fs=7.5,
          color=GREY, style="italic")
    ax.set_ylim(6, 100)
    save(fig, "sequence_diagram.png")


def main():
    print("Generating report diagrams...")
    gantt_chart()
    system_architecture()
    dfd_level0()
    dfd_level1()
    use_case_diagram()
    class_diagram()
    sequence_diagram()
    print(f"Done - 7 figures in {OUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
