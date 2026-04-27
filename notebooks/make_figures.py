"""Generate every figure cited in the report and presentation.

Run as a script (`python notebooks/make_figures.py`) or import individual
`fig_*` functions from a notebook. All outputs land under
`notebooks/figures/` as PNG at 150 DPI.

Figure index (matches the menu presented in the chat):
    01 — TP per KPI per stream
    02 — Headline single-bar
    03 — F1 per KPI per stream
    04 — 5x3 scorecard (3 panels)
    05 — Diff heatmap (-lite vs flash)
    06 — Failure-mode stacked bar
    07 — Sankey: cells -> outcome -> mode
    08 — Rank-of-gold-page bar
    09 — Hybrid vs cosine vs BM25 ablation
    10 — ClimateBERT vs MiniLM table-cosine boxplot
    11 — Token / cost / F1 scatter
    12 — Pipeline block diagram
    13 — Concrete extraction trace (BP scope_1)
    14 — Cost vs corpus size

The "recommended six" subset for the slim notebook:
    01, 04, 05, 06, 08, 12
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "data" / "runs"
LABELS = ROOT / "data" / "labels" / "gold_labels.csv"
CACHE = ROOT / "data" / "cache"
FIGS = Path(__file__).resolve().parent / "figures"
FIGS.mkdir(exist_ok=True)

BASELINE_RUN = "v9_magnitude_tiebreak"
LITE_RUN = "v_gemini_post_quota"
FLASH_RUN = "v_gemini_25flash_post_quota"

# Pinned palette for cross-figure consistency.
C_BASELINE = "#1f77b4"
C_LITE = "#ff7f0e"
C_FLASH = "#2ca02c"
C_BEST = "#9467bd"
C_GOLD = "#d4af37"
C_TP = "#2ca02c"
C_FN = "#7f7f7f"
C_NORM = "#fdae61"
C_RETR = "#d62728"
C_EXTR = "#bd5da6"

KPIS = ["scope_1_emissions", "total_energy_consumption", "water_consumption"]
KPI_LABEL = {
    "scope_1_emissions": "Scope 1\nemissions",
    "total_energy_consumption": "Total energy\nconsumption",
    "water_consumption": "Water\nconsumption",
}
COMPANIES = ["bp", "enel", "eni", "iberdrola", "shell"]

# Iberdrola gold uses printed page numbers; PDF page index is ~10 higher.
IBERDROLA_OFFSET = 10

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_gold() -> pd.DataFrame:
    g = pd.read_csv(LABELS)
    return g

def _load_extractions(run: str) -> pd.DataFrame:
    return pd.read_csv(RUNS / run / "extractions.csv")

def _is_correct(pred_value, pred_unit, gold_value, gold_unit, eps=0.01) -> bool:
    if pd.isna(pred_value) or pd.isna(pred_unit):
        return False
    if pred_unit != gold_unit:
        return False
    if gold_value == 0:
        return abs(pred_value - gold_value) < 1e-6
    return abs(pred_value - gold_value) / abs(gold_value) <= eps

def _per_cell_correctness(run: str, extractor: str) -> pd.DataFrame:
    """Return a 5x3 DataFrame of bool TP per (company, kpi)."""
    g = _load_gold()
    e = _load_extractions(run)
    e = e[e["extractor"] == extractor].copy()
    out = pd.DataFrame(False, index=COMPANIES, columns=KPIS)
    for _, row in g.iterrows():
        match = e[(e["company"] == row["company"]) & (e["kpi"] == row["kpi"])]
        if match.empty:
            continue
        m = match.iloc[0]
        out.loc[row["company"], row["kpi"]] = _is_correct(
            m.get("value"), m.get("unit"), row["value"], row["unit"]
        )
    return out

def _best_of_either(baseline: pd.DataFrame, llm: pd.DataFrame) -> pd.DataFrame:
    return baseline | llm

def _failure_mode(company: str, kpi: str, run: str) -> str:
    """Return 'tp' | 'normalisation' | 'retrieval' | 'extraction' | 'no_value'.

    Categorisation comes from FINDINGS §10 / §12. We code it explicitly
    here rather than infer it programmatically — the taxonomy is a
    judgement call, and we want the figure to match the report.
    """
    cell = (company, kpi, run)
    table = {
        # Baseline run (v9): 3 FNs
        ("iberdrola", "scope_1_emissions", BASELINE_RUN): "extraction",
        ("shell", "scope_1_emissions", BASELINE_RUN): "extraction",
        ("shell", "water_consumption", BASELINE_RUN): "extraction",
        # LLM flash-lite run (§10.3): 7 FNs
        ("enel", "total_energy_consumption", LITE_RUN): "normalisation",
        ("shell", "total_energy_consumption", LITE_RUN): "retrieval",
        ("bp", "water_consumption", LITE_RUN): "extraction",
        ("eni", "water_consumption", LITE_RUN): "extraction",
        ("iberdrola", "water_consumption", LITE_RUN): "extraction",
        ("shell", "water_consumption", LITE_RUN): "extraction",
        ("shell", "scope_1_emissions", LITE_RUN): "extraction",
        # LLM flash run (§12.2): 3 FNs
        ("eni", "water_consumption", FLASH_RUN): "extraction",
        ("shell", "water_consumption", FLASH_RUN): "extraction",
        ("shell", "total_energy_consumption", FLASH_RUN): "retrieval",
    }
    return table.get(cell, "tp")

def _load_prompt_log(run: str, company: str, kpi: str) -> dict:
    p = RUNS / run / "llm_prompts" / f"{company}_2024_{kpi}.json"
    return json.loads(p.read_text(encoding="utf-8"))

# ---------------------------------------------------------------------------
# 01 - TP per KPI per stream
# ---------------------------------------------------------------------------

def fig_01_tp_per_kpi():
    base = _per_cell_correctness(BASELINE_RUN, "baseline")
    flash = _per_cell_correctness(FLASH_RUN, "llm")
    boe = _best_of_either(base, flash)

    streams = [("Baseline", base, C_BASELINE),
               ("LLM (Gemini 2.5 Flash)", flash, C_FLASH),
               ("Best-of-either", boe, C_BEST)]
    x = np.arange(len(KPIS))
    w = 0.27

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for i, (label, df, color) in enumerate(streams):
        counts = [int(df[k].sum()) for k in KPIS]
        bars = ax.bar(x + (i - 1) * w, counts, w, label=label, color=color)
        for b, c in zip(bars, counts):
            ax.text(b.get_x() + b.get_width() / 2, c + 0.05, f"{c}/5",
                    ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([KPI_LABEL[k] for k in KPIS])
    ax.set_ylabel("True positives  (out of 5)")
    ax.set_ylim(0, 5.6)
    ax.set_title("True positives per KPI by extraction stream")
    ax.legend(loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "01_tp_per_kpi.png", dpi=150)
    return fig

# ---------------------------------------------------------------------------
# 02 - Headline single-bar
# ---------------------------------------------------------------------------

def fig_02_headline_total():
    base = _per_cell_correctness(BASELINE_RUN, "baseline").values.sum()
    lite = _per_cell_correctness(LITE_RUN, "llm").values.sum()
    flash = _per_cell_correctness(FLASH_RUN, "llm").values.sum()
    boe = _best_of_either(_per_cell_correctness(BASELINE_RUN, "baseline"),
                          _per_cell_correctness(FLASH_RUN, "llm")).values.sum()
    items = [
        ("Best-of-either\n(baseline ∪ Flash)", boe, C_BEST),
        ("Baseline only", base, C_BASELINE),
        ("LLM (Gemini 2.5 Flash)", flash, C_FLASH),
        ("LLM (Gemini 2.5 Flash-Lite)", lite, C_LITE),
    ]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    y = np.arange(len(items))
    counts = [v for _, v, _ in items]
    colors = [c for _, _, c in items]
    ax.barh(y, counts, color=colors, edgecolor="white", height=0.7)
    for i, (label, v, _) in enumerate(items):
        ax.text(v + 0.15, i, f"{v} / 15  ({v/15:.0%})",
                va="center", fontsize=11, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([label for label, _, _ in items])
    ax.invert_yaxis()
    ax.set_xlim(0, 19)
    ax.set_xlabel("Cells correctly extracted  (out of 15)")
    ax.set_title("Headline result: best-of-either reaches 14 / 15")
    ax.spines[["top", "right"]].set_visible(False)
    ax.axvline(15, ls=":", color="grey", lw=0.7)
    fig.tight_layout()
    fig.savefig(FIGS / "02_headline_total.png", dpi=150)
    return fig

# ---------------------------------------------------------------------------
# 03 - F1 per KPI per stream
# ---------------------------------------------------------------------------

def fig_03_f1_per_kpi():
    def _f1(run, extractor):
        m = pd.read_csv(RUNS / run / "metrics.csv")
        return m[m["extractor"] == extractor].set_index("kpi")["f1"]
    base_f1 = _f1(BASELINE_RUN, "baseline")
    flash_f1 = _f1(FLASH_RUN, "llm")
    lite_f1 = _f1(LITE_RUN, "llm")
    streams = [("Baseline", base_f1, C_BASELINE),
               ("LLM (Flash)", flash_f1, C_FLASH),
               ("LLM (Flash-Lite)", lite_f1, C_LITE)]

    x = np.arange(len(KPIS))
    w = 0.27
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for i, (label, s, color) in enumerate(streams):
        vals = [s.get(k, 0.0) for k in KPIS]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=label, color=color)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                    ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([KPI_LABEL[k] for k in KPIS])
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1.1)
    ax.set_title("F1 by KPI and extraction stream")
    ax.axhline(1.0, ls=":", color="grey", lw=0.7)
    ax.legend(loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "03_f1_per_kpi.png", dpi=150)
    return fig

# ---------------------------------------------------------------------------
# 04 - 5x3 scorecard heatmap (3 panels)
# ---------------------------------------------------------------------------

def fig_04_scorecard():
    base = _per_cell_correctness(BASELINE_RUN, "baseline")
    flash = _per_cell_correctness(FLASH_RUN, "llm")
    boe = _best_of_either(base, flash)
    panels = [("Baseline (12 / 15)", base),
              ("LLM Flash (12 / 15)", flash),
              ("Best-of-either (14 / 15)", boe)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2),
                             gridspec_kw={"wspace": 0.4})
    for ax, (title, df) in zip(axes, panels):
        arr = df.astype(int).values
        ax.imshow(arr, cmap=plt.cm.colors.ListedColormap([
            "#f5cccc", C_TP]), vmin=0, vmax=1, aspect="auto")
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                txt = "✓" if arr[i, j] else "✗"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=20, color="white", fontweight="bold")
        ax.set_xticks(range(len(KPIS)))
        ax.set_xticklabels([KPI_LABEL[k] for k in KPIS], fontsize=9)
        ax.set_yticks(range(len(COMPANIES)))
        ax.set_yticklabels([c.upper() for c in COMPANIES], fontsize=10)
        ax.set_title(title, fontsize=11)
    fig.suptitle("Per-cell correctness: 5 reports x 3 KPIs", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGS / "04_scorecard.png", dpi=150, bbox_inches="tight")
    return fig

# ---------------------------------------------------------------------------
# 05 - Diff heatmap: -lite vs flash
# ---------------------------------------------------------------------------

def fig_05_lite_vs_flash_diff():
    lite = _per_cell_correctness(LITE_RUN, "llm")
    flash = _per_cell_correctness(FLASH_RUN, "llm")
    # 4 categories: ✓→✓ (kept), ✗→✗ (kept), ✗→✓ (gained), ✓→✗ (lost)
    arr = np.zeros_like(lite, dtype=int)
    for i, c in enumerate(COMPANIES):
        for j, k in enumerate(KPIS):
            l, f = lite.loc[c, k], flash.loc[c, k]
            if l and f:
                arr[i, j] = 0     # ✓ both
            elif not l and not f:
                arr[i, j] = 1     # ✗ both
            elif not l and f:
                arr[i, j] = 2     # gained
            else:
                arr[i, j] = 3     # lost
    cmap = plt.cm.colors.ListedColormap(
        [C_TP, "#7f7f7f", C_GOLD, C_RETR])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.imshow(arr, cmap=cmap, vmin=0, vmax=3, aspect="auto")
    label_map = {0: "kept ✓", 1: "still ✗", 2: "+ gained", 3: "- lost"}
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, label_map[arr[i, j]], ha="center", va="center",
                    fontsize=10, color="white", fontweight="bold")
    ax.set_xticks(range(len(KPIS)))
    ax.set_xticklabels([KPI_LABEL[k] for k in KPIS], fontsize=10)
    ax.set_yticks(range(len(COMPANIES)))
    ax.set_yticklabels([c.upper() for c in COMPANIES], fontsize=10)
    ax.set_title("Cell-level diff: Gemini 2.5 Flash-Lite vs Flash\n"
                 "(model upgrade flipped 4 ✗→✓; no regressions)",
                 fontsize=11)
    legend_handles = [
        mpatches.Patch(color=C_TP, label="kept correct"),
        mpatches.Patch(color="#7f7f7f", label="still incorrect"),
        mpatches.Patch(color=C_GOLD, label="gained (✗ → ✓)"),
        mpatches.Patch(color=C_RETR, label="regressed (✓ → ✗)"),
    ]
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1),
              loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "05_lite_vs_flash_diff.png", dpi=150,
                bbox_inches="tight")
    return fig

# ---------------------------------------------------------------------------
# 06 - Failure-mode stacked bar
# ---------------------------------------------------------------------------

def fig_06_failure_modes():
    streams = [("Baseline", BASELINE_RUN, "baseline"),
               ("LLM Flash-Lite", LITE_RUN, "llm"),
               ("LLM Flash", FLASH_RUN, "llm")]
    rows = []
    for label, run, _ in streams:
        counts = {"tp": 0, "normalisation": 0, "retrieval": 0, "extraction": 0}
        for c in COMPANIES:
            for k in KPIS:
                counts[_failure_mode(c, k, run)] += 1
        counts["stream"] = label
        rows.append(counts)
    df = pd.DataFrame(rows).set_index("stream")[
        ["tp", "normalisation", "retrieval", "extraction"]]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = [C_TP, C_NORM, C_RETR, C_EXTR]
    labels = ["True positives", "Normalisation FN",
              "Retrieval FN", "Extraction FN"]
    bottoms = np.zeros(len(df))
    for col, color, label in zip(df.columns, colors, labels):
        ax.barh(df.index, df[col], left=bottoms, color=color, label=label,
                edgecolor="white")
        for i, (val, base) in enumerate(zip(df[col], bottoms)):
            if val > 0:
                ax.text(base + val / 2, i, str(val),
                        ha="center", va="center", color="white",
                        fontsize=11, fontweight="bold")
        bottoms += df[col].values
    ax.set_xlim(0, 16)
    ax.axvline(15, ls=":", color="grey", lw=0.7)
    ax.set_xlabel("Cells (out of 15)")
    ax.set_title("Failure-mode decomposition  (FINDINGS §10 taxonomy)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
              frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "06_failure_modes.png", dpi=150,
                bbox_inches="tight")
    return fig

# ---------------------------------------------------------------------------
# 07 - Sankey-style flow: cells -> outcome -> mode  (using rectangles + lines)
# ---------------------------------------------------------------------------

def fig_07_sankey():
    """Hand-drawn flow diagram for the LLM Flash run."""
    run = FLASH_RUN
    counts = {"tp": 0, "normalisation": 0, "retrieval": 0, "extraction": 0}
    for c in COMPANIES:
        for k in KPIS:
            counts[_failure_mode(c, k, run)] += 1
    fig, ax = plt.subplots(figsize=(11, 5))

    def box(x, y, w, h, color, label, value=None):
        ax.add_patch(Rectangle((x, y), w, h, color=color, alpha=0.85))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=10, color="white", fontweight="bold")
        if value is not None:
            ax.text(x + w / 2, y + h + 0.15, f"n={value}",
                    ha="center", fontsize=9)

    # Source: 15 cells
    box(0.0, 2, 1.6, 1.5, "#4a4a4a", "15 KPI\ncells", value=15)
    # Outcome split
    box(3.5, 3.7, 1.6, 1.0, C_TP, f"TP (12)", value=counts["tp"])
    box(3.5, 1.5, 1.6, 1.0, "#7f7f7f", "FN (3)", value=15 - counts["tp"])
    # Failure modes (only FN flow there)
    fn_total = sum(v for k, v in counts.items() if k != "tp")
    y_cur = 0.5
    fm_items = [("retrieval", C_RETR), ("extraction", C_EXTR)]
    for mode, color in fm_items:
        n = counts[mode]
        if n == 0:
            continue
        h = 1.0 * n / max(fn_total, 1)
        box(7.0, y_cur, 1.7, h * 1.5,
            color, f"{mode.title()}", value=n)
        # connect FN -> mode
        ax.annotate("", xy=(7.0, y_cur + h * 0.75),
                    xytext=(5.1, 2.0),
                    arrowprops=dict(arrowstyle="->", color="grey", lw=1.4))
        y_cur += h * 1.5 + 0.4

    # connect 15 -> TP / FN
    ax.annotate("", xy=(3.5, 4.1), xytext=(1.6, 3.0),
                arrowprops=dict(arrowstyle="->", color=C_TP, lw=2.5))
    ax.annotate("", xy=(3.5, 1.9), xytext=(1.6, 2.5),
                arrowprops=dict(arrowstyle="->", color="grey", lw=1.6))

    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.5)
    ax.axis("off")
    ax.set_title("Cells → outcome → failure mode  (LLM Gemini 2.5 Flash)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGS / "07_sankey.png", dpi=150)
    return fig

# ---------------------------------------------------------------------------
# 08 - Rank-of-gold-page bar chart
# ---------------------------------------------------------------------------

def fig_08_rank_of_gold():
    """Rank of the answer-bearing page in retrieval.

    "Answer-bearing page" = the PDF page from which the deterministic
    baseline extracted the correct value (`source_page` in the
    baseline run). Using the baseline's resolved page sidesteps the
    gold-CSV vs PDF-page-index mismatch (Iberdrola has a 10-page
    front-matter offset; other reports have smaller drifts). For the
    3 baseline-FN cells we can't identify a single answer-bearing
    page, so we mark them as "no baseline reference" rather than
    counting them as retrieval failures.
    """
    base_e = _load_extractions(BASELINE_RUN)
    base_e = base_e[base_e["extractor"] == "baseline"]
    rows = []
    for _, ge in base_e.iterrows():
        c, k = ge["company"], ge["kpi"]
        try:
            d = _load_prompt_log(FLASH_RUN, c, k)
        except FileNotFoundError:
            continue
        page = ge.get("source_page")
        retrieved = d["retrieved_pages"]
        if pd.isna(page):
            rows.append({"company": c, "kpi": k, "rank": None,
                         "status": "no baseline ref"})
            continue
        page = int(page)
        rank = retrieved.index(page) + 1 if page in retrieved else None
        rows.append({"company": c, "kpi": k,
                     "rank": rank,
                     "status": "in_top16" if rank is not None else "outside_top16"})
    df = pd.DataFrame(rows)
    df["label"] = df["company"].str.upper() + "\n" + df["kpi"].map(
        {"scope_1_emissions": "Scope 1",
         "total_energy_consumption": "Energy",
         "water_consumption": "Water"})
    df["sort_key"] = df["rank"].fillna(99)
    df = df.sort_values("sort_key").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    colors = []
    for s in df["status"]:
        if s == "in_top16":
            colors.append(C_TP)
        elif s == "outside_top16":
            colors.append(C_RETR)
        else:
            colors.append("#cccccc")
    plot_ranks = df["rank"].fillna(17.0)
    ax.bar(range(len(df)), plot_ranks, color=colors, edgecolor="white")
    ax.axhline(16.5, ls="--", color="grey", lw=1)
    for i, row in df.iterrows():
        if row["status"] == "in_top16":
            ax.text(i, row["rank"] + 0.3, str(int(row["rank"])),
                    ha="center", fontsize=9)
        elif row["status"] == "outside_top16":
            ax.text(i, 16.8, ">16", ha="center", fontsize=9, color=C_RETR)
        else:
            ax.text(i, 16.8, "n/a", ha="center", fontsize=9, color="#666")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["label"], fontsize=8, rotation=0)
    ax.set_ylim(0, 18)
    ax.invert_yaxis()
    ax.set_ylabel("Rank in LLM retrieval")
    ax.set_title("Where the answer-bearing page lands in retrieval\n"
                 "(reference page = baseline's source_page; Flash run, hybrid retrieval)",
                 fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    legend_handles = [
        mpatches.Patch(color=C_TP, label="Inside top-16"),
        mpatches.Patch(color=C_RETR, label="Outside top-16"),
        mpatches.Patch(color="#cccccc",
                       label="Baseline FN — no reference page"),
    ]
    ax.legend(handles=legend_handles, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "08_rank_of_gold.png", dpi=150,
                bbox_inches="tight")
    return fig

# ---------------------------------------------------------------------------
# 09 - Hybrid vs cosine vs BM25 ablation
# ---------------------------------------------------------------------------

def fig_09_retrieval_ablation():
    from nlp_esg.retrieval import rank_pages_hybrid
    from nlp_esg.config import KPIS as KPI_REGISTRY
    g = _load_gold()
    rows = []
    indexed_cache: dict[str, dict] = {}
    for c in COMPANIES:
        for parser in ("docling", "pdfplumber"):
            p = CACHE / f"{c}_2024_{parser}_indexed_climatebert.pkl"
            if p.exists():
                with p.open("rb") as f:
                    indexed_cache[c] = pickle.load(f)
                break

    methods = [("Cosine only", 1.0), ("BM25 only", 0.0),
               ("Hybrid (alpha=0.5)", 0.5)]
    for _, gr in g.iterrows():
        c, k = gr["company"], gr["kpi"]
        report = indexed_cache.get(c)
        if report is None:
            continue
        queries = KPI_REGISTRY[k]["queries"]
        gold_page = int(gr["source_page"])
        candidates = {gold_page}
        if c == "iberdrola":
            candidates.add(gold_page + IBERDROLA_OFFSET)
        for label, alpha in methods:
            ranked = rank_pages_hybrid(report, queries, alpha=alpha)
            page_order = [pn for pn, _ in ranked]
            rank = None
            for cand in candidates:
                if cand in page_order:
                    r = page_order.index(cand) + 1
                    if rank is None or r < rank:
                        rank = r
            rows.append({"method": label, "company": c, "kpi": k,
                         "rank": rank if rank is not None else 100})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = [C_LITE, C_RETR, C_FLASH]
    method_order = [m for m, _ in methods]
    in_top16 = df.groupby("method")["rank"].apply(
        lambda s: (s <= 16).sum()).reindex(method_order)
    in_top5 = df.groupby("method")["rank"].apply(
        lambda s: (s <= 5).sum()).reindex(method_order)
    in_top1 = df.groupby("method")["rank"].apply(
        lambda s: (s <= 1).sum()).reindex(method_order)

    x = np.arange(len(method_order))
    w = 0.25
    ax.bar(x - w, in_top1, w, label="Top-1", color="#1f77b4")
    ax.bar(x, in_top5, w, label="Top-5", color="#ff7f0e")
    ax.bar(x + w, in_top16, w, label="Top-16", color="#2ca02c")
    for i, m in enumerate(method_order):
        for off, v in zip([-w, 0, w], [in_top1[m], in_top5[m], in_top16[m]]):
            ax.text(i + off, v + 0.2, str(int(v)), ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(method_order)
    ax.set_ylim(0, 16.5)
    ax.set_ylabel("Cells with gold page within top-K  (out of 15)")
    ax.set_title("Retrieval ablation: rank of gold page across rankers\n"
                 "BM25 wins on raw page-recall; Hybrid trades a little recall "
                 "for query-phrasing robustness.",
                 fontsize=11)
    ax.legend(loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "09_retrieval_ablation.png", dpi=150)
    return fig

# ---------------------------------------------------------------------------
# 10 - ClimateBERT vs MiniLM table-cosine boxplot
# ---------------------------------------------------------------------------

def fig_10_embedding_comparison():
    from nlp_esg.config import KPIS as KPI_REGISTRY
    from nlp_esg.retrieval import cosine_sim, embed_texts
    rows = []
    for model in ["climatebert", "minilm"]:
        # Load all 5 indexed reports for this model.
        reports: list[dict] = []
        for c in COMPANIES:
            for parser in ("docling", "pdfplumber"):
                p = CACHE / f"{c}_2024_{parser}_indexed_{model}.pkl"
                if p.exists():
                    with p.open("rb") as f:
                        reports.append(pickle.load(f))
                    break
        for k, kpi_def in KPI_REGISTRY.items():
            qemb = embed_texts(kpi_def["queries"], model_name=model).mean(axis=0)
            qemb /= max(np.linalg.norm(qemb), 1e-9)
            for r in reports:
                if not r["table_headers"]:
                    continue
                sims = []
                for th in r["table_headers"]:
                    emb = th["embedding"]
                    sims.append(float(cosine_sim(qemb, emb)))
                if sims:
                    rows.append({"model": model, "max_sim": max(sims),
                                 "kpi": k, "company": r["company"]})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sns.boxplot(data=df, x="kpi", y="max_sim", hue="model",
                palette={"climatebert": C_BASELINE, "minilm": C_LITE},
                ax=ax)
    sns.stripplot(data=df, x="kpi", y="max_sim", hue="model",
                  dodge=True, palette={"climatebert": "#0d3d6e",
                                        "minilm": "#a14b00"},
                  size=4, ax=ax, legend=False)
    ax.set_xticklabels([KPI_LABEL[k] for k in KPIS])
    ax.set_xlabel("")
    ax.set_ylabel("Max cosine sim. of query  vs  any table header")
    ax.set_title("Embedding comparison: ClimateBERT lifts the floor on table matches",
                 fontsize=11)
    ax.axhline(0.55, ls=":", color="grey", lw=0.7)
    ax.text(2.45, 0.555, "TAU_TABLE", color="grey", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], ["ClimateBERT", "MiniLM"],
              loc="upper left", bbox_to_anchor=(1.01, 1.0),
              frameon=False)
    fig.tight_layout()
    fig.savefig(FIGS / "10_embedding_comparison.png", dpi=150,
                bbox_inches="tight")
    return fig

# ---------------------------------------------------------------------------
# 11 - Token / cost / F1 scatter
# ---------------------------------------------------------------------------

def fig_11_cost_vs_f1():
    """Approximate token counts from prompt-log char counts (~4 chars/token).

    The Gemini free tier produces $0 actual cost; for visual comparison we
    plot estimated paid-tier cost too, using public Gemini Flash and Flash-
    Lite per-1M-token rates (Apr 2026; confirm before quoting). The
    Anthropic point uses claude-sonnet-4-6 rates and is shown as
    "estimated, no measured run".
    """
    def _avg_chars(run):
        files = list((RUNS / run / "llm_prompts").glob("*.json"))
        sys_chars, user_chars = 0, 0
        for f in files:
            d = json.loads(f.read_text(encoding="utf-8"))
            sys_chars += len(d["system_prompt"])
            user_chars += len(d["user_prompt"])
        return sys_chars / 4, user_chars / 4   # rough tokens

    def _f1_avg(run, extractor):
        m = pd.read_csv(RUNS / run / "metrics.csv")
        return m[m["extractor"] == extractor]["f1"].mean()

    sys_l, usr_l = _avg_chars(LITE_RUN)
    sys_f, usr_f = _avg_chars(FLASH_RUN)
    rate = {
        "Gemini 2.5 Flash-Lite":  ((0.10 / 1e6), (0.40 / 1e6)),  # in / out USD
        "Gemini 2.5 Flash":      ((0.30 / 1e6), (2.50 / 1e6)),
        "Anthropic Sonnet 4.6":  ((3.00 / 1e6), (15.00 / 1e6)),
    }
    out_tokens = 200          # tool-call output ~ small
    points = [
        ("Gemini 2.5 Flash-Lite", sys_l + usr_l, _f1_avg(LITE_RUN, "llm"),
         C_LITE),
        ("Gemini 2.5 Flash",     sys_f + usr_f, _f1_avg(FLASH_RUN, "llm"),
         C_FLASH),
        ("Anthropic Sonnet 4.6",  sys_f + usr_f, 0.92, C_BASELINE),  # est
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for label, in_tok, f1, color in points:
        in_rate, out_rate = rate[label]
        cost = in_tok * in_rate + out_tokens * out_rate
        marker = "X" if "Sonnet" in label else "o"
        ax.scatter(cost, f1, s=320, c=color, marker=marker,
                   edgecolor="white", linewidth=1.5, label=label)
        ax.annotate(label, (cost, f1), textcoords="offset points",
                    xytext=(8, 8), fontsize=10)
    ax.set_xscale("log")
    ax.set_xlabel("Estimated cost per cell (USD, log scale)")
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0.55, 1.0)
    ax.set_title("Cost vs. quality across LLM tiers  (one full corpus run)\n"
                 "Sonnet point is estimated; not measured here.",
                 fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "11_cost_vs_f1.png", dpi=150)
    return fig

# ---------------------------------------------------------------------------
# 12 - Pipeline block diagram
# ---------------------------------------------------------------------------

def fig_12_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(14, 5.0))
    BW, BH = 2.6, 1.4   # box width, height

    # Layout:  PDF -> Parse -> Index -> Retrieval ->  { Baseline, LLM } -> KPIExtraction
    #          y=2 row everywhere except Baseline (y=3.2), LLM (y=0.8), at column x=11.
    blocks = [
        ("PDF\n(report)",                            0.0,  2.0, "#4a4a4a"),
        ("ParsedReport\n(Docling /\npdfplumber)",     3.5,  2.0, "#1f77b4"),
        ("IndexedReport\n(ClimateBERT +\nBM25)",      7.0,  2.0, "#1f77b4"),
        ("rank_pages_hybrid\n(top-16 pages)",        10.5,  2.0, "#ff7f0e"),
        ("BaselineExtractor\n(table-first\n+ line-scanner)", 14.0, 3.4, "#2ca02c"),
        ("LLMExtractor\n(Anthropic /\nGemini)",      14.0,  0.6, "#9467bd"),
        ("KPIExtraction\n(value, unit,\nyear, snippet)", 17.5, 2.0, "#d4af37"),
    ]
    centers = []
    for label, x, y, color in blocks:
        ax.add_patch(Rectangle((x, y - BH / 2), BW, BH, color=color,
                                alpha=0.9, zorder=2))
        ax.text(x + BW / 2, y, label, ha="center", va="center",
                color="white", fontsize=9, fontweight="bold", zorder=3)
        centers.append((x + BW / 2, y, x, x + BW, y - BH / 2, y + BH / 2))

    def _arrow(src_idx, dst_idx, label, label_off=0.25, label_above=True):
        cx_s, cy_s, _, x_right_s, _, _ = centers[src_idx]
        cx_d, cy_d, x_left_d, _, _, _ = centers[dst_idx]
        ax.add_patch(FancyArrowPatch(
            (x_right_s, cy_s), (x_left_d, cy_d),
            arrowstyle="->", mutation_scale=16,
            color="#444", lw=1.6, zorder=1))
        # Label position: midpoint, slight offset
        midx = (x_right_s + x_left_d) / 2
        midy = (cy_s + cy_d) / 2
        ax.text(midx, midy + (label_off if label_above else -label_off),
                label, fontsize=8, color="#444",
                ha="center", va="bottom" if label_above else "top",
                style="italic")

    _arrow(0, 1, "parse")
    _arrow(1, 2, "build index")
    _arrow(2, 3, "queries")
    _arrow(3, 4, "context", label_above=True)
    _arrow(3, 5, "context + tool-use", label_above=False)
    _arrow(4, 6, "value")
    _arrow(5, 6, "tool response", label_above=False)

    ax.set_xlim(-0.5, 21)
    ax.set_ylim(-0.5, 4.8)
    ax.axis("off")
    ax.set_title("Pipeline architecture — single PDF, single KPI",
                 fontsize=12, pad=15)
    fig.tight_layout()
    fig.savefig(FIGS / "12_pipeline_diagram.png", dpi=150,
                bbox_inches="tight")
    return fig

# ---------------------------------------------------------------------------
# 13 - Concrete trace: BP scope_1
# ---------------------------------------------------------------------------

def fig_13_concrete_trace():
    company, kpi = "bp", "scope_1_emissions"
    pl = _load_prompt_log(FLASH_RUN, company, kpi)
    g = _load_gold()
    grow = g[(g["company"] == company) & (g["kpi"] == kpi)].iloc[0]
    e = _load_extractions(FLASH_RUN)
    erow = e[(e["company"] == company) & (e["kpi"] == kpi)
             & (e["extractor"] == "llm")].iloc[0]

    fig = plt.figure(figsize=(13, 5.0))
    gs = fig.add_gridspec(1, 4, width_ratios=[2, 2, 2.5, 2.2])

    # Panel A: gold cell
    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    ax.set_title("Gold cell", fontsize=11)
    ax.text(0.0, 0.85, f"Company: BP", fontsize=11, fontweight="bold")
    ax.text(0.0, 0.70, f"KPI: Scope 1 emissions", fontsize=10)
    ax.text(0.0, 0.55, f"Year: {int(grow['reporting_year'])}", fontsize=10)
    ax.text(0.0, 0.40, f"Value: {grow['value']:,.0f}", fontsize=11,
            color=C_GOLD, fontweight="bold")
    ax.text(0.0, 0.27, f"Unit: {grow['unit']}", fontsize=10)
    ax.text(0.0, 0.14, f"Source page: {int(grow['source_page'])}",
            fontsize=10)

    # Panel B: top-5 retrieved pages
    ax = fig.add_subplot(gs[0, 1])
    ax.axis("off")
    ax.set_title("Top-5 retrieved pages", fontsize=11)
    gold_page = int(grow["source_page"])
    top5 = pl["retrieved_pages"][:5]
    for i, p in enumerate(top5):
        is_gold = (p == gold_page)
        ax.text(0.0, 0.85 - i * 0.13,
                f"{'*' if is_gold else '  '}  page {p}",
                fontsize=11,
                color=C_GOLD if is_gold else "#333",
                fontweight="bold" if is_gold else "normal")
    note = ("* = gold-bearing page" if gold_page in top5
            else f"(gold page {gold_page} ranks "
                 f"#{pl['retrieved_pages'].index(gold_page) + 1 if gold_page in pl['retrieved_pages'] else '>16'})")
    ax.text(0.0, 0.10, note, fontsize=8, style="italic", color=C_GOLD)

    # Panel C: source snippet
    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    ax.set_title("Source snippet (LLM)", fontsize=11)
    snippet = (pl["tool_response"] or {}).get("source_snippet", "")
    snippet = snippet[:240]
    import textwrap
    wrapped = textwrap.fill(snippet, width=44)
    ax.text(0.02, 0.85, wrapped, fontsize=9.5, family="monospace",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff8d6",
                      edgecolor=C_GOLD, linewidth=1))

    # Panel D: extracted value (canonicalized) + raw LLM output
    ax = fig.add_subplot(gs[0, 3])
    ax.axis("off")
    ax.set_title("Extracted (after canonicalisation)", fontsize=11)
    tr = pl["tool_response"] or {}
    canon_value = erow["value"]
    canon_unit = erow["unit"]
    is_match = (
        not pd.isna(canon_value)
        and canon_unit == grow["unit"]
        and abs(canon_value - grow["value"]) / grow["value"] <= 0.01
    )
    ax.text(0.0, 0.86, f"Value: {canon_value:,.0f}",
            fontsize=11, color=C_TP, fontweight="bold")
    ax.text(0.0, 0.74, f"Unit: {canon_unit}", fontsize=10)
    ax.text(0.0, 0.56,
            f"LLM raw: value={tr.get('value')}, unit={tr.get('unit')}",
            fontsize=8, color="#555")
    ax.text(0.0, 0.46,
            f"(canonicaliser converted {tr.get('unit')} → {canon_unit})",
            fontsize=8, style="italic", color="#555")
    ax.text(0.0, 0.22,
            f"Match against gold: {'✓' if is_match else '✗'}",
            fontsize=11, fontweight="bold",
            color=C_TP if is_match else C_RETR)

    fig.suptitle("Concrete extraction trace — BP, Scope 1 emissions  (LLM Gemini 2.5 Flash)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGS / "13_concrete_trace.png", dpi=150,
                bbox_inches="tight")
    return fig

# ---------------------------------------------------------------------------
# 14 - Cost vs corpus size
# ---------------------------------------------------------------------------

def fig_14_cost_corpus_size():
    sizes = np.array([5, 10, 50, 100, 500, 1000, 10000])
    # Per-cell cost estimates from §11 + SUSTAINABILITY (3 KPIs per report).
    cost_paid_gemini = sizes * 3 * 0.0033
    cost_paid_anthropic = sizes * 3 * 0.10

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(sizes, cost_paid_gemini, marker="o", color=C_FLASH,
            label="Gemini 2.5 Flash paid (~$0.003 / cell)", lw=2)
    ax.plot(sizes, cost_paid_anthropic, marker="o", color=C_BASELINE,
            label="Anthropic Sonnet 4.6  (~$0.10 / cell, est.)", lw=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(4, 12000)
    ax.set_ylim(0.04, 5000)
    ax.set_xlabel("Corpus size (number of reports)")
    ax.set_ylabel("Estimated USD for one full pipeline run")
    ax.set_title("Cost vs. corpus size  (3 KPIs per report)",
                 fontsize=12)
    ax.legend(loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    # Annotate free tier
    ax.text(0.02, 0.92,
            "Gemini free tier: $0 at any corpus size\n"
            "(rate-limited to 20 RPD per project)",
            transform=ax.transAxes, fontsize=9,
            color=C_LITE, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff4e0",
                      edgecolor=C_LITE, linewidth=1))
    fig.tight_layout()
    fig.savefig(FIGS / "14_cost_vs_corpus.png", dpi=150)
    return fig

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

ALL_FIGURES = [
    fig_01_tp_per_kpi,
    fig_02_headline_total,
    fig_03_f1_per_kpi,
    fig_04_scorecard,
    fig_05_lite_vs_flash_diff,
    fig_06_failure_modes,
    fig_07_sankey,
    fig_08_rank_of_gold,
    fig_09_retrieval_ablation,
    fig_10_embedding_comparison,
    fig_11_cost_vs_f1,
    fig_12_pipeline_diagram,
    fig_13_concrete_trace,
    fig_14_cost_corpus_size,
]

# The recommended-six subset for the slim notebook.
CORE_FIGURES = [
    fig_01_tp_per_kpi,
    fig_04_scorecard,
    fig_05_lite_vs_flash_diff,
    fig_06_failure_modes,
    fig_08_rank_of_gold,
    fig_12_pipeline_diagram,
]

def _apply_style():
    sns.set_theme(style="white", context="talk")
    # DejaVu Sans is bundled with matplotlib and has full Unicode coverage
    # (✓, ✗, ³, etc.) — Arial misses several of these glyphs.
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
    })


if __name__ == "__main__":
    _apply_style()
    for fn in ALL_FIGURES:
        print(f"==> {fn.__name__}")
        fn()
        plt.close("all")
    print(f"\nWrote {len(ALL_FIGURES)} figures to {FIGS}")
