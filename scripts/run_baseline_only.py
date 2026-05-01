"""Baseline-only pipeline runner.

Runs the deterministic baseline extractor against the cached
`IndexedReport` for every PDF in `data/reports/`, evaluates against the
gold labels in `data/labels/gold_labels.csv`, and writes results under
`data/runs/<run_tag>/` (default tag: `v_baseline_only`).

Use this when:
  - You want to validate the rule-based baseline without spending Gemini
    or Anthropic API credits.
  - The LLM provider's daily quota is exhausted.
  - You're iterating on baseline-only changes (regex, negative tokens,
    table-row scoring) and don't want LLM calls in the loop.

Run from the **repository root** so the relative `data/` paths resolve:

    cd /path/to/nlp-esg-extractor
    python scripts/run_baseline_only.py
    python scripts/run_baseline_only.py --run-tag my_baseline_run

The first invocation parses + indexes any PDF that doesn't yet have a
cache entry under `data/cache/`. Subsequent invocations reuse the
cached `IndexedReport`s — typical wall-clock is a few seconds.
"""
from __future__ import annotations
import argparse
import dataclasses
import logging
import pickle
from pathlib import Path

import pandas as pd

from nlp_esg.config import CACHE_DIR, KPI_KEYS, REPORTS_DIR, RUNS_DIR
from nlp_esg.evaluate import evaluate
from nlp_esg.extractors.baseline import BaselineExtractor
from nlp_esg.ingest import _parse_filename, _parse_with_pdfplumber
from nlp_esg.ingest_docling import parse_with_docling
from nlp_esg.pipeline import load_gold_labels
from nlp_esg.retrieval import build_index

log = logging.getLogger("baseline_only")


def _parsed_report_for(pdf: Path):
    """Force docling-first parsing, ignoring any pre-existing pdfplumber cache.

    Mirrors the dispatcher behaviour of `nlp_esg.ingest.parse_pdf` but
    only honours the docling cache; on cache miss it runs docling fresh
    and falls back to pdfplumber on docling failure. This isolates the
    Docling-track behaviour from any stale pdfplumber caches that might
    coexist on disk.
    """
    company, year = _parse_filename(pdf)
    docling_cache = CACHE_DIR / f"{company}_{year}_docling.pkl"
    pdfplumber_cache = CACHE_DIR / f"{company}_{year}_pdfplumber.pkl"

    if docling_cache.exists() and docling_cache.stat().st_mtime >= pdf.stat().st_mtime:
        with docling_cache.open("rb") as f:
            return pickle.load(f)

    report = parse_with_docling(pdf)
    used_parser = "docling"
    if report is None or not report.get("pages"):
        log.warning("docling failed for %s; falling back to pdfplumber", pdf.name)
        report = _parse_with_pdfplumber(pdf)
        used_parser = "pdfplumber"

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{company}_{year}_{used_parser}.pkl"
    with cache_path.open("wb") as f:
        pickle.dump(report, f)
    return report


def main(run_tag: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    out_dir = RUNS_DIR / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    extractor = BaselineExtractor()
    extractions = []
    pdfs = sorted(REPORTS_DIR.glob("*.pdf"))
    if not pdfs:
        log.error("no PDFs found in %s", REPORTS_DIR)
        return
    log.info("== %d PDFs to process ==", len(pdfs))

    for pdf in pdfs:
        log.info("processing %s", pdf.name)
        try:
            parsed = _parsed_report_for(pdf)
        except Exception as e:
            log.error("parse failed for %s: %s", pdf.name, e)
            continue
        indexed = build_index(parsed)
        for kpi_key in KPI_KEYS:
            row = extractor.extract(indexed, kpi_key)
            row.run_tag = run_tag
            extractions.append(row)

    golds = load_gold_labels()
    rows = []
    for kpi in KPI_KEYS:
        preds = [e for e in extractions if e.extractor == "baseline" and e.kpi == kpi]
        kpi_golds = [g for g in golds if g["kpi"] == kpi]
        rows.append(evaluate(preds, kpi_golds, extractor="baseline", kpi=kpi))
    metrics = pd.DataFrame(rows)

    print("\n=== Baseline-only metrics ===")
    print(metrics.to_string(index=False))
    total_tp = int(metrics["tp"].sum())
    print(f"\nBaseline total: {total_tp}/15 (TP); F1 macro = {metrics['f1'].mean():.3f}")

    # Persist
    pd.DataFrame([dataclasses.asdict(e) for e in extractions]).to_csv(
        out_dir / "extractions.csv", index=False
    )
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    print(f"\nPersisted to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline-only pipeline runner")
    parser.add_argument(
        "--run-tag", default="v_baseline_only",
        help="Output directory name under data/runs/ (default: v_baseline_only)",
    )
    args = parser.parse_args()
    main(run_tag=args.run_tag)
