"""One-off runner: full pipeline with Docling enabled, per-PDF timings.

Mirrors `nlp_esg.pipeline.main` but instruments the parse+index stage so
we can see exactly how long each PDF spends in Docling vs ClimateBERT
embedding. Caches the parsed-report and indexed-report on disk just like
the regular pipeline.
"""
from __future__ import annotations
import argparse
import dataclasses
import logging
import time
from pathlib import Path

import pandas as pd

from nlp_esg.compare import build_comparison_table
from nlp_esg.config import CACHE_DIR, KPI_KEYS, LABELS_DIR, REPORTS_DIR, RUNS_DIR
from nlp_esg.evaluate import evaluate
from nlp_esg.ingest import _parse_filename, _parse_with_pdfplumber
from nlp_esg.ingest_docling import parse_with_docling
from nlp_esg.pipeline import (
    _persist_run,
    load_gold_labels,
    run_evaluation,
    run_extraction,
)
from nlp_esg.retrieval import build_index
import pickle

log = logging.getLogger("run_docling_full")


def _force_docling_parse(pdf: Path):
    """Force Docling-first parsing, ignoring any existing pdfplumber cache.

    The stock `nlp_esg.ingest.parse_pdf` reads pdfplumber cache as a
    second-chance fallback; that masks the docling experiment because
    the corpus already has pdfplumber caches on disk. Here we look at
    only the docling cache. Cache miss → run docling fresh; on docling
    failure, fall back to a fresh pdfplumber parse (no cache lookup,
    so we still see honest behaviour for that PDF).
    """
    company, year = _parse_filename(pdf)
    docling_cache = CACHE_DIR / f"{company}_{year}_docling.pkl"
    pdfplumber_cache = CACHE_DIR / f"{company}_{year}_pdfplumber.pkl"

    if docling_cache.exists() and docling_cache.stat().st_mtime >= pdf.stat().st_mtime:
        with docling_cache.open("rb") as f:
            return pickle.load(f), True

    report = parse_with_docling(pdf)
    if report is None or not report.get("pages"):
        log.warning("docling failed for %s; falling back to pdfplumber", pdf.name)
        report = _parse_with_pdfplumber(pdf)
        cache_path = pdfplumber_cache
    else:
        cache_path = docling_cache

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(report, f)
    return report, False


def load_indexed_reports_timed(reports_dir: Path = REPORTS_DIR):
    """Force docling-first parsing and time each PDF's parse + index step."""
    indexed = []
    timings: list[dict] = []
    pdfs = sorted(reports_dir.glob("*.pdf"))
    log.info("== %d PDFs to process ==", len(pdfs))
    for pdf in pdfs:
        log.info("== %s (%.1f MB) ==", pdf.name, pdf.stat().st_size / 1e6)
        try:
            t_parse = time.time()
            parsed, parse_cache_hit = _force_docling_parse(pdf)
            dt_parse = time.time() - t_parse
        except Exception as e:
            log.error("parse failed for %s: %s", pdf.name, e)
            timings.append({"pdf": pdf.name, "parse_s": -1, "index_s": -1, "parser": "error"})
            continue

        t_index = time.time()
        idx = build_index(parsed)
        dt_index = time.time() - t_index
        indexed.append(idx)

        timings.append({
            "pdf": pdf.name,
            "parser": parsed["parser"],
            "n_pages": len(parsed["pages"]),
            "n_tables": len(parsed["tables"]),
            "parse_s": round(dt_parse, 1),
            "parse_cache_hit": parse_cache_hit,
            "index_s": round(dt_index, 1),
            "total_s": round(dt_parse + dt_index, 1),
        })
        log.info(
            "   parser=%s pages=%d tables=%d parse=%.1fs (cache_hit=%s) index=%.1fs",
            parsed["parser"], len(parsed["pages"]), len(parsed["tables"]),
            dt_parse, parse_cache_hit, dt_index,
        )
    return indexed, timings


def main(run_tag: str) -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    overall_t0 = time.time()
    indexed, timings = load_indexed_reports_timed()
    if not indexed:
        log.error("no reports loaded")
        return

    parse_index_total = time.time() - overall_t0
    log.info("== Parse+index complete in %.1fs (%.1f min) ==",
             parse_index_total, parse_index_total / 60)
    log.info("Per-PDF timings:")
    for t in timings:
        log.info("  %s: %s", t["pdf"], t)

    prompt_log_dir = RUNS_DIR / run_tag / "llm_prompts"
    extr_t0 = time.time()
    extractions = run_extraction(
        indexed, include_llm=True, prompt_log_dir=prompt_log_dir,
    )
    for e in extractions:
        e.run_tag = run_tag
    extr_dt = time.time() - extr_t0
    log.info("== Extraction complete in %.1fs ==", extr_dt)

    for extractor in ("baseline", "llm"):
        df = build_comparison_table(extractions, extractor=extractor)
        print(f"\n=== {extractor} comparison table ===")
        print(df)

    golds = load_gold_labels()
    metrics_df = None
    if golds:
        metrics_df = run_evaluation(extractions, golds)
        print("\n=== Evaluation ===")
        print(metrics_df)

    _persist_run(extractions, metrics_df, run_tag, RUNS_DIR)

    # Persist timings alongside extractions/metrics.
    out_dir = RUNS_DIR / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(timings).to_csv(out_dir / "parse_timings.csv", index=False)

    overall_dt = time.time() - overall_t0
    log.info("== Pipeline finished in %.1fs (%.1f min) ==", overall_dt, overall_dt / 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", default="v_docling_full")
    args = parser.parse_args()
    main(run_tag=args.run_tag)
