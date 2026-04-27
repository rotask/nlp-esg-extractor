"""Top-level orchestration for the ESG KPI extraction run.

`main` loads PDFs from `data/reports/`, builds an `IndexedReport` for
each (parsed pages + sentence/table-header embeddings), runs both the
deterministic baseline and the LLM extractor across every KPI,
evaluates against `data/labels/gold_labels.csv`, and persists the
artifacts under `data/runs/<run_tag>/` (`extractions.csv`,
`metrics.csv`, plus per-(company, kpi) prompt logs in
`llm_prompts/`).
"""
from __future__ import annotations
import argparse
import dataclasses
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from nlp_esg.compare import build_comparison_table
from nlp_esg.config import KPI_KEYS, LABELS_DIR, REPORTS_DIR, RUNS_DIR
from nlp_esg.evaluate import evaluate
from nlp_esg.extractors.baseline import BaselineExtractor
from nlp_esg.extractors.llm import LLMExtractor
from nlp_esg.ingest import parse_pdf
from nlp_esg.retrieval import build_index
from nlp_esg.types import KPIExtraction

log = logging.getLogger(__name__)


def load_indexed_reports(reports_dir: Path = REPORTS_DIR) -> list:
    indexed = []
    for pdf in sorted(reports_dir.glob("*.pdf")):
        try:
            parsed = parse_pdf(pdf)
        except Exception as e:
            log.error("Failed to parse %s: %s", pdf.name, e)
            continue
        indexed.append(build_index(parsed))
    return indexed


def run_extraction(
    indexed_reports: Iterable,
    include_llm: bool = True,
    prompt_log_dir: Path | None = None,
) -> list[KPIExtraction]:
    baseline = BaselineExtractor()
    llm = LLMExtractor(prompt_log_dir=prompt_log_dir) if include_llm else None

    out: list[KPIExtraction] = []
    for report in indexed_reports:
        for kpi_key in KPI_KEYS:
            out.append(baseline.extract(report, kpi_key))
            if llm is not None:
                out.append(llm.extract(report, kpi_key))
    return out


def load_gold_labels(path: Path | None = None) -> list[dict]:
    path = path or (LABELS_DIR / "gold_labels.csv")
    if not path.exists():
        return []
    df = pd.read_csv(path)
    df = df.where(pd.notna(df), None)
    for col in ("report_year", "reporting_year"):
        if col in df.columns:
            df[col] = df[col].apply(lambda v: int(v) if v is not None else None)
    return df.to_dict(orient="records")


def run_evaluation(
    extractions: list[KPIExtraction], golds: list[dict]
) -> pd.DataFrame:
    rows = []
    for extractor in sorted({e.extractor for e in extractions}):
        for kpi in KPI_KEYS:
            preds = [e for e in extractions if e.extractor == extractor and e.kpi == kpi]
            kpi_golds = [g for g in golds if g["kpi"] == kpi]
            rows.append(evaluate(preds, kpi_golds, extractor=extractor, kpi=kpi))
    return pd.DataFrame(rows)


def _persist_run(
    extractions: list[KPIExtraction],
    metrics_df: pd.DataFrame | None,
    run_tag: str,
    runs_dir: Path,
) -> None:
    out_dir = runs_dir / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [dataclasses.asdict(e) for e in extractions]
    pd.DataFrame(rows).to_csv(out_dir / "extractions.csv", index=False)
    if metrics_df is not None:
        metrics_df.to_csv(out_dir / "metrics.csv", index=False)


def main(run_tag: str = "v2_docling") -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    indexed = load_indexed_reports()
    log.info("Loaded %d reports", len(indexed))
    if not indexed:
        log.error("No reports found in %s", REPORTS_DIR)
        return

    prompt_log_dir = RUNS_DIR / run_tag / "llm_prompts"
    extractions = run_extraction(
        indexed, include_llm=True, prompt_log_dir=prompt_log_dir,
    )
    for e in extractions:
        e.run_tag = run_tag

    for extractor in ("baseline", "llm"):
        df = build_comparison_table(extractions, extractor=extractor)
        print(f"\n=== {extractor} comparison table ===")
        print(df)

    golds = load_gold_labels()
    metrics_df: pd.DataFrame | None = None
    if golds:
        metrics_df = run_evaluation(extractions, golds)
        print("\n=== Evaluation ===")
        print(metrics_df)
    else:
        log.warning("No gold labels found at %s — skipping evaluation",
                    LABELS_DIR / "gold_labels.csv")

    _persist_run(extractions, metrics_df, run_tag, RUNS_DIR)
    log.info("Persisted run to %s/%s/", RUNS_DIR, run_tag)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", default="v2_docling")
    args = parser.parse_args()
    main(run_tag=args.run_tag)
