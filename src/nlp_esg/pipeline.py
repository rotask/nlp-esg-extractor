from __future__ import annotations
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from nlp_esg.compare import build_comparison_table
from nlp_esg.config import KPI_KEYS, LABELS_DIR, REPORTS_DIR
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
    indexed_reports: Iterable, include_llm: bool = True
) -> list[KPIExtraction]:
    baseline = BaselineExtractor()
    llm = LLMExtractor() if include_llm else None

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
    # Coerce year fields to int where non-null
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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    indexed = load_indexed_reports()
    log.info("Loaded %d reports", len(indexed))
    if not indexed:
        log.error("No reports found in %s", REPORTS_DIR)
        return

    extractions = run_extraction(indexed, include_llm=True)

    for extractor in ("baseline", "llm"):
        df = build_comparison_table(extractions, extractor=extractor)
        print(f"\n=== {extractor} comparison table ===")
        print(df)

    golds = load_gold_labels()
    if golds:
        metrics = run_evaluation(extractions, golds)
        print("\n=== Evaluation ===")
        print(metrics)
    else:
        log.warning("No gold labels found at %s — skipping evaluation",
                    LABELS_DIR / "gold_labels.csv")


if __name__ == "__main__":
    main()
