import os
from pathlib import Path
import pytest

from nlp_esg.config import REPORTS_DIR

_PDFS = list(REPORTS_DIR.glob("*.pdf")) if REPORTS_DIR.exists() else []

pytestmark = pytest.mark.skipif(
    not _PDFS,
    reason="No PDFs in data/reports — skipping real-PDF integration test.",
)


def test_real_pdf_pipeline_runs_end_to_end():
    from nlp_esg.ingest import parse_pdf
    from nlp_esg.retrieval import build_index
    from nlp_esg.extractors.baseline import BaselineExtractor

    pdf = _PDFS[0]
    parsed = parse_pdf(pdf)
    indexed = build_index(parsed)
    ext = BaselineExtractor()
    # Just check extraction runs without exception for all 3 KPIs.
    for kpi in ("scope_1_emissions", "renewable_energy", "water_consumption"):
        result = ext.extract(indexed, kpi)
        assert result.extractor == "baseline"
