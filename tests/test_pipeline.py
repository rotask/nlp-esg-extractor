from unittest.mock import patch
import pandas as pd
from nlp_esg.pipeline import run_extraction


def test_run_extraction_produces_rows_per_report_and_kpi(
    fake_embed, report_with_table, report_sentence_only
):
    from nlp_esg.retrieval import build_index
    indexed_reports = [build_index(report_with_table), build_index(report_sentence_only)]

    # Mock the LLM extractor at class level so it returns deterministic "not reported".
    from nlp_esg.types import KPIExtraction

    def fake_llm_extract(self, report, kpi_key):
        return KPIExtraction(
            company=report["company"], report_year=report["report_year"], kpi=kpi_key,
            value=None, unit=None, reporting_year=None,
            source_snippet=None, source_page=None, confidence=None,
            extractor="llm", flags=[],
        )

    with patch("nlp_esg.extractors.llm.LLMExtractor.extract", fake_llm_extract):
        extractions = run_extraction(indexed_reports, include_llm=True)

    # 2 reports x 3 KPIs x 2 extractors = 12 rows
    assert len(extractions) == 12
    assert {e.extractor for e in extractions} == {"baseline", "llm"}
    assert {e.kpi for e in extractions} == {
        "scope_1_emissions", "total_energy_consumption", "water_consumption",
    }
