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


def test_main_persists_extractions_with_run_tag(tmp_path, monkeypatch):
    from nlp_esg import pipeline
    from nlp_esg.types import KPIExtraction

    monkeypatch.setattr(pipeline, "load_indexed_reports", lambda: ["fake"])
    fake_extractions = [
        KPIExtraction(
            company="bp", report_year=2024, kpi="scope_1_emissions",
            value=33.7e6, unit="tCO2e", reporting_year=2024,
            source_snippet=None, source_page=None, confidence=0.9,
            extractor="llm",
        )
    ]
    monkeypatch.setattr(
        pipeline, "run_extraction",
        lambda r, include_llm=True, prompt_log_dir=None: fake_extractions,
    )
    monkeypatch.setattr(pipeline, "load_gold_labels", lambda: [])
    monkeypatch.setattr(pipeline, "RUNS_DIR", tmp_path)

    pipeline.main(run_tag="v2_test")

    csv_path = tmp_path / "v2_test" / "extractions.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert "run_tag" in df.columns
    assert (df["run_tag"] == "v2_test").all()
