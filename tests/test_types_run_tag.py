from nlp_esg.types import KPIExtraction


def test_run_tag_defaults_to_none():
    e = KPIExtraction(
        company="bp", report_year=2024, kpi="scope_1_emissions",
        value=None, unit=None, reporting_year=None,
        source_snippet=None, source_page=None, confidence=None,
        extractor="baseline",
    )
    assert e.run_tag is None


def test_run_tag_can_be_set():
    e = KPIExtraction(
        company="bp", report_year=2024, kpi="scope_1_emissions",
        value=33.7e6, unit="tCO2e", reporting_year=2024,
        source_snippet="snip", source_page=12, confidence=0.9,
        extractor="llm", run_tag="v2_docling",
    )
    assert e.run_tag == "v2_docling"
