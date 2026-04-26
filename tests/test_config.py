from nlp_esg.config import KPIS, KPI_KEYS
from nlp_esg.types import KPIExtraction


def test_kpi_registry_has_three_keys():
    assert set(KPI_KEYS) == {"scope_1_emissions", "total_energy_consumption", "water_consumption"}


def test_each_kpi_has_required_fields():
    required = {"query", "unit_family", "canonical_unit", "plausible_range"}
    for key in KPI_KEYS:
        assert required.issubset(KPIS[key].keys()), f"{key} missing fields"


def test_plausible_ranges_are_ordered():
    for key in KPI_KEYS:
        lo, hi = KPIS[key]["plausible_range"]
        assert lo < hi


def test_kpi_extraction_dataclass_instantiates():
    x = KPIExtraction(
        company="acme", report_year=2024, kpi="scope_1_emissions",
        value=1000.0, unit="tCO2e", reporting_year=2024,
        source_snippet="table@page 5: Scope 1 | 1,000",
        source_page=5, confidence=0.8, extractor="baseline", flags=[],
    )
    assert x.value == 1000.0
    assert x.flags == []


def test_kpi_extraction_not_reported_is_allowed():
    x = KPIExtraction(
        company="acme", report_year=2024, kpi="water_consumption",
        value=None, unit=None, reporting_year=None,
        source_snippet=None, source_page=None, confidence=None,
        extractor="baseline", flags=[],
    )
    assert x.value is None


def test_every_kpi_has_queries_list():
    for kpi_key in KPI_KEYS:
        kpi = KPIS[kpi_key]
        assert "queries" in kpi, f"{kpi_key} missing 'queries'"
        assert isinstance(kpi["queries"], list)
        assert len(kpi["queries"]) >= 2, f"{kpi_key} needs at least 2 queries"
        assert all(isinstance(q, str) and q.strip() for q in kpi["queries"])


def test_single_query_field_still_present():
    for kpi_key in KPI_KEYS:
        assert "query" in KPIS[kpi_key]
