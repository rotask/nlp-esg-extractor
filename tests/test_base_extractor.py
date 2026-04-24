import pytest
from nlp_esg.extractors.base import Extractor


def test_extractor_is_abstract():
    with pytest.raises(TypeError):
        Extractor()  # type: ignore[abstract]


def test_concrete_subclass_works():
    from nlp_esg.types import KPIExtraction

    class Dummy(Extractor):
        def extract(self, report, kpi_key):
            return KPIExtraction(
                company=report["company"], report_year=report["report_year"],
                kpi=kpi_key, value=None, unit=None, reporting_year=None,
                source_snippet=None, source_page=None, confidence=None,
                extractor="dummy", flags=[],
            )

    d = Dummy()
    result = d.extract({"company": "x", "report_year": 2024}, "scope_1_emissions")
    assert result.extractor == "dummy"
