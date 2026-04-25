import os
import pytest

pytestmark = pytest.mark.skipif(
    not (os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("RUN_INTEGRATION")),
    reason="Set ANTHROPIC_API_KEY and RUN_INTEGRATION=1 to run integration tests.",
)


def test_llm_extractor_real_call():
    from nlp_esg.extractors.llm import LLMExtractor

    stub = {
        "company": "acme", "report_year": 2024,
        "pages": [],
        "tables": [],
        "sentences": [
            {"page_num": 1,
             "text": "Scope 1 direct greenhouse gas emissions in 2024 were 45,678 tCO2e.",
             "embedding": None},
        ],
        "table_headers": [],
    }
    ext = LLMExtractor()
    result = ext.extract(stub, "scope_1_emissions")
    assert result.value is not None
    assert abs(result.value - 45678.0) / 45678.0 < 0.01
