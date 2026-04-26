from unittest.mock import MagicMock, patch
import pytest

from nlp_esg.extractors.llm import LLMExtractor


class _FakeToolUse:
    def __init__(self, name: str, input_: dict):
        self.type = "tool_use"
        self.name = name
        self.input = input_


class _FakeResponse:
    def __init__(self, content):
        self.content = content
        self.stop_reason = "tool_use"


def _valid_response(value=45678.0, unit="tCO2e", year=2024, snippet="...", confidence=0.9):
    return _FakeResponse([_FakeToolUse("record_kpi", {
        "value": value, "unit": unit, "reporting_year": year,
        "source_snippet": snippet, "confidence": confidence,
    })])


def _not_reported_response():
    return _FakeResponse([_FakeToolUse("record_kpi", {
        "value": None, "unit": None, "reporting_year": None,
        "source_snippet": None, "confidence": 0.9,
    })])


@pytest.fixture
def indexed_stub():
    return {
        "company": "acme", "report_year": 2024,
        "pages": [{"page_num": 5, "text": "Scope 1 was 45,678 tCO2e in 2024."}],
        "tables": [],
        "sentences": [{"page_num": 5, "text": "Scope 1 was 45,678 tCO2e in 2024.",
                       "embedding": None}],
        "table_headers": [],
    }


def test_llm_parses_valid_tool_use(indexed_stub):
    with patch("nlp_esg.extractors.llm.Anthropic") as mock_cls:
        client = MagicMock()
        client.messages.create.return_value = _valid_response()
        mock_cls.return_value = client

        ext = LLMExtractor()
        with patch.object(ext, "_build_context", return_value="context"):
            result = ext.extract(indexed_stub, "scope_1_emissions")

    assert result.value == pytest.approx(45678.0)
    assert result.unit == "tCO2e"
    assert result.reporting_year == 2024
    assert result.extractor == "llm"


def test_llm_handles_not_reported(indexed_stub):
    with patch("nlp_esg.extractors.llm.Anthropic") as mock_cls:
        client = MagicMock()
        client.messages.create.return_value = _not_reported_response()
        mock_cls.return_value = client

        ext = LLMExtractor()
        with patch.object(ext, "_build_context", return_value="context"):
            result = ext.extract(indexed_stub, "scope_1_emissions")

    assert result.value is None
    assert result.unit is None


def test_llm_rejects_value_without_unit(indexed_stub):
    bad = _FakeResponse([_FakeToolUse("record_kpi", {
        "value": 45678.0, "unit": None, "reporting_year": 2024,
        "source_snippet": "...", "confidence": 0.9,
    })])
    with patch("nlp_esg.extractors.llm.Anthropic") as mock_cls:
        client = MagicMock()
        client.messages.create.return_value = bad
        mock_cls.return_value = client

        ext = LLMExtractor()
        with patch.object(ext, "_build_context", return_value="context"):
            result = ext.extract(indexed_stub, "scope_1_emissions")

    assert result.value is None
    assert "llm_missing_unit" in result.flags


def test_llm_rejects_out_of_range(indexed_stub):
    bad = _FakeResponse([_FakeToolUse("record_kpi", {
        "value": 0.01, "unit": "tCO2e", "reporting_year": 2024,
        "source_snippet": "...", "confidence": 0.9,
    })])
    with patch("nlp_esg.extractors.llm.Anthropic") as mock_cls:
        client = MagicMock()
        client.messages.create.return_value = bad
        mock_cls.return_value = client

        ext = LLMExtractor()
        with patch.object(ext, "_build_context", return_value="context"):
            result = ext.extract(indexed_stub, "scope_1_emissions")

    assert result.value is None
    assert "out_of_range" in result.flags


def test_llm_retries_on_api_error(indexed_stub):
    with patch("nlp_esg.extractors.llm.Anthropic") as mock_cls:
        client = MagicMock()
        client.messages.create.side_effect = [
            Exception("transient"),
            _valid_response(),
        ]
        mock_cls.return_value = client

        ext = LLMExtractor(max_retries=2, retry_base_delay=0)
        with patch.object(ext, "_build_context", return_value="context"):
            result = ext.extract(indexed_stub, "scope_1_emissions")

    assert result.value == pytest.approx(45678.0)
    assert client.messages.create.call_count == 2


def test_llm_gives_up_after_max_retries(indexed_stub):
    with patch("nlp_esg.extractors.llm.Anthropic") as mock_cls:
        client = MagicMock()
        client.messages.create.side_effect = Exception("always fail")
        mock_cls.return_value = client

        ext = LLMExtractor(max_retries=2, retry_base_delay=0)
        with patch.object(ext, "_build_context", return_value="context"):
            result = ext.extract(indexed_stub, "scope_1_emissions")

    assert result.value is None
    assert "api_error" in result.flags


def test_llm_build_context_uses_multi_query_hybrid(monkeypatch):
    """LLMExtractor._build_context should consult kpi['queries'] (plural)."""
    from nlp_esg.extractors.llm import LLMExtractor
    captured = {}

    def fake_hybrid(report, queries, **kw):
        captured["queries"] = list(queries)
        return [(p["page_num"], 1.0) for p in report["pages"]]

    monkeypatch.setattr("nlp_esg.extractors.llm.rank_pages_hybrid", fake_hybrid)

    indexed = {
        "company": "x", "report_year": 2024, "parser": "pdfplumber",
        "pages": [{"page_num": 1, "text": "page one"}],
        "sentences": [], "table_headers": [], "tables": [],
    }
    ext = LLMExtractor()
    ctx = ext._build_context(
        indexed,
        kpi_query="Scope 1 direct greenhouse gas emissions",
        kpi_unit_family=["tCO2e"],
        kpi_queries=[
            "Total gross Scope 1 GHG emissions",
            "Scope 1 (direct) emissions",
        ],
    )
    assert "Total gross Scope 1 GHG emissions" in captured["queries"]
    assert "Scope 1 (direct) emissions" in captured["queries"]
    assert "page one" in ctx
