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


def test_cache_key_changes_when_system_prompt_changes(monkeypatch, tmp_path):
    """Same (model, kpi, user_prompt) but different system_prompt -> different key."""
    from nlp_esg.extractors import llm as llm_mod

    ext = llm_mod.LLMExtractor()
    k1 = ext._cache_key("scope_1_emissions", "user-text", "system-A")
    k2 = ext._cache_key("scope_1_emissions", "user-text", "system-B")
    assert k1 != k2


def test_cache_key_stable_when_inputs_match():
    from nlp_esg.extractors import llm as llm_mod

    ext = llm_mod.LLMExtractor()
    k1 = ext._cache_key("scope_1_emissions", "user-text", "system-A")
    k2 = ext._cache_key("scope_1_emissions", "user-text", "system-A")
    assert k1 == k2


def test_provider_defaults_to_anthropic(monkeypatch):
    """With no env var and no kwarg, LLMExtractor uses Anthropic — backwards compat."""
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    from nlp_esg.extractors.llm import LLMExtractor
    ext = LLMExtractor()
    assert ext.provider == "anthropic"


def test_provider_reads_env_when_not_passed(monkeypatch):
    """LLM_PROVIDER and GEMINI_MODEL env vars drive the default selection."""
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    from nlp_esg.extractors.llm import LLMExtractor
    ext = LLMExtractor()
    assert ext.provider == "gemini"
    assert ext.model == "gemini-2.5-flash"


def test_explicit_kwarg_overrides_env(monkeypatch):
    """An explicit provider= kwarg always wins over the env var."""
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    from nlp_esg.extractors.llm import LLMExtractor
    ext = LLMExtractor(provider="anthropic", model="claude-sonnet-4-6")
    assert ext.provider == "anthropic"
    assert ext.model == "claude-sonnet-4-6"


def test_provider_can_be_set_to_gemini():
    """LLMExtractor accepts provider='gemini' and stores it."""
    from nlp_esg.extractors.llm import LLMExtractor
    ext = LLMExtractor(provider="gemini", model="gemini-2.0-flash")
    assert ext.provider == "gemini"
    assert ext.model == "gemini-2.0-flash"


def test_invalid_provider_raises():
    """Unknown providers should fail fast at construction."""
    from nlp_esg.extractors.llm import LLMExtractor
    with pytest.raises(ValueError, match="provider"):
        LLMExtractor(provider="not-a-real-provider")


def test_gemini_provider_parses_function_call_response():
    """Gemini provider should call generate_content and return the function_call args."""
    from unittest.mock import MagicMock, patch
    from nlp_esg.extractors.llm import LLMExtractor

    # Build a fake Gemini response: candidates[0].content.parts[0].function_call
    function_call = MagicMock()
    function_call.name = "record_kpi"
    function_call.args = {
        "value": 33700000.0, "unit": "tCO2e", "reporting_year": 2024,
        "source_snippet": "Total Scope 1 ...", "confidence": 0.92,
    }
    part = MagicMock(function_call=function_call)
    candidate = MagicMock()
    candidate.content = MagicMock(parts=[part])
    response = MagicMock(candidates=[candidate])

    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = response

    with patch("nlp_esg.extractors.llm._gemini_client", return_value=fake_client):
        ext = LLMExtractor(provider="gemini", model="gemini-2.0-flash")
        result = ext._call_with_retry(
            user_prompt="ctx",
            kpi_key="scope_1_emissions",
            system_prompt="rules",
        )

    assert result == {
        "value": 33700000.0, "unit": "tCO2e", "reporting_year": 2024,
        "source_snippet": "Total Scope 1 ...", "confidence": 0.92,
    }
    # Verify the Gemini SDK was called with the right model
    call_kwargs = fake_client.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == "gemini-2.0-flash"


def test_gemini_provider_handles_no_function_call():
    """If Gemini returns a response without a function_call block, return None."""
    from unittest.mock import MagicMock, patch
    from nlp_esg.extractors.llm import LLMExtractor

    # Part has no function_call
    part = MagicMock()
    part.function_call = None
    response = MagicMock(candidates=[MagicMock(content=MagicMock(parts=[part]))])

    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = response
    with patch("nlp_esg.extractors.llm._gemini_client", return_value=fake_client):
        ext = LLMExtractor(provider="gemini", model="gemini-2.0-flash",
                            max_retries=1, retry_base_delay=0)
        result = ext._call_with_retry("ctx", "scope_1_emissions", "rules")
    assert result is None


def test_gemini_throttles_between_consecutive_calls(monkeypatch):
    """Two consecutive Gemini calls should sleep at least min_call_interval_s
    apart to stay under the 10 RPM free-tier rate limit."""
    from unittest.mock import MagicMock, patch
    from nlp_esg.extractors import llm as llm_mod

    fake_call = MagicMock()
    fake_call.name = "record_kpi"
    fake_call.args = {"value": 1, "unit": "tCO2e", "reporting_year": 2024,
                       "source_snippet": "x", "confidence": 0.9}
    response = MagicMock(candidates=[MagicMock(content=MagicMock(
        parts=[MagicMock(function_call=fake_call)]))])
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = response

    sleeps: list[float] = []
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: sleeps.append(s))

    with patch("nlp_esg.extractors.llm._gemini_client", return_value=fake_client):
        ext = llm_mod.LLMExtractor(
            provider="gemini", model="gemini-2.5-flash-lite",
            min_call_interval_s=6.5,
        )
        ext._call_with_retry("p1", "scope_1_emissions", "sys")
        ext._call_with_retry("p2", "scope_1_emissions", "sys")

    # Between two successive successful calls the throttle should have slept ≈ 6.5s
    assert any(s >= 6.0 for s in sleeps), f"no throttle sleep observed (sleeps={sleeps})"


def test_anthropic_does_not_throttle(monkeypatch):
    """Anthropic provider has its own rate-limit handling; no inter-call sleep."""
    from unittest.mock import MagicMock, patch
    from nlp_esg.extractors.llm import LLMExtractor

    block = MagicMock()
    block.type = "tool_use"
    block.name = "record_kpi"
    block.input = {"value": 1, "unit": "tCO2e", "reporting_year": 2024,
                    "source_snippet": "x", "confidence": 0.9}
    response = MagicMock(content=[block])

    sleeps: list[float] = []
    import nlp_esg.extractors.llm as llm_mod
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: sleeps.append(s))

    with patch("nlp_esg.extractors.llm.Anthropic") as mock_cls:
        client = MagicMock()
        client.messages.create.return_value = response
        mock_cls.return_value = client
        # No min_call_interval_s passed — Anthropic default is 0 (no throttle).
        ext = LLMExtractor(provider="anthropic")
        ext._call_with_retry("p1", "scope_1_emissions", "sys")
        ext._call_with_retry("p2", "scope_1_emissions", "sys")

    # Anthropic with default settings should not have triggered the throttle
    assert all(s < 6.0 for s in sleeps), f"Anthropic should not throttle (sleeps={sleeps})"


def test_gemini_provider_retries_on_exception():
    """Gemini call failures retry up to max_retries with backoff."""
    from unittest.mock import MagicMock, patch
    from nlp_esg.extractors.llm import LLMExtractor

    function_call = MagicMock()
    function_call.name = "record_kpi"
    function_call.args = {"value": 1, "unit": "tCO2e", "reporting_year": 2024,
                          "source_snippet": "x", "confidence": 0.9}
    success_response = MagicMock(candidates=[MagicMock(content=MagicMock(
        parts=[MagicMock(function_call=function_call)]))])

    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = [
        Exception("transient"), success_response,
    ]
    with patch("nlp_esg.extractors.llm._gemini_client", return_value=fake_client):
        ext = LLMExtractor(provider="gemini", model="gemini-2.0-flash",
                            max_retries=2, retry_base_delay=0)
        result = ext._call_with_retry("ctx", "scope_1_emissions", "rules")
    assert result is not None
    assert result["value"] == 1
    assert fake_client.models.generate_content.call_count == 2


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
