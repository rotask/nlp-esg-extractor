from __future__ import annotations
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from anthropic import Anthropic

from nlp_esg.config import ANTHROPIC_MODEL, CACHE_DIR, KPIS
from nlp_esg.extractors.base import Extractor
from nlp_esg.normalize import canonicalize_unit, normalize_co2, to_canonical_value
from nlp_esg.retrieval import embed_texts, rank_pages_cosine, rank_pages_hybrid
from nlp_esg.types import KPIExtraction

log = logging.getLogger(__name__)

_TOOL_SCHEMA = {
    "name": "record_kpi",
    "description": "Record the extracted KPI value. Use null for value when the KPI is not reported.",
    "input_schema": {
        "type": "object",
        "properties": {
            "value": {"type": ["number", "null"]},
            "unit": {"type": ["string", "null"]},
            "reporting_year": {"type": ["integer", "null"]},
            "source_snippet": {"type": ["string", "null"]},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": ["value", "unit", "reporting_year", "source_snippet", "confidence"],
    },
}

# Gemini uses a different tool format; same fields, slightly different shape.
_GEMINI_TOOLS = [{
    "function_declarations": [{
        "name": "record_kpi",
        "description": "Record the extracted KPI value. Use null for value when the KPI is not reported.",
        "parameters": {
            "type": "object",
            "properties": {
                "value": {"type": "number", "nullable": True},
                "unit": {"type": "string", "nullable": True},
                "reporting_year": {"type": "integer", "nullable": True},
                "source_snippet": {"type": "string", "nullable": True},
                "confidence": {"type": "number"},
            },
            "required": ["value", "unit", "reporting_year", "source_snippet", "confidence"],
        },
    }],
}]
_GEMINI_TOOL_CONFIG = {
    "function_calling_config": {
        "mode": "ANY",
        "allowed_function_names": ["record_kpi"],
    }
}

_VALID_PROVIDERS = ("anthropic", "gemini")


def _gemini_client():
    """Lazy import + create a Gemini client. Module-level so tests can patch it."""
    from google import genai
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY (or GOOGLE_API_KEY) is not set; cannot create Gemini client."
        )
    return genai.Client(api_key=api_key)

_SYSTEM_PROMPT = """You are an information-extraction assistant for ESG sustainability reports.

You will be given:
- A KPI to extract.
- The list of acceptable units for that KPI.
- Passages of text from a corporate sustainability report.

Your job: return the KPI's MOST RECENT year value by calling the `record_kpi` tool.

Rules:
- If the document shows multiple year columns (e.g. "2021 2022 2023 2024 2025"), pick the LATEST year's value.
- Pick the CONSOLIDATED / TOTAL group-level value — not a subsidiary, business segment, or sub-component breakdown.
- Pick the ABSOLUTE total — not an intensity ratio (e.g. tCO2e/$, MWh/€, l/kWh) or a percentage.
- For Scope 1 emissions: pick "Total gross Scope 1 GHG emissions" / "Scope 1 (direct) greenhouse gas emissions". If the report distinguishes "consolidated" from "operational control + non-consolidated entities" (ESRS-aligned reporting, common in Shell/Eni), pick the LARGER ESRS-aligned figure that includes operated non-consolidated entities, not the consolidated-only sub-total. Never pick Scope 2, Scope 3, methane-only, intensity, or net (Scope 1+2 combined).
- For total energy consumption: pick the company-wide total energy CONSUMED — not energy produced from renewables, fuel-only, or electricity-only sub-totals. If the report distinguishes "operational control" from "ESRS-aligned + non-consolidated", pick the LARGER ESRS-aligned figure.
- For water consumption: ONLY pick a value labelled "water CONSUMPTION" or "freshwater CONSUMPTION" or "net water consumption". REJECT every value labelled "withdrawal", "withdrawn", "discharge", "discharged", "recycled", "reclaimed", "reused", "produced water", or "wastewater". This rule is STRICT — when in doubt between consumption and withdrawal, return null.
- Multiply out magnitude prefixes yourself: if the document says "82.0 million m³" return value=82000000, unit=m3. If it says "32,141 thousand m³" return value=32141000, unit=m3. If it says "168.59 TWh" you may either return value=168.59, unit=TWh OR multiply out — but NEVER write a unit that contains a magnitude word (e.g. never "million m3", "thousand m3", "Mm3", "million cubic metres"). The unit MUST be a single base unit from the listed acceptable units.
- If the KPI is not reported clearly in this passage, call the tool with value=null, unit=null.
- Never guess or infer from breakdowns if there is no consolidated total — use value=null.
- `reporting_year` is the year the value refers to (the most recent column).
- `source_snippet` is the exact quoted text supporting the value (max 200 chars)."""


class LLMExtractor(Extractor):
    name = "llm"

    def __init__(
        self,
        model: str | None = None,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        provider: str | None = None,
        min_call_interval_s: float | None = None,
    ):
        if provider is None:
            provider = os.environ.get("LLM_PROVIDER", "anthropic")
        if provider not in _VALID_PROVIDERS:
            raise ValueError(
                f"Unknown provider {provider!r}; expected one of {_VALID_PROVIDERS}"
            )
        if model is None:
            if provider == "gemini":
                model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
            else:
                model = os.environ.get("ANTHROPIC_MODEL", ANTHROPIC_MODEL)
        if min_call_interval_s is None:
            # Gemini free tier is 10 RPM — sleep 6.5s between calls to stay under.
            # Anthropic has its own rate-limit handling and a higher cap; no throttle.
            min_call_interval_s = 6.5 if provider == "gemini" else 0.0
        self.provider = provider
        self.model = model
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.min_call_interval_s = min_call_interval_s
        self._last_call_time = 0.0
        self._client: Anthropic | None = None

    @property
    def client(self) -> Anthropic:
        if self._client is None:
            self._client = Anthropic()
        return self._client

    def _cache_key(self, kpi_key: str, user_prompt: str, system_prompt: str) -> str:
        """SHA256 of (model, kpi, system_prompt, user_prompt).

        Including the system prompt ensures that prompt-rule changes
        (e.g. new disambiguation rules) invalidate cached responses
        rather than silently serving stale extractions.
        """
        return hashlib.sha256(
            f"{self.model}|{kpi_key}|{system_prompt}|{user_prompt}".encode()
        ).hexdigest()

    def _build_context(
        self,
        report: Any,
        kpi_query: str,
        kpi_unit_family: list[str] | None = None,
        kpi_queries: list[str] | None = None,
    ) -> str:
        """Return full page text + tables for top-K pages, ranked by hybrid
        BM25 + cosine retrieval across multiple KPI query phrasings.

        Falls back to single-query cosine retrieval when no kpi_queries are
        provided (e.g. older callers that only pass kpi_query).
        """
        if kpi_queries:
            ranked = rank_pages_hybrid(report, list(kpi_queries))
        else:
            query_emb = embed_texts([kpi_query])[0]
            ranked = rank_pages_cosine(report, query_emb, unit_tokens=kpi_unit_family)
        top_pages = ranked[:12]
        top_page_nums = {pn for pn, _ in top_pages}

        pages_by_num = {p["page_num"]: p for p in report["pages"]}
        parts: list[str] = []
        for pn, _ in top_pages:
            page = pages_by_num.get(pn)
            if page is None:
                continue
            parts.append(f"=== Page {pn} ===")
            parts.append(normalize_co2(page["text"])[:4000])
            parts.append("")

        # Append any tables on the selected pages (cleaner than text for some).
        for t in report["tables"]:
            if t["page_num"] not in top_page_nums:
                continue
            parts.append(f"[Table @ page {t['page_num']}]")
            parts.append(" | ".join(t["headers"]))
            for row in t["rows"]:
                parts.append(" | ".join(row))
            parts.append("")

        return "\n".join(parts)

    def extract(self, report: Any, kpi_key: str) -> KPIExtraction:
        kpi = KPIS[kpi_key]
        flags: list[str] = []

        context = self._build_context(
            report,
            kpi["query"],
            kpi_unit_family=kpi["unit_family"],
            kpi_queries=kpi.get("queries"),
        )

        user_prompt = (
            f"KPI to extract: {kpi['query']}\n"
            f"Acceptable units: {', '.join(kpi['unit_family'])}\n\n"
            f"Document excerpts:\n{context}"
        )

        cache_key = self._cache_key(kpi_key, user_prompt, _SYSTEM_PROMPT)
        cache_path = CACHE_DIR / "llm" / f"{cache_key}.json"
        tool_input: dict | None = None
        if cache_path.exists():
            try:
                with cache_path.open(encoding="utf-8") as f:
                    tool_input = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                log.warning("corrupt llm cache at %s (%s); refetching", cache_path, e)
        if tool_input is None:
            tool_input = self._call_with_retry(user_prompt, kpi_key)
            if tool_input is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with cache_path.open("w", encoding="utf-8") as f:
                    json.dump(tool_input, f)

        if tool_input is None:
            return self._not_reported(report, kpi_key, flags=["api_error"])

        value = tool_input.get("value")
        unit = tool_input.get("unit")
        reporting_year = tool_input.get("reporting_year")
        snippet = tool_input.get("source_snippet")
        confidence = tool_input.get("confidence")

        if value is None:
            return self._not_reported(report, kpi_key, flags=flags)

        if unit is None:
            return self._not_reported(
                report, kpi_key, flags=[*flags, "llm_missing_unit"]
            )

        try:
            canonicalize_unit(unit)
            canonical_value = to_canonical_value(value, unit, kpi["canonical_unit"])
        except ValueError:
            return self._not_reported(
                report, kpi_key, flags=[*flags, "unit_unknown"]
            )

        lo, hi = kpi["plausible_range"]
        if not (lo <= canonical_value <= hi):
            return self._not_reported(
                report, kpi_key, flags=[*flags, "out_of_range"]
            )

        return KPIExtraction(
            company=report["company"],
            report_year=report["report_year"],
            kpi=kpi_key,
            value=canonical_value,
            unit=kpi["canonical_unit"],
            reporting_year=report["report_year"],
            source_snippet=snippet,
            source_page=None,
            confidence=confidence,
            extractor=self.name,
            flags=flags,
        )

    def _call_with_retry(
        self, user_prompt: str, kpi_key: str, system_prompt: str = _SYSTEM_PROMPT,
    ) -> dict | None:
        for attempt in range(self.max_retries):
            self._throttle_if_needed()
            try:
                if self.provider == "anthropic":
                    result = self._call_anthropic_once(user_prompt, kpi_key, system_prompt)
                elif self.provider == "gemini":
                    result = self._call_gemini_once(user_prompt, kpi_key, system_prompt)
                else:
                    raise ValueError(f"Unsupported provider {self.provider!r}")
                self._last_call_time = time.time()
                return result
            except Exception as e:
                self._last_call_time = time.time()
                log.warning("LLM call failed (attempt %d/%d): %s",
                            attempt + 1, self.max_retries, e)
                # On 429 rate-limit errors, the per-minute window is the only
                # recovery — back off long enough to clear it.
                if attempt < self.max_retries - 1:
                    err_str = str(e)
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        time.sleep(60.0)
                    else:
                        time.sleep(self.retry_base_delay * (2 ** attempt))
        return None

    def _throttle_if_needed(self) -> None:
        """Sleep so that successive calls are ≥ min_call_interval_s apart."""
        if self.min_call_interval_s <= 0 or self._last_call_time == 0.0:
            return
        elapsed = time.time() - self._last_call_time
        wait = self.min_call_interval_s - elapsed
        if wait > 0:
            time.sleep(wait)

    def _call_anthropic_once(
        self, user_prompt: str, kpi_key: str, system_prompt: str
    ) -> dict | None:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            temperature=0,
            system=[{
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }],
            tools=[_TOOL_SCHEMA],
            tool_choice={"type": "tool", "name": "record_kpi"},
            messages=[{"role": "user", "content": user_prompt}],
        )
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use" and block.name == "record_kpi":
                return dict(block.input)
        log.warning("LLM response had no tool_use block for %s", kpi_key)
        return None

    def _call_gemini_once(
        self, user_prompt: str, kpi_key: str, system_prompt: str
    ) -> dict | None:
        client = _gemini_client()
        config = {
            "system_instruction": system_prompt,
            "temperature": 0,
            "max_output_tokens": 500,
            "tools": _GEMINI_TOOLS,
            "tool_config": _GEMINI_TOOL_CONFIG,
        }
        resp = client.models.generate_content(
            model=self.model, contents=user_prompt, config=config,
        )
        for cand in (resp.candidates or []):
            content = getattr(cand, "content", None)
            for part in getattr(content, "parts", []) or []:
                fc = getattr(part, "function_call", None)
                if fc and getattr(fc, "name", None) == "record_kpi":
                    args = getattr(fc, "args", None) or {}
                    return dict(args)
        log.warning("Gemini response had no function_call for %s", kpi_key)
        return None

    def _not_reported(self, report: Any, kpi_key: str, flags: list[str]) -> KPIExtraction:
        return KPIExtraction(
            company=report["company"],
            report_year=report["report_year"],
            kpi=kpi_key,
            value=None, unit=None, reporting_year=None,
            source_snippet=None, source_page=None, confidence=None,
            extractor=self.name, flags=flags,
        )
