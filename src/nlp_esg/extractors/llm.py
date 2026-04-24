from __future__ import annotations
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

from anthropic import Anthropic

from nlp_esg.config import ANTHROPIC_MODEL, CACHE_DIR, KPIS
from nlp_esg.extractors.base import Extractor
from nlp_esg.normalize import canonicalize_unit, to_canonical_value
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

_SYSTEM_PROMPT = """You are an information-extraction assistant for ESG sustainability reports.

You will be given:
- A KPI to extract.
- The list of acceptable units for that KPI.
- A passage of text from a corporate sustainability report.

Your job: return the KPI's current-year value by calling the `record_kpi` tool.

Rules:
- Return the value in whatever unit the document uses — do not convert.
- The unit MUST be one of the acceptable units listed.
- If the KPI is not reported in this passage, call the tool with value=null, unit=null.
- Never guess or infer from subsidiary breakdowns if there's no consolidated total — use value=null.
- `reporting_year` is the year the value refers to, not the publication year.
- `source_snippet` is the exact quoted text supporting the value."""


class LLMExtractor(Extractor):
    name = "llm"

    def __init__(
        self,
        model: str = ANTHROPIC_MODEL,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
    ):
        self.model = model
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self._client: Anthropic | None = None

    @property
    def client(self) -> Anthropic:
        if self._client is None:
            self._client = Anthropic()
        return self._client

    def _build_context(self, report: Any, kpi_query: str) -> str:
        """Concatenate top table pages + top-k sentences as the LLM context."""
        parts: list[str] = []
        for t in report["tables"]:
            parts.append(f"[Table @ page {t['page_num']}]")
            parts.append(" | ".join(t["headers"]))
            for row in t["rows"]:
                parts.append(" | ".join(row))
            parts.append("")

        for s in report["sentences"]:
            parts.append(f"[Page {s['page_num']}] {s['text']}")

        return "\n".join(parts)

    def extract(self, report: Any, kpi_key: str) -> KPIExtraction:
        kpi = KPIS[kpi_key]
        flags: list[str] = []

        context = self._build_context(report, kpi["query"])

        user_prompt = (
            f"KPI to extract: {kpi['query']}\n"
            f"Acceptable units: {', '.join(kpi['unit_family'])}\n\n"
            f"Document excerpts:\n{context}"
        )

        cache_key = hashlib.sha256(
            f"{self.model}|{kpi_key}|{user_prompt}".encode()
        ).hexdigest()
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
            reporting_year=reporting_year,
            source_snippet=snippet,
            source_page=None,
            confidence=confidence,
            extractor=self.name,
            flags=flags,
        )

    def _call_with_retry(self, user_prompt: str, kpi_key: str) -> dict | None:
        for attempt in range(self.max_retries):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=500,
                    temperature=0,
                    system=[{
                        "type": "text",
                        "text": _SYSTEM_PROMPT,
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
            except Exception as e:
                log.warning("LLM call failed (attempt %d/%d): %s",
                            attempt + 1, self.max_retries, e)
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_base_delay * (2 ** attempt))
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
