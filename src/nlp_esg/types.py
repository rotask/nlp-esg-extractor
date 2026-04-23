from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class KPIExtraction:
    company: str
    report_year: int
    kpi: str
    value: float | None
    unit: str | None
    reporting_year: int | None
    source_snippet: str | None
    source_page: int | None
    confidence: float | None
    extractor: str
    flags: list[str] = field(default_factory=list)
