"""Shared dataclasses and TypedDicts.

`KPIExtraction` is the canonical row produced by both extractors and
serialised into `extractions.csv`. The TypedDicts for `ParsedReport`,
`IndexedReport`, `Page`, `TableEntry`, `Sentence`, and
`TableHeaderEmb` live in `ingest.py` / `retrieval.py` next to the
code that produces them.
"""
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
    run_tag: str | None = None
