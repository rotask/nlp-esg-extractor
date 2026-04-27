"""PDF ingest dispatcher.

`parse_pdf` tries Docling first via `ingest_docling.parse_with_docling`
and falls back to the in-file `_parse_with_pdfplumber` on `None` /
empty / quality-check failure. The `NLP_ESG_DISABLE_DOCLING=1` env
var short-circuits Docling entirely. The on-disk pickle cache name
includes the parser tag (`{company}_{year}_{parser}.pkl`) so v1
(pdfplumber) and v2 (Docling) caches coexist and do not collide.
"""
from __future__ import annotations
import logging
import pickle
import re
from pathlib import Path
from typing import TypedDict

import pdfplumber

from nlp_esg.config import CACHE_DIR

log = logging.getLogger(__name__)

_FILENAME_RE = re.compile(r"^(?P<company>[a-z0-9_\-]+)_(?P<year>\d{4})\.pdf$", re.IGNORECASE)


class Page(TypedDict):
    page_num: int
    text: str


class TableEntry(TypedDict):
    page_num: int
    headers: list[str]
    rows: list[list[str]]


class ParsedReport(TypedDict):
    company: str
    report_year: int
    parser: str
    pages: list[Page]
    tables: list[TableEntry]


def _parse_filename(path: Path) -> tuple[str, int]:
    m = _FILENAME_RE.match(path.name)
    if not m:
        raise ValueError(
            f"Report filename {path.name!r} does not match {{company}}_{{year}}.pdf"
        )
    return m.group("company").lower(), int(m.group("year"))


# Imported here (after ParsedReport / _parse_filename are defined) to avoid
# a partial-module circular-import cycle: ingest_docling imports from ingest.
from nlp_esg.ingest_docling import parse_with_docling  # noqa: E402


def _parse_with_pdfplumber(path: Path) -> ParsedReport:
    """Parse a PDF with pdfplumber. No caching — the caller handles cache."""
    company, year = _parse_filename(path)
    pages: list[Page] = []
    tables: list[TableEntry] = []

    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append({"page_num": i, "text": text})

            for raw in page.extract_tables() or []:
                if not raw:
                    continue
                headers = [(c or "").strip() for c in raw[0]]
                rows = [[(c or "").strip() for c in row] for row in raw[1:]]
                tables.append({"page_num": i, "headers": headers, "rows": rows})

    return {
        "company": company,
        "report_year": year,
        "parser": "pdfplumber",
        "pages": pages,
        "tables": tables,
    }


def parse_pdf(path: Path, use_cache: bool = True) -> ParsedReport:
    """Parse a PDF into pages + tables. Tries Docling first, falls back to pdfplumber.

    Cache filename includes the parser tag so v1 (pdfplumber-only) and v2
    (Docling-first) caches coexist on disk.
    """
    company, year = _parse_filename(path)
    docling_cache = CACHE_DIR / f"{company}_{year}_docling.pkl"
    pdfplumber_cache = CACHE_DIR / f"{company}_{year}_pdfplumber.pkl"

    if use_cache:
        for cache_path in (docling_cache, pdfplumber_cache):
            if cache_path.exists() and cache_path.stat().st_mtime >= path.stat().st_mtime:
                with cache_path.open("rb") as f:
                    return pickle.load(f)

    report = parse_with_docling(path)
    used_parser = "docling"
    if report is None or not report["pages"]:
        log.warning("docling failed for %s; falling back to pdfplumber", path.name)
        report = _parse_with_pdfplumber(path)
        used_parser = "pdfplumber"

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = CACHE_DIR / f"{company}_{year}_{used_parser}.pkl"
        with cache_path.open("wb") as f:
            pickle.dump(report, f)

    return report
