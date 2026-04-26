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


def parse_pdf(path: Path, use_cache: bool = True) -> ParsedReport:
    """Parse a PDF into pages + tables. Caches to data/cache based on mtime."""
    company, year = _parse_filename(path)
    cache_path = CACHE_DIR / f"{company}_{year}.pkl"

    if use_cache and cache_path.exists() and cache_path.stat().st_mtime >= path.stat().st_mtime:
        with cache_path.open("rb") as f:
            return pickle.load(f)

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

    report: ParsedReport = {
        "company": company,
        "report_year": year,
        "parser": "pdfplumber",
        "pages": pages,
        "tables": tables,
    }

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as f:
            pickle.dump(report, f)

    return report
