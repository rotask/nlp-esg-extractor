from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from nlp_esg.ingest import ParsedReport

log = logging.getLogger(__name__)


def _docling_disabled() -> bool:
    """Honour the NLP_ESG_DISABLE_DOCLING env var.

    On some machines Docling's layout model segfaults (std::bad_alloc /
    SIGSEGV) on long ESG PDFs, which kills the host Python process and
    bypasses our Python-level fallback. The env var lets the user skip
    Docling entirely without touching code.
    """
    val = os.environ.get("NLP_ESG_DISABLE_DOCLING", "").strip().lower()
    return val in {"1", "true", "yes", "on"}

try:
    from docling.document_converter import DocumentConverter  # noqa: F401
except ImportError:  # pragma: no cover - import-time fallback
    DocumentConverter = None  # type: ignore[assignment]


_DOCLING_MAX_FILE_BYTES = 15 * 1024 * 1024  # 15 MB


def parse_with_docling(path: Path) -> "ParsedReport | None":
    """Parse a PDF with Docling. Return None on any failure so the caller can fall back."""
    # Imported lazily to avoid a circular import with nlp_esg.ingest.
    from nlp_esg.ingest import _parse_filename

    if _docling_disabled():
        log.info("docling disabled via NLP_ESG_DISABLE_DOCLING; falling back")
        return None

    if DocumentConverter is None:
        log.warning("docling not installed; caller should fall back")
        return None

    # Docling's layout model has unbounded memory growth on long PDFs and
    # OOMs (std::bad_alloc) past ~28 pages on bigger reports. Skip up-front
    # for oversized inputs so the dispatcher falls back without burning
    # 5-10 minutes per file.
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    if size > _DOCLING_MAX_FILE_BYTES:
        log.warning(
            "skipping docling for %s (%.1f MB > %.1f MB threshold)",
            path.name, size / 1e6, _DOCLING_MAX_FILE_BYTES / 1e6,
        )
        return None

    try:
        company, year = _parse_filename(path)
    except ValueError as e:
        log.warning("filename parse failed for %s: %s", path.name, e)
        return None

    try:
        converter = DocumentConverter()
        result = converter.convert(str(path))
        doc = result.document
    except Exception as e:
        log.warning("docling failed to convert %s: %s", path.name, e)
        return None

    n_pages = _doc_num_pages(doc)
    if n_pages == 0:
        log.warning("docling returned 0 pages for %s", path.name)
        return None

    pages = []
    for page_no in range(1, n_pages + 1):
        try:
            text = doc.export_to_markdown(page_no=page_no)
        except Exception:
            text = ""
        pages.append({"page_num": page_no, "text": text or ""})

    # Quality check: Docling can OOM mid-document on long PDFs and silently
    # emit empty pages thereafter. If too many pages are empty/junk, return
    # None so the caller falls back to pdfplumber for the whole report.
    substantive = sum(1 for p in pages if len((p["text"] or "").strip()) >= 100)
    if n_pages > 0 and substantive / n_pages < 0.5:
        log.warning(
            "docling produced %d/%d substantive pages for %s — falling back",
            substantive, n_pages, path.name,
        )
        return None

    tables = []
    for item, _level in _safe_iterate(doc):
        if getattr(item, "label", None) != "table":
            continue
        page_no = _table_page(item)
        if page_no is None:
            continue
        try:
            df = item.export_to_dataframe()
            headers = [str(c) for c in list(df.columns)]
            rows = [
                [("" if v is None else str(v)) for v in row]
                for row in df.values.tolist()
            ]
        except Exception:
            continue
        tables.append({"page_num": page_no, "headers": headers, "rows": rows})

    return {
        "company": company,
        "report_year": year,
        "parser": "docling",
        "pages": pages,
        "tables": tables,
    }


def _doc_num_pages(doc: Any) -> int:
    try:
        np = doc.num_pages
        return np() if callable(np) else int(np)
    except Exception:
        pass
    try:
        return len(doc.pages)
    except Exception:
        return 0


def _safe_iterate(doc: Any):
    try:
        yield from doc.iterate_items()
    except Exception:
        return


def _table_page(table_item: Any) -> int | None:
    prov = getattr(table_item, "prov", None) or []
    for p in prov:
        page_no = getattr(p, "page_no", None)
        if isinstance(page_no, int):
            return page_no
    return None
