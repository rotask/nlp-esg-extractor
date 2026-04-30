"""Docling-backed PDF parser, page-range batched.

Returns a `ParsedReport` (text + tables) or `None` so the caller can
fall back to pdfplumber. Page-range batching keeps peak memory bounded
by processing `_PAGE_BATCH_SIZE` pages at a time and forcing
`gc.collect()` between batches; this is what lets long ESG reports
(>500 pages) get through on a 16 GB laptop where the prior single-shot
path SIGSEGV'd in the C++ layout model. Two safety nets remain: a
file-size guard for pathological inputs, and a post-parse
"majority-empty pages" check that bails when fewer than half the pages
have substantive text. The `NLP_ESG_DISABLE_DOCLING=1` env var
short-circuits Docling entirely.
"""
from __future__ import annotations
import gc
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
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.accelerator_options import (
        AcceleratorDevice,
        AcceleratorOptions,
    )
except ImportError:  # pragma: no cover - import-time fallback
    DocumentConverter = None  # type: ignore[assignment]
    PdfFormatOption = None  # type: ignore[assignment]
    InputFormat = None  # type: ignore[assignment]
    PdfPipelineOptions = None  # type: ignore[assignment]
    AcceleratorDevice = None  # type: ignore[assignment]
    AcceleratorOptions = None  # type: ignore[assignment]


_DOCLING_MAX_FILE_BYTES = 100 * 1024 * 1024  # 100 MB

def _default_batch_size() -> int:
    """20 on CPU, 10 on a small (≤8 GB) GPU.

    The RTX 4050 / RTX 4060 8GB class throws std::bad_alloc on image-heavy
    pages when running larger batches; halving the batch size stops the
    bad_alloc without giving back too much GPU throughput.
    """
    try:
        import torch
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            return 10 if vram_gb < 9 else 20
    except Exception:
        pass
    return 20


# Page batch size for the layout/structure pipeline. Defaults adapt to
# device and VRAM; override via NLP_ESG_DOCLING_BATCH_SIZE for tuning.
_PAGE_BATCH_SIZE = int(
    os.environ.get("NLP_ESG_DOCLING_BATCH_SIZE", str(_default_batch_size()))
)


def _free_cuda_cache() -> None:
    """Release any cached CUDA blocks back to the driver. No-op on CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _count_pdf_pages(path: Path) -> int:
    """Count pages cheaply without invoking the Docling layout model.

    Returns 0 on any failure; the caller treats that as "couldn't probe"
    and falls through to a single-shot conversion path that doesn't need
    a page count up-front.
    """
    try:
        import pypdfium2 as pdfium
    except ImportError:
        return 0
    try:
        with pdfium.PdfDocument(str(path)) as pdf:
            return len(pdf)
    except Exception as e:
        log.debug("could not count pages in %s: %s", path.name, e)
        return 0


def _rss_mb() -> float:
    """Best-effort current-process RSS in MB. Returns 0.0 if psutil missing."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def _resolve_accelerator_device() -> "AcceleratorDevice":
    """Pick CUDA when available, else CPU. Honour NLP_ESG_DOCLING_DEVICE override."""
    override = os.environ.get("NLP_ESG_DOCLING_DEVICE", "").strip().lower()
    if override == "cpu":
        return AcceleratorDevice.CPU
    if override == "cuda":
        return AcceleratorDevice.CUDA
    try:
        import torch
        if torch.cuda.is_available():
            return AcceleratorDevice.CUDA
    except Exception:
        pass
    return AcceleratorDevice.CPU


def _make_pipeline_options() -> "PdfPipelineOptions":
    """Lighter pipeline: no OCR, keep table structure (we need it for KPIs).

    Explicitly sets the accelerator device — Docling's `auto` is not always
    reliable when the torch install briefly mismatches the driver. Override
    with `NLP_ESG_DOCLING_DEVICE=cpu|cuda` if needed.
    """
    opts = PdfPipelineOptions()
    opts.do_ocr = False
    opts.do_table_structure = True
    device = _resolve_accelerator_device()
    opts.accelerator_options = AcceleratorOptions(device=device)
    return opts


def _make_converter() -> "DocumentConverter":
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=_make_pipeline_options()),
        }
    )


def parse_with_docling(path: Path) -> "ParsedReport | None":
    """Parse a PDF with Docling in page-range batches.

    Falls back to a single-shot conversion when the page count can't be
    determined (e.g. a malformed input PDF). Returns None on any
    unrecoverable failure so the caller (`ingest.parse_pdf`) can fall back
    to pdfplumber for the whole report.
    """
    from nlp_esg.ingest import _parse_filename

    if _docling_disabled():
        log.info("docling disabled via NLP_ESG_DISABLE_DOCLING; falling back")
        return None
    if DocumentConverter is None:
        log.warning("docling not installed; caller should fall back")
        return None

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

    n_pages_total = _count_pdf_pages(path)
    if n_pages_total <= 0:
        return _parse_single_shot(path, company, year)

    log.info(
        "docling batch parse: %s (%d pages, batch=%d, RSS=%.0f MB)",
        path.name, n_pages_total, _PAGE_BATCH_SIZE, _rss_mb(),
    )

    pages: list[dict] = []
    tables: list[dict] = []
    try:
        converter = _make_converter()
    except Exception as e:
        log.warning("docling converter init failed for %s: %s", path.name, e)
        return None

    for batch_start in range(1, n_pages_total + 1, _PAGE_BATCH_SIZE):
        batch_end = min(batch_start + _PAGE_BATCH_SIZE - 1, n_pages_total)
        try:
            result = converter.convert(str(path), page_range=(batch_start, batch_end))
            doc = result.document
        except Exception as e:
            log.warning(
                "docling batch %d-%d failed for %s: %s",
                batch_start, batch_end, path.name, e,
            )
            return None

        _collect_pages_and_tables(doc, batch_start, batch_end, pages, tables)
        del result, doc
        gc.collect()
        _free_cuda_cache()
        log.info(
            "  batch %d-%d done; total pages=%d tables=%d RSS=%.0f MB",
            batch_start, batch_end, len(pages), len(tables), _rss_mb(),
        )

    if _is_majority_empty(pages):
        log.warning(
            "docling produced too many empty pages for %s — falling back",
            path.name,
        )
        return None

    return {
        "company": company,
        "report_year": year,
        "parser": "docling",
        "pages": pages,
        "tables": tables,
    }


def _parse_single_shot(
    path: Path, company: str, year: int
) -> "ParsedReport | None":
    """Single-shot conversion (legacy path) when page-count probing fails."""
    try:
        converter = _make_converter()
        result = converter.convert(str(path))
        doc = result.document
    except Exception as e:
        log.warning("docling failed to convert %s: %s", path.name, e)
        return None

    n_pages = _doc_num_pages(doc)
    if n_pages == 0:
        log.warning("docling returned 0 pages for %s", path.name)
        return None

    pages: list[dict] = []
    tables: list[dict] = []
    _collect_pages_and_tables(doc, 1, n_pages, pages, tables)

    if _is_majority_empty(pages):
        log.warning(
            "docling produced too many empty pages for %s — falling back",
            path.name,
        )
        return None

    return {
        "company": company,
        "report_year": year,
        "parser": "docling",
        "pages": pages,
        "tables": tables,
    }


def _collect_pages_and_tables(
    doc: Any,
    expected_lo: int,
    expected_hi: int,
    pages: list[dict],
    tables: list[dict],
) -> None:
    """Append batch pages and any tables on those pages to the running lists.

    Handles two page-numbering conventions Docling can emit when
    `page_range=(lo, hi)` is set: original-PDF numbering preserved, or
    1..N within the batch. Falls back to a positional remap when the
    keys come back renumbered.
    """
    try:
        present_keys = sorted(int(k) for k in (getattr(doc, "pages", None) or {}).keys())
    except Exception:
        present_keys = []

    if not present_keys:
        n = _doc_num_pages(doc)
        present_keys = list(range(1, max(0, n) + 1))

    preserves_numbering = bool(present_keys) and min(present_keys) >= expected_lo

    for i, doc_page_no in enumerate(present_keys):
        try:
            text = doc.export_to_markdown(page_no=doc_page_no)
        except Exception:
            text = ""
        original_pn = doc_page_no if preserves_numbering else expected_lo + i
        pages.append({"page_num": original_pn, "text": text or ""})

    for item, _level in _safe_iterate(doc):
        if getattr(item, "label", None) != "table":
            continue
        raw_pn = _table_page(item)
        if raw_pn is None:
            continue
        if preserves_numbering:
            page_no = raw_pn
        elif 1 <= raw_pn <= len(present_keys):
            page_no = expected_lo + (raw_pn - 1)
        else:
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


def _is_majority_empty(pages: list[dict]) -> bool:
    n = len(pages)
    if n == 0:
        return True
    substantive = sum(1 for p in pages if len((p["text"] or "").strip()) >= 100)
    return substantive / n < 0.5


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
