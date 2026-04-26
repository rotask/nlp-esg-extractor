# NLP ESG Iteration 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the iteration-2 design from `docs/superpowers/specs/2026-04-26-nlp-esg-iteration2-design.md` — Docling ingest with pdfplumber fallback, multi-query reciprocal-rank-fusion retrieval, BM25+cosine hybrid ranking, lone-subscript-2 normalisation, and LLM cache-key fix that makes the existing system-prompt-v2 take effect.

**Architecture:** Additive iteration over the existing pipeline. New `ingest_docling.py` parser; `ingest.py` becomes a dispatcher. Page-ranking logic lifts from `extractors/llm.py` into `retrieval.py` so a new RRF/BM25 hybrid can share it. Baseline gets a row-label-aware embedding tweak. LLM cache key gains the system prompt as a component. New CLI flag `--run-tag` persists extractions per run for the multi-run comparison.

**Tech Stack:** Python 3.11, pdfplumber (existing), Docling (new), sentence-transformers + ClimateBERT (existing), rank-bm25 (new), Anthropic SDK (existing), pandas, pytest.

---

## Pre-flight

The repo currently has uncommitted changes in `src/nlp_esg/extractors/baseline.py`, `extractors/llm.py`, `normalize.py`, `pipeline.py`, `retrieval.py` (per `git status`). Those changes contain the in-flight system-prompt-v2 work and the partially-extracted page-ranking logic. The first task captures them in a coherent commit so the rest of the plan starts from a clean working tree.

Engineers running this plan in a fresh worktree: skip Task 0 if your worktree was created from a clean `master`. Otherwise execute it.

---

## Task 0: Snapshot existing uncommitted work

**Files:**
- All 5 files listed in pre-flight section.

- [ ] **Step 1: Confirm the uncommitted state matches expectations**

Run:
```bash
git status
git diff --stat
```
Expected: 5 modified files (`baseline.py`, `llm.py`, `normalize.py`, `pipeline.py`, `retrieval.py`).

- [ ] **Step 2: Run the existing test suite to confirm green-on-disk**

Run: `pytest -q --ignore=tests/test_integration_llm.py --ignore=tests/test_integration_real_pdf.py`
Expected: all tests pass.

- [ ] **Step 3: Commit the existing work as a single descriptive snapshot**

```bash
git add src/nlp_esg/extractors/baseline.py src/nlp_esg/extractors/llm.py src/nlp_esg/normalize.py src/nlp_esg/pipeline.py src/nlp_esg/retrieval.py
git commit -m "feat: page-level retrieval + system prompt v2 (in-flight)

Captures iteration-2 work prepared during the prior session:
- LLM extractor _build_context switched to page-level retrieval with
  unit-presence boost (FINDINGS.md §2.1)
- System prompt expanded to v2 with ESRS aggregation,
  magnitude-prefix, and water-consumption-vs-withdrawal rules
- Baseline extractor: row-0 promotion when headers lack a year
- normalize_co2: CO2 inline + suffix-after + next-line patterns

This commit is the starting point for the iteration-2 plan.
The remaining changes (Docling, BM25, multi-query RRF, cache-key
fix) follow in subsequent commits."
```

- [ ] **Step 4: Verify clean tree**

Run: `git status`
Expected: `nothing to commit, working tree clean` (untracked `.claude/`, `data/reports/`, `docs/FINDINGS.md` are fine — those aren't modifications).

---

## Task 1: Add `parser` field to `ParsedReport` and `run_tag` to `KPIExtraction`

**Why:** Both are referenced by every later task (Docling cache filename, multi-run comparison). Add them first so signatures are stable for the rest of the plan.

**Files:**
- Modify: `src/nlp_esg/ingest.py` (TypedDict)
- Modify: `src/nlp_esg/types.py` (dataclass)
- Modify: `tests/test_ingest.py` (existing assertions)
- Test: `tests/test_types_run_tag.py` (new)

- [ ] **Step 1: Write the failing test for `KPIExtraction.run_tag`**

Create `tests/test_types_run_tag.py`:
```python
from nlp_esg.types import KPIExtraction


def test_run_tag_defaults_to_none():
    e = KPIExtraction(
        company="bp", report_year=2024, kpi="scope_1_emissions",
        value=None, unit=None, reporting_year=None,
        source_snippet=None, source_page=None, confidence=None,
        extractor="baseline",
    )
    assert e.run_tag is None


def test_run_tag_can_be_set():
    e = KPIExtraction(
        company="bp", report_year=2024, kpi="scope_1_emissions",
        value=33.7e6, unit="tCO2e", reporting_year=2024,
        source_snippet="snip", source_page=12, confidence=0.9,
        extractor="llm", run_tag="v2_docling",
    )
    assert e.run_tag == "v2_docling"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_types_run_tag.py -v`
Expected: FAIL — `KPIExtraction` has no `run_tag` field.

- [ ] **Step 3: Add `run_tag` to `KPIExtraction`**

Edit `src/nlp_esg/types.py` — add `run_tag: str | None = None` after `flags`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_types_run_tag.py -v`
Expected: PASS.

- [ ] **Step 5: Add `parser` to `ParsedReport` TypedDict**

Edit `src/nlp_esg/ingest.py` — change `ParsedReport`:
```python
class ParsedReport(TypedDict):
    company: str
    report_year: int
    parser: str
    pages: list[Page]
    tables: list[TableEntry]
```

In the existing `parse_pdf`, set `"parser": "pdfplumber"` in the `report: ParsedReport = {...}` dict literal (alongside the existing keys).

- [ ] **Step 6: Confirm existing tests still pass**

Run: `pytest -q --ignore=tests/test_integration_llm.py --ignore=tests/test_integration_real_pdf.py`
Expected: all pass. If any test constructs a `ParsedReport` literal manually (likely `tests/test_ingest.py`, `tests/test_baseline.py`), update the literal to include `"parser": "pdfplumber"`.

- [ ] **Step 7: Commit**

```bash
git add src/nlp_esg/types.py src/nlp_esg/ingest.py tests/test_types_run_tag.py tests/test_ingest.py tests/test_baseline.py
git commit -m "feat(types): add ParsedReport.parser and KPIExtraction.run_tag"
```
(Drop any test files from the `git add` above that you didn't actually modify.)

---

## Task 2: Add `queries: list[str]` to KPI registry

**Why:** Multi-query retrieval (Task 8) needs the additional query phrasings. The single-string `query` field stays so the existing LLM extractor and baseline keep working.

**Files:**
- Modify: `src/nlp_esg/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_config.py`:
```python
from nlp_esg.config import KPIS, KPI_KEYS


def test_every_kpi_has_queries_list():
    for kpi_key in KPI_KEYS:
        kpi = KPIS[kpi_key]
        assert "queries" in kpi, f"{kpi_key} missing 'queries'"
        assert isinstance(kpi["queries"], list)
        assert len(kpi["queries"]) >= 2, f"{kpi_key} needs at least 2 queries"
        assert all(isinstance(q, str) and q.strip() for q in kpi["queries"])


def test_single_query_field_still_present():
    for kpi_key in KPI_KEYS:
        assert "query" in KPIS[kpi_key]
```

- [ ] **Step 2: Run test to verify the first test fails**

Run: `pytest tests/test_config.py -v`
Expected: `test_every_kpi_has_queries_list` FAILS, `test_single_query_field_still_present` PASSES.

- [ ] **Step 3: Add the `queries` field to each KPI**

Edit `src/nlp_esg/config.py`:
```python
KPIS: dict[str, dict] = {
    "scope_1_emissions": {
        "query": "Scope 1 direct greenhouse gas emissions",
        "queries": [
            "Total gross Scope 1 GHG emissions",
            "Scope 1 direct greenhouse gas emissions tCO2e",
            "Scope 1 (direct) emissions",
        ],
        "unit_family": ["tCO2e", "ktCO2e", "MtCO2e", "t CO2-eq", "t CO2e", "tonnes CO2e"],
        "canonical_unit": "tCO2e",
        "plausible_range": (1e2, 1e9),
    },
    "total_energy_consumption": {
        "query": "Total energy consumption",
        "queries": [
            "Total energy consumption MWh",
            "Energy consumption GWh",
            "Total energy consumed across operations",
        ],
        "unit_family": ["MWh", "GWh", "TWh", "GJ", "TJ", "PJ", "kWh"],
        "canonical_unit": "MWh",
        "plausible_range": (1e2, 1e9),
    },
    "water_consumption": {
        "query": "Total water consumption withdrawal",
        "queries": [
            "Total freshwater consumption million m3",
            "Water consumption m3",
            "Total water consumed",
        ],
        "unit_family": ["m3", "m³", "ML", "megaliters", "megalitres", "kL", "thousand m3", "cubic metres"],
        "canonical_unit": "m3",
        "plausible_range": (1e1, 1e10),
    },
}
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_config.py -v`
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/nlp_esg/config.py tests/test_config.py
git commit -m "feat(config): add per-KPI 'queries' list for multi-query retrieval"
```

---

## Task 3: Lone-subscript-2 normalisation regex

**Why:** Findings §3.5 — BP's columnar layout produces `MtCOe ... \n2` with the subscript "2" on its own line. Current `_CO2_NEXT_LINE` covers `MtCOe ... \n2eq` but not the bare-2 case.

**Files:**
- Modify: `src/nlp_esg/normalize.py`
- Test: `tests/test_normalize.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_normalize.py`:
```python
def test_lone_subscript_2_on_own_line():
    """BP pattern: 'MtCOe ... \\n2 \\n' — subscript-2 alone on its own line."""
    raw = "Scope 1 (direct) greenhouse gas MtCOe 33.2 30.4 31.1 32.8 33.7\n2\nemissions k l GHG"
    out = normalize_co2(raw)
    assert "MtCO2e" in out
    # The lone "2" line is consumed (or merged) — there should not be a
    # standalone "\n2\n" remaining after the unit fix.
    assert "MtCOe" not in out  # the pre-fix pattern is gone


def test_lone_subscript_2_does_not_corrupt_normal_text():
    """A '2' on its own line that is NOT preceded by a CO unit must not change."""
    raw = "Methodology section.\n2\nThis paragraph discusses scope 1."
    out = normalize_co2(raw)
    assert out == raw
```

- [ ] **Step 2: Run tests to verify the first fails**

Run: `pytest tests/test_normalize.py::test_lone_subscript_2_on_own_line -v`
Expected: FAIL — `MtCOe` is still present in the output.

- [ ] **Step 3: Add the regex and call it from `normalize_co2`**

Edit `src/nlp_esg/normalize.py` — add the new pattern alongside the existing `_CO2_*` patterns:
```python
_CO2_LONE_SUBSCRIPT = re.compile(
    r"(?P<unit>(?:M[tT]|[kKgG][tT]|[tT])?CO)e([^\n]*)\n\s*2(?=\s*\n)"
)
```

In the existing `normalize_co2(text)` function, add the new substitution after the existing patterns and before the function returns:
```python
text = _CO2_LONE_SUBSCRIPT.sub(r"\g<unit>2e\2", text)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_normalize.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/nlp_esg/normalize.py tests/test_normalize.py
git commit -m "feat(normalize): handle CO2 lone-subscript-2 on its own line"
```

---

## Task 4: Refactor — lift pdfplumber parsing to a private helper

**Why:** Task 6 will turn `parse_pdf` into a Docling-first dispatcher. Lifting the existing logic now (no behaviour change) makes that diff clean and testable.

**Files:**
- Modify: `src/nlp_esg/ingest.py`

- [ ] **Step 1: Run the existing ingest tests to confirm green starting state**

Run: `pytest tests/test_ingest.py -v`
Expected: all pass.

- [ ] **Step 2: Refactor `parse_pdf` to delegate to a private helper**

Edit `src/nlp_esg/ingest.py`. Add a new private function and reduce `parse_pdf` to a delegating wrapper. The exported behaviour is unchanged.

```python
def _parse_with_pdfplumber(path: Path) -> ParsedReport:
    """Parse a PDF with pdfplumber. No caching — caller handles cache."""
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
    """Parse a PDF into pages + tables. Caches to data/cache based on mtime."""
    company, year = _parse_filename(path)
    cache_path = CACHE_DIR / f"{company}_{year}_pdfplumber.pkl"
    if use_cache and cache_path.exists() and cache_path.stat().st_mtime >= path.stat().st_mtime:
        with cache_path.open("rb") as f:
            return pickle.load(f)
    report = _parse_with_pdfplumber(path)
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as f:
            pickle.dump(report, f)
    return report
```

Note the cache filename now includes `_pdfplumber`. The v1 cache files (`bp_2024.pkl`, etc.) will be ignored — first run on each PDF re-parses with pdfplumber and writes the new filename. That's the expected behaviour.

- [ ] **Step 3: Run all tests**

Run: `pytest -q --ignore=tests/test_integration_llm.py --ignore=tests/test_integration_real_pdf.py`
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add src/nlp_esg/ingest.py
git commit -m "refactor(ingest): lift pdfplumber parsing into _parse_with_pdfplumber"
```

---

## Task 5: Add Docling parser

**Why:** Findings §3.1 — pdfplumber is the bottleneck for the baseline. Docling preserves multi-column reading order and emits properly-shaped tables.

**Files:**
- Create: `src/nlp_esg/ingest_docling.py`
- Modify: `pyproject.toml`
- Test: `tests/test_ingest_docling.py`

- [ ] **Step 1: Add `docling` to dependencies**

Edit `pyproject.toml` — add `"docling>=2,<3"` to the `dependencies = [...]` list.

Then install:
```bash
pip install -e .
```
Expected: docling installs (~500MB–1GB models will download lazily on first parse, not at install time).

- [ ] **Step 2: Write the failing test**

Create `tests/test_ingest_docling.py`:
```python
from unittest.mock import MagicMock, patch

import pytest

from nlp_esg.ingest_docling import parse_with_docling


def _fake_docling_doc():
    """Build a minimal mock that mimics what parse_with_docling consumes."""
    table_item = MagicMock()
    table_item.label = "table"
    table_item.prov = [MagicMock(page_no=2)]
    df_mock = MagicMock()
    df_mock.columns = ["", "2024", "2023"]
    df_mock.values.tolist.return_value = [["Total Scope 1 GHG (MtCO2eq)", "33.7", "32.8"]]
    table_item.export_to_dataframe.return_value = df_mock

    doc = MagicMock()
    doc.num_pages.return_value = 3
    doc.export_to_markdown.side_effect = lambda page_no=None: f"# Page {page_no}\nsome text"
    doc.iterate_items.return_value = [(table_item, 0)]
    return doc


def test_parse_with_docling_returns_parsed_report(tmp_path):
    pdf = tmp_path / "demo_2024.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    converter = MagicMock()
    converter.convert.return_value = MagicMock(document=_fake_docling_doc())
    with patch("nlp_esg.ingest_docling.DocumentConverter", return_value=converter):
        report = parse_with_docling(pdf)
    assert report is not None
    assert report["parser"] == "docling"
    assert report["company"] == "demo"
    assert report["report_year"] == 2024
    assert len(report["pages"]) == 3
    assert any(t["page_num"] == 2 for t in report["tables"])


def test_parse_with_docling_returns_none_on_exception(tmp_path):
    pdf = tmp_path / "broken_2024.pdf"
    pdf.write_bytes(b"not a pdf")
    converter = MagicMock()
    converter.convert.side_effect = RuntimeError("boom")
    with patch("nlp_esg.ingest_docling.DocumentConverter", return_value=converter):
        report = parse_with_docling(pdf)
    assert report is None


def test_parse_with_docling_returns_none_on_empty_pages(tmp_path):
    pdf = tmp_path / "empty_2024.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    doc = MagicMock()
    doc.num_pages.return_value = 0
    doc.iterate_items.return_value = []
    converter = MagicMock()
    converter.convert.return_value = MagicMock(document=doc)
    with patch("nlp_esg.ingest_docling.DocumentConverter", return_value=converter):
        report = parse_with_docling(pdf)
    assert report is None
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_ingest_docling.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nlp_esg.ingest_docling'`.

- [ ] **Step 4: Implement `ingest_docling.py`**

Create `src/nlp_esg/ingest_docling.py`:
```python
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any

from nlp_esg.ingest import ParsedReport, _parse_filename

log = logging.getLogger(__name__)


def parse_with_docling(path: Path) -> ParsedReport | None:
    """Parse a PDF with Docling. Return None on any failure so the caller can fall back."""
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as e:
        log.warning("docling not available (%s); caller should fall back", e)
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

    try:
        n_pages = doc.num_pages() if callable(getattr(doc, "num_pages", None)) else len(doc.pages)
    except Exception:
        n_pages = 0
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
            rows = [[("" if v is None else str(v)) for v in row] for row in df.values.tolist()]
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
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_ingest_docling.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/nlp_esg/ingest_docling.py tests/test_ingest_docling.py
git commit -m "feat(ingest): add Docling parser as alternative to pdfplumber"
```

---

## Task 6: Wire the dispatcher in `parse_pdf`

**Why:** The dispatcher tries Docling first, falls back to pdfplumber on failure. Cache filename gains the parser tag so v1 and v2 caches coexist.

**Files:**
- Modify: `src/nlp_esg/ingest.py`
- Test: `tests/test_ingest.py`

- [ ] **Step 1: Write the failing tests for the dispatcher**

Append to `tests/test_ingest.py`:
```python
from unittest.mock import patch

from nlp_esg.ingest import parse_pdf


def test_parse_pdf_uses_docling_on_success(tmp_path, monkeypatch):
    pdf = tmp_path / "succ_2024.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    docling_report = {
        "company": "succ", "report_year": 2024, "parser": "docling",
        "pages": [{"page_num": 1, "text": "hello"}], "tables": [],
    }
    monkeypatch.setattr(
        "nlp_esg.ingest.parse_with_docling",
        lambda p: docling_report,
    )
    out = parse_pdf(pdf, use_cache=False)
    assert out["parser"] == "docling"


def test_parse_pdf_falls_back_to_pdfplumber_when_docling_returns_none(
    tmp_path, monkeypatch
):
    pdf = tmp_path / "fall_2024.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    monkeypatch.setattr("nlp_esg.ingest.parse_with_docling", lambda p: None)
    fallback_report = {
        "company": "fall", "report_year": 2024, "parser": "pdfplumber",
        "pages": [{"page_num": 1, "text": "fallback"}], "tables": [],
    }
    monkeypatch.setattr(
        "nlp_esg.ingest._parse_with_pdfplumber",
        lambda p: fallback_report,
    )
    out = parse_pdf(pdf, use_cache=False)
    assert out["parser"] == "pdfplumber"


def test_parse_pdf_falls_back_when_docling_returns_empty_pages(
    tmp_path, monkeypatch
):
    pdf = tmp_path / "emp_2024.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    monkeypatch.setattr(
        "nlp_esg.ingest.parse_with_docling",
        lambda p: {"company": "emp", "report_year": 2024, "parser": "docling",
                   "pages": [], "tables": []},
    )
    fallback_report = {
        "company": "emp", "report_year": 2024, "parser": "pdfplumber",
        "pages": [{"page_num": 1, "text": "fallback"}], "tables": [],
    }
    monkeypatch.setattr(
        "nlp_esg.ingest._parse_with_pdfplumber",
        lambda p: fallback_report,
    )
    out = parse_pdf(pdf, use_cache=False)
    assert out["parser"] == "pdfplumber"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ingest.py -v -k "docling or fall"`
Expected: 3 FAILS (`parse_pdf` does not yet try Docling).

- [ ] **Step 3: Update `parse_pdf` to dispatch**

Edit `src/nlp_esg/ingest.py`:
```python
from nlp_esg.ingest_docling import parse_with_docling


def parse_pdf(path: Path, use_cache: bool = True) -> ParsedReport:
    company, year = _parse_filename(path)

    # Try cache for either parser, preferring docling.
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
```

- [ ] **Step 4: Run all non-integration tests**

Run: `pytest -q --ignore=tests/test_integration_llm.py --ignore=tests/test_integration_real_pdf.py`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/nlp_esg/ingest.py tests/test_ingest.py
git commit -m "feat(ingest): dispatch parse_pdf — Docling first, pdfplumber fallback"
```

---

## Task 7: Lift page-ranking out of `LLMExtractor` into `retrieval.rank_pages_cosine`

**Why:** The page-ranking logic (findings §2.1) currently lives in `LLMExtractor._build_context`. The new RRF + BM25 hybrid (Tasks 8–9) needs to share it. Lift it now with **identical behaviour** so this commit is purely a refactor.

**Files:**
- Modify: `src/nlp_esg/retrieval.py`
- Modify: `src/nlp_esg/extractors/llm.py`
- Test: `tests/test_retrieval.py`

- [ ] **Step 1: Write the failing test for `rank_pages_cosine`**

Append to `tests/test_retrieval.py`:
```python
import numpy as np

from nlp_esg.retrieval import rank_pages_cosine


def _ix(*, pages, sentences, table_headers, tables):
    return {
        "company": "x", "report_year": 2024, "parser": "pdfplumber",
        "pages": pages, "sentences": sentences,
        "table_headers": table_headers, "tables": tables,
    }


def test_rank_pages_cosine_returns_pages_sorted_by_max_sim():
    q = np.array([1.0, 0.0], dtype=np.float32)
    e_high = np.array([1.0, 0.0], dtype=np.float32)
    e_low = np.array([0.0, 1.0], dtype=np.float32)
    indexed = _ix(
        pages=[{"page_num": 1, "text": "a"}, {"page_num": 2, "text": "b"}],
        sentences=[
            {"page_num": 1, "text": "low", "embedding": e_low},
            {"page_num": 2, "text": "high", "embedding": e_high},
        ],
        table_headers=[],
        tables=[],
    )
    ranked = rank_pages_cosine(indexed, q)
    assert ranked[0][0] == 2
    assert ranked[1][0] == 1


def test_rank_pages_cosine_unit_presence_boost():
    q = np.array([1.0, 0.0], dtype=np.float32)
    e_mid = np.array([0.5, 0.5], dtype=np.float32)
    indexed = _ix(
        pages=[
            {"page_num": 1, "text": "narrative no unit"},
            {"page_num": 2, "text": "data with MtCO2e marker"},
        ],
        sentences=[
            {"page_num": 1, "text": "x", "embedding": e_mid},
            {"page_num": 2, "text": "y", "embedding": e_mid},
        ],
        table_headers=[], tables=[],
    )
    ranked = rank_pages_cosine(indexed, q, unit_tokens=["mtco2e"])
    assert ranked[0][0] == 2  # boosted page wins despite same cosine
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_retrieval.py -v -k rank_pages_cosine`
Expected: FAIL — function does not exist.

- [ ] **Step 3: Add `rank_pages_cosine` to `retrieval.py`**

Append to `src/nlp_esg/retrieval.py`:
```python
def rank_pages_cosine(
    report: "IndexedReport",
    query_emb: np.ndarray,
    unit_tokens: list[str] | None = None,
) -> list[tuple[int, float]]:
    """Score each page by max sentence/table-header cosine sim to query.

    Optional small unit-presence bonus (+0.1) for pages whose normalised text
    contains a KPI unit token. Returns [(page_num, score)] sorted descending.
    """
    page_max: dict[int, float] = {}
    for s in report["sentences"]:
        sim = cosine_sim(query_emb, s["embedding"])
        pn = s["page_num"]
        if sim > page_max.get(pn, -1.0):
            page_max[pn] = sim
    for th in report["table_headers"]:
        sim = cosine_sim(query_emb, th["embedding"])
        pn = report["tables"][th["table_idx"]]["page_num"]
        if sim > page_max.get(pn, -1.0):
            page_max[pn] = sim

    tokens = [u.lower() for u in (unit_tokens or [])]
    scored: list[tuple[int, float]] = []
    for p in report["pages"]:
        base = page_max.get(p["page_num"], 0.0)
        bonus = 0.0
        if tokens:
            text_l = normalize_co2(p["text"]).lower()
            if any(u in text_l for u in tokens):
                bonus = 0.1
        scored.append((p["page_num"], base + bonus))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
```

- [ ] **Step 4: Update `LLMExtractor._build_context` to use it**

Edit `src/nlp_esg/extractors/llm.py` — replace the page-scoring block in `_build_context` with a call to `rank_pages_cosine`:
```python
from nlp_esg.retrieval import rank_pages_cosine

# inside _build_context:
query_emb = embed_texts([kpi_query])[0]
ranked = rank_pages_cosine(report, query_emb, unit_tokens=kpi_unit_family)
top_pages = ranked[:12]
top_page_nums = {pn for pn, _ in top_pages}
# ... rest unchanged: building parts list from top_pages ...
```

Delete the now-unused `cosine_sim` import inside `llm.py` if it's no longer referenced.

- [ ] **Step 5: Run all tests**

Run: `pytest -q --ignore=tests/test_integration_llm.py --ignore=tests/test_integration_real_pdf.py`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/nlp_esg/retrieval.py src/nlp_esg/extractors/llm.py tests/test_retrieval.py
git commit -m "refactor(retrieval): lift page ranking into rank_pages_cosine"
```

---

## Task 8: Multi-query reciprocal-rank-fusion ranker

**Why:** Findings §3.2 — single-string queries miss reporting-phrasing variation. RRF fuses ranks across multiple queries.

**Files:**
- Modify: `src/nlp_esg/retrieval.py`
- Test: `tests/test_retrieval.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_retrieval.py`:
```python
def test_rrf_combines_ranks():
    """A page that's rank-2 in two queries should beat a page that's rank-1 in one."""
    from nlp_esg.retrieval import _rrf_combine
    rankings = [
        [(1, 0.9), (2, 0.8), (3, 0.5)],   # query A: page 1 wins
        [(2, 0.95), (1, 0.7), (3, 0.4)],  # query B: page 2 wins
    ]
    fused = _rrf_combine(rankings, k=60)
    by_page = dict(fused)
    assert by_page[2] > by_page[3]  # page 2 ranks #1 once and #2 once -> higher than page 3
    # Pages 1 and 2 each appear once at rank-1 and once at rank-2; tie expected.
    assert abs(by_page[1] - by_page[2]) < 1e-9


def test_rank_pages_rrf_uses_multiple_queries(monkeypatch):
    """rank_pages_rrf calls rank_pages_cosine once per query and fuses."""
    import numpy as np
    from nlp_esg.retrieval import rank_pages_rrf
    indexed = {
        "company": "x", "report_year": 2024, "parser": "pdfplumber",
        "pages": [{"page_num": i, "text": ""} for i in (1, 2, 3)],
        "sentences": [], "table_headers": [], "tables": [],
    }
    fake_embed = lambda texts: np.array([[1.0, 0.0]] * len(texts), dtype=np.float32)
    monkeypatch.setattr("nlp_esg.retrieval.embed_texts", fake_embed)

    rankings = iter([
        [(1, 0.9), (2, 0.5), (3, 0.1)],
        [(2, 0.9), (1, 0.5), (3, 0.1)],
    ])
    monkeypatch.setattr(
        "nlp_esg.retrieval.rank_pages_cosine",
        lambda *a, **kw: next(rankings),
    )
    out = rank_pages_rrf(indexed, ["q1", "q2"])
    assert out[0][0] in (1, 2)  # both tie at #1, third place is page 3
    assert out[-1][0] == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_retrieval.py -v -k rrf`
Expected: FAIL — `_rrf_combine` and `rank_pages_rrf` do not exist.

- [ ] **Step 3: Implement RRF**

Append to `src/nlp_esg/retrieval.py`:
```python
from collections import defaultdict


def _rrf_combine(
    rankings: list[list[tuple[int, float]]], k: int = 60
) -> list[tuple[int, float]]:
    """Reciprocal-rank fusion. Each ranking is [(page_num, score)] sorted desc."""
    fused: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, (page_num, _) in enumerate(ranking):
            fused[page_num] += 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: -x[1])


def rank_pages_rrf(
    report: "IndexedReport",
    queries: list[str],
    unit_tokens: list[str] | None = None,
    k: int = 60,
) -> list[tuple[int, float]]:
    """Multi-query page ranking via reciprocal-rank fusion."""
    if not queries:
        return []
    rankings = []
    for q in queries:
        q_emb = embed_texts([q])[0]
        rankings.append(rank_pages_cosine(report, q_emb, unit_tokens=unit_tokens))
    return _rrf_combine(rankings, k=k)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_retrieval.py -v -k rrf`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/nlp_esg/retrieval.py tests/test_retrieval.py
git commit -m "feat(retrieval): multi-query reciprocal-rank-fusion page ranking"
```

---

## Task 9: Hybrid BM25 + cosine page ranking

**Why:** Findings §3.3 — embeddings cluster narrative ESG pages too tightly. BM25 rewards rare-token overlap (e.g. `MtCO2e`, `33.7`), surfacing data pages.

**Files:**
- Modify: `src/nlp_esg/retrieval.py`
- Modify: `pyproject.toml`
- Test: `tests/test_retrieval.py`

- [ ] **Step 1: Add `rank-bm25` dependency**

Edit `pyproject.toml` — add `"rank-bm25>=0.2,<1"` to the dependencies list, then `pip install -e .`.

- [ ] **Step 2: Write the failing test**

Append to `tests/test_retrieval.py`:
```python
def test_bm25_picks_page_with_rare_token(monkeypatch):
    """Hybrid ranking should surface the page containing the rare query token."""
    import numpy as np
    from nlp_esg.retrieval import rank_pages_hybrid
    indexed = {
        "company": "x", "report_year": 2024, "parser": "pdfplumber",
        "pages": [
            {"page_num": 1, "text": "long narrative paragraph about climate strategy"},
            {"page_num": 2, "text": "Total Scope 1 GHG emissions MtCO2e 33.7 in 2024"},
            {"page_num": 3, "text": "another narrative page about governance"},
        ],
        "sentences": [
            {"page_num": p, "text": "x",
             "embedding": np.array([0.5, 0.5], dtype=np.float32)}
            for p in (1, 2, 3)
        ],
        "table_headers": [], "tables": [],
    }
    fake_embed = lambda texts: np.array([[0.5, 0.5]] * len(texts), dtype=np.float32)
    monkeypatch.setattr("nlp_esg.retrieval.embed_texts", fake_embed)
    out = rank_pages_hybrid(indexed, ["Scope 1 emissions MtCO2e"])
    assert out[0][0] == 2
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_retrieval.py -v -k hybrid`
Expected: FAIL — `rank_pages_hybrid` does not exist.

- [ ] **Step 4: Implement hybrid scoring**

Append to `src/nlp_esg/retrieval.py`:
```python
import re as _re_bm25
from rank_bm25 import BM25Okapi

_BM25_TOK_RE = _re_bm25.compile(r"[A-Za-z0-9]+")


def _tokenize_for_bm25(text: str) -> list[str]:
    return _BM25_TOK_RE.findall(text.lower())


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def rank_pages_hybrid(
    report: "IndexedReport",
    queries: list[str],
    alpha: float = 0.5,
    rrf_k: int = 60,
) -> list[tuple[int, float]]:
    """alpha * cosine_rrf + (1 - alpha) * bm25, both min-max normalised in [0,1]."""
    if not queries:
        return []

    # Cosine via RRF across the multiple queries.
    rrf_ranking = rank_pages_rrf(report, queries, k=rrf_k)
    rrf_by_page = dict(rrf_ranking)

    # BM25 over normalised page texts.
    page_nums = [p["page_num"] for p in report["pages"]]
    corpus = [_tokenize_for_bm25(normalize_co2(p["text"])) for p in report["pages"]]
    if not any(corpus):
        return rrf_ranking
    bm25 = BM25Okapi(corpus)
    query_tokens = _tokenize_for_bm25(" ".join(queries))
    bm25_scores = list(bm25.get_scores(query_tokens))

    # Min-max within report so alpha has consistent meaning.
    rrf_vals = [rrf_by_page.get(pn, 0.0) for pn in page_nums]
    rrf_norm = _minmax(rrf_vals)
    bm25_norm = _minmax(bm25_scores)

    fused = [
        (pn, alpha * r + (1 - alpha) * b)
        for pn, r, b in zip(page_nums, rrf_norm, bm25_norm)
    ]
    fused.sort(key=lambda x: x[1], reverse=True)
    return fused
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_retrieval.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/nlp_esg/retrieval.py tests/test_retrieval.py
git commit -m "feat(retrieval): hybrid BM25 + cosine RRF page ranking"
```

---

## Task 10: Switch `LLMExtractor` to use the hybrid ranker

**Why:** The hybrid ranker is what produces measurably better page selection. The old single-query + unit-presence boost path is replaced.

**Files:**
- Modify: `src/nlp_esg/extractors/llm.py`
- Test: `tests/test_llm.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_llm.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm.py::test_llm_build_context_uses_multi_query_hybrid -v`
Expected: FAIL — `_build_context` doesn't accept `kpi_queries`.

- [ ] **Step 3: Update `_build_context` and the `extract` caller**

Edit `src/nlp_esg/extractors/llm.py`:

```python
from nlp_esg.retrieval import rank_pages_hybrid

class LLMExtractor(Extractor):
    # ...
    def _build_context(
        self,
        report: Any,
        kpi_query: str,
        kpi_unit_family: list[str] | None = None,
        kpi_queries: list[str] | None = None,
    ) -> str:
        queries = list(kpi_queries) if kpi_queries else [kpi_query]
        ranked = rank_pages_hybrid(report, queries)
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
        # ...
        context = self._build_context(
            report,
            kpi["query"],
            kpi_unit_family=kpi["unit_family"],
            kpi_queries=kpi.get("queries"),
        )
        # ... rest unchanged
```

- [ ] **Step 4: Run all tests**

Run: `pytest -q --ignore=tests/test_integration_llm.py --ignore=tests/test_integration_real_pdf.py`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/nlp_esg/extractors/llm.py tests/test_llm.py
git commit -m "feat(llm): use multi-query hybrid retrieval for context"
```

---

## Task 11: Row-label-aware table-header embedding (baseline)

**Why:** Findings §1.1 — Eni-style tables have semantic content in `row[0]` (the label) rather than headers. Including the first 5 row labels in the embedded header string makes those tables retrievable.

**Files:**
- Modify: `src/nlp_esg/retrieval.py`
- Test: `tests/test_retrieval.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_retrieval.py`:
```python
def test_build_index_table_header_includes_row_labels(monkeypatch):
    """Embedded header_string should include row[0] of first 5 rows."""
    from nlp_esg.retrieval import build_index

    captured: list[list[str]] = []
    def fake_embed(texts, model_name=None):
        import numpy as np
        captured.append(list(texts))
        return np.zeros((len(texts), 4), dtype=np.float32)
    monkeypatch.setattr("nlp_esg.retrieval.embed_texts", fake_embed)

    parsed = {
        "company": "eni", "report_year": 2024, "parser": "pdfplumber",
        "pages": [{"page_num": 1, "text": "x"}],
        "tables": [{
            "page_num": 1,
            "headers": ["", "2024", "2023"],
            "rows": [
                ["Total gross Scope 1 GHG emissions (MtCO2eq)", "18.95", "20.20"],
                ["Methane emissions", "0.5", "0.6"],
            ],
        }],
    }
    build_index(parsed)
    # The second embed_texts call is for table headers.
    header_strings = captured[1]
    assert any("Total gross Scope 1 GHG emissions" in hs for hs in header_strings)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_retrieval.py::test_build_index_table_header_includes_row_labels -v`
Expected: FAIL — current `build_index` only embeds headers, not row labels.

- [ ] **Step 3: Update `build_index`**

In `src/nlp_esg/retrieval.py`, change the header-string construction inside `build_index`:
```python
header_strings: list[str] = []
for t in report["tables"]:
    parts = [h for h in t["headers"] if h]
    parts.extend(row[0] for row in t["rows"][:5] if row and row[0])
    header_strings.append(" | ".join(parts))
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_retrieval.py -v`
Expected: all PASS. Existing tests that snapshot `header_string` may need their fixtures updated; do so minimally.

- [ ] **Step 5: Commit**

```bash
git add src/nlp_esg/retrieval.py tests/test_retrieval.py
git commit -m "feat(retrieval): include first 5 row labels in table header embedding"
```

---

## Task 12: Add system prompt to LLM cache key

**Why:** This is the load-bearing change for system-prompt v2 to take effect. Without it, prompt-v2 runs read v1 cached responses and the change is invisible.

**Files:**
- Modify: `src/nlp_esg/extractors/llm.py`
- Test: `tests/test_llm.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_llm.py`:
```python
def test_cache_key_changes_when_system_prompt_changes(monkeypatch, tmp_path):
    """Same (model, kpi, user_prompt) but different system_prompt -> different cache file."""
    from nlp_esg.extractors import llm as llm_mod
    monkeypatch.setattr(llm_mod, "CACHE_DIR", tmp_path)

    ext = llm_mod.LLMExtractor()
    k1 = ext._cache_key("scope_1_emissions", "user-text", "system-A")
    k2 = ext._cache_key("scope_1_emissions", "user-text", "system-B")
    assert k1 != k2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm.py::test_cache_key_changes_when_system_prompt_changes -v`
Expected: FAIL — `_cache_key` does not exist.

- [ ] **Step 3: Refactor cache-key creation into a method and add the system prompt**

Edit `src/nlp_esg/extractors/llm.py` — extract the inline `hashlib.sha256(...)` into a method that includes the system prompt:

```python
def _cache_key(self, kpi_key: str, user_prompt: str, system_prompt: str) -> str:
    return hashlib.sha256(
        f"{self.model}|{kpi_key}|{system_prompt}|{user_prompt}".encode()
    ).hexdigest()
```

Update the caller in `extract`:
```python
cache_key = self._cache_key(kpi_key, user_prompt, _SYSTEM_PROMPT)
cache_path = CACHE_DIR / "llm" / f"{cache_key}.json"
```

- [ ] **Step 4: Run all tests**

Run: `pytest -q --ignore=tests/test_integration_llm.py --ignore=tests/test_integration_real_pdf.py`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/nlp_esg/extractors/llm.py tests/test_llm.py
git commit -m "fix(llm): include system prompt in cache key

System-prompt changes (e.g. iteration-2 disambiguation rules) must
invalidate cached responses. The previous key — (model, kpi, user_prompt)
— silently served v1 responses to v2 calls."
```

---

## Task 13: Persist run-tagged extractions and metrics to disk

**Why:** The multi-run comparison table (Task 14) needs both runs on disk. Today `pipeline.main` only prints DataFrames.

**Files:**
- Modify: `src/nlp_esg/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pipeline.py`:
```python
def test_main_persists_extractions_with_run_tag(tmp_path, monkeypatch):
    from nlp_esg import pipeline
    from nlp_esg.types import KPIExtraction

    monkeypatch.setattr(pipeline, "load_indexed_reports", lambda: ["fake"])
    fake_extractions = [
        KPIExtraction(
            company="bp", report_year=2024, kpi="scope_1_emissions",
            value=33.7e6, unit="tCO2e", reporting_year=2024,
            source_snippet=None, source_page=None, confidence=0.9,
            extractor="llm",
        )
    ]
    monkeypatch.setattr(pipeline, "run_extraction", lambda r, include_llm=True: fake_extractions)
    monkeypatch.setattr(pipeline, "load_gold_labels", lambda: [])
    monkeypatch.setattr(pipeline, "RUNS_DIR", tmp_path)

    pipeline.main(run_tag="v2_test")

    csv_path = tmp_path / "v2_test" / "extractions.csv"
    assert csv_path.exists()
    import pandas as pd
    df = pd.read_csv(csv_path)
    assert "run_tag" in df.columns
    assert (df["run_tag"] == "v2_test").all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline.py::test_main_persists_extractions_with_run_tag -v`
Expected: FAIL.

- [ ] **Step 3: Add `RUNS_DIR` to config and `run_tag` plumbing in `pipeline.py`**

Edit `src/nlp_esg/config.py` — add:
```python
RUNS_DIR = DATA_DIR / "runs"
```

Edit `src/nlp_esg/pipeline.py`:
```python
import argparse
import dataclasses
from nlp_esg.config import RUNS_DIR

def _persist_run(extractions, metrics_df, run_tag, runs_dir):
    out_dir = runs_dir / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [dataclasses.asdict(e) for e in extractions]
    pd.DataFrame(rows).to_csv(out_dir / "extractions.csv", index=False)
    if metrics_df is not None:
        metrics_df.to_csv(out_dir / "metrics.csv", index=False)


def main(run_tag: str = "v2_docling") -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    indexed = load_indexed_reports()
    log.info("Loaded %d reports", len(indexed))
    if not indexed:
        log.error("No reports found in %s", REPORTS_DIR)
        return

    extractions = run_extraction(indexed, include_llm=True)
    for e in extractions:
        e.run_tag = run_tag

    for extractor in ("baseline", "llm"):
        df = build_comparison_table(extractions, extractor=extractor)
        print(f"\n=== {extractor} comparison table ===")
        print(df)

    golds = load_gold_labels()
    metrics_df = None
    if golds:
        metrics_df = run_evaluation(extractions, golds)
        print("\n=== Evaluation ===")
        print(metrics_df)
    else:
        log.warning("No gold labels found — skipping evaluation")

    _persist_run(extractions, metrics_df, run_tag, RUNS_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", default="v2_docling")
    args = parser.parse_args()
    main(run_tag=args.run_tag)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_pipeline.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/nlp_esg/config.py src/nlp_esg/pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): persist run-tagged extractions to data/runs/<tag>/"
```

---

## Task 14: Multi-run comparison helper

**Why:** The writeup table compares v1 (pdfplumber, baseline 0/15, LLM 8/15) against v2 (Docling+RRF+BM25, target ≥11/15). `compare.py` builds it from the per-run CSVs.

**Files:**
- Modify: `src/nlp_esg/compare.py`
- Test: `tests/test_compare.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_compare.py`:
```python
def test_build_run_comparison_joins_two_runs(tmp_path):
    import pandas as pd
    from nlp_esg.compare import build_run_comparison

    (tmp_path / "v1").mkdir()
    (tmp_path / "v2").mkdir()
    pd.DataFrame([
        {"company": "bp", "report_year": 2024, "kpi": "scope_1_emissions",
         "extractor": "llm", "value": 33.7e6, "unit": "tCO2e",
         "reporting_year": 2024, "run_tag": "v1"},
    ]).to_csv(tmp_path / "v1" / "extractions.csv", index=False)
    pd.DataFrame([
        {"company": "bp", "report_year": 2024, "kpi": "scope_1_emissions",
         "extractor": "llm", "value": 33.7e6, "unit": "tCO2e",
         "reporting_year": 2024, "run_tag": "v2"},
    ]).to_csv(tmp_path / "v2" / "extractions.csv", index=False)

    df = build_run_comparison(["v1", "v2"], runs_dir=tmp_path)
    assert ("bp", "scope_1_emissions") in {(r["company"], r["kpi"]) for _, r in df.iterrows()}
    cols = set(df.columns)
    assert "v1_llm_value" in cols
    assert "v2_llm_value" in cols
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_compare.py::test_build_run_comparison_joins_two_runs -v`
Expected: FAIL — `build_run_comparison` does not exist.

- [ ] **Step 3: Implement `build_run_comparison`**

Append to `src/nlp_esg/compare.py`:
```python
from pathlib import Path

from nlp_esg.config import RUNS_DIR


def build_run_comparison(
    runs: list[str], runs_dir: Path = RUNS_DIR
) -> pd.DataFrame:
    """Join per-run extractions on (company, report_year, kpi). One column per
    (run_tag, extractor) pair, value-only for compactness."""
    frames = []
    for run in runs:
        path = runs_dir / run / "extractions.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        wide = df.pivot_table(
            index=["company", "report_year", "kpi"],
            columns="extractor",
            values="value",
            aggfunc="first",
        ).reset_index()
        wide.columns = [
            f"{run}_{c}_value" if c not in ("company", "report_year", "kpi") else c
            for c in wide.columns
        ]
        frames.append(wide)
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on=["company", "report_year", "kpi"], how="outer")
    return out
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_compare.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/nlp_esg/compare.py tests/test_compare.py
git commit -m "feat(compare): build_run_comparison joins per-run extractions"
```

---

## Task 15: Run iteration 2 end-to-end and append findings

**Why:** Numbers, not promises. This is the deliverable.

**Files:**
- Modify: `docs/FINDINGS.md`
- Modify: `notebooks/demo.ipynb` (optional, time permitting)

- [ ] **Step 1: Run the v1 baseline reproduction**

(Skip if you already have the v1 extractions on disk.)

```bash
git checkout master -- src/nlp_esg
python -m nlp_esg.pipeline --run-tag v1_pdfplumber
```

This regenerates the v1 extractions under `data/runs/v1_pdfplumber/`. Note: this requires the `--run-tag` plumbing from Task 13 — if running v1 reproduction, run it after Task 13 is merged but with the *baseline* code temporarily checked out, then return to HEAD.

If checkout is impractical, copy the existing v1 numbers from FINDINGS.md §2.6 manually into `data/runs/v1_pdfplumber/metrics.csv` and skip this step.

- [ ] **Step 2: Run iteration 2**

```bash
python -m nlp_esg.pipeline --run-tag v2_docling
```

Expected:
- First run downloads Docling models (~500MB–1GB) — log it in the writeup.
- Per-PDF Docling parse: ~30s–2min depending on PDF size.
- 15 LLM API calls (5 reports × 3 KPIs), all cache misses (system prompt is part of the key now).
- `data/runs/v2_docling/extractions.csv` and `metrics.csv` produced.
- Console prints the comparison table and the metrics table.

- [ ] **Step 3: Inspect per-cell deltas**

Open a Python REPL or a quick notebook cell:
```python
from nlp_esg.compare import build_run_comparison
df = build_run_comparison(["v1_pdfplumber", "v2_docling"])
print(df.to_string())
```

Sanity-check against the predictions in the spec §5.7 and §6.5.

- [ ] **Step 4: Append iteration-2 section to FINDINGS.md**

Edit `docs/FINDINGS.md` — append a `## 5. Iteration 2 (2026-04-26)` section with subsections:
- 5.1 What changed (one-paragraph summary referencing tasks above)
- 5.2 What worked (per-cell flips that went green; cite the bucket)
- 5.3 What still does not work (any cell that stayed red; honest analysis)
- 5.4 Headline numbers vs gold (P/R/F1 table; v1 vs v2 row deltas)
- 5.5 Cost (rough Anthropic spend; how many cache hits would be possible on re-run)

Write what actually happened, not what the spec predicted. If results diverge from §5.7's "expected outcome" table, explain why.

- [ ] **Step 5: Commit findings + run artefacts**

```bash
git add docs/FINDINGS.md data/runs/v2_docling/metrics.csv
# extractions.csv may contain large source_snippet fields — judge whether to commit it
git commit -m "docs: iteration 2 results — Docling + multi-query RRF + system prompt v2"
```

---

## Self-review summary

Spec coverage check (every spec section maps to at least one task):

| Spec section | Task(s) |
| --- | --- |
| §3.1 In-scope item 1 (Docling + fallback) | 4, 5, 6 |
| §3.1 item 2 (multi-query RRF) | 2, 8 |
| §3.1 item 3 (BM25 hybrid) | 9 |
| §3.1 item 4 (lone-subscript regex) | 3 |
| §3.1 item 5 (system prompt v2) | 0 (already in repo, snapshotted) |
| §3.1 item 6 (system prompt in cache key) | 12 |
| §3.1 item 7 (run-tagged extractions + comparison) | 1, 13, 14 |
| §3.1 item 8 (FINDINGS.md update) | 15 |
| §4.1 module changes | 4, 5, 6, 7, 8, 9, 10, 11, 13, 14 |
| §4.2 data contracts (parser, run_tag, queries) | 1, 2 |
| §4.3 cache layout | 4, 6 |
| §5.5 baseline row-label embedding | 11 |
| §5.5 drop value-as-header detector | (no task — explicit non-action; not implementing it is the action) |
| §5.6 LLM extractor prompt v2 | 0 |
| §5.6 cache key fix | 12 |
| §5.7 pipeline + compare | 13, 14 |
| §6 error handling | 5, 6 (Docling failure modes), 9 (empty-corpus fallback) |
| §7 testing | every task includes tests |

No spec section is uncovered. Every code-step task contains the actual code, not a placeholder. Method signatures referenced across tasks (`rank_pages_cosine`, `rank_pages_rrf`, `rank_pages_hybrid`, `_build_context(..., kpi_queries=...)`, `_cache_key`) are consistent.
