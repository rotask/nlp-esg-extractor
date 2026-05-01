# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A two-extractor pipeline for pulling three numerical KPIs (Scope 1 emissions, total energy consumption, water consumption) out of corporate sustainability PDFs and producing a comparison table with P/R/F1 metrics against hand-labelled gold. Coursework deliverable; the iteration history lives in `docs/FINDINGS.md`.

## Run

```bash
# Default — runs Docling-first ingest, then both extractors, then evaluation.
python -m nlp_esg.pipeline --run-tag v_my_run

# Required on this machine: Docling's layout model SIGSEGVs on long PDFs in
# this corpus. Fall back to pdfplumber-only via env var.
NLP_ESG_DISABLE_DOCLING=1 python -m nlp_esg.pipeline --run-tag v_my_run
```

Output: `data/runs/<tag>/extractions.csv` + `metrics.csv`. The pipeline also prints comparison tables to stdout. Multi-run side-by-side joins via `compare.build_run_comparison(["v1", "v2"])`.

## Test

```bash
pytest -q --ignore=tests/test_integration_llm.py --ignore=tests/test_integration_real_pdf.py
pytest tests/test_normalize.py::test_lone_subscript_2_on_own_line -v   # single test
RUN_INTEGRATION=1 pytest                                                # opt-in real-API/real-PDF
```

The non-integration suite must stay under ~30 s on CPU; do not introduce tests that load embeddings or hit the API.

## Architecture (the parts you can't get from reading one file)

**Ingest dispatcher in `ingest.parse_pdf`**: tries `ingest_docling.parse_with_docling` first, falls back to the in-file `_parse_with_pdfplumber` on `None`/empty/quality-check failure. Returns a `ParsedReport` TypedDict with a `parser` field tagging which path won. Cache filename includes the parser (`{company}_{year}_{parser}.pkl`) so v1/v2 caches coexist. The `NLP_ESG_DISABLE_DOCLING=1` env var short-circuits Docling at the top of `parse_with_docling`.

**Two-tier cache**: `parse_pdf` caches the `ParsedReport` (text + tables); `retrieval.build_index` caches the `IndexedReport` (text + tables + per-sentence + per-table-header embeddings) under `{company}_{year}_{parser}_indexed_{model}.pkl`. ClimateBERT forward passes are slow on CPU (~5–15 min per long report), so leaving the index cache uninvalidated is critical for iteration speed. Invalidation happens implicitly when the parser tag or model name changes.

**Page-level retrieval shared by both extractors**: `retrieval.rank_pages_hybrid(report, queries, alpha=0.5)` combines reciprocal-rank-fusion across multiple KPI query phrasings (cosine over ClimateBERT sentence/table-header embeddings) with BM25 over normalised page text, both min-max scaled to [0,1]. The KPI registry in `config.KPIS` carries a `queries: list[str]` of 2–4 phrasings per KPI plus a `negative_tokens` list (e.g. `withdrawal` for water — rejects rows/lines that match unwanted metrics).

**Baseline extractor (`extractors/baseline.py`)**:
1. Table-first search over `report["table_headers"]` (cosine ≥ `TAU_TABLE`), with year-column selection via `_find_year_col` and unit canonicalisation. Negative-token filter rejects rows like "Energy production from renewable sources" when extracting total consumption.
2. Falls back to a **page-level line scanner** (`_scan_lines_for_kpi`): rank pages with `rank_pages_hybrid`, scan each line for KPI keywords + negative tokens + a `(value, unit)` parse via `normalize.parse_value`. Year-column awareness via `_pick_year_column_value`: when a year-row sits within ±5 lines, pick the column position of the most-recent year, taking the **last N** parsed numbers from the data line (skips noise digits like "1" in "Scope 1") and preserving magnitude via ratio scaling.

**LLM extractor (`extractors/llm.py`)**: Anthropic SDK with strict tool-use schema (`tool_choice={"type":"tool","name":"record_kpi"}`). Uses the **same** `rank_pages_hybrid` retrieval to build context (top-12 pages + same-page tables, 4000-char-per-page cap). Cache key is `sha256(model | kpi | system_prompt | user_prompt)` — system prompt MUST be in the key so that prompt-rule changes (ESRS aggregation, withdrawal-vs-consumption, magnitude prefix) invalidate the cache. Without this, prompt edits silently serve stale v1 responses (this was the load-bearing bug fixed in iteration 2). `reporting_year` is overwritten with `report["report_year"]` after extraction — the gold convention is the filename year regardless of which column the value comes from.

**`parse_value` patterns** (`normalize.py`): forward `(num)(mag?)(unit)`, magnitude-reverse `(mag)(unit)(num)`, plain-reverse `(unit)(num)`. Separator between unit and number is `[^A-Za-z0-9]{1,8}` so pdfplumber's flattened table text (`"(MWh) 84,399,860"`, `"million m3 | 47.3"`) parses. The `_normalize_for_parse` pre-pass splits PDF rendering quirks — `millionm 3` → `million m3`, `.000 m3` → `thousand m3`. `_NUMBER_IN_TEXT_RE` deliberately drops space-as-thousands so `"269 289"` is read as two values, not `269,289`.

**CO₂ normalisation** (`normalize.normalize_co2`): four regex passes that fix `CO 2 e`, `COe2`, `MtCO ... \n2eq`, and `MtCOe ... \n2` (lone subscript on its own line — BP's columnar ESG datasheet pattern).

## Gotchas

- **Gold page numbers ≠ PDF page indices.** `data/labels/gold_labels.csv` `source_page` uses the report's printed page numbers (which start after front matter). For Iberdrola the offset is ~10 pages. The runtime pipeline doesn't use the hint — retrieval finds the page — so this only matters when investigating gold by hand.
- **Docling on this machine SIGSEGVs.** Skip via `NLP_ESG_DISABLE_DOCLING=1`. The 15 MB file-size guard in `ingest_docling.py` and the post-parse "majority-empty pages" check exist for environments where Docling sometimes works.
- **The LLM cache key change requires re-running** — old cached responses under the v1 key are still on disk at `data/cache/llm/*.json` but won't be served because the new key includes the system prompt. Don't manually port them; it defeats the validation.
- **`build_index` table-header embedding includes the first 5 row labels** alongside headers, because Eni-style tables put the KPI label in `row[0]` and have year-only headers (`["", "2025", "2024"]`).

## Where the design and history live

- `docs/superpowers/specs/2026-04-26-nlp-esg-iteration2-design.md` — current iteration design.
- `docs/superpowers/plans/2026-04-26-nlp-esg-iteration2.md` — task-by-task implementation plan.
- `docs/FINDINGS.md` — what failed, what worked, and why, across all iterations. Read this before starting structural work; many "obvious" improvements have already been tried and recorded as not viable.
