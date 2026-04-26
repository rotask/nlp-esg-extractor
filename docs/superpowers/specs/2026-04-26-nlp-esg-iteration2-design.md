# NLP ESG KPI Extraction — Iteration 2 Design

**Date:** 2026-04-26
**Project:** Team B — NLP track
**Predecessor spec:** [2026-04-23-nlp-esg-kpi-extraction-design.md](./2026-04-23-nlp-esg-kpi-extraction-design.md)
**Predecessor results:** [docs/FINDINGS.md](../../FINDINGS.md) (baseline 0/15, LLM 8/15, aggregate F1 ≈ 0.66)

## 1. Goal

Lift extraction quality on the existing 5-report × 3-KPI evaluation set by addressing the failure modes recorded in `FINDINGS.md`. Realistic target: baseline 0 → 6–9/15, LLM 8 → 11–13/15, aggregate F1 ≈ 0.66 → ≈ 0.85.

The pipeline shape, KPI registry semantics, evaluation rubric, and dataclass contracts from the predecessor spec are unchanged. This iteration is additive.

## 2. Decisions made during brainstorming

Recorded so the implementation plan does not relitigate them.

| Decision | Choice | Reason |
| --- | --- | --- |
| Focus | Both extractors, balanced | Coursework comparison reads cleanest if both improve; Docling helps both paths anyway |
| Docling integration | Primary, pdfplumber fallback | Keeps the existing 8/15 LLM result as a floor; safer than full replacement given Shell's complex layout |
| AMD GPU / torch-directml | Skip entirely | LLM API call dominates wall time; embedding is cached after first run; DirectML on `transformers` is fragile |
| LLM verify pass (findings §3.6) | Skip | Credits are tight; finish system-prompt v2 work instead (findings §2.6 noted prompt v2 was started but never completed) |
| Larger gold set (findings §3.7) | Skip | Stay at 5 × 3 = 15; coursework deadline does not justify expansion |
| Page-image / chart fallback (§3.8) | Skip | None of the 5 reports require it for the 3 chosen KPIs |
| Confidence calibration (§3.9) | Skip | Out of scope for this iteration |
| Value-as-header table fusion (§3.4) | Skip | Docling makes the bug moot; if Docling falls back, accept the cell stays a baseline miss |

## 3. Scope

### 3.1 In

1. **Docling ingest** with pdfplumber fallback (findings §3.1).
2. **Multi-query retrieval with reciprocal-rank fusion** (findings §3.2).
3. **Hybrid BM25 + cosine page ranking** (findings §3.3) — replaces the ad-hoc `+0.1` unit-presence boost.
4. **Lone-subscript-2 normalisation regex** (findings §3.5).
5. **System prompt v2** for the LLM extractor — three new disambiguation rule blocks targeting the failure buckets recorded in findings §2.6.
6. **System prompt added to the LLM cache key** so prompt changes invalidate cache.
7. **Run-tagged extractions** + a multi-run comparison table for the writeup.
8. **Updated `FINDINGS.md`** with an "Iteration 2" section appended after the run completes.

### 3.2 Out

See decision table in §2 — explicitly excluded items are not implemented.

## 4. Architecture

### 4.1 Module changes

```
src/nlp_esg/
├── ingest.py                 [MODIFIED]   parse_pdf() now tries Docling → falls back to pdfplumber; existing pdfplumber code stays in this file as the fallback branch
├── ingest_docling.py         [NEW]        Docling parsing path
├── retrieval.py              [MODIFIED]   RRF + BM25 hybrid
├── normalize.py              [MODIFIED]   _CO2_LONE_SUBSCRIPT pattern added
├── extractors/
│   ├── baseline.py           [MODIFIED]   row-label embedding; drop value-as-header detector
│   └── llm.py                [MODIFIED]   system prompt v2; cache key includes system prompt
├── compare.py                [MODIFIED]   build_run_comparison(runs)
├── pipeline.py               [MODIFIED]   --run-tag CLI flag; per-run output dir
├── config.py                 [MODIFIED]   KPIs gain queries: list[str]
└── types.py                  [MODIFIED]   KPIExtraction.run_tag: str | None
```

No new top-level modules besides `ingest_docling.py`. `ingest.py` becomes a thin dispatcher (~20 LoC).

### 4.2 Data contracts

**`ParsedReport` gains one field:**

```python
ParsedReport = {
    "company": str,
    "report_year": int,
    "parser": "docling" | "pdfplumber",   # NEW
    "pages": [{"page_num": int, "text": str}],
    "tables": [{"page_num": int, "headers": list[str], "rows": list[list[str]]}],
}
```

**`KPIExtraction` gains one optional field:**

```python
@dataclass
class KPIExtraction:
    ...                                    # all existing fields unchanged
    run_tag: str | None = None             # NEW; e.g., "v1_pdfplumber" or "v2_docling"
```

**`KPIS` registry gains one field:**

```python
KPIS["scope_1_emissions"]["queries"] = [
    "Total gross Scope 1 GHG emissions",
    "Scope 1 direct greenhouse gas emissions tCO2e",
    "Scope 1 (direct) emissions",
]
KPIS["total_energy_consumption"]["queries"] = [
    "Total energy consumption MWh",
    "Energy consumption GWh",
    "Total energy consumed across operations",
]
KPIS["water_consumption"]["queries"] = [
    "Total freshwater consumption million m3",
    "Water consumption m3",
    "Total water consumed",
]
```

The existing single-string `query` field stays for any caller that wants one query.

### 4.3 Cache layout

Parsed-report cache filename gains a parser tag so v1 and v2 coexist:

```
data/cache/
├── bp_2024.pkl                       # v1, pdfplumber (existing)
├── bp_2024_docling.pkl               # v2, Docling (new on first run)
├── bp_2024_pdfplumber.pkl            # v2 fallback if Docling fails (rare)
├── ...
└── llm/
    └── {sha256_of_(model|kpi|system|user)}.json   # cache key gains system prompt
```

LLM cache from v1 is preserved on disk — different cache keys for v2 (system prompt and retrieval-context both differ), so v2 hits the API. No manual cache deletion.

## 5. Component specifications

### 5.1 `ingest_docling.py` (new)

```python
def parse_with_docling(pdf_path: Path) -> ParsedReport | None:
    """Returns ParsedReport on success, None on failure."""
```

- Uses `docling.document_converter.DocumentConverter`.
- Walks `DoclingDocument.iterate_items()`, groups items by `page_no`.
- Per page `text`: markdown export of that page's items, preserving column reading order.
- Per `TableItem`: reconstruct `{page_num, headers, rows}` via `table_item.export_to_dataframe()`.
- Returns `None` on import error, conversion exception, or zero-page output.

### 5.2 `ingest.py` (modified)

The existing public function `parse_pdf(path: Path, use_cache: bool = True) -> ParsedReport` is updated; signature is unchanged. The existing pdfplumber logic moves inside a private `_parse_with_pdfplumber(path)` in the same file. Dispatcher logic:

```python
def parse_pdf(path: Path, use_cache: bool = True) -> ParsedReport:
    # cache lookup as today, but the cache path now includes parser tag
    parsed = parse_with_docling(path)
    if parsed is None or not parsed["pages"]:
        log.warning("docling failed for %s; falling back to pdfplumber", path.name)
        parsed = _parse_with_pdfplumber(path)
        parsed["parser"] = "pdfplumber"
    else:
        parsed["parser"] = "docling"
    return parsed
```

Cache filename gains the parser tag: `data/cache/{company}_{year}_{parser}.pkl`. The v1 parser-less pkl files (`bp_2024.pkl` etc.) are not read by the new pipeline; they remain on disk only as historical artefacts and can be deleted by the user at will.

### 5.3 `retrieval.py` (modified)

**Multi-query RRF:**

```python
def rank_pages_rrf(report: IndexedReport, queries: list[str], k: int = 60) -> list[tuple[int, float]]:
    rankings = [embed_and_rank_pages(report, q) for q in queries]
    fused: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, (page_num, _) in enumerate(ranking):
            fused[page_num] += 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: -x[1])
```

**Hybrid score:**

```python
score = 0.5 * cosine_rrf_normalised + 0.5 * bm25_normalised
```

- Both halves min-max normalised to `[0, 1]` within the report so α = 0.5 has consistent meaning.
- α = 0.5 fixed; not tuned (overfitting risk on n=15).
- `k = 60` fixed (canonical RRF default).
- BM25 over `normalize_co2`-normalised page text, tokenised with `re.findall(r"[A-Za-z0-9]+", text.lower())`. Library: `rank-bm25`.
- Removes the `+0.1` unit-presence boost from the v1 `retrieval.py` — BM25 already rewards rare unit tokens.

### 5.4 `normalize.py` (modified)

Add one regex covering the BP "lone subscript on its own line" pattern:

```python
_CO2_LONE_SUBSCRIPT = re.compile(
    r"(?P<unit>(?:M[tT]|[kKgG][tT]|[tT])?CO)e([^\n]*)\n\s*2(?=\s*\n)"
)

def normalize_co2(text: str) -> str:
    text = _existing_patterns(text)
    text = _CO2_LONE_SUBSCRIPT.sub(r"\g<unit>2e\2", text)
    return text
```

Existing call sites in retrieval and baseline sentence fallback are unchanged — they automatically pick up the new pattern.

### 5.5 `extractors/baseline.py` (modified)

Two surgical changes; the rest of the file is unchanged:

1. **Row-label-aware table embedding.** Embed `header_string = " | ".join(headers + [row[0] for row in rows[:5]])` instead of headers only. Picks up Eni-style `row[0] = "Total gross Scope 1 GHG emissions"`.
2. **No new value-as-header detector.** Findings §3.4 is explicitly skipped (Docling preserves table shape, so the pdfplumber-only bug it addressed becomes moot; pdfplumber-fallback misses are accepted).

The plausible-range filter, unit-family allow-list, and sentence fallback regex stay as-is.

### 5.6 `extractors/llm.py` (modified)

**System prompt v2** appends three rule blocks to the v1 prompt:

```
WATER METRIC SELECTION:
- "Withdrawal" / "withdrawn" / "abstracted" / "intake" → REJECT, even if labelled "total".
- "Consumption" / "consumed" / "net consumption" → ACCEPT.
- "Discharge" / "discharged" / "released" → REJECT.
- If a report only reports withdrawal and never consumption, return null with
  source_snippet showing the withdrawal line so the human can adjudicate.
- The word "total" attached to withdrawal/discharge does not promote it to consumption.

AGGREGATION SCOPE (when a report distinguishes consolidated-only vs ESRS / operated-non-consolidated figures):
- Prefer the ESRS-aligned / operated-non-consolidated-included figure.
- It is usually the LARGER number on the same line, often footnoted with
  "(ESRS aligned)" or "including operated non-consolidated entities".
- If the report only gives consolidated-only, take it.
- If only segment-level breakdowns with no consolidated total, return null.

UNIT FIELD RULES:
- Multiply out magnitude prefixes yourself in the value field.
  "269 million MWh" → value=269000000, unit="MWh".
  "1,275 ML"        → value=1275,    unit="ML".
- NEVER write a magnitude word ("million", "thousand", "billion") in the unit field.
  Only the canonical unit token (MWh, GWh, tCO2e, MtCO2e, m3, ML, etc.).
- If the unit appears in the report as "million cubic metres", the unit field is "m3"
  and the value is multiplied by 1,000,000.
```

**Cache key** changes from `sha256(f"{model}|{kpi_key}|{user_prompt}")` to `sha256(f"{model}|{kpi_key}|{system_prompt}|{user_prompt}")`. One-line change in `_cache_path()`.

Everything else (`tool_choice`, `temperature=0`, `record_kpi` schema, `reporting_year=report["report_year"]` override, retry/backoff, range/unit/`flags` post-processing) is unchanged.

### 5.7 `pipeline.py` and `compare.py` (modified)

`pipeline.py` today only prints comparison and metrics DataFrames; it does not persist extractions. This iteration adds a **new** persistence step:

- `main()` accepts `--run-tag <name>` (default `"v2_docling"`).
- After `run_extraction`, write the full extractions list to `data/runs/{run_tag}/extractions.csv` and the metrics table to `data/runs/{run_tag}/metrics.csv`.
- The existing print-to-stdout behaviour for `build_comparison_table` and `run_evaluation` stays.
- Each `KPIExtraction` is stamped with `run_tag` before persistence so `compare.py` can rebuild the multi-run view from the CSVs alone.

`compare.py` gains `build_run_comparison(runs: list[str]) -> DataFrame` that reads `data/runs/{run}/extractions.csv` for each run, joins them on `(company, kpi)` and emits the side-by-side comparison table described in §6 of the brainstorming. The roll-up P/R/F1 table is produced by reusing the existing `evaluate.py` per run and concatenating.

## 6. Error handling

Inherits all v1 error handling from the predecessor spec. Iteration-specific additions:

- **Docling import failure** → log at `WARNING`, fall back to pdfplumber for the entire run.
- **Docling per-PDF failure** → log at `WARNING`, fall back to pdfplumber for that PDF only; other PDFs continue with Docling.
- **Docling partial-page failure** → keep the pages Docling successfully parsed; do not fall back the whole report (an 80-page Docling parse is likely better than a 100-page pdfplumber parse).
- **BM25 corpus empty** (a report with zero pages of normalised text after parsing) → fall back to cosine-only ranking; flag the report.
- **Multi-query retrieval where all queries return zero hits** → fall back to single-query retrieval with the v1 `query` string.

## 7. Testing

The full `pytest` suite (excluding integration) must continue to run in under 10 seconds with no network and no model loading.

### 7.1 New unit tests

- `tests/test_ingest_docling.py` — patches `DocumentConverter` to return a canned `DoclingDocument`-shaped mock; asserts `parser="docling"` and page count > 0.
- `tests/test_ingest_fallback.py` — patches `parse_with_docling` to return `None`; asserts the dispatcher produces a `ParsedReport` with `parser="pdfplumber"`.
- `tests/test_retrieval_rrf.py` — synthetic rankings: a page that's #2 in two query rankings beats a page that's #1 in only one.
- `tests/test_retrieval_bm25.py` — 3-page synthetic report where the rare token appears only on page 2; assert page 2 ranks first under hybrid scoring.
- `tests/test_normalize_co2_lone_subscript.py` — the BP pattern + a no-op string round-trip case.
- `tests/test_baseline_row_label.py` — Eni-style fixture where `row[0]` is the label; assert the table-header embedding includes the row label.
- `tests/test_llm_cache_key_includes_system.py` — same `(model, kpi, user_prompt)` with two different system prompts produces different cache files.
- `tests/test_llm_system_prompt_v2_snapshot.py` — string equality against `tests/fixtures/system_prompt_v2.txt` so prompt changes show up as reviewable diffs.
- `tests/test_compare_run_comparison.py` — synthetic two-run dataframes; assert the joined comparison table has the expected columns and shape.

### 7.2 Existing tests

Existing baseline / LLM / evaluate / normalize tests stay green. Updates only where signatures changed (e.g., `ParsedReport` now has a `parser` field — fixture builders updated to include it).

### 7.3 Integration tests (skippable, opt-in via `RUN_INTEGRATION=1`)

- `test_integration_docling_real_pdf.py` — parses one of the 5 real PDFs with Docling, asserts page count > 0 and at least one table emitted. Skipped without `RUN_INTEGRATION=1`.

## 8. Dependencies

Added to `pyproject.toml`:

```
docling>=2,<3
rank-bm25>=0.2,<1
```

Docling on first run downloads its layout / OCR models (~500MB–1GB) into `~/.cache/docling/`; the README gets a one-paragraph note flagging this. No GPU dependency.

## 9. Deliverables

- The updated pipeline produces `data/runs/v2_docling/extractions.csv`.
- The notebook gains two cells: the multi-run comparison table and one-per-bucket qualitative example.
- `docs/FINDINGS.md` gains a "5. Iteration 2 (2026-04-26)" section after the run completes, with the same shape as sections 1–3 (what didn't work, what worked, headline numbers, remaining errors).
- Both v1 and v2 extractions remain on disk for reproducibility.

## 10. Open questions

None at time of writing.
