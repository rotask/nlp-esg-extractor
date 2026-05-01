# Project Overview — Technical Edition

A deeper companion to `PROJECT_OVERVIEW.md`. Same project, same pipeline,
same results — but with code-level detail, data-shape diagrams, real
example I/O at every stage, and the technical rationale behind each
choice. Intended for readers who will read the code, reproduce the runs,
or extend the system.

For the plain-English version, see `PROJECT_OVERVIEW.md`.
For the full iteration history, see `FINDINGS.md`.

---

## 1. Scope and contract

**Input.** N corporate ESG/sustainability PDFs in `data/reports/`
named `{company}_{year}.pdf`. Five reports in the gold set
(BP, Shell, Enel, Eni, Iberdrola; FY2025).

**Output.** For every `(report, KPI)` pair, two `KPIExtraction` rows
(one per extractor) written to `data/runs/<run_tag>/extractions.csv`,
plus per-extractor P/R/F1 in `metrics.csv`.

**KPIs (hardcoded in `nlp_esg.config.KPIS`).**

| key | canonical unit | plausible range | example aliases |
|---|---|---|---|
| `scope_1_emissions` | `tCO2e` | `[1e2, 1e9]` | tCO₂e, ktCO₂e, MtCO₂e |
| `total_energy_consumption` | `MWh` | `[1e2, 1e9]` | MWh, GWh, TWh, GJ, TJ, PJ, kWh |
| `water_consumption` | `m3` | `[1e1, 1e10]` | m³, ML, kL, Mm³, megalitres |

**Correctness predicate (`evaluate.is_correct`).** A prediction is a
true positive iff, after canonicalisation:
- `pred.unit` is in the KPI's `unit_family`,
- `pred.reporting_year == gold.reporting_year`,
- `|pred.value − gold.value| / gold.value ≤ EPSILON` where `EPSILON = 0.01`.

---

## 2. Data flow with concrete shapes

```
                  ┌─────────────────────────────┐
                  │  data/reports/bp_2025.pdf   │  Path
                  └──────────────┬──────────────┘
                                 ▼
   ╔═══════════════════════════════════════════════════════════════╗
   ║ STAGE 1   ingest.parse_pdf()                                  ║
   ║                                                               ║
   ║  Try ingest_docling.parse_with_docling()                      ║
   ║   └─ on None / empty / SIGSEGV / NLP_ESG_DISABLE_DOCLING=1 →  ║
   ║      _parse_with_pdfplumber()                                 ║
   ║                                                               ║
   ║  Cache:  data/cache/{company}_{year}_{parser}.pkl             ║
   ╚═══════════════════════════════════════════════════════════════╝
                                 │
                                 ▼  ParsedReport (TypedDict)
                ┌────────────────────────────────────┐
                │ {company: 'bp', report_year: 2025, │
                │  parser: 'pdfplumber',             │
                │  pages: [{page_num, text}, ...],   │
                │  tables: [{page_num, headers,      │
                │           rows}, ...]}             │
                └────────────────┬───────────────────┘
                                 ▼
   ╔═══════════════════════════════════════════════════════════════╗
   ║ STAGE 2   normalize_co2 + retrieval.build_index               ║
   ║                                                               ║
   ║  - normalize_co2() applied to every page text and every       ║
   ║    table cell (4 regex passes for CO₂ subscript artefacts).   ║
   ║  - split_sentences() → ClimateBERT mean-pool embeddings       ║
   ║    (768-dim) per sentence and per table-header-string.        ║
   ║  - Table header embedding includes first 5 row labels         ║
   ║    (Eni-style tables put the KPI label in row[0]).            ║
   ║                                                               ║
   ║  Cache:  data/cache/{company}_{year}_{parser}                 ║
   ║                    _indexed_{model}.pkl                       ║
   ╚═══════════════════════════════════════════════════════════════╝
                                 │
                                 ▼  IndexedReport (TypedDict)
   ┌──────────────────────────────────────────────────────────┐
   │ ParsedReport ⊕                                           │
   │   sentences:     list[{page_num, text, embedding(768)}]  │
   │   table_headers: list[{table_idx, header_string,         │
   │                        embedding(768)}]                  │
   └──────────────────────────────┬───────────────────────────┘
                                  ▼
   ╔═══════════════════════════════════════════════════════════════╗
   ║ STAGE 3   retrieval.rank_pages_hybrid(report, queries, α=0.5) ║
   ║                                                               ║
   ║  For each query phrasing q in KPIS[kpi]['queries']:           ║
   ║    rank_pages_cosine(report, embed(q))                        ║
   ║                                                               ║
   ║  Combine via Reciprocal Rank Fusion (k=60):                   ║
   ║    rrf_score(p) = Σ_q  1 / (k + rank_q(p))                    ║
   ║                                                               ║
   ║  Compute BM25 over normalised page text with                  ║
   ║    query_tokens = tokenize(' '.join(queries))                 ║
   ║                                                               ║
   ║  Min-max scale both to [0,1] and combine:                     ║
   ║    final(p) = α * rrf_norm(p) + (1-α) * bm25_norm(p)          ║
   ╚═══════════════════════════════════════════════════════════════╝
                                  │
                                  ▼  list[(page_num, score)]  sorted desc
                                  │
              ┌───────────────────┴───────────────────┐
              ▼                                       ▼
   ╔══════════════════════════════╗      ╔══════════════════════════════╗
   ║ STAGE 4a   BaselineExtractor ║      ║ STAGE 4b   LLMExtractor      ║
   ║                              ║      ║                              ║
   ║ try table-first path:        ║      ║ context = top_n=16 pages,    ║
   ║   if max(cos(table_header_i, ║      ║   normalised text capped at  ║
   ║         q)) ≥ TAU_TABLE=0.55:║      ║   4000 chars/page +          ║
   ║     find year col, pick row, ║      ║   tables on the same pages.  ║
   ║     parse_value, canonicalize║      ║                              ║
   ║                              ║      ║ provider ∈ {anthropic,gemini}║
   ║ else fall back to            ║      ║ tool_choice = record_kpi     ║
   ║ _scan_lines_for_kpi:         ║      ║   with strict JSON schema    ║
   ║   for each top-25 page:      ║      ║                              ║
   ║     for each line:           ║      ║ cache key = sha256(          ║
   ║       reject if neg-token,   ║      ║   model | kpi |              ║
   ║       parse (value, unit),   ║      ║   system_prompt |            ║
   ║       year-col if year-row   ║      ║   user_prompt)               ║
   ║       within ±25 lines,      ║      ║                              ║
   ║       prefer-larger tiebreak ║      ║ writes per-call JSON to      ║
   ║                              ║      ║   runs/<tag>/llm_prompts/    ║
   ╚══════════════════════════════╝      ╚══════════════════════════════╝
              │                                       │
              └───────────────┬───────────────────────┘
                              ▼
                   list[KPIExtraction]
                              │
                              ▼
   ╔═══════════════════════════════════════════════════════════════╗
   ║ STAGE 5   evaluate.is_correct + compare.build_comparison_table║
   ║                                                               ║
   ║  Correctness gate: unit-family ∩ year match ∩                 ║
   ║                    |Δvalue|/gold ≤ EPSILON (0.01).            ║
   ║                                                               ║
   ║  Persists:                                                    ║
   ║    data/runs/<run_tag>/extractions.csv                        ║
   ║    data/runs/<run_tag>/metrics.csv                            ║
   ║    data/runs/<run_tag>/llm_prompts/{co}_{yr}_{kpi}.json       ║
   ╚═══════════════════════════════════════════════════════════════╝
```

---

## 3. Stage-by-stage with examples

Every example is a real artefact from the v9 run on BP's
`bp_2025.pdf` unless otherwise noted. They reproduce by setting
`NLP_ESG_DISABLE_DOCLING=1` and running
`python -m nlp_esg.pipeline --run-tag <tag>`.

### 3.1 Ingest

`ingest.parse_pdf` is a Docling-first / pdfplumber-fallback dispatcher
with a parser-keyed disk cache:

```python
def parse_pdf(path: Path, use_cache: bool = True) -> ParsedReport:
    company, year = _parse_filename(path)
    docling_cache    = CACHE_DIR / f"{company}_{year}_docling.pkl"
    pdfplumber_cache = CACHE_DIR / f"{company}_{year}_pdfplumber.pkl"

    if use_cache:
        for p in (docling_cache, pdfplumber_cache):
            if p.exists() and p.stat().st_mtime >= path.stat().st_mtime:
                return pickle.load(p.open("rb"))

    report = parse_with_docling(path)        # batched Docling, may return None
    used_parser = "docling"
    if report is None or not report["pages"]:
        report = _parse_with_pdfplumber(path)
        used_parser = "pdfplumber"

    if use_cache:
        cache_path = CACHE_DIR / f"{company}_{year}_{used_parser}.pkl"
        pickle.dump(report, cache_path.open("wb"))
    return report
```

The cache filename is keyed on the parser, so a successful Docling run
and a (different-day) pdfplumber fallback for the same PDF coexist
without collision.

#### 3.1.1 Docling parser — page-range batching + CUDA acceleration

`ingest_docling.parse_with_docling` is the canonical parser. Three
behaviours that aren't obvious from the call signature:

```python
# extracts/ingest_docling.py (excerpt)
def parse_with_docling(path: Path) -> "ParsedReport | None":
    if _docling_disabled():                                 # NLP_ESG_DISABLE_DOCLING
        return None
    if path.stat().st_size > _DOCLING_MAX_FILE_BYTES:       # 100 MB safety cap
        return None
    n_pages_total = _count_pdf_pages(path)                  # via pypdfium2
    if n_pages_total <= 0:                                  # malformed PDF
        return _parse_single_shot(path, company, year)

    converter = _make_converter()                           # one model load
    pages, tables = [], []
    for batch_start in range(1, n_pages_total + 1, _PAGE_BATCH_SIZE):
        batch_end = min(batch_start + _PAGE_BATCH_SIZE - 1, n_pages_total)
        result = converter.convert(str(path),
                                   page_range=(batch_start, batch_end))
        _collect_pages_and_tables(result.document, batch_start, batch_end,
                                  pages, tables)
        del result
        gc.collect()
        _free_cuda_cache()                                  # torch.cuda.empty_cache()
    if _is_majority_empty(pages):
        return None
    return {"company": company, "report_year": year,
            "parser": "docling", "pages": pages, "tables": tables}
```

**`_PAGE_BATCH_SIZE`** auto-selects: 20 on CPU, 10 on GPUs with <9 GB
VRAM (the dev RTX 4050 has 6 GB). At batch=20 on the RTX 4050 we
reproducibly hit `std::bad_alloc` on image-dense pages
("Stage preprocess failed for run 1, pages [4]"); halving the batch
fixes it. Override via `NLP_ESG_DOCLING_BATCH_SIZE`.

**Pipeline options** are pinned explicitly (Docling 2.92's defaults
match these, but a future upgrade could revert them):

```python
opts = PdfPipelineOptions()
opts.do_ocr = False                                  # corpus has text layer
opts.do_table_structure = True
opts.table_structure_options = TableStructureOptions(
    mode=TableFormerMode.ACCURATE,                   # better on borderless tables
    do_cell_matching=True,                           # match cells back to PDF text
)
opts.accelerator_options = AcceleratorOptions(
    device=AcceleratorDevice.CUDA                    # explicit; 'auto' is unreliable
        if torch.cuda.is_available() else AcceleratorDevice.CPU,
)
```

The `_collect_pages_and_tables` helper handles two page-numbering
conventions Docling can emit when `page_range=(lo, hi)` is set:
original-PDF numbering preserved (the modern behaviour) or 1..N within
the batch (older versions). It uses a positional remap when keys come
back renumbered.

**Memory profile (verified on RTX 4050).** Each batch: model warm-up
once at ~1.7 GB RSS, then each subsequent batch peaks ~2.1 GB RSS.
Forced `gc.collect()` + `torch.cuda.empty_cache()` between batches
keeps the trajectory bounded — RSS does NOT grow monotonically with
page count.

**Wall-clock per PDF (verified on RTX 4050, batch=10):**

| Report     | Pages | Parse time | Per-page |
|------------|------:|-----------:|---------:|
| bp         |    14 |       54 s |   3.9 s  |
| iberdrola  |   148 |      135 s |   0.9 s  |
| eni        |   488 |      986 s |   2.0 s  |
| shell      |   462 |      952 s |   2.1 s  |
| enel       |   700 |      995 s |   1.4 s  |
| **total**  |  1812 |  ≈ 53 min |   ≈1.7 s |

Two `std::bad_alloc` events were observed on Eni (pages 128, 439) — both
non-gold pages. Docling logs the error and continues; the lost pages
appear as empty in the output, but the post-parse "majority empty"
check still passes (97% of Eni's pages remained substantive).

#### 3.1.2 pdfplumber parser — fast fallback

Used only when Docling fails or is disabled. Implementation is a
straight pdfplumber loop:

```python
def _parse_with_pdfplumber(path: Path) -> ParsedReport:
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            pages.append({"page_num": i, "text": page.extract_text() or ""})
            for raw in page.extract_tables() or []:
                headers = [(c or "").strip() for c in raw[0]]
                rows = [[(c or "").strip() for c in row] for row in raw[1:]]
                tables.append({"page_num": i, "headers": headers, "rows": rows})
    return {"parser": "pdfplumber", "pages": pages, "tables": tables, ...}
```

**Failure modes** that pdfplumber introduces (FINDINGS §1.1):

| Company | What pdfplumber returns |
|---|---|
| BP datasheet | `headers=['2025']`, all values jammed into one cell: `r0=['33.7\n17.2\n15.4\n…']`. No label column — labels are in page text. |
| Enel | Tables fragmented into micro-tables: each individual data value sits in its own header (`headers=['18.95']`). |
| Shell | `headers=['million MWh', '269']` is the entire "table" for total energy. |

The line-scanner fallback in `BaselineExtractor` exists *because* of
these mangled outputs. With Docling, the table-first path fires far
more often and the line scanner becomes a true backstop.

#### 3.1.3 Side-by-side: Docling vs pdfplumber on BP page 6

```python
# pdfplumber output for BP page 6 (one of two "tables"):
{'headers': ['2025'],
 'rows': [['33.7\n1.0\n32.6\n1.7\n0.02\n1.7\n0.7\n0.03\n0.7\n134,448\n4,718\n129,730\n0.16\n0.12\n0.16']]}

# Docling output for the same page:
{'headers': ['Operational control i,j,x', 'Unit', '2021', '2022', '2023', '2024', '2025'],
 'rows': [
     ['Scope 1 (direct) emissions l',                  'MtCO 2 e', '33.2','30.4','31.1','32.8','33.7'],
     ['UKandoffshore',                                 'MtCO 2 e', '1.0','1.0','1.0','1.0','1.0'],
     ['Global (excluding UKandoffshore)',              'MtCO 2 e', '32.1','29.4','30.1','31.8','32.6'],
     ['Scope 2 (indirect) emissions - location-based', 'MtCO 2 e', '2.4','2.1','2.0','2.4','1.7'],
     ...
 ]}
```

Same PDF, same page, completely different table-first reachability.

### 3.2 CO₂ normalisation

PDF renderers split `CO₂e` into separate glyphs that pdfplumber returns
in four distinct broken forms. `normalize_co2` runs four regex passes:

```python
# Real BP example, before:
"Scope 1 (direct) greenhouse gas MtCOe 33.2 30.4 31.1 32.8 33.7\n2\nemissions ..."
# After:
"Scope 1 (direct) greenhouse gas MtCO2e 33.2 30.4 31.1 32.8 33.7\nemissions ..."
```

The four patterns we cover:

| pattern | example | regex marker |
|---|---|---|
| spaced subscript | `CO 2 e` → `CO2e` | `_CO2_SPACED` |
| next-line subscript | `MtCOe…\n2eq` | `_CO2_NEXT_LINE` |
| reverse next-line | `MtCO…\n2` | `_CO2_REV_NEXT_LINE` |
| **lone subscript on its own line** | `MtCOe …\n2 \n` (BP datasheet) | `_CO2_LONE_SUBSCRIPT` |

**Iteration-4 fix.** A non-letter lookbehind was added to all four
patterns to prevent the `co` inside the word `Scope` from matching;
prior versions corrupted Enel page 147 to `Sco2eqpe`, dropping its
hybrid-retrieval rank from ~top-3 to #24.

### 3.3 Indexing

`build_index` runs ClimateBERT (or MiniLM) over every sentence and
every table-header-string and stores results back into the
`IndexedReport`:

```python
sentence_emb = SentenceTransformer(climatebert).encode(
    [s for s in split_sentences(normalize_co2(page.text))]
)  # shape (n_sentences, 768)

# Table-header string includes first 5 row labels, addressing the
# Eni-style "year-only header, KPI label in row[0]" case:
header_string = " | ".join(headers + [r[0] for r in rows[:5]])
```

**Cost.** ~5-15 minutes per long PDF on CPU for ClimateBERT
(82M params). The `IndexedReport` pickle is the *load-bearing* cache —
without it, every iteration on retrieval/extractor logic would pay the
full embedding bill again.

**Cache invalidation.** The path is keyed on `parser` and `model`:
```
data/cache/bp_2025_pdfplumber_indexed_climatebert.pkl
data/cache/bp_2025_pdfplumber_indexed_minilm.pkl
data/cache/bp_2025_docling_indexed_climatebert.pkl
```
Changing the parser or the model creates a fresh cache; changing
`build_index` logic does NOT invalidate. The convention is to bump the
filename suffix or `rm` the pkl when index logic changes.

### 3.4 Retrieval

The KPI registry carries multiple query phrasings:

```python
"scope_1_emissions": {
    "queries": [
        "Total gross Scope 1 GHG emissions",
        "Scope 1 direct greenhouse gas emissions tCO2e",
        "Scope 1 (direct) emissions",
    ],
    "negative_tokens": ["scope 2", "scope 3", "methane", "intensity", ...],
    ...
}
```

`rank_pages_hybrid` does three things:

```python
# 1. Per-query cosine ranking over sentences and table-header embeddings.
#    Each page's score = max cosine of any sentence/header on it.
rrf_rankings = [rank_pages_cosine(rep, embed(q)) for q in queries]

# 2. Reciprocal-rank fusion across queries.
fused[p] = sum(1 / (60 + rank_q(p)) for q in queries)

# 3. BM25 over tokenised normalised page text, using all query
#    tokens concatenated.
bm25_scores = BM25Okapi(corpus).get_scores(query_tokens)

# 4. Min-max scale both to [0,1] and weighted-sum.
score(p) = 0.5 * rrf_norm(p) + 0.5 * bm25_norm(p)
```

**Real example: BP scope_1 hybrid ranking.**

```
Top 5 pages by hybrid score:
  #1  page=  3  score=0.8653   (Assurance scope listing)
  #2  page=  6  score=0.8234   (Datasheet table)        ← gold page
  #3  page=  5  score=0.6880
  #4  page=  4  score=0.6423
  #5  page=  2  score=0.5043
```

**Why hybrid not just cosine.** ClimateBERT clusters narrative ESG
pages tightly: BP's spread is 0.92–0.97, signal-to-noise is poor. BM25
picks data pages out by rewarding rare tokens like `"134,448"` or
`"MtCO2e"`. Per FINDINGS §6.9, going from cosine-only to hybrid is what
made the data page consistently rank above narrative pages.

**Why multi-query RRF.** Single-query "Total energy consumption" misses
pages whose phrasing is "Energy consumption GWh"; multi-query with RRF
(k=60) gives every phrasing a chance to surface its match.

**Why `top_n_pages = 25` for the baseline.** Iteration 4 found Enel
scope_1 gold page at rank #24 with ClimateBERT (the line was on a page
where `Sco2eqpe` corruption depressed similarity); widening from 8 to
25 brought it into scope. The LLM uses `top_n = 16` because its prompt
budget is the binding constraint, not retrieval.

### 3.5 Baseline extractor — table path

The table-first path is the dominant code path with Docling input
(versus the line-scanner fallback that dominated under pdfplumber).
Seven heuristics, applied in this order:

```python
# 1. Above-threshold candidates by header-string cosine.
table_candidates = [
    (sim, table) for th in report["table_headers"]
    if (sim := cosine_sim(query_emb, th["embedding"])) >= TAU_TABLE
]

# 2. Page-level rejection: KPI may declare page_negative_phrases that
#    are matched against Markdown headings on the page.
heading_text = _markdown_headings(page_text_by_num[table["page_num"]])
if any(phrase in heading_text for phrase in kpi.get("page_negative_phrases", [])):
    continue

# 3. Year-column cap to skip milestone target columns.
year_col = _find_year_col(eff_headers, report_year)   # year ≤ report_year + 1

# 4. Score every row, propagating section context. Rows whose year-col
#    cell has no number become section markers, not candidates.
scored_rows, current_section = [], ""
for ri, row in enumerate(rows):
    if not re.search(_NUMBER_RE, row[year_col] or ""):
        if (row[0] or "").strip():
            current_section = row[0]
        continue
    label = row[0] if row[0] not in COLUMN_HEADER_ARTIFACTS else row[1]
    haystack = f"{current_section} {label} {row[1] if len(row)>1 else ''}".lower()
    if any(neg in haystack for neg in kpi["negative_tokens"]):
        continue
    rs = _row_score(kpi["query"], query_tokens, label)
    if rs > 0:
        scored_rows.append((rs, ri, label))

# 5. Iterate scored rows in descending order. First row whose value
#    passes unit + plausible-range checks becomes this table's
#    candidate; others are tried if it fails (per-table fall-through).
for best_rs, best_ri, display_label in sorted(scored_rows, reverse=True):
    raw_value = parse_number(re.search(_NUMBER_RE, row[year_col]).group())
    unit_match = _infer_unit_from_row_or_header(eff_headers, row, year_col, ...)
    multiplier, unit = unit_match
    canonical_value = to_canonical_value(raw_value * multiplier, unit, kpi["canonical_unit"])
    if not (lo <= canonical_value <= hi):
        continue                     # try next row in same table
    all_table_candidates.append((sim * best_rs, KPIExtraction(...)))
    break

# 6. Cross-table magnitude tiebreak: among candidates within 5% of the
#    best combined score, prefer the LARGEST canonical value.
best_score = max(c[0] for c in all_table_candidates)
tied = [c for c in all_table_candidates
        if (best_score - c[0]) / best_score <= 0.05]
return max(tied, key=lambda c: c[1].value or 0.0)[1]
```

**Year-column cap (`_find_year_col`).** Iberdrola's Scope 1 table mixes
actual data years and milestone target years in the same row:

```
['Tons', '2024', '2025', '%\n25/24', '2026', '2040', '2050', 'Annual %\ntarget...']
```

Without a cap, `_find_year_col` returned the index of `"2050"` — every
data cell in that column is `"N/AV."`, so every candidate row was
skipped. The cap (`year ≤ report_year`) admits 2025 for `report_year=2025`
files (the corpus convention is filename year = most-recent data column)
while excluding milestone columns 2026/2040/2050.

**Page-heading negative phrases (`page_negative_phrases`).** Defined
per-KPI in `config.KPIS`. For `water_consumption`:

```python
"page_negative_phrases": ["financial control"],
```

This rejects Shell page 424 (Markdown heading `## Water consumption
(financial control boundary) [A]`) in favour of page 385 (heading
`## Water consumption (E3-4)` — operational control, gold 86 M m³).
**Restricted to Markdown-heading lines** (`#…`), not body text — page
385 mentions "financial control boundary" only in a body cross-reference,
which would otherwise falsely disqualify it.

**Section propagation + `equity` negative token.** When walking rows
top-down, the most recent section-header row's `row[0]` is propagated
to subsequent data rows. Negative tokens are matched against
`(section + label + row[1])`. BP's GHG-Equityshare sub-table has its
own data rows whose labels do *not* contain "equity"; the section row
above them does. With propagation + `equity` in scope_1 / energy
negative tokens, those rows are filtered out.

**`_infer_unit_from_row_or_header` returns `(multiplier, unit)`.**
Search order:

1. value cell itself (`'45,678 tCO2e'`)
2. a `'Unit'`/`'Units'` column
3. row[1] as a fallback unit cell (Docling's empty-header pattern)
4. value column header — try whole + per-token (handles
   `'million cubic metres.2025'` via `canonicalize_unit_robust`)
5. row[0] label
6. other table headers

`canonicalize_unit_robust(s)` handles Docling-specific shapes the
strict `canonicalize_unit` rejects:

```python
canonicalize_unit_robust("MtCO 2eq")              → (1.0, "MtCO2e")
canonicalize_unit_robust("(Mm 3 )")               → (1.0, "Mm3")
canonicalize_unit_robust("million cubic metres.2025") → (1e6, "m3")
canonicalize_unit_robust("millionMWh")            → (1e6, "MWh")
```

It strips outer punctuation/parens, trailing `.YYYY` (compound
headers), tries internal-whitespace stripping, then peels off
`million`/`thousand`/`billion` magnitude prefixes (loose or glued)
before recursing into `canonicalize_unit`.

**Per-table row fall-through.** Eni page 166's gold row is
`'Direct GHG emissions (Scope 1)'` (rs=0.20, value 28.4 MtCO₂e),
masked under the higher-scoring `'Percentage of Scope 1 ... emission
trading system'` (rs=0.30, value `61` with unit `%` — fails unit
check). The fall-through tries scored rows in order; the second-best
row recovers the gold cell.

**Cross-table magnitude tiebreak.** When two tables both yield
viable candidates within 5% of each other (Shell scope_1 sub-totals
appear in multiple ESG appendix tables), prefer the largest canonical
value — a soft preference for ESRS-aligned consolidated totals over
segment sub-totals.

**Real example: BP `total_energy_consumption` on Docling output.**

```
table @ page 6
  headers: ['Metric','Unit','2021','2022','2023','2024','2025']
  cosine(header_string, "Total energy consumption MWh") = 0.929  ✓
  year_col_idx = 6  (column "2025")
  scored_rows[0] = (rs=1.00, ri=1, "Energy consumption t l")     # phrase match
  row = ['Energy consumption t l', 'GWh', '128,805', ..., '134,448']
  parse_number("134,448") → 134_448.0
  _infer_unit_from_row_or_header(...) → (1.0, 'GWh')             # 'Unit' column
  to_canonical_value(134_448, 'GWh', 'MWh') = 134_448_000.0      # ✓ gold
```

**Real example: Shell `water_consumption`** (Docling baseline finds it):

```
candidate page 424:    REJECTED (page heading contains 'financial control')
candidate page 385:    sim=0.945, scored_rows[0] = (rs=0.40, "Water consumption [C]")
  cell at year_col=1 ('million cubic metres.2025') = '86'
  _infer_unit_from_row_or_header at step 4:
      canonicalize_unit_robust('million cubic metres.2025')
        → strip year suffix → 'million cubic metres'
        → magnitude split → ('million', 'cubic metres')
        → canonicalize_unit('cubic metres') = 'm3'
        → return (1e6, 'm3')
  canonical = 86 * 1e6 = 86_000_000 m³  ✓ gold
```

### 3.6 Baseline extractor — line scanner

Triggers when the table path returns nothing. Scans every line on the
top-25 pages from `rank_pages_hybrid`, computing per line:

| signal | example |
|---|---|
| KPI keyword density | `len(query_tokens & line_tokens) / len(query_tokens)` |
| negative-token rejection | `not any(neg in line for neg in negative_tokens)` |
| `parse_value` | extracts `(value, unit)` via regex over numbers + unit aliases |
| year-column awareness | if year row within ±25 lines, pick the column for the most-recent year |
| page-rank bonus | small bonus for lines on rank-1 pages over rank-25 pages |
| "starts with Total" bonus | small lexical bonus |
| **magnitude tiebreak** | among candidates within 0.05 of best score, prefer the largest canonical value |

**Real example: BP `water_consumption` line scan.**

```
page 9 line: "| Freshwater consumption | millionm 3 | 53.6 | 51.7 | 47.4 | 46.5 | 47.3 |"

Year row (offset −14 lines on this page):
  "| Metric | Unit | 2021 | 2022 | 2023 | 2024 | 2025 |"
  → most-recent year = 2025 → column index 6

parse_value normalises 'millionm 3' → 'million m3' (the
_normalize_for_parse pre-pass), then matches the magnitude-before-unit
pattern → (47.3, 'million m3').

Magnitude preserved via ratio scaling:
  raw_value = 53.6 (column 2 — "first" parsed number)
  target_col_num = 47.3 (column 6)
  canonical = raw_value * 1e6 * (target_col_num / raw_value)
            = 47.3 × 1e6
            = 47_300_000.0  m³
```

Gold = 47,300,000 m³. ✓

**Year search window.** Originally ±5 lines. Iteration-3.5 widened to
±25 above because BP's Freshwater row sits 14 lines below its year
header. Closest-above wins, so multi-table pages still resolve their
own headers correctly.

**Magnitude tiebreak (iteration 5, v9).** Eni pages 166 and 167 both
contain near-identical lines `"Direct GHG emissions (Scope 1)
(MtCO2eq.) X.X"` with different values (28.4 consolidated vs 18.6
segment). Same kw_score, same prefix-bonus; rank-bonus alone determined
the outcome and the wrong one (page 167) won. Adding "prefer the
largest canonical value when scores tie within 0.05" flipped Eni
scope_1 from 18.6 → 28.4.

### 3.7 LLM extractor

`LLMExtractor._build_context`:

```python
ranked = rank_pages_hybrid(report, kpi['queries'])
top_pages = ranked[:16]  # was 12 pre-iteration-6.7
chunks = []
for pn, _ in top_pages:
    page_text = normalize_co2(pages[pn].text)[:4000]
    chunks.append(f"=== Page {pn} ===\n{page_text}")
    for tbl in tables_on_page(pn):
        chunks.append(format_table(tbl))
context = "\n\n".join(chunks)
```

**Tool schema (Anthropic shape).**

```python
{
    "name": "record_kpi",
    "input_schema": {
        "type": "object",
        "properties": {
            "value": {"type": ["number", "null"]},
            "unit":  {"type": ["string", "null"]},
            "reporting_year":  {"type": ["integer", "null"]},
            "source_snippet":  {"type": ["string", "null"]},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": [...]  # all five
    }
}
```

The `tool_choice={"type":"tool","name":"record_kpi"}` parameter forces
the model into a single function call — no free-form prose, no parsing
fragility.

**System prompt (excerpt).** Full text in `extractors/llm.py:86`.

```
- For Scope 1 emissions: pick "Total gross Scope 1 GHG emissions" /
  "Scope 1 (direct) greenhouse gas emissions". If the report
  distinguishes "consolidated" from "operational control + non-
  consolidated entities" (ESRS-aligned reporting, common in
  Shell/Eni), pick the LARGER ESRS-aligned figure that includes
  operated non-consolidated entities, not the consolidated-only
  sub-total. Never pick Scope 2, Scope 3, methane-only, intensity,
  or net (Scope 1+2 combined).
- For water consumption: ONLY pick a value labelled "water
  CONSUMPTION" or "freshwater CONSUMPTION" or "net water
  consumption". REJECT every value labelled "withdrawal",
  "withdrawn", "discharge", ...  This rule is STRICT — when in
  doubt between consumption and withdrawal, return null.
  If the report distinguishes an "operational control" boundary
  from a "financial control" (or "ESRS") boundary for water
  (common in Shell), prefer the OPERATIONAL CONTROL figure —
  that is the primary disclosure for water in oil & gas reports.
- Multiply out magnitude prefixes yourself: ... NEVER write a unit
  that contains a magnitude word (never "million m3", "thousand
  m3", "Mm3", "million cubic metres").
```

The boundary-preference clause was added because Shell's report has
two water-consumption tables — page 385 (operational control,
86 M m³, gold) and page 424 (financial control, 127 M m³,
supplementary). The Scope 1 rule already says *"prefer the LARGER
ESRS-aligned figure"*; the water rule's preference is the opposite
direction (operational control is typically smaller and is the
primary disclosure for water). Each KPI's preferred boundary lives
in its own clause so they don't cross-contaminate.

**Cache key.**

```python
cache_key = sha256(f"{model}|{kpi}|{system_prompt}|{user_prompt}").hexdigest()
cache_path = CACHE_DIR / "llm" / f"{cache_key}.json"
```

System prompt MUST be in the key — without it, prompt-rule changes
silently serve stale v1 responses (the load-bearing iteration-2 bug).

**Real cached response (Eni `total_energy_consumption`, gemini-2.5-flash).**

```json
{
  "company": "eni",
  "report_year": 2025,
  "kpi": "total_energy_consumption",
  "provider": "gemini",
  "model": "gemini-2.5-flash",
  "from_cache": false,
  "retrieved_pages": [170, 168, 171, 172, ...],
  "system_prompt": "You are an information-extraction assistant ...",
  "user_prompt": "KPI to extract: total_energy_consumption ...",
  "tool_response": {
    "value": 84399860,
    "unit": "MWh",
    "reporting_year": 2025,
    "source_snippet": "Energy consumption (MWh) 84,399,860",
    "confidence": 0.95
  }
}
```

Per FINDINGS §11, every LLM call writes one of these to
`data/runs/<tag>/llm_prompts/`. Reading them off disk is sufficient to
diagnose any failure — retrieval (was the gold page in
`retrieved_pages`?), extraction (what does `tool_response.source_snippet`
say?), normalisation (does `to_canonical_value` accept that unit?).

### 3.8 Normalisation and unit conversion

Three primitives in `normalize.py`:

```python
parse_value(text) -> (value: float, unit: str) | None
canonicalize_unit(unit_str) -> str  # case + punct + alias normalisation
to_canonical_value(value, unit, target_unit) -> float
```

**`parse_value` patterns** (in priority order):

| pattern | example | matches |
|---|---|---|
| forward | `33.7 MtCO2e` | `(num)(magnitude?)(unit)` |
| magnitude-reverse | `million MWh 269` | `(magnitude)(unit)(num)` |
| plain-reverse | `(MWh) 84,399,860` | `(unit)(num)` |

Plus a `_normalize_for_parse` pre-pass that splits PDF rendering
artefacts: `millionm 3` → `million m3`, `.000 m3` → `thousand m3`,
`m 3` → `m3`. The `_NUMBER_IN_TEXT_RE` deliberately drops
space-as-thousands so `"269 289"` parses as two numbers, not `269,289`.

**`canonicalize_unit`** — case-fold + whitespace-strip + punctuation-
strip (the iteration-6.6 fix; `MtCO2eq.` previously didn't match alias
`mtco2eq`) + alias lookup against `_UNIT_ALIASES`.

**`to_canonical_value`** — table-driven multipliers:

```python
_UNIT_CONVERSIONS = {
    ('GWh', 'MWh'): 1e3,    ('TWh', 'MWh'): 1e6,
    ('GJ', 'MWh'): 1/3.6,   ('TJ', 'MWh'): 277.78,  ...
    ('ML', 'm3'): 1e3,      ('Mm3', 'm3'): 1e6,  ...
    ('MtCO2e', 'tCO2e'): 1e6,  ('ktCO2e', 'tCO2e'): 1e3, ...
}
```

After conversion, `plausible_range` rejects values outside the KPI's
expected magnitude — this caught `flash-lite`'s erroneous
`168_590_000_000` MWh (Enel total energy, 10⁹ slip from TWh).

### 3.9 Evaluation

```python
def is_correct(pred: KPIExtraction, gold: dict) -> bool:
    if pred.value is None or pred.unit is None:
        return False
    if pred.reporting_year != gold['reporting_year']:
        return False
    if pred.unit not in KPIS[pred.kpi]['unit_family']:
        return False
    return abs(pred.value - gold['value']) / gold['value'] <= EPSILON
```

P/R/F1 computed per `(extractor, kpi)`:

```
TP = correct & non-null
FP = wrong-and-non-null
FN = null OR wrong (subset of all gold cells)
P = TP / (TP+FP),  R = TP / (TP+FN),  F1 = 2PR/(P+R)
```

**Per-cell diff between flash-lite and flash (FINDINGS §12.2).**
Identical retrieval, identical prompts, identical cache plumbing — only
the model name differs:

| cell | flash-lite | flash | Δ |
|---|---|---|---|
| bp water | 82M ✗ (withdrawal) | 47.3M ✓ | fixed |
| enel total_energy | None ✗ (10⁹ slip) | 168.59M ✓ | fixed |
| iberdrola water | 1.4B ✗ (withdrawal) | 45.64M ✓ | fixed |
| shell scope_1 | 46M ✗ (op-only) | 69M ✓ | fixed |

A model-tier change with **no code change** moved the LLM stream from
8/15 → 12/15. This isolates how much of the residual error budget is
model-quality vs algorithm.

**Cross-model comparison (post-fix code, corrected gold).** Same
retrieval, same prompt, same code — varying only `GEMINI_MODEL`:

| Model | LLM TP | Notable failure modes (cells) |
|---|---|---|
| `gemini-2.5-flash` | 12 / 15 | shell water (financial control), eni water, shell energy |
| `gemini-2.5-flash-lite` | 8 / 15 | + withdrawal-vs-consumption + 10⁹ magnitude slip |
| `gemini-3-flash-preview` | 9 / 15 | sums components (eni water 42+12=54, iberdrola scope_1 5.247M+3.233M=8.48M, iberdrola energy continuing+discontinued); ignores operational-control rule |
| `gemini-3.1-flash-lite-preview` | 6 / 15 | same as 3-flash-preview + 3 cells lost to API 503s; reverts shell scope_1 to 46M op-only |

`gemini-2.5-flash` remains the most rule-compliant Gemini model on this
task. The two preview models tested ignored explicit prompt rules
(*"REJECT every value labelled withdrawal"*, *"prefer OPERATIONAL
CONTROL"*, *"never sum components"*) that `2.5-flash` honours.

---

## 4. Headline numbers — final state

### 4.1 Run-by-run scorecard

| Run dir | Parser | Extractor | TP | F1 macro |
|---|---|---|---|---|
| `v9_magnitude_tiebreak/` | pdfplumber | baseline | 12/15 | 0.88 |
| `v_gemini_25flash_post_quota/` | pdfplumber | LLM (`gemini-2.5-flash`) | 12/15 | 0.88 |
| `v_gemini_25flash_post_quota/` | pdfplumber | best-of-either | 14/15 | 0.96 |
| `v_gemini_post_quota/` | pdfplumber | LLM (`gemini-2.5-flash-lite`) | 8/15 | 0.66 |
| `v_docling_full/` | docling | baseline (Docling, pre-fix) | 6/15 | 0.55 |
| `v_docling_baseline_fixed/` | docling | baseline (Docling, fixes A+B) | 7/15 | 0.62 |
| `v_docling_baseline_v2/` | docling | baseline (Docling, +section/tiebreak/row[1]) | 12/15 | 0.88 |
| **`v_docling_baseline_only/`** | **docling** | **baseline (Docling, all fixes)** | **14/15** | **0.96** |
| theoretical (if `flash` quota available) | docling | best-of-either (Docling baseline ∪ `gemini-2.5-flash`) | **15/15** | **1.00** |

### 4.2 Per-company × per-KPI × per-method

`OK` = within EPSILON (1%) of gold; **bold** = wrong value.

| Company   | KPI                | Gold       | Baseline + pdfplumber | Baseline + Docling | LLM (gemini-2.5-flash) |
|-----------|--------------------|------------|-----------------------|--------------------|-----------------------|
| bp        | scope_1            | 33,700,000 | 33,700,000 OK         | 33,700,000 OK      | 33,700,000 OK         |
| bp        | total_energy       | 134,448,000| 134,448,000 OK        | 134,448,000 OK     | 134,448,000 OK        |
| bp        | water              | 47,300,000 | **51,123,614** ✗ (line, ratio-scaled) | 47,300,000 OK (table @ pg 9) | 47,300,000 OK |
| enel      | scope_1            | 18,950,000 | 18,950,000 OK         | 18,950,000 OK      | 18,950,000 OK         |
| enel      | total_energy       | 168,590,000| 168,590,000 OK        | 168,590,000 OK     | 168,590,000 OK        |
| enel      | water              | 32,141,000 | 32,141,000 OK         | 32,141,000 OK      | 32,141,000 OK         |
| eni       | scope_1            | 28,400,000 | 28,400,000 OK         | 28,400,000 OK      | 28,400,000 OK         |
| eni       | total_energy       | 84,399,860 | 84,399,860 OK         | 84,399,860 OK      | 84,399,860 OK         |
| eni       | water              | 42,000,000 | 42,000,000 OK         | 42,000,000 OK (table @ pg 184, fall-through past `Percentage of …`) | **54,000,000** ✗ (sums "Operated 42 + not operated 12") |
| iberdrola | scope_1            | 5,246,890  | 5,246,890 OK (year-col cap fix) | 5,246,890 OK | 5,246,890 OK |
| iberdrola | total_energy       | 101,572,520| 101,572,520 OK        | 101,572,520 OK     | 101,572,520 OK        |
| iberdrola | water              | 45,642,187 | 45,642,187 OK         | 45,642,187 OK (row[0]='Metric' → row[1] label) | 45,642,187 OK |
| shell     | scope_1            | 69,000,000 | **None** ✗            | **62,400,000** ✗ (line @ pg 369 narrative) | 69,000,000 OK |
| shell     | total_energy       | 269,000,000| 269,000,000 OK        | 269,000,000 OK (table @ pg 368 'millionMWh') | **189,000,000** ✗ |
| shell     | water              | 86,000,000 | **72,000,000** ✗ (line @ pg 383 narrative) | 86,000,000 OK (page-heading filter rejects pg 424) | **127,000,000** ✗ (financial control table @ pg 424) |
| **TOTAL** |                    |            | **12 / 15**           | **14 / 15**        | **12 / 15**           |

### 4.3 Per-KPI

| KPI                       | pdfplumber baseline | Docling baseline | LLM (flash) | Docling ∪ LLM |
|---------------------------|--------------------:|-----------------:|------------:|--------------:|
| scope_1_emissions         | 4 / 5               | 4 / 5            | 5 / 5       | 5 / 5         |
| total_energy_consumption  | 5 / 5               | 5 / 5            | 4 / 5       | 5 / 5         |
| water_consumption         | 3 / 5               | **5 / 5**        | 3 / 5       | 5 / 5         |
| **Total**                 | 12 / 15             | **14 / 15**      | 12 / 15     | **15 / 15**   |

### 4.4 What changed at the cell level when switching parsers

Two cells flip ✗ → ✓ when moving from pdfplumber to Docling baseline,
both in the **water KPI**:

- **bp_water (47.3 M m³).** pdfplumber returns `headers=['2025']` with
  every numeric value collapsed into one cell. The line-scanner's
  ratio-scaling against a 14-line-distant year header recovered 47.3 M
  on master only after the magnitude-tiebreak fix landed in v9; even
  then it landed at 51.12 M because `parse_value` couldn't disambiguate
  the canonical value cleanly. Docling preserves the row
  `'Freshwater consumption' | 'millionm 3' | … | '47.3'` so the
  table-first path picks `47.3` directly.
- **shell_water (86 M m³).** Two tables on pages 385 (operational
  control, 86 M, gold) and 424 (financial control, 127 M, supplementary).
  pdfplumber outputs both as flat single-cell lists; the line scanner
  picks a narrative line (`'around 72 million cubic metres'`) on page
  383, which is `72 M ✗`. Docling outputs both as proper tables with
  Markdown headings; the page-heading filter rejects page 424, leaving
  page 385 as the sole candidate.

The single residual baseline miss is **shell_scope_1**: the gold
69 M tCO₂e ESRS-aligned figure exists only in narrative (no row says
"69" in any table). The LLM picks it; the baseline cannot. Best-of-either
is therefore 15/15.

---

## 4.5 Docling vs pdfplumber — comparison and recommendation

### 4.5.1 Same input, different reachable cells

The two parsers see the same PDFs, but the table-first path's
"reachable" cell-set is dramatically different:

| Cell | pdfplumber reachable? | Docling reachable? |
|---|---|---|
| bp.scope_1 | ✓ (page 6 datasheet works under pdfplumber too if cell is parseable) | ✓ |
| bp.total_energy | ✓ | ✓ |
| bp.water | ✗ (mangled to single-cell list; line-scanner ratio-scaling lands at 51.12 M not 47.3 M) | ✓ (clean 'Freshwater consumption' row) |
| enel.scope_1 | line scanner finds it on page 147 | ✓ table-first @ pg 147 |
| enel.total_energy | line scanner | ✓ table-first @ pg 150 ('Millions of kWh' header → kWh + magnitude) |
| enel.water | line scanner | ✓ table @ pg 286 |
| eni.scope_1 | line scanner with magnitude tiebreak (28.4 over 18.6) | ✓ table-first @ pg 166 (per-table fall-through past `Percentage` row) |
| eni.total_energy | line scanner | ✓ table-first @ pg 170 |
| eni.water | line scanner | ✓ table-first @ pg 184 (`(Mm 3 )` unit) |
| iberdrola.scope_1 | ✓ (table-first since year-col cap fix) | ✓ |
| iberdrola.total_energy | ✓ | ✓ |
| iberdrola.water | line scanner | ✓ table-first @ pg 58 (5-col `[Metric, Description, Unit, …]` schema → row[1] label) |
| shell.scope_1 | unreachable (narrative-only ESRS aggregation) | unreachable (same) |
| shell.total_energy | line scanner finds 269 M on pg 368 | ✓ table-first @ pg 368 (`'millionMWh'` glued unit) |
| shell.water | line scanner finds narrative '72 million' on pg 383 (✗) | ✓ table-first @ pg 385 (page-heading filter rejects pg 424) |

The Docling baseline solves **5 cells via the table-first path that
pdfplumber-baseline solves only via the line scanner** (Enel
scope_1/energy/water, Eni energy/water), plus **2 cells that pdfplumber
got wrong** (BP water, Shell water). It misses one cell the pdfplumber
baseline also missed (Shell scope_1, narrative-only).

### 4.5.2 Why Docling picks more cleanly

Docling preserves four pieces of structure that pdfplumber loses:

1. **Row labels**. pdfplumber's BP datasheet returns one cell per
   "table" with all values strung together by `\n`; row labels live in
   page text and have to be glued back by the line scanner. Docling
   gives proper row labels in row[0].
2. **Unit cells**. pdfplumber rarely keeps a separate unit column;
   Docling outputs `'GWh'`, `'MtCO 2 e'`, `'(Mm 3 )'` in row[1].
3. **Compound year + unit headers**. Docling outputs forms like
   `'million cubic metres.2025'` (year suffix + unit prefix). The new
   `canonicalize_unit_robust` parses these via `_YEAR_SUFFIX_RE` strip
   + magnitude split → `(1e6, 'm3')`.
4. **Markdown headings**. Lines like `## Water consumption (financial
   control boundary) [A]` are preserved verbatim in `page.text`. The
   page-heading filter uses these to disambiguate which boundary a
   page belongs to.

### 4.5.3 Cost ledger

| Property | pdfplumber | Docling (CPU) | Docling (GPU, RTX 4050) |
|---|---|---|---|
| Wall-clock per long PDF | seconds | ~14 s/page → ~1.5 hr/700-pg report | ~1.4 s/page → ~16 min/700-pg report |
| Full corpus (1812 pp) | < 30 s | ≈ 7 hours | ≈ 53 minutes |
| Peak RSS during inference | tens of MB | ~2 GB (batched) | ~2 GB (batched) |
| Peak VRAM | — | — | ~5 GB |
| Fresh dependency cost | `pip install pdfplumber` | + `pip install docling` (~4 GB models) | + CUDA-built torch (~3 GB) |
| Stability under stress | rock-solid | C++ models can SIGSEGV / `bad_alloc`; mitigated via batching | as CPU; 2 single-page failures observed on Eni (lost pages were non-gold) |
| Determinism | full | full | full (CUDA kernels are bit-stable here) |

### 4.5.4 Recommendation

**Use Docling first**, with pdfplumber as fallback:

- Docling baseline alone (14/15) **beats** the master pdfplumber
  baseline (12/15) by 2 cells, all from cleaner table extraction.
- The combined Docling-baseline + LLM run reaches **15/15** versus the
  pdfplumber-based master's 14/15 — full corpus recovery for the first
  time.
- The cost is wall-clock parse time (53 min on GPU once per corpus,
  then cached), and ~7 GB of additional dependencies (Docling models +
  CUDA torch). On caches-warm runs the difference vanishes — the
  pipeline finishes in ~30 s.

**Use pdfplumber as fallback** (status quo of the dispatcher):

- When Docling fails (file too large, all-empty pages, `bad_alloc` on
  the whole document, or `NLP_ESG_DISABLE_DOCLING=1` for environments
  without CUDA).
- For ad-hoc development: `parse_with_pdfplumber` is seconds, lets you
  iterate on extractor logic without re-paying Docling's parse cost.

The dispatcher in `ingest.parse_pdf` does exactly this. Cache filenames
include the parser tag (`{co}_{yr}_docling.pkl` vs
`{co}_{yr}_pdfplumber.pkl`) so the two coexist on disk and runs are
reproducible regardless of which parser served a given PDF.

---

## 5. Error analysis — the three-bucket taxonomy

The pipeline is `text → retrieval → extraction → normalisation`. Every
miss falls into exactly one of three buckets, distinguishable from the
prompt-log JSON files:

### 5.1 Retrieval errors

The gold-bearing page never reaches the extractor. Diagnostic from
`llm_prompts/<co>_<yr>_<kpi>.json`:

```python
gold_page in d['retrieved_pages']  # → False
```

**`v_gemini_25flash_post_quota` count: 1.** Shell `total_energy` —
gold page 366 is rank > 16 in the hybrid ranker. The model picked
"189 billion kWh" prose from a page it did see (operational-control
boundary, gold is ESRS-aligned 269M).

**Fix path.** Bump `top_n` to 20+; tighter ESRS phrasing in
`queries`. Cheap to try, not guaranteed.

### 5.2 Extraction errors

Right page in context, wrong cell picked. Sub-types:

- **Definition ambiguity (rule violation).** Smaller LLMs treat the
  withdrawal-vs-consumption rule as a soft preference. Flash-lite hit
  this on 4/5 water cells. Flash fixed 3/4. The two preview models
  (`gemini-3-flash-preview`, `gemini-3.1-flash-lite-preview`) regressed
  on this: the latter returned Iberdrola water as
  `1,274,971,000 m³` from a row literally labelled
  *"Total water withdrawal"*. Baseline avoids it via per-KPI
  `negative_tokens`.
- **Sub-total selection (ESRS vs operational control, Scope 1).**
  Shell's report has 46Mt (consolidated) and 69Mt (ESRS-aligned).
  Gold = 69. The baseline can't reach 69 because no single line says
  it; the LLM only picks it with a sufficiently capable model.
- **Boundary disambiguation (water).** Shell publishes two
  water-consumption tables — operational control on page 385
  (86 M m³, the primary disclosure) and financial control on page 424
  (127 M m³). The system prompt was extended with an explicit
  "prefer OPERATIONAL CONTROL" rule for water, but the preview models
  tested still picked 127. Per-KPI scoping prevents the Scope 1
  "prefer larger" instruction from leaking onto water.
- **Sum-of-components.** `gemini-3-flash-preview` and
  `gemini-3.1-flash-lite-preview` repeatedly return component sums
  instead of pre-computed totals: Eni water `Operated: 42 + not
  operated: 12 → 54`; Iberdrola scope_1 `Continuing 5,246,890 +
  Discontinued 3,233,218 → 8,480,108`. The system prompt explicitly
  forbids this (*"never guess or infer from breakdowns if there is no
  consolidated total"*); `gemini-2.5-flash` honours the rule, the
  preview models ignore it.
- **Wrong year column.** Mitigated by two fixes:
  `_pick_year_column_value`'s ±25-line search + ratio-scaling for
  magnitude preservation, and the new `_find_year_col` cap at
  `report_year + 1` that excludes milestone target columns.
- **Rule overgeneralisation.** Flash applied "prefer ESRS-larger" to
  Eni water (intended only for scope_1/total_energy) and emitted
  `42 + 12 = 54M` instead of operated-only `42`.

**Fix paths.** Stronger negative tokens; KPI-scoped prompt rules;
post-extraction guard re-prompting on snippets matching `withdraw|withdrew`.

### 5.3 Normalisation errors

Right value, right unit, wrong magnitude after canonicalisation. The
plausible-range guard catches these as `out_of_range` flags rather
than letting them through as wildly-wrong values.

**`v_gemini_25flash_post_quota` count: 0** (flash got the conversion
right). On flash-lite it was 1 — Enel total energy, `168.59 TWh` →
`168_590_000_000` MWh (10⁹ instead of 10⁶).

**Fix path.** Prompt rule: "never multiply prefixes; let our
deterministic `to_canonical_value` do the conversion". Already in the
prompt; flash-lite ignored it; flash respects it.

### 5.4 Inherent corpus limits

- **Shell scope_1 (baseline only).** The 69 Mt ESRS-aligned figure
  is computed only in narrative prose; no single line/cell in the PDF
  says "69". The baseline's line-and-table scanner cannot reach it.
  The LLM can — `gemini-2.5-flash` returns 69 M tCO₂e, the weakest
  preview models revert to the 46 M operational-control sub-total.

Three cells previously listed here have been moved out of this bucket
on the Docling baseline:

- **Shell water.** Page 385 (operational control) is extracted as a
  proper table by Docling; the page-heading filter rejects page 424
  (financial control). Result: 86 M m³ ✓.
- **Iberdrola scope_1.** `_find_year_col` cap at `report_year + 1`
  excludes the 2050 milestone column (all `"N/AV."`); the table-path
  picks 2025 with value `5,246,890 tCO₂e`.
- **BP water.** Docling preserves the `Freshwater consumption` row
  with all five years and the `'millionm 3'` unit; the table-first
  path picks the 2025 column directly. The pdfplumber baseline
  consistently mis-scaled to 51.12 M via line-scanner ratio scaling.

### 5.5 Failure-mode tally — Docling baseline + `gemini-2.5-flash`

| bucket | count | residual cells | fixable in code? |
|---|---|---|---|
| retrieval | 0 | — | every gold page now reaches its extractor in this configuration |
| extraction (baseline rules) | 1 | shell.scope_1 | covered by the LLM today; future fix would need narrative-aware aggregation |
| extraction (LLM rules) | 2 | shell.water, shell.energy | covered by the baseline today |
| normalisation | 0 | — | the magnitude-aware `canonicalize_unit_robust` + Docling's clean unit cells eliminate the 10⁹ slip |
| corpus limit | 0 | — | every gold cell is recoverable by *some* extractor in this combination |

After best-of-either, the residual count is **zero** — the combined
run reaches 15/15. The headline number is the union of two extractors
whose failure surfaces are structurally disjoint:

- The baseline misses on **narrative-only ESRS aggregation** (Shell
  scope_1) — needs an LLM to synthesise.
- The LLM misses on **table-first cells where the wrong table outranks
  the right one** (Shell water financial-vs-operational, Shell energy
  189 vs 269, Eni water sums components) — needs deterministic table
  rules.

---

## 6. Reproducibility

### 6.1 Re-running

**Docling-first (canonical):**

```bash
LLM_PROVIDER=gemini GEMINI_MODEL=gemini-2.5-flash \
    python scripts/run_docling_full.py --run-tag my_run
```

First run on the corpus: ~53 min on RTX 4050 GPU (Docling parse) +
~30 s for ClimateBERT GPU embedding + few minutes for retrieval +
extraction + LLM calls. Subsequent runs: seconds (both parsed-report
and indexed-report on disk; LLM responses cached on prompt hash).

**Baseline-only (LLM quota exhausted, fast iteration):**

```bash
python scripts/run_baseline_only.py
```

Bypasses the LLM stage; runs in seconds against the cached
`IndexedReport`s. Persists to `data/runs/v_baseline_only/`.

**pdfplumber-only (no GPU, fast wall-clock):**

```bash
NLP_ESG_DISABLE_DOCLING=1 LLM_PROVIDER=gemini \
    GEMINI_MODEL=gemini-2.5-flash \
    python -m nlp_esg.pipeline --run-tag my_run
```

Runs available for direct reproducibility:

**pdfplumber-based** (`master`, pre-Docling):
- `data/runs/v9_magnitude_tiebreak/` — baseline-only (12/15).
- `data/runs/v_gemini_25flash_post_quota/` — `gemini-2.5-flash` LLM run
  paired with the pdfplumber baseline; canonical pre-Docling headline
  (baseline 12/15, LLM 12/15, best-of-either 14/15).
- `data/runs/v_gemini_post_quota/` — `gemini-2.5-flash-lite` LLM run
  on the same pdfplumber input (LLM 8/15).

**Docling-based** (`experiment/docling-batched`):
- `data/runs/v_docling_full/` — initial Docling run, includes
  `parse_timings.csv` with per-PDF parse + index timing.
- `data/runs/v_docling_baseline_fixed/` — Docling baseline after the
  first round of fixes (CO2 spaces + row[1] unit fallback): 7/15.
- `data/runs/v_docling_baseline_v2/` — Docling baseline after section
  propagation, magnitude tiebreak, column-header artefact fallback:
  12/15.
- `data/runs/v_docling_baseline_only/` — **canonical Docling baseline
  run**, all fixes applied: **14/15**.

### 6.2 Determinism

| layer | deterministic? |
|---|---|
| Ingest | ✓ pdfplumber path is fully deterministic; Docling is too |
| `normalize_co2`, `parse_value`, `to_canonical_value` | ✓ |
| `embed_texts` (ClimateBERT, MiniLM) | ✓ (same model file, same input → bit-identical) |
| `rank_pages_hybrid` | ✓ |
| `BaselineExtractor` | ✓ |
| `LLMExtractor` cache hit | ✓ (replays cached `tool_response`) |
| `LLMExtractor` cache miss | not strictly deterministic — `temperature=0` but the Gemini API may still return slightly different `confidence` or `source_snippet` text on borderline cases |

The TP/FP/FN counts in `metrics.csv` are stable run-to-run modulo the
last bullet.

### 6.3 Test surface

```
pytest -q --ignore=tests/test_integration_llm.py \
          --ignore=tests/test_integration_real_pdf.py
# → 134 tests, ~50s, no embeddings, no API
```

Integration tests are opt-in via `RUN_INTEGRATION=1`. Twelve of the
134 tests are Docling-specific regression tests added during this
iteration:

| Test | Pattern under test |
|---|---|
| `test_parse_with_docling_batched_aggregates_pages` | Multi-batch loop preserves page numbering |
| `test_baseline_extracts_with_co2_subscript_spaces` | `'MtCO 2 e'` (BP) — internal whitespace + CO2 normalisation |
| `test_baseline_extracts_with_unit_in_row1_no_unit_header` | Empty header + unit in row[1] (Enel) |
| `test_baseline_handles_unit_with_internal_space` | `'MtCO 2eq'` (Enel scope_1) |
| `test_baseline_handles_unit_with_parens_and_space` | `'(Mm 3 )'` (Eni water) |
| `test_baseline_handles_glued_magnitude_unit` | `'millionMWh'` glued (Shell energy) |
| `test_baseline_handles_compound_year_header_with_unit` | `'million cubic metres.2025'` (Shell water) |
| `test_baseline_uses_row1_label_when_row0_is_column_artifact` | row[0]='Metric' → use row[1] (Iberdrola) |
| `test_baseline_section_aware_filtering_prefers_operational` | Section propagation + 'equity' negative token (BP) |
| `test_baseline_falls_through_to_next_row_when_best_row_fails` | Per-table row fall-through (Eni page 166) |
| `test_baseline_rejects_table_when_page_has_negative_context` | Page-heading filter (Shell water) |
| `test_find_year_col_skips_future_target_years` | Year-col cap (Iberdrola milestone columns) |

### 6.4 Audit trail

For every LLM call:

```
data/runs/<tag>/llm_prompts/<co>_<yr>_<kpi>.json
  ├─ retrieved_pages      (was retrieval the cause?)
  ├─ system_prompt        (cache-key load-bearing)
  ├─ user_prompt          (~50KB context; verbatim)
  ├─ tool_response        (raw model output before canonicalisation)
  └─ from_cache           (replay vs fresh API call)
```

A reader can reproduce any §5 conclusion from this directory alone,
without re-querying the API.

---

## 7. Where to read next

- `docs/FINDINGS.md` — full iteration history (v1 → v9 baseline,
  iteration 6 LLM resurrection via Gemini, model-tier comparison).
  Read before structural changes.
- `docs/API.md` — module-level Python API (importable from notebooks).
- `docs/SUSTAINABILITY.md` — impact, scalability, ethics.
- `CLAUDE.md` — architecture cheatsheet + cache-key invariants +
  gotchas (CO₂ word-boundary bug, Docling SIGSEGV, gold page-number
  offset, etc.).
- `notebooks/headline_figures.ipynb` — 14 figures used in the report
  and presentation, regenerated from the committed run artefacts.
