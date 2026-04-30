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
(BP, Shell, Enel, Eni, Iberdrola; FY2024).

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
                  │  data/reports/bp_2024.pdf   │  Path
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
                │ {company: 'bp', report_year: 2024, │
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
`bp_2024.pdf` unless otherwise noted. They reproduce by setting
`NLP_ESG_DISABLE_DOCLING=1` and running
`python -m nlp_esg.pipeline --run-tag <tag>`.

### 3.1 Ingest

`ingest.parse_pdf` is a dispatcher:

```python
def parse_pdf(pdf_path: Path) -> ParsedReport:
    if not os.environ.get("NLP_ESG_DISABLE_DOCLING"):
        try:
            r = parse_with_docling(pdf_path)
            if r is not None and _quality_ok(r):
                return r
        except Exception as e:
            log.warning("Docling failed (%s); falling back", e)
    return _parse_with_pdfplumber(pdf_path)
```

The cache key includes the parser tag so v1 (pdfplumber-only) and v2
(Docling-or-pdfplumber) caches can coexist.

**Example output (BP, abbreviated).**

```python
ParsedReport({
  'company': 'bp',
  'report_year': 2024,
  'parser': 'pdfplumber',
  'pages': [
    {'page_num': 1, 'text': 'bp Sustainability Report 2025\n\n...'},
    {'page_num': 6, 'text': 'Net zero Greenhouse gas emissions ...'},
    ...
  ],
  'tables': [
    {'page_num': 6,
     'headers': ['Metric', 'Unit', '2021', '2022', '2023', '2024', '2025'],
     'rows': [
       ['Scope 1 (direct) GHG emissions', 'MtCO2e', '32.8', '...', '33.7'],
       ['Energy consumption t l',         'GWh',    '128,805', ..., '134,448'],
       ...
     ]},
    ...
  ],
})
```

**Failure modes pdfplumber introduces** (FINDINGS §1.1):

| Company | What pdfplumber returns |
|---|---|
| BP datasheet | `headers=['2025']`, all values jammed into one cell: `r0=['33.7\n17.2\n15.4\n…']`. No label column — labels are in page text. |
| Enel | Tables fragmented into micro-tables: each individual data value sits in its own header (`headers=['18.95']`). |
| Shell | `headers=['million MWh', '269']` is the entire "table" for total energy. |

The line-scanner fallback exists because of these.

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
data/cache/bp_2024_pdfplumber_indexed_climatebert.pkl
data/cache/bp_2024_pdfplumber_indexed_minilm.pkl
data/cache/bp_2024_docling_indexed_climatebert.pkl
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

When `max(cosine(table_header_i, query)) ≥ TAU_TABLE = 0.55`, the
table path fires:

```python
# Pick the best-matching table header.
best_table = argmax cosine(table_headers, embed(query))

# Find the year column. Most-recent year wins, but candidates are
# capped at report_year + 1 to skip future target columns
# (Iberdrola tables have 2024/2025 alongside 2026/2040/2050).
year_col_idx = _find_year_col(headers, report_year)

# Score each row by query-overlap on row[0] (the row label).
best_row = argmax row_score(query_tokens, row[0])

# Reject if any negative_token appears in row[0].
if any(neg in row[0].lower() for neg in negative_tokens):
    continue

# Parse value + unit, canonicalize.
value, unit = parse_value(best_row[year_col_idx])
```

**Year-column cap (post-fix).** `_find_year_col` originally returned
the column index of the most-recent year matched by `\b(19|20)\d{2}\b`
in the effective headers. Iberdrola's Scope 1 table mixes actual data
years and milestone target years in the same row:

```
['Tons', '2024', '2025', '%\n25/24', '2026', '2040', '2050', 'Annual %\ntarget...']
```

Without a cap, `_find_year_col` returned the index of `"2050"` — and
every data cell in that column is `"N/AV."`, so the table-path
silently skipped every candidate row. The fix:

```python
if year > report_year + 1:
    continue                       # skip target / milestone years
```

This recovers Iberdrola Scope 1 (gold = 5,246,890 tCO₂e) for the
baseline. The cap also tolerates "2024-named files that cover FY2025"
because the `+ 1` window still admits `2025` for `report_year=2024`.

**Real example: BP `total_energy_consumption`.**

```
table @ page 6
  headers: ['Metric','Unit','2021','2022','2023','2024','2025']
  cosine(header_string, "Total energy consumption MWh") = 0.929  ✓
  year_col_idx = 6  (column "2025")
  best_row = ['Energy consumption t l', 'GWh', '128,805', ..., '134,448']
  row[6] = "134,448" → parse_value → (134_448.0, 'GWh')
  to_canonical_value(134_448, 'GWh', 'MWh') = 134_448_000.0

KPIExtraction(
  value=134_448_000.0, unit='MWh', reporting_year=2024,
  source_page=6,
  source_snippet='table@page 6: Energy consumption t l | 134,448',
  extractor='baseline')
```

Gold = 134,448,000 MWh. ✓

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
  "report_year": 2024,
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
    "reporting_year": 2024,
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

## 4. Headline numbers — final state (post-fix, corrected gold)

| Run dir | Extractor | TP | F1 |
|---|---|---|---|
| post-fix baseline (any of `v_corrected_gold/`, `v_after_fixes/`, `v_gemini31_lite/`) | baseline | 12/15 | 0.88 |
| `v_corrected_gold/` | LLM (`gemini-2.5-flash`) | 12/15 | 0.88 |
| `v_after_fixes/` | LLM (`gemini-3-flash-preview`) | 9/15 | 0.74 |
| `v_gemini31_lite/` | LLM (`gemini-3.1-flash-lite-preview`) | 6/15 | 0.54 |
| **best-of-either (baseline ∪ `gemini-2.5-flash`)** | — | **14/15** | **0.96** |

| KPI | Baseline TP | LLM-flash TP | Best-of-either |
|---|---|---|---|
| scope_1_emissions | 4 / 5 | 5 / 5 | 5 / 5 |
| total_energy_consumption | 5 / 5 | 4 / 5 | 5 / 5 |
| water_consumption | 3 / 5 | 3 / 5 | 4 / 5 |
| **total** | **12 / 15** | **12 / 15** | **14 / 15** |

The baseline gain (scope_1 3/5 → 4/5) comes from the year-column cap
fix in `_find_year_col` — Iberdrola scope_1 is now baseline-extractable.
Water dropped 4/5 → 3/5 because the corrected gold (Shell water 26 →
86 M m³) is no longer matched by the baseline's 72 M narrative line.

The single unrecovered cell is **Shell water (gold = 86 M m³,
operational-control boundary)**. The data IS extractable — Shell publishes
two boundary-tagged tables (`page 385: 86 M (operational)`,
`page 424: 127 M (financial)`). The baseline picks a narrative sentence
on page 383 (`"around 72 million cubic metres"`); `gemini-2.5-flash` picks
the financial-control table. This is a Bucket-B disambiguation failure,
not a corpus limit.

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

Two cells previously listed here have been moved out of this bucket:

- **Shell water.** With the gold corrected to 86 M m³ (operational-
  control boundary), the value is in an extractable table on page 385.
  The misextraction is now a Bucket-B boundary-disambiguation failure
  (which of two boundary-tagged tables to pick) — not corpus-bound.
- **Iberdrola scope_1 (baseline).** Solved by capping `_find_year_col`
  at `report_year + 1`. Previously the table-path selected the 2050
  target column (where every cell is `"N/AV."`) and silently failed;
  now it selects 2025 and recovers `5,246,890 tCO₂e`.

### 5.5 Failure-mode tally for the canonical run

For the post-fix canonical run (baseline + `gemini-2.5-flash`):

| bucket | count | fixable in code? |
|---|---|---|
| retrieval | 1 | partially (top_n + queries) |
| extraction (rules) | 2 | yes (prompt edits / negative tokens / post-extraction guard) |
| normalisation | 0 | already fixed by tier change |
| corpus limit | 1 | not without a stronger LLM (Shell scope_1 ESRS aggregation has no single line/cell to extract) |

The 14/15 best-of-either headline is the **union** of two extractors
that fail in structurally different ways. Reporting only one number
hides this.

---

## 6. Reproducibility

### 6.1 Re-running

```bash
NLP_ESG_DISABLE_DOCLING=1 LLM_PROVIDER=gemini \
    GEMINI_MODEL=gemini-2.5-flash \
    python -m nlp_esg.pipeline --run-tag my_run
```

First run: ~5-15 min per long PDF (ClimateBERT embedding pass) +
~3-5 min for retrieval + extraction + LLM calls. Subsequent runs: seconds
(both `IndexedReport` and LLM responses cached on disk).

Runs available for direct reproducibility:
- `data/runs/v9_magnitude_tiebreak/` — pre-fix baseline-only.
- `data/runs/v_gemini_25flash_post_quota/` — pre-fix `gemini-2.5-flash`
  LLM run referenced in the original headline.
- `data/runs/v_gemini_post_quota/` — `gemini-2.5-flash-lite` (kept for
  §5/error-mode discussion).
- `data/runs/v_corrected_gold/` — `gemini-2.5-flash`, gold-corrected,
  pre-code-fix; LLM 12/15.
- `data/runs/v_after_fixes/` — `gemini-3-flash-preview`, gold-corrected,
  with year-col cap + water boundary rule; baseline 12/15, LLM 9/15.
- `data/runs/v_gemini31_lite/` — `gemini-3.1-flash-lite-preview`, same
  code; baseline 12/15, LLM 6/15 (3 cells lost to API 503s).

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
# → 122 tests, ~30s, no embeddings, no API
```

Integration tests are opt-in via `RUN_INTEGRATION=1`.

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
