# NLP ESG KPI Extraction — Findings

Three KPIs extracted from five FY2024/2025 corporate ESG/Sustainability reports
(BP, Shell, Enel, Eni, Iberdrola): Scope 1 emissions, total energy
consumption, water consumption.

Two extractors were compared:
1. **Baseline** — embedding-retrieval over pdfplumber tables/sentences with
   regex-based number/unit parsing.
2. **LLM** — Claude Sonnet 4.6 with retrieved context and a strict tool-use
   schema.

This document records what failed, what succeeded, and why — and the
improvements that would push the system further.

---

## 1. What did not work

### 1.1 Naive table-row matching (the original baseline design)

The first iteration assumed pdfplumber would return tables shaped like:

```
headers = ["KPI label",  "2023",  "2024",  "2025"]
rows    = [["Scope 1 emissions", "30.4", "32.8", "33.7"]]
```

…and that the extractor only had to find the right row, then read the
right year column. **None of the five PDFs match this shape.** Each
breaks the assumption in a different way:

| Company | Failure mode |
| --- | --- |
| BP (ESG Datasheet) | Single-column tables with `headers=['2025']` and **all values jammed into a single newline-delimited cell**: `r0=['33.7\n17.2\n15.4\n…']`. There is no label column at all — labels live in the page text. |
| Enel | pdfplumber **fragments tables into micro-tables**: the year row `['', '2025', '2024 Change']` becomes one table with empty rows; each individual data value becomes its own micro-table where the value sits in the *header* (`headers=['18.95']`, the actual scope-1 figure). |
| Shell | Same fragmentation — `headers=['million MWh', '269']` is the entire "table" for total energy (269 million MWh). The "value" is the second header cell, not a row cell. |
| Eni | Year header survives, but `row[0]` is itself a numeric value, not a row label: `row=['28.4', '26.2']`. The label is implicit in the surrounding text. |
| Iberdrola | Properly structured tables exist (e.g. `[Source, 2025, 2024]` with a labelled row), **but the actual gold values for total-energy live on pages 35-36 inside running text, not in any table pdfplumber recognises**. |

Concrete consequence: of 15 (company × KPI) extraction attempts in the
final baseline run, only **one** produced any value (Iberdrola
total-energy, 88.19 M MWh from page 46) — and that one was the wrong
row ("Energy production from renewable sources" instead of "Total
energy consumption"), counted as a false positive.

### 1.2 Regex value-parsing on raw sentences

Falling back from tables to a sentence-level scan also failed. The
sentence splitter (`(?<=[.!?])\s+(?=[A-Z(])`) splits on
sentence-ending punctuation, but data lines extracted from PDFs almost
never end in punctuation — they end in a number or a unit. The top-K
sentences by cosine similarity to the KPI query were therefore narrative
fragments like:

```
Scope 1 GHG Emissions.
Greenhouse gas emissions Information relating to greenhouse gas emissions.
Total water withdrawals decreased (-9.6%) for generation.
```

…all of which are semantically close to the query but contain no
extractable `(value, unit)` pair. Across all 5 companies × 3 KPIs the
sentence path returned `parse_value=None` for every top-K sentence —
i.e. **0% recall from the sentence fallback**.

### 1.3 CO₂ subscript artefacts breaking unit parsing

PDF renderers split `CO₂e` into baseline + subscript glyphs that
pdfplumber returns as separate tokens or even separate lines. The same
substring shows up as `CO 2 e`, `MtCOe ... \n2eq`, or `MtCOe ... \n2`
(orphan subscript). `parse_value` rejects every such occurrence because
the unit alias table only contains `tCO2e` / `MtCO2e` etc.

A `normalize_co2()` pass was added that fixes the first two patterns
(`CO 2 e` and the `MtCOe…\n2eq` subscript-on-next-line case), but the
"orphan subscript-2 alone on its own line" pattern from BP's columnar
layout is not yet covered.

### 1.4 First LLM context format (before refinement)

The LLM extractor's first context-building strategy concatenated
**every** parsed table and **every** sentence in the report. For Enel
(811 tables, 5,111 sentences) and Shell (713 tables, 8,718 sentences)
this produced prompts in the hundreds of thousands of tokens, blowing
through the 30,000-tokens-per-minute Anthropic rate limit on the first
KPI of the first company and triggering an API error.

The second iteration trimmed to top-10 tables + top-20 sentences — fast
enough, but useless: as section 1.1/1.2 explain, the relevant data is
*not* in the highest-similarity tables/sentences, because it sits in
unstructured page text or in micro-tables whose headers are bare
numbers with no semantic content for the embedding to match on.

### 1.5 LLM `reporting_year` mismatch with gold

Even when the LLM extracted the correct value, the evaluator scored it
as a false positive. The gold labels use `reporting_year=2024` for
every row (matching the PDF filename), but the value itself is the
*most recent* column in the report, often labelled "2025" in the table
header. The LLM faithfully reported `reporting_year=2025` — and
`is_correct()` strictly compares `pred.reporting_year ==
gold.reporting_year`, marking the prediction wrong despite the value
being exactly right.

---

## 2. What did work

### 2.1 Page-level retrieval with full page text + unit-presence boost

The breakthrough for the LLM extractor was **stop sending tables and
sentences, send raw page text**. The actual KPI values live in
unstructured layouts that pdfplumber and the sentence splitter both
mangle, but the page text preserves the original token sequence. After
running `normalize_co2()`, lines like:

```
Scope 1 (direct) greenhouse gas MtCO2e 33.2 30.4 31.1 32.8 33.7
Total gross Scope 1 GHG emissions(1) MtCO2eq 18.95 20.20 (1.25) -6.2%
Energy consumption GWh 128,805 121,697 124,770 129,872 134,448
```

…are exactly the kind of input Claude can read at a glance.

The retrieval logic that surfaces the right page:

1. For each page, take the maximum cosine similarity of any sentence
   embedding or table-header embedding on that page against the KPI
   query embedding.
2. Add a small (+0.1) bonus to pages whose normalised text contains at
   least one of the KPI's accepted unit tokens (`tCO2e`, `MWh`, `m3`
   etc.). In dense ESG datasheets — BP especially — many pages cluster
   around 0.94-0.97 similarity, and the unit-presence signal reliably
   bumps the actual data page above the narrative pages.
3. Take top-12 pages by the boosted score; emit each page's normalised
   text (capped to 4,000 chars), followed by any pdfplumber tables on
   the same pages (as a backup signal — sometimes table rows survive
   even when text layout breaks).

For BP specifically, this single change pushed scope-1 from extracting
"5" (a footnote/row index) to extracting the correct **33,700,000
tCO2e** with the snippet `Scope 1 (direct) emissions l MtCOe 33.2 30.4
31.1 32.8 33.7` cited as evidence.

### 2.2 Tightened system prompt with explicit disambiguation rules

Sustainability reports contain dozens of values that *look* like
candidates for any given KPI:

- Scope 1 total vs. methane only vs. business-segment breakdown
- Energy consumed vs. energy *produced* from renewables vs.
  fuel-only sub-totals
- Water *withdrawn* vs. water *consumed* vs. water *discharged*

The system prompt now spells out which to pick:

```
- For Scope 1 emissions: pick "Total gross Scope 1" / "Scope 1 (direct)
  GHG emissions" — not Scope 2, Scope 3, or methane-only.
- For total energy consumption: pick the company-wide total energy
  CONSUMED — not energy produced from renewables, fuel-only, or
  electricity-only sub-totals.
- For water consumption: pick total water CONSUMED — not withdrawal,
  discharge, or recycled water.
- Pick the ABSOLUTE total — not an intensity ratio (e.g. tCO2e/$,
  MWh/€, l/kWh) or a percentage.
- If multiple year columns are present, pick the LATEST year's value.
```

These rules are KPI-specific because sustainability terminology is
inconsistent across reports — each rule was added in response to a
real misextraction observed during iteration (e.g. Iberdrola initially
returned the renewable-energy production figure of 88.19 M MWh, which
*sounds* like total energy but is a sub-component).

### 2.3 Strict tool-use schema (forced JSON)

`tool_choice={"type": "tool", "name": "record_kpi"}` forces Claude to
emit a structured `record_kpi` tool call rather than free-form text.
This eliminates an entire class of failure (regex-parsing the model's
prose), keeps cache keys deterministic, and makes the cost of a "value
not reported" outcome explicit (`value=null, unit=null`) instead of
implicit.

### 2.4 Overriding `reporting_year` to `report["report_year"]`

Once it became clear the gold convention was "use the filename year
regardless of which column the value comes from", the fix was a
one-line override: `reporting_year=report["report_year"]`. The
LLM-reported year is discarded — the only signal we trust from the
LLM is the value and the unit. This unblocked the correct extractions
that had been counting as false positives.

### 2.5 Caching keyed on `(model, kpi_key, user_prompt)`

The LLM extractor caches each tool-input dict to disk under a
SHA256 hash of `f"{model}|{kpi_key}|{user_prompt}"`. Cache hits skip
the API call entirely. After one full run completes, re-running the
pipeline costs zero API tokens — useful for iterating on the
*evaluation* logic (e.g. the `reporting_year` override fix) without
re-spending on the *extraction*.

### 2.6 End-to-end results across all 5 companies

A full run against all 5 reports × 3 KPIs (15 cells) was completed
with the page-level retrieval LLM extractor. Headline numbers vs the
hand-labelled gold:

| Extractor | KPI | TP | FP | FN | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | scope_1_emissions | 0 | 0 | 5 | 0.00 | 0.00 | 0.00 |
| baseline | total_energy_consumption | 0 | 1 | 4 | 0.00 | 0.00 | 0.00 |
| baseline | water_consumption | 0 | 0 | 5 | 0.00 | 0.00 | 0.00 |
| **llm** | **scope_1_emissions** | **3** | **0** | **2** | **1.00** | **0.60** | **0.75** |
| **llm** | **total_energy_consumption** | **4** | **1** | **0** | **0.80** | **1.00** | **0.89** |
| **llm** | **water_consumption** | **1** | **1** | **3** | **0.50** | **0.25** | **0.33** |

**Aggregate LLM**: 8 / 15 correct (macro-F1 ≈ 0.66; precision 0.77,
recall 0.62). The baseline contributed 0 true positives — the single
non-null extraction was the wrong row (renewables production instead
of total energy).

Per-cell breakdown:

| Company | scope_1 | total_energy | water |
| --- | --- | --- | --- |
| BP | ✅ 33.7 M tCO2e | ✅ 134.4 M MWh | ❌ picked withdrawal (82 M m³) instead of consumption (47.3 M m³) |
| Enel | ✅ 18.95 M tCO2e | ✅ 168.6 M MWh | ✅ 32.1 M m³ |
| Eni | ❌ none ("no standalone consolidated total") | ✅ 84.4 M MWh | ❌ picked withdrawal (821 Mm³) |
| Iberdrola | ✅ 5.25 M tCO2e | ✅ 101.6 M MWh | ❌ picked withdrawal (1,275 ML) |
| Shell | ❌ picked op-control 46 (gold needs ESRS+non-consolidated 69) | ❌ picked 189 (gold = 269) | ❌ unit "million cubic metres" not canonicalised |

The remaining errors fall into three buckets:

1. **Withdrawal vs consumption confusion** (BP water, Eni water,
   Iberdrola water). The LLM reads "Total freshwater withdrawal" and
   picks it because the value is large and labelled "Total"; the rule
   "reject withdrawal" was added to the system prompt but proved
   insufficient — at least one report repeats the rejected metric so
   prominently that the model still chose it.
2. **ESRS-aligned vs operational-control aggregation** (Shell scope_1,
   Shell total_energy, Eni scope_1). When a report distinguishes
   "consolidated entities only" from "consolidated + operated
   non-consolidated entities (ESRS-aligned)", the gold uses the
   higher ESRS-aligned figure but the LLM either picks the lower
   consolidated-only figure (Shell scope_1: 46 vs gold 69) or returns
   null because no single line matches the canonical phrasing (Eni
   scope_1).
3. **Magnitude-prefixed unit canonicalisation** (Shell water). The
   model returned `unit="million cubic metres"`, which is not in the
   alias table; the unit-rejection branch nullified the otherwise
   correct extraction.

A second iteration of the system prompt (stricter
withdrawal-vs-consumption language; "multiply out magnitude prefixes
yourself, never write a magnitude word in the unit field"; explicit
ESRS-aligned guidance) was prepared but only a partial re-run completed
before the API credit balance was exhausted. The five cached
extractions from earlier iterations are preserved in
`data/cache/llm/*.json`.

---

## 3. Potential improvements

The order roughly tracks expected ROI, highest first.

### 3.1 Better PDF parsing layer (highest impact)

pdfplumber is the bottleneck for the *baseline* path. Two-column
sustainability reports interleave columns in the extracted text, table
detection fragments multi-row tables into many micro-tables, and
subscripts are split off their parent line. A modern document-AI
parser would unlock the baseline:

- **Docling** or **Unstructured.io** preserve column reading order and
  emit structured table objects with row/column spans intact.
- **pdfplumber's table_settings tuning** (explicit_vertical_lines,
  intersection_tolerance, edge_min_length) can be calibrated per
  document layout to reduce fragmentation, though it requires per-PDF
  inspection.
- **OCR fallback via Tesseract** for any page where text extraction
  yields suspiciously few tokens — sometimes "scanned" sustainability
  PDFs land in the corpus and pdfplumber returns nothing.

Without changing extractor logic, switching to Docling would
plausibly take baseline recall from ~0% to >50%.

### 3.2 KPI-specific retrieval queries, not bare KPI names

The current system uses the KPI name as the embedding query
(`"Scope 1 direct greenhouse gas emissions"`). This collides with
narrative pages that describe the methodology rather than report a
value. Better retrieval queries would include known reporting
phrasings:

- For scope_1: `["Total gross Scope 1 GHG emissions", "Scope 1
  (direct) emissions tCO2e"]` — embed both, retrieve union.
- For total_energy: `["Total energy consumption MWh", "Energy
  consumption GWh"]`.
- For water: `["Freshwater consumption million m3", "Water consumption
  m3"]`.

Multi-query retrieval with reciprocal-rank fusion would reduce the
cases where the data page just barely misses top-K.

### 3.3 Hybrid lexical + semantic ranking (BM25 + cosine)

ClimateBERT embeddings cluster narrative ESG pages too tightly (BP's
spread is 0.92-0.97). A BM25 score that rewards rare-token overlap —
e.g. "33.7" or "MtCO2e" appearing on a page — would surface data
pages without needing the unit-presence boost hack. Standard
hybrid retrieval (sparse + dense with weighted sum) is well-studied
and easy to add.

### 3.4 Extend the baseline extractor for "value-as-header" tables

The Enel/Shell pattern where pdfplumber emits `headers=['18.95']` or
`headers=['million MWh', '269']` could be handled with a special
table-shape detector: when a table has a single-cell header that is a
plausible numeric value (or unit + number pair), and an adjacent
sibling table on the same page contains a year header row, fuse them
into a `(label, year, value)` triple. This would lift Enel/Shell
baseline recall without touching the LLM path.

### 3.5 Add `_CO2_LONE_SUBSCRIPT` regex to `normalize_co2`

Cover the BP pattern where the subscript "2" sits alone on its own
line, e.g.:

```
Scope 1 (direct) greenhouse gas MtCOe 33.2 30.4 31.1 32.8 33.7 ...
2
emissions k l GHG – Equity share h,p
```

The regex `(M[tT]|[kKgG][tT]|[tT])?COe[^\n]*\n\s*2(?=\s*\n)` matched
at line boundaries would let the baseline parse unit-bearing values
on these pages.

### 3.6 Multi-pass LLM (retrieve → verify)

A second LLM call to verify each extracted value against the cited
snippet would catch cases where the model picks a sub-component or
out-of-context number. This roughly doubles cost but plausibly takes
precision from "good" to "near-perfect" without extra retrieval work.

### 3.7 Larger gold set + per-company adjudication notes

Five companies × 3 KPIs = 15 hand-labelled values is a tight
evaluation set for a problem this open-ended; adding 15-20 more
companies would let coverage and unit-handling be measured properly,
and would make P/R/F1 numbers less noisy. Recording which line in the
PDF the human used (with line-level citation) would also enable
position-level scoring, not just value-level scoring.

### 3.8 Page-image fallback for charts

Some sustainability reports communicate KPIs primarily through
infographics/charts; the textual data is either absent from the PDF or
locked inside a rasterised graphic. A multimodal LLM call (Claude with
image input on the rendered page) could recover those cases. Cost
impact is non-trivial — probably gated on whichever pages text
extraction returns nothing useful.

### 3.9 Confidence calibration

Both extractors emit a `confidence` score, but the values aren't
calibrated against actual correctness — the baseline uses
`table_sim * row_score`, the LLM emits a self-reported number. A
calibration step (Platt scaling or isotonic regression on a
held-out slice) would let the comparison table flag low-confidence
extractions for human review.

---

## 4. Headline takeaway

For these five reports, **structural pdfplumber-based table
extraction is fundamentally insufficient** — the documents are too
heterogeneous to fit any one schema. The viable path was to treat
extraction as **page-level RAG**: retrieve the most likely page using
embeddings + a unit-presence signal, then let an LLM with a strict
tool-use schema and KPI-specific disambiguation rules read the
page text and pick the right number. The baseline survives in the
final pipeline as a comparison/diagnostic — not because it works,
but because its failure modes illustrate exactly why the LLM path is
needed.

---

## 5. Iteration 2 (2026-04-26)

Implementation plan: `docs/superpowers/plans/2026-04-26-nlp-esg-iteration2.md`.
Design: `docs/superpowers/specs/2026-04-26-nlp-esg-iteration2-design.md`.

### 5.1 What changed

Eight code commits (Tasks 1–14) implemented the design:

1. `parser` field on `ParsedReport`, `run_tag` on `KPIExtraction` —
   foundation for multi-parser dispatch and run-tagged persistence.
2. Per-KPI `queries: list[str]` in the registry — three reporting
   phrasings per KPI taken from observed report wording.
3. `_CO2_LONE_SUBSCRIPT` regex covering the BP "MtCOe ... \n2 \n"
   pattern (§3.5).
4. `_parse_with_pdfplumber` lifted into a private helper.
5. `ingest_docling.parse_with_docling` + dispatcher in
   `parse_pdf` — Docling first, pdfplumber fallback.
6. `rank_pages_cosine` lifted out of `LLMExtractor` into
   `retrieval.py`.
7. `rank_pages_rrf` (multi-query reciprocal-rank fusion).
8. `rank_pages_hybrid` (RRF + BM25 hybrid, both min-max
   normalised, α = 0.5).
9. `LLMExtractor._build_context` switched to multi-query hybrid
   retrieval.
10. `build_index` table-header embedding now includes the first
    five row labels — addresses Eni-style tables where row[0]
    carries the KPI semantics.
11. **System prompt added to the LLM cache key.** This is the
    load-bearing fix: the v1 cache key was
    `(model, kpi, user_prompt)`, so the prompt-v2 work prepared
    in §2.6 silently served v1 responses. The new key is
    `(model, kpi, system_prompt, user_prompt)`.
12. `--run-tag` CLI flag, `data/runs/<tag>/extractions.csv`
    persistence, `build_run_comparison` helper.

Two follow-up commits adapted to environmental constraints:

- File-size guard: skip Docling for PDFs above 15 MB.
- `NLP_ESG_DISABLE_DOCLING=1` env var to disable Docling entirely.

### 5.2 What worked

**Baseline now extracts BP `total_energy_consumption` correctly: 134,448,000 MWh (gold = 134.4 M MWh).** Iteration 1 baseline missed every cell (0/15). The lift came from three changes acting together:

- `_CO2_LONE_SUBSCRIPT` and the existing CO₂ patterns made the BP
  page parseable instead of garbled.
- The hybrid BM25 + RRF page ranker surfaced the right BP page.
  ClimateBERT cosine similarity put the data page in a tight
  cluster around 0.94–0.97 with narrative pages; BM25 picked it
  out by rewarding the exact "MWh" / "134,448" tokens.
- Multi-query retrieval (three phrasings instead of one) added
  enough rank-diversity to keep the data page in the top 12.

**The architecture works end-to-end**: Docling-first dispatch with
pdfplumber fallback, run-tagged extraction persistence, multi-run
comparison join, and the LLM-cache-key fix all landed and are
covered by 99 passing tests.

### 5.3 What did not work

**Docling crashed.** The C++ layout model has a memory bug that
manifests as `std::bad_alloc` past ~28 pages on long PDFs and
escalates to a process-killing `SIGSEGV` on Eni (12 MB) — exit code
139, no Python-level fallback possible.

Mitigation path was three commits deep:

1. Add a quality-check fallback: if more than 50 % of pages come
   back empty after Docling parse, return `None` so the dispatcher
   falls back. Catches the OOM-but-still-returns-something case.
2. File-size guard: skip Docling up-front for files above 15 MB so
   we don't burn 5–10 minutes per file before the quality check
   fires (caught Enel 42 MB and Shell 21 MB).
3. After confirming the segfault still hit on Eni at 12 MB despite
   the size guard, the `NLP_ESG_DISABLE_DOCLING=1` env var was
   added and the v2 run completed with pdfplumber for all 5 PDFs.

The Docling code path remains in the codebase for environments
where the layout model behaves; on this corpus + machine the
fallback is the only working configuration.

**LLM credits exhausted.** Every v2 LLM call returned a 400
`invalid_request_error` ("credit balance is too low"). Every v2 LLM
extraction is `value=None, flags=["api_error"]`. The 8/15 LLM result
from iteration 1 stands as the LLM benchmark; the v2 LLM numbers
in the table below are 0/15 strictly because the API never returned
a non-error response, not because the prompt or retrieval changes
regressed quality.

### 5.4 Headline numbers vs gold

| Run | Extractor | KPI | TP | FP | FN | Precision | Recall | F1 | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v1 | baseline | scope_1 | 0 | 0 | 5 | 0.00 | 0.00 | 0.00 | |
| v1 | baseline | total_energy | 0 | 1 | 4 | 0.00 | 0.00 | 0.00 | |
| v1 | baseline | water | 0 | 0 | 5 | 0.00 | 0.00 | 0.00 | |
| **v2** | **baseline** | **scope_1** | **0** | **0** | **5** | **0.00** | **0.00** | **0.00** | unchanged |
| **v2** | **baseline** | **total_energy** | **1** | **1** | **3** | **0.50** | **0.25** | **0.33** | **+1 TP (BP)** |
| **v2** | **baseline** | **water** | **0** | **0** | **5** | **0.00** | **0.00** | **0.00** | unchanged |
| v1 | llm | scope_1 | 3 | 0 | 2 | 1.00 | 0.60 | 0.75 | cached, blocked from re-run |
| v1 | llm | total_energy | 4 | 1 | 0 | 0.80 | 1.00 | 0.89 | cached, blocked from re-run |
| v1 | llm | water | 1 | 1 | 3 | 0.50 | 0.25 | 0.33 | cached, blocked from re-run |
| v2 | llm | scope_1 | 0 | 0 | 5 | 0.00 | 0.00 | 0.00 | API errored on every cell |
| v2 | llm | total_energy | 0 | 0 | 5 | 0.00 | 0.00 | 0.00 | API errored on every cell |
| v2 | llm | water | 0 | 0 | 5 | 0.00 | 0.00 | 0.00 | API errored on every cell |

Aggregate baseline: 0/15 → **1/15** TP. Aggregate LLM (v1 cached):
**8/15** TP, unchanged.

Per-cell v2 baseline detail:

| Company | KPI | v2 baseline value | Verdict |
| --- | --- | --- | --- |
| BP | total_energy | 134,448,000 MWh | ✅ matches gold (134.4 M MWh) |
| Iberdrola | total_energy | 88,190,000 MWh | ❌ FP (renewable production, not consumption — same row picked in v1) |
| All other (company × KPI) | — | None | not extracted |

### 5.5 Pipeline state on disk

- `data/runs/v2_no_docling/extractions.csv` — 30 rows
  (5 reports × 3 KPIs × 2 extractors). LLM rows have
  `flags=['api_error']`.
- `data/runs/v2_no_docling/metrics.csv` — the table above (machine-readable).
- `data/cache/{company}_{year}_pdfplumber.pkl` — fresh v2 caches.
- `data/cache/llm/*.json` — five v1 cached responses preserved
  but not consulted by v2 (different cache keys).

### 5.6 What remains for v2 LLM evaluation

When credits are restored, re-running the pipeline should produce
~11–13/15 LLM TPs based on the rule-by-rule analysis in §2.6:

- **Bucket 1 (water consumption)**: BP / Eni / Iberdrola water
  cells should flip green or to honest-null on the new
  withdrawal-vs-consumption rule.
- **Bucket 2 (ESRS aggregation)**: Shell scope_1 should flip from
  46 → 69 ESRS-aligned. Eni scope_1 may stay null if no
  consolidated total exists.
- **Bucket 3 (magnitude prefixes)**: Shell water and Shell
  total_energy should flip green on the multiply-out-magnitude
  rule.

Re-run command (one line, no code change needed):

```
NLP_ESG_DISABLE_DOCLING=1 python -m nlp_esg.pipeline --run-tag v2_llm
```

The cache-key fix guarantees these will be fresh API calls under
prompt v2 — the iteration-1 cached responses will not silently be
served.

### 5.7 Headline takeaway for this iteration

The retrieval and normalisation work landed and demonstrably
unlocked the first baseline TP this corpus has ever produced. The
LLM-side improvements are wired and tested but unobservable in this
run because of the API credit constraint; the cache-key fix means
the v1 cached responses will not contaminate the v2 evaluation
when credits return. Docling did not work on this corpus + machine
and is now opt-in via env var rather than load-bearing.
