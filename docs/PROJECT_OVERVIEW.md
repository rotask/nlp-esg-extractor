# Project Overview — NLP ESG KPI Extraction

A plain-English walkthrough of what the project does, how it does it, what
results we got, and where it failed. Written for colleagues across mixed
backgrounds; intended as a primer to help draft the final report.

---

## 1. The problem in one paragraph

Big companies publish annual **Sustainability/ESG reports** — long PDFs (200–700
pages) describing their environmental performance. Three numbers in those
reports matter for almost every comparison: **Scope 1 emissions** (direct
greenhouse gases the company emits, in tonnes of CO₂-equivalent),
**total energy consumption** (in MWh), and **water consumption** (in m³).
Today an analyst has to read each report by hand to pull these out. The
documents have no shared schema — values can sit in a table, in a paragraph,
or even inside an infographic. We built a system that automates this
extraction for five companies (BP, Shell, Enel, Eni, Iberdrola; FY2024) and
measures how accurate it is against numbers we labelled by hand.

---

## 2. Pipeline workflow (read this first)

The system is one command (`python -m nlp_esg.pipeline`) that runs five
stages in order. Each stage feeds the next.

```
                         ┌──────────────────────┐
                         │  ESG report PDFs     │
                         │  data/reports/*.pdf  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                ┌──────────────────────────────────────┐
       (1)      │  INGEST                              │
                │  - Try Docling (smart parser);       │
                │    fall back to pdfplumber.          │
                │  - Output: pages of text + tables.   │
                └──────────────────┬───────────────────┘
                                   │
                                   ▼
                ┌──────────────────────────────────────┐
       (2)      │  CLEAN & INDEX                       │
                │  - Fix CO₂ subscript glitches.       │
                │  - Embed every sentence and table    │
                │    header with ClimateBERT.          │
                │  - Cache to disk (slow first run,    │
                │    instant after).                   │
                └──────────────────┬───────────────────┘
                                   │
                                   ▼
                ┌──────────────────────────────────────┐
       (3)      │  RETRIEVE relevant pages             │
                │  - For each KPI, ask 3 phrasings.    │
                │  - Combine semantic similarity       │
                │    (ClimateBERT) + keyword score     │
                │    (BM25). Take top ~16-25 pages.    │
                └──────────────────┬───────────────────┘
                                   │
                ┌──────────────────┴───────────────────┐
                ▼                                      ▼
   ┌──────────────────────────┐          ┌──────────────────────────┐
(4a)│  BASELINE EXTRACTOR      │       (4b)│  LLM EXTRACTOR           │
   │  Deterministic rules:     │          │  Send the top pages to a │
   │   - Search tables first.  │          │  large language model    │
   │   - Else scan lines for   │          │  (Gemini 2.5-flash)      │
   │     KPI keywords + units. │          │  with strict JSON tool   │
   │   - Pick latest-year col. │          │  schema and KPI rules.   │
   │   - Reject "withdrawal",  │          │  Cache responses on      │
   │     "renewable" etc.      │          │  prompt-hash.            │
   └──────────────┬────────────┘          └──────────────┬───────────┘
                  │                                      │
                  └────────────────┬─────────────────────┘
                                   ▼
                ┌──────────────────────────────────────┐
       (5)      │  NORMALISE & EVALUATE                │
                │  - Convert all values to canonical   │
                │    units (tCO2e, MWh, m³).           │
                │  - Compare to hand-labelled gold;    │
                │    write metrics.csv (P / R / F1).   │
                │  - Persist extractions.csv +         │
                │    LLM prompt logs for audit.        │
                └──────────────────────────────────────┘
```

**Key idea:** we don't ask the system to read 700 pages. Stage 3 picks
~20 candidate pages; stages 4a/4b only see those. The two extractors are
independent and run side-by-side so we can compare their answers.

---

## 3. The headline result

Five companies × three KPIs = **15 gold cells**.

| Extractor                          | Correct (TP) | F1   | What it is |
|------------------------------------|--------------|------|------------|
| Deterministic baseline             | 12 / 15      | 0.88 | Rules + regex over retrieved pages |
| LLM (Gemini 2.5-flash)             | 12 / 15      | 0.88 | Sends pages to a language model with strict instructions |
| **Best-of-either (combined)**      | **14 / 15**  | **0.96** | Take the correct answer from whichever extractor got it |

The two extractors fail on **different** cells, so combining them recovers
14 of 15. The single unrecovered cell is **Shell water (gold = 26 M m³)** —
the value lives inside a chart/infographic that no text-based parser can read.

For context, the baseline started at **0/15** in iteration 1 and reached
12/15 only after several rounds of debugging (recorded in `docs/FINDINGS.md`).
This document focuses on the final state, not the journey.

---

## 4. Technical implementation — the pieces explained

### 4.1 Ingest (turning PDFs into text)

PDFs are messy. Words and numbers come back in odd orders, columns get
interleaved, and special characters like the subscript "2" in CO₂ split
into separate tokens. We use two parsers:

- **Docling** — IBM's modern document AI. Preserves table structure
  cleanly when it works. Unfortunately on our development machine it
  segfaulted on long PDFs, so we ship a flag (`NLP_ESG_DISABLE_DOCLING=1`)
  to skip it.
- **pdfplumber** — older, simpler library. Less structured output but
  reliable. This is the working configuration on our hardware.

Output: a `ParsedReport` containing each page's text and any detected tables.

### 4.2 Cleaning the text

Two specific PDF artefacts caused real bugs:

- **CO₂ subscript splitting.** PDF renderers often store `CO₂e` as three
  glyphs: `CO`, then a baseline-shifted `2`, then `e`. pdfplumber returns
  these as `"CO 2 e"` or even with `"2"` on the next line. We have four
  regex passes (`normalize_co2`) that stitch them back together. One of
  them ("lone subscript on its own line") is specific to BP's columnar
  ESG datasheet.
- **Number/unit rendering quirks.** `"millionm 3"` (where the m³ broke
  apart), `"(MWh) 84,399,860"` (parenthesised unit before the number),
  `".000 m3"` (orphan thousands marker). The `parse_value` function in
  `normalize.py` handles these patterns explicitly.

### 4.3 Retrieval (finding the right page)

Embedding every sentence and table header in 700 pages and comparing to
a query gets us close but not perfect. We use a **hybrid** approach that
combines two signals:

- **Semantic similarity** via **ClimateBERT** embeddings (a 768-dim
  language model trained on climate-domain text). For each KPI we use
  three query phrasings (e.g. *"Total gross Scope 1 GHG emissions"*,
  *"Scope 1 (direct) emissions"*) and merge their rankings via
  Reciprocal Rank Fusion.
- **BM25** — classic lexical scoring that rewards rare-token overlap.
  Pages containing literal tokens like `"MWh"` or `"33.7"` get a boost.

Both scores are normalised to [0, 1] and averaged (α = 0.5). The top 25
pages go to the baseline; the top 16 go to the LLM.

We also tested a smaller general-purpose embedding model (**MiniLM**)
and found ClimateBERT wins by **+1 TP** — entirely because it recognises
domain phrases like *"Operational control boundary"* as similar to
*"Total energy consumption"* even though no words overlap.

### 4.4 The baseline extractor (deterministic rules)

Two paths:

1. **Table-first.** Find a table whose header embedding matches the KPI
   query above a threshold (cosine ≥ 0.55). Look at the row labels for
   one matching the KPI. Identify the year column (latest year). Read
   the value, parse the unit, convert to canonical units.
2. **Line-scan fallback.** When no table row works, scan every line on
   the top retrieved pages. For each line, check: does it have a
   KPI-related keyword? Does it have a numeric value with a recognised
   unit? Does it contain a forbidden token (e.g. "withdrawal" when we
   want "consumption")? If a year header is nearby, pick the column
   for the most-recent year. Pick the highest-scoring line across all
   top pages, with a tiebreak that prefers the *largest* candidate
   value (consolidated totals beat segment sub-totals).

Crucial detail: each KPI carries **negative tokens** — words that
disqualify a match. For water consumption these include `withdrawal`,
`discharge`, `recycled`. For total energy: `renewable`, `produced`,
`intensity`. These rules came from observing actual misextractions.

### 4.5 The LLM extractor

The same retrieved pages get sent to a Large Language Model
(**Gemini 2.5-flash** is the canonical version; Anthropic Claude is
also wired up). Three design choices matter:

- **Strict tool-use schema.** We force the model to call a function
  named `record_kpi(value, unit, reporting_year, source_snippet,
  confidence)` instead of writing free-form prose. This eliminates
  parsing errors and makes outputs auditable.
- **Disambiguation rules in the system prompt.** We tell it explicitly:
  *Pick "Total gross Scope 1" not Scope 2 or methane-only. Pick
  consumption not withdrawal. Pick the latest year. Don't multiply
  prefixes — write `168.59 TWh`, don't compute the MWh value yourself.*
- **Cache keyed on the full prompt.** SHA-256 of `(model | KPI |
  system_prompt | user_prompt)`. Edits to the prompt automatically
  invalidate stale cached responses — without this we silently served
  v1 answers under v2 rules (the load-bearing bug from iteration 2).

Every LLM call is logged to `data/runs/<tag>/llm_prompts/*.json` with
the retrieved pages, the full prompt, and the raw tool response. This
makes any failure debuggable from disk without re-running the API.

### 4.6 Normalisation and evaluation

Values come back in different units (GWh, TWh, ML, Mm³, etc.). A small
unit-alias table converts each to the canonical unit (MWh, m³, tCO₂e).
A prediction is **correct** if, after canonicalisation:

- the unit family matches gold,
- the reporting year matches gold,
- `|predicted − gold| / gold ≤ 0.01`.

Evaluation produces precision, recall, and F1 per `(extractor, KPI)`
and writes `metrics.csv`.

---

## 5. Results in detail

### 5.1 Per-KPI performance (best LLM run, gemini-2.5-flash)

| KPI                       | Baseline TP | LLM TP | Best-of-either |
|---------------------------|-------------|--------|----------------|
| Scope 1 emissions         | 3 / 5       | 5 / 5  | 5 / 5          |
| Total energy consumption  | 5 / 5       | 4 / 5  | 5 / 5          |
| Water consumption         | 4 / 5       | 3 / 5  | 4 / 5          |
| **Total**                 | **12 / 15** | **12 / 15** | **14 / 15** |

**Total energy consumption is solved.** Both extractors get 4–5/5;
combined, all five companies are correct.

### 5.2 What each extractor uniquely contributes

- The **baseline** uniquely solves cells where the value is in a
  cleanly-structured table or labelled line. Its strength is determinism
  — same input, same output, no API cost, fully auditable rules.
- The **LLM** uniquely solves **Iberdrola Scope 1** (gold = 5,246,890 tCO₂e).
  This value lives on a row that pdfplumber flattens into bare numbers
  with no nearby label: `2 5,179,674 5,246,890 1.3 N/AV. N/AV.`. Only
  the LLM, by reading the section heading several lines above, can
  associate the row with "Gross Scope 1 GHG emissions" and pick the
  right column for 2025.

This is the value of running both — they fail on different cells.

### 5.3 Cost and latency

| Component             | Time / cost                                    |
|-----------------------|------------------------------------------------|
| First-time embedding  | 5–15 minutes per long PDF (CPU, one-off)      |
| Subsequent runs       | Seconds (cached embeddings)                    |
| LLM API (Gemini free) | ~$0 within free tier (10 RPM, 20 RPD)         |
| Full pipeline run     | ~3–5 minutes for all 5 reports × 3 KPIs        |

---

## 6. Error analysis (report-worthy failure modes)

We classify every error into one of three buckets, because each has a
different remedy. This separation is what the facilitator asked for and
is more useful than a single F1 number.

### 6.1 Bucket A — Retrieval errors

The right page never reaches the extractor. The system simply doesn't
see the data.

- **Shell total energy (gold = 269 M MWh).** The gold value sits on
  page 366 of Shell's Sustainability Report. Our hybrid retrieval ranks
  it outside the top 16 pages, so neither the baseline's line scanner
  nor the LLM's prompt contains it. Both extractors instead pick a
  similar-sounding "189 billion kWh" prose figure (operational-control
  boundary, not the ESRS-aligned total).
- **Fix path.** Wider retrieval window (top-25 already helped Enel),
  better KPI query phrasings, or a different ranker. Cheap to try; not
  guaranteed to work because Shell uses very specific ESRS terminology
  in headings.

### 6.2 Bucket B — Extraction errors

The right page *is* in context, but the extractor picks the wrong
number. Three sub-types:

- **Definition ambiguity (most common).**
  - *Withdrawal vs. consumption.* All five companies report water
    figures multiple times — once as "withdrawal" (water taken in),
    once as "consumption" (water actually used). Gold is consumption.
    The smaller LLM (`flash-lite`) picked withdrawal in 4/5 water
    cells despite explicit rules. The larger model (`flash`) follows
    the rule on 3 of those 4. The baseline avoids it via negative
    tokens (`withdrawal`, `discharged`, `recycled`).
  - *Operational control vs. ESRS-aligned aggregation.* Shell reports
    "Scope 1 = 46 Mt" (consolidated entities only) and "ESRS-aligned
    Scope 1 = 69 Mt" (includes operated-but-not-consolidated
    entities). Gold is the larger ESRS figure. The baseline can't
    reach it because no single line says "69"; the LLM can only get
    it with a sufficiently capable model.
- **Wrong year column.** The line contains 5 yearly values; the
  extractor picks the wrong one. The baseline addresses this by
  searching ±25 lines for a year header and picking the column matching
  the most recent year; magnitude is preserved through ratio-scaling.
- **Rule overgeneralisation.** The "ESRS-aligned, prefer larger figure"
  rule was written for Scope 1 / energy. The bigger LLM applied it to
  water too, summing operated + non-operated columns. Gold for Eni
  water uses operated-only. Fix: scope the rule to specific KPIs in
  the prompt.

### 6.3 Bucket C — Normalisation errors

The right value and unit are identified, but the magnitude conversion is
wrong.

- **Enel total energy (flash-lite).** The model correctly read
  `"168.59 TWh"`, correctly chose unit `MWh`, but multiplied by 10⁹
  instead of 10⁶. The plausible-range guard (`[100, 1e9]` MWh) caught
  the value as out-of-range and discarded it.
- **Fix.** Forbid the LLM from multiplying prefixes — accept "168.59 TWh"
  as `value=168.59, unit=TWh` and let our deterministic
  `to_canonical_value` do the conversion. The bigger model (`flash`)
  fixed this on its own; the rule still belongs in the prompt for
  defensive reasons.

### 6.4 Inherent corpus limits (not fixable in code)

- **Shell water (gold = 26 M m³).** The gold value lives inside an
  **infographic** on page 122. No text-based extractor (Docling,
  pdfplumber, or LLM consuming their text) sees the digit "26". The
  caption says *"million cubic metres fresh-water consumption"* but the
  number is rendered as a graphic element. To recover this cell we'd
  need OCR over the rendered page or a multimodal LLM with image input.
- **Iberdrola Scope 1 (baseline only).** The data row in pdfplumber's
  output is `"2 5,179,674 5,246,890 1.3 N/AV. N/AV."` — column-flattened,
  no label on the same line. The LLM solves this by looking at the
  section heading several lines above; the baseline's line-by-line
  scanner can't.

### 6.5 Error breakdown summary

For the canonical run (`v_gemini_25flash_post_quota`, baseline + Gemini
2.5-flash):

| Bucket            | Count | Fixable in code? |
|-------------------|-------|------------------|
| Retrieval         | 1     | Partially (top_n + queries) |
| Extraction (rules)| 2     | Yes (prompt edits / negative tokens) |
| Normalisation     | 0     | — (already fixed by the bigger model) |
| Corpus limit      | 1     | No without multimodal/OCR |

The headline 14/15 (best-of-either) is therefore **not a blunt
aggregate** — it is the union of two extractors whose failures are
structurally different. One mistake (Shell water) is an artefact of the
input data, not the algorithm.

---

## 7. What this means for the writeup

Three takeaways are worth highlighting in the final report:

1. **Hybrid retrieval matters more than any single component.** Going
   from semantic-only to BM25+semantic is what unlocked the baseline.
   Going from one query phrasing to three (with rank fusion) is what
   surfaced the right pages reliably.
2. **The baseline and the LLM are complementary, not redundant.** They
   fail on different cells. Reporting only the LLM number hides the
   fact that the baseline solves Shell total-energy cleanly while the
   LLM doesn't (retrieval-bounded), and the LLM solves Iberdrola
   Scope 1 cleanly while the baseline can't (column-flattening).
3. **Model tier matters but isn't a substitute for engineering.**
   Switching `gemini-2.5-flash-lite` → `gemini-2.5-flash` lifted the
   LLM stream from 8/15 to 12/15 with **no code change**. But the
   remaining errors are a mix of retrieval limits, prompt-rule edge
   cases, and corpus limits — none of which a bigger model fixes for
   free.

---

## 8. Where the canonical artefacts live

- `data/runs/v9_magnitude_tiebreak/` — best baseline-only run.
- `data/runs/v_gemini_25flash_post_quota/` — best LLM run; this is the
  one referenced by the headline 12/15 LLM and 14/15 best-of-either.
- `data/runs/v_gemini_post_quota/` — flash-lite version retained for
  the error-mode discussion (its failures are pedagogically clearer).
- `data/labels/gold_labels.csv` — 15 hand-labelled values with source
  page numbers and adjudication notes.
- `data/runs/<tag>/llm_prompts/*.json` — full prompt + retrieved pages
  + tool response for every LLM call. Sufficient to reproduce any
  cited failure without re-querying the API.

For deeper detail see `docs/FINDINGS.md` (full iteration history) and
`docs/SUSTAINABILITY.md` (impact, scalability, ethics).
