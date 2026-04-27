# Sustainability Assessment

This document is the Sustainability Assessment deliverable for the
NLP ESG KPI Extraction project (Team B). It addresses the five items
required by the project completion checklist: impact metrics, UN SDG
alignment, scalability, future work, and ethical considerations. All
numeric claims are either grounded in the committed run artefacts
under `data/runs/`, derived by the back-of-envelope math shown
in-line, or explicitly flagged as estimates.

## 1. Why this project exists

Corporate ESG (Environmental, Social, Governance) disclosures are
the primary public mechanism by which companies report their
environmental performance — greenhouse gas emissions, energy use,
water consumption, waste, biodiversity, and so on. Under the EU
Corporate Sustainability Reporting Directive (CSRD, in force from
financial year 2024 for large undertakings) and its accompanying
European Sustainability Reporting Standards (ESRS), most large EU
companies are now legally required to publish these figures in
annual sustainability reports.

In practice these reports are 200–700-page PDFs with no shared
schema. Definitions, units, scopes, and consolidation boundaries
vary across companies; the same figure can appear once in a
narrative paragraph, once in a table on a different page, and once
inside an infographic image. Today, comparing a single KPI across
issuers requires hand-extraction by an analyst — locate the page,
validate the unit and reporting boundary, transcribe, and convert.
This is slow, error-prone, and does not scale to the corpus sizes
needed for sector-level or index-wide analytics.

This project automates the *extraction* layer for three numerical
KPIs (Scope 1 emissions, total energy consumption, water
consumption) on a five-report evaluation corpus (BP, Shell, Enel,
Eni, Iberdrola, FY2024). It is deliberately framed as a *building
block* for downstream analytics — not an analyst replacement.
Cells that materially affect regulatory filings or investment
decisions still require human review; the system's job is to
narrow the search space and propose normalised values with
auditable source snippets.

## 2. Impact metrics

Five concrete, measurable metrics are defined below. For each, the
subsections specify (a) how it is measured today on the in-repo
corpus, (b) the threshold at which the project's status would
change, and (c) what data would be needed to re-measure it on a
new corpus.

### 2.a Extraction accuracy (F1)

**Definition.** Macro F1 of extractor predictions against
hand-labelled gold values in `data/labels/gold_labels.csv`,
computed by `src/nlp_esg/evaluate.py` per `(extractor, KPI)` and
aggregated.

**Today.** From `data/runs/v_gemini_25flash_post_quota/metrics.csv`
(15 gold cells = 5 reports × 3 KPIs):

| Extractor                           | TP    | F1   |
|-------------------------------------|-------|------|
| Deterministic baseline              | 12/15 | 0.88 |
| LLM (`gemini-2.5-flash`)            | 12/15 | 0.88 |
| Best-of-either (baseline ∪ LLM)     | 14/15 | 0.96 |

**Threshold for "production ready" (proposed).** F1 ≥ 0.95 with
per-KPI recall ≥ 0.9 on a corpus of at least 30 reports × 3 KPIs
(= 90 cells). A single best-of-either run currently meets the
F1 ≥ 0.95 bar but the corpus is too small for the bar to be
statistically meaningful — at n = 15 the difference between 0.88
and 0.96 is two cells.

**To re-measure on a new corpus.** For each new (company,
report-year) pair: drop the PDF into `data/reports/` named
`{company}_{year}.pdf`; add three rows to `gold_labels.csv` (one
per KPI, with the printed page number, value, and unit); rerun
the pipeline. The harness handles everything else.

### 2.b Analyst time saved per report

**Definition.** Hand-extraction time per `(KPI, report)` cell minus
machine processing time per cell, summed across the corpus.

**Today (estimated).** A baseline of ~30 minutes per cell for a
human analyst is a rough order of magnitude — locate the page in a
500-page PDF, confirm reporting boundary (operational vs ESRS,
withdrawal vs consumption), convert units, transcribe. The
machine pipeline processes 5 reports × 3 KPIs end-to-end in roughly
3–5 minutes when the embedding cache is warm (verified by timing
`v_gemini_25flash_post_quota` reruns; cold first-ingest of a single
500-page PDF takes 5–15 minutes on CPU because of ClimateBERT
embedding computation, and is amortised across all subsequent
runs by the disk cache).

Speedup math, end to end, on the 15-cell corpus once the index
cache is warm:

```
human:   15 cells × 30 min/cell  = 450 min  (~7.5 hr)
machine:  5 min / 15 cells       = 0.33 min/cell
speedup: 30 min / 0.33 min       ≈ 90×
```

The 30-minute-per-cell figure is an estimate. A formal measurement
would require timing several analysts on the same corpus blind, and
is out of scope for the coursework.

**Threshold.** A 10× speedup at maintained accuracy is the threshold
at which automated first-pass extraction becomes operationally
attractive (the "review the 5% disagreements" workflow displaces
the "transcribe everything" workflow). The current best-of-either
run clears that bar.

**To re-measure on a new corpus.** Time the cold-ingest run
(`time python -m nlp_esg.pipeline ...`) and divide by cell count.
The human side requires a separate study.

### 2.c Cost per extraction

**Definition.** Marginal cost per `(KPI, report)` cell of running
the LLM stream, in USD. The baseline stream is free at inference
time (deterministic regex + cached embeddings), so this metric
applies to the LLM extractor.

**Today, by tier.**

*Free tier (Gemini 2.5 Flash, 20 requests-per-day per project):*
the headline `v_gemini_25flash_post_quota` run is $0. The 15-cell
corpus fits within the daily quota (15 calls), and there is no
per-token charge on the free tier. This is the run reproduced by
the headline 14/15 best-of-either result.

*Paid tier (Anthropic `claude-sonnet-4-6`):* per-cell back-of-envelope:

```
context size:   ~10 K input tokens  (top-12 pages × ~800 chars/page,
                                     ~4 chars/token after compaction)
output size:    ~200 output tokens   (tool-use record_kpi arguments)
input price:    $3.00 per 1 M input tokens   (Claude Sonnet 4.6, public
                                              Anthropic pricing as of
                                              2026-04; verify current rate
                                              before committing budget)
output price:   $15.00 per 1 M output tokens (same source)

per cell:       10_000 × 3/1_000_000  + 200 × 15/1_000_000
              = $0.030 + $0.003 ≈ $0.033

15-cell corpus: ~$0.50 per full run.
```

*Paid tier (Gemini 2.5 Flash, beyond free quota):* roughly an order
of magnitude cheaper than Sonnet 4.6 on input tokens; ballpark
~$0.05 per 15-cell run. (Pricing varies; not relied on for the
headline run.)

**Comparison to analyst hour.** At an estimated $15–50/hour fully
loaded analyst cost (rough order of magnitude — depends on region,
seniority, and whether overheads are counted) and ~30 min/cell, the
human cost is $7.50–25 per cell. The paid LLM stream at ~$0.03/cell
is therefore ~250–800× cheaper; the free tier is unbounded-times
cheaper but throughput-capped.

**Threshold.** At 100,000 cells (e.g. an index-wide annual run),
the paid-tier projection of ~$3,000 must be compared against
storage, infrastructure, and human-review costs to determine
whether automation is net-positive. Below that scale the LLM cost
is rounding error against the dev-and-eval engineering cost.

### 2.d Reproducibility (auditability)

**Definition.** Fraction of LLM extractions for which the prompt,
retrieved pages, and tool response are persisted in
`data/runs/<tag>/llm_prompts/<company>_<year>_<kpi>.json` (per
`docs/FINDINGS.md` §11).

**Today.** 15/15 = 100% on the headline run
(`data/runs/v_gemini_25flash_post_quota/llm_prompts/`). Each JSON
contains `system_prompt`, `user_prompt`, `retrieved_pages`,
`tool_response.value`, `tool_response.unit`,
`tool_response.source_snippet`, and `from_cache`. This means every
LLM cell can be audited against the source PDF without re-running
the API.

The baseline extractor's cells include a `source_snippet` and
`source_page` in `extractions.csv` (verified by inspecting
`v_gemini_25flash_post_quota/extractions.csv`), giving the same
auditability for the deterministic stream.

**Threshold.** 100% is the only acceptable bar for an extraction
pipeline whose outputs may flow into regulated reporting. The
project meets it.

**To re-measure on a new corpus.** Confirm the
`llm_prompts/` directory contains one file per
`(company, report_year, kpi)` tuple, and that
`extractions.csv` has a non-empty `source_snippet` column for
every non-null prediction.

### 2.e KPI coverage of corpus

**Definition.** Fraction of `(company, KPI)` cells in the corpus
for which any extractor returns a non-null value that matches gold
within tolerance.

**Today.** 14/15 = 93.3% (best-of-either, headline run). The single
unrecovered cell is Shell water consumption: gold is 26 Mm³ inside
an SVG/raster infographic in the Shell 2024 sustainability report,
which no text-based extractor in either stream can read (`docs/FINDINGS.md`
§10.2 and §12.2).

**Threshold.** 95% coverage at 0.95 F1 is the proposed
production-ready bar. Reaching it on this corpus requires either
OCR (to crack the Shell infographic) or relaxing the gold to
accept the page-text "72 Mm³ fresh water consumed" alternative
(which is a different metric — not relaxable).

**To re-measure on a new corpus.** Coverage = (number of cells
with non-null union prediction matching gold) / (number of gold
cells). Computed from `extractions.csv` plus `gold_labels.csv` by
the existing harness.

## 3. UN Sustainable Development Goals alignment

This is a coursework deliverable; alignments below are framed
honestly as topical, not as quantified contributions to the
underlying targets. The SDG target numbers are real and verifiable
on the UN's published list; the wording paraphrases the official
text rather than quoting it verbatim.

### Primary alignment

**SDG 12 — Responsible Consumption and Production.** Target 12.6
addresses encouraging large companies to integrate sustainability
information into their reporting cycle. Mandatory ESG disclosure
under EU CSRD/ESRS is the regulatory expression of that goal in
Europe. Automating extraction does not change *whether* companies
report — that is a regulatory question — but it changes whether
the reported information is *usable* downstream. By converting
unstructured PDF disclosures into machine-readable, normalised,
auditable tuples, the project supports the analytical layer that
turns disclosure into accountability.

### Secondary alignment

**SDG 13 — Climate Action.** Target 13.2 concerns integrating
climate measures into policy and planning. Scope 1 emissions —
direct GHG emissions from owned or controlled sources, defined by
the Greenhouse Gas Protocol — are the foundational input to
corporate climate accounting. Comparable Scope 1 figures across
issuers are a precondition for sector benchmarking, transition-
plan validation, and net-zero progress tracking. The pipeline
extracts one of the three KPIs that matter most here.

### Tertiary alignment

**SDG 6 — Clean Water and Sanitation.** Target 6.4 concerns water-
use efficiency. Water consumption is one of the three KPIs the
pipeline extracts. The honest qualifier is that the project
measures *disclosed* water consumption, which is a coarse and
boundary-sensitive figure (operational vs financial-control vs
ESRS-aligned, withdrawal vs consumption); even on this small
corpus the gold annotator had to make non-trivial disambiguation
calls (see §6 below and `docs/FINDINGS.md` §12.2).

**SDG 7 — Affordable and Clean Energy.** Total energy consumption
(the second KPI) is one input to energy-intensity ratios used in
SDG 7 monitoring at the firm level.

### Adjacent alignment

**SDG 9 — Industry, Innovation and Infrastructure.** The project
contributes — at the coursework scale — a small piece of analytic
infrastructure (a reproducible KPI-extraction harness with
auditable outputs) to the broader research ecosystem around ESG
analytics.

The project does **not** claim alignment to SDGs 1, 2, 3, 4, 5, 8,
10, 11, 14, 15, 16, or 17. Several of those (e.g. SDG 8 on decent
work) appear in the social pillar of broader ESG taxonomies, but
the pipeline only handles environmental KPIs.

## 4. Scalability analysis

### 4.1 What scales linearly

- **Corpus size at fixed KPI count.** Pipeline runtime is
  `O(reports × KPIs)`. Per-report cost on CPU breaks down as: PDF
  parsing (seconds), embedding the report (5–15 minutes on long
  PDFs, the dominant first-ingest term), retrieval per KPI
  (sub-second once embeddings are cached), baseline extraction
  (sub-second), LLM extraction (one provider round-trip, typically
  2–8 seconds at temperature 0).
- **Re-runs on the same corpus.** Both the parsed `ParsedReport`
  and the `IndexedReport` (text + tables + ClimateBERT
  embeddings) are cached on disk under `data/cache/`, keyed on
  `{company}_{year}_{parser}` and
  `{company}_{year}_{parser}_indexed_{model}` respectively. LLM
  responses are cached under `data/cache/llm/` keyed on
  `sha256(model | kpi | system_prompt | user_prompt)`. A second
  run on the same corpus with the same prompts and models is
  dominated by cache hits and finishes in seconds. This was
  observed empirically across the 10+ committed run tags under
  `data/runs/`.

### 4.2 What does not scale yet

- **CPU-only embeddings.** ClimateBERT inference is run on CPU on
  the dev machine. There is no GPU code path. First-ingest of a
  single 500-page PDF can take 5–15 minutes (verified). On a
  corpus of 100 reports this becomes ~10–25 hours of cold-start
  embedding work — annoying, but a one-time cost amortised across
  all subsequent runs.
- **LLM rate limits.** Gemini's free tier caps at 20 requests per
  day per project, which is sufficient for 5 reports × 3 KPIs = 15
  calls but not for 100 reports × 3 KPIs = 300 calls per day.
  Sustained operation at corpus sizes above ~6 reports/day requires
  a paid tier.
- **Gold-label statistical power.** With only 15 gold cells, the
  difference between "F1 = 0.88" and "F1 = 0.96" is two cells. Any
  claim about per-KPI accuracy or per-extractor superiority is
  underpowered until the gold set grows to ~100 cells. This is the
  single biggest credibility limit on the headline numbers.

### 4.3 What needs structural change to scale

- **KPI ontology.** Three KPIs are hard-coded in `src/nlp_esg/config.py`
  via the `KPIS` registry, each with `queries`, `unit_family`,
  `plausible_range`, and `negative_tokens`. Adding new KPIs (Scope 2,
  Scope 3, waste, biodiversity, social KPIs) requires extending the
  registry — no model retraining, no architectural change — but
  every new KPI also needs ~10–20 hand-labelled gold cells to
  establish reliable per-KPI F1.
- **Multi-language corpora.** ClimateBERT is English-only. German
  DAX issuers and French CAC 40 issuers often publish in their
  respective national languages. Supporting them requires either
  per-language embedding models (multilingual MiniLM, multilingual
  BGE, etc.) or a translate-then-extract pipeline. Both options
  introduce normalisation risk in the unit and definition layer.
- **PDF heterogeneity.** This is the single biggest ingest blocker.
  pdfplumber flattens column-split table rows in inconsistent
  ways (the `eni`-style tables with KPI labels in `row[0]` were
  the recurring case driving normalisation work; see
  `docs/FINDINGS.md` §10.2). Docling's C++ layout model SIGSEGVs
  on long PDFs on the dev machine — fine on server-class hardware
  but unstable enough that the headline runs use
  `NLP_ESG_DISABLE_DOCLING=1`. A robust ingest layer that combines
  Docling-grade layout with deterministic fallback is the highest-
  leverage scaling work outside the LLM stream.
- **Definitional drift.** As the gold set grows beyond a single
  reporting year and a single regulatory regime (ESRS), the
  "which Scope 1?" question proliferates: ESRS-aligned vs
  operational-control vs financial-control, equity-share vs
  consolidated, etc. The current system prompt encodes one set of
  preferences (`pick the LARGER ESRS-aligned figure`); scaling
  beyond European 2024 reports requires a more flexible
  rule-engine or per-regime prompt variants.

### 4.4 Cost and throughput projection

Ballpark scaling, holding KPI count at 3 and assuming the embedding
cache is warm for re-runs (which it would be in production
operation):

| Corpus size       | Cold-ingest wall-clock      | Pipeline wall-clock (warm) | LLM cost (paid Sonnet 4.6) | LLM cost (paid Gemini 2.5 Flash) |
|-------------------|-----------------------------|----------------------------|----------------------------|---------------------------------|
| 5 reports (today) | ~30–60 min                  | ~3–5 min                   | ~$0.50                     | ~$0.05                          |
| 100 reports       | ~10–20 hr                   | ~1–2 hr                    | ~$10                       | ~$1                             |
| 10,000 reports    | ~50–80 days (CPU only)      | ~5–10 days                 | ~$1,000                    | ~$100                           |

*Assumptions made for the 10,000-report row, all rough:*
1. Linear scaling of per-report wall-clock (verified up to 5
   reports; not verified at 10,000).
2. Per-cell LLM cost held at the §2.c estimates (~$0.033 Sonnet,
   ~$0.003 Gemini Flash). Actual production costs will diverge
   with prompt-caching discounts, batching, retries, and provider
   pricing changes.
3. CPU-only inference; GPU acceleration would compress the
   cold-ingest column substantially.
4. No rate-limit headroom included. A real 10,000-report run needs
   either Anthropic enterprise quota or Gemini paid quota.

## 5. Future work

Ordered roughly by ROI on the headline accuracy and coverage
metrics, highest first.

1. **Targeted prompt fix for `eni` water rule-overgeneralisation.**
   `docs/FINDINGS.md` §12.2 documents that `gemini-2.5-flash`
   over-applies the "pick the LARGER ESRS-aligned figure" rule
   from Scope 1 to water consumption, summing the operated and
   non-operated columns to 54 instead of taking the operated 42.
   A one-line system-prompt edit limiting that rule to scope_1
   and total_energy should fix this cell at zero infrastructure
   cost. Highest ROI (one cell, zero cost, deterministic).
2. **OCR fallback for Shell-style infographics.** The single
   unrecovered cell across all extractors is Shell water
   (gold 26 Mm³, embedded in an SVG/raster infographic). Adding
   Tesseract or Docling-OCR to the ingest layer for image-only
   regions of pages flagged by the layout model would lift
   best-of-either from 14/15 to 15/15 in principle. Pre-condition:
   stable Docling.
3. **Stable Docling integration on production hardware.** The
   layout model SIGSEGVs on long PDFs on the dev machine but works
   on server-class hardware. Deploying on stable infrastructure
   is expected to lift table-extraction quality on the 30–40% of
   table cells where pdfplumber currently flattens columns. This
   is the highest-leverage *upstream* fix for both extractors.
4. **Eval harness expansion.** Currently per-`(extractor, KPI)` F1
   only. Concrete additions: per-`(company, KPI)` breakdown, an
   "best-of-either" oracle row that the harness emits directly so
   the headline 14/15 number is reproducible from
   `metrics.csv` alone (it is currently computed by hand by reading
   `extractions.csv`), and automatic error-mode classification
   using the §10 normalisation/retrieval/extraction taxonomy.
5. **KPI registry expansion.** Add Scope 2 (location- and
   market-based variants), Scope 3 (the 15 categories from the
   GHG Protocol), waste, and biodiversity indicators. Each new
   KPI needs a registry entry and 10–20 hand-labelled gold cells.
   Low risk to existing KPIs because the registry isolates
   per-KPI configuration.
6. **Multi-language support.** Either swap ClimateBERT for a
   multilingual sustainability-tuned embedding model, or add a
   per-language router that delegates to MiniLM-multilingual on
   non-English reports. Scoping work; not on the critical path
   for the English ESG corpus.
7. **Active-learning gold expansion.** When the pipeline ingests
   a new corpus, surface low-confidence or extractor-disagreement
   cells for human review and roll the resolved values into the
   gold set. Closes the loop on §4.2's "gold-label statistical
   power" gap.
8. **Production interface.** The current surface is CLI plus a
   Python library (see `docs/API.md`). A small REST API or a
   Streamlit app would broaden access for non-developers. Not
   on the accuracy critical path.

## 6. Ethical considerations

### 6.1 Hallucination risk

Large language models can confidently emit values that do not
appear in the source. Three project mitigations make this
tractable but not eliminated:

- **Strict tool-use schema.** The LLM extractor uses the provider
  SDK's tool-use mode with `tool_choice` forcing the
  `record_kpi` tool, whose schema requires a `source_snippet`
  string field. Free-form responses are not accepted.
- **Plausible-range guard.** `src/nlp_esg/normalize.py` carries a
  `plausible_range` per KPI in `config.KPIS`. Values outside the
  range are flagged `out_of_range` and dropped (e.g. the `enel`
  total_energy normalisation slip in `docs/FINDINGS.md` §10.3
  was caught this way: the model produced 168.59 × 10⁹ MWh, which
  the range guard rejected).
- **Prompt-and-context logging.** Every LLM extraction writes its
  full system prompt, user prompt, retrieved page numbers, and
  tool response to disk under `data/runs/<tag>/llm_prompts/`
  (`docs/FINDINGS.md` §11). Every claim is auditable from the
  artefacts alone, without re-running the API.

**Residual risk.** A value that is in-range and accompanied by a
plausible-looking snippet but is still wrong. The clearest
documented case is `eni` water in the `gemini-2.5-flash` run:
the model emitted 54 Mm³ from a snippet showing `Mm3 42 12 45 9`,
having summed the operated and non-operated columns
(`docs/FINDINGS.md` §12.2). The snippet is real, the arithmetic is
defensible under one reading of the prompt, but the gold value is
42 Mm³. This kind of error survives all three mitigations above
and requires either a rule change or downstream human review.

### 6.2 Bias in gold labels

Gold labels are produced by a single annotator. Several cells
involve a real disambiguation choice:

- **Shell Scope 1.** The report distinguishes operational-control
  Scope 1 (46 Mt) from ESRS-aligned Scope 1 including
  non-consolidated entities (69 Mt). Gold is 69 Mt; the system
  prompt instructs the model to prefer the larger ESRS-aligned
  figure.
- **Shell water.** The report distinguishes financial-control
  water consumption (127 Mm³), operational-boundary water
  consumption (26 Mm³), and "fresh water consumed" prose
  (72 Mm³). Gold is 26 Mm³.

Each of these is a defensible value for "Scope 1 emissions" or
"water consumption" under a different reporting boundary. The
annotator's choice is the ground truth used for evaluation, and
the system prompt encodes the same preferences — but a different
annotator following different guidance (e.g. CDP-aligned rather
than ESRS-aligned) could produce different gold labels and
therefore different F1 numbers on the same predictions. This is
a real limit on the absolute meaning of the headline F1.

### 6.3 Environmental cost of LLM inference

Back-of-envelope, for one full 15-cell LLM run:

```
input  tokens per cell:  ~10,000
output tokens per cell:  ~200
cells per run:           15
total tokens per run:    ~153,000  (predominantly input)
```

Public estimates of cloud-LLM CO2 footprint vary widely
(commonly cited ranges run from sub-gram to several grams CO2e per
1 M tokens for Flash-class models; figures depend heavily on the
data centre's grid mix, which the user does not control). Using a
deliberately conservative ballpark of ~5 g CO2e per 1 M tokens for
output and ~0.5 g per 1 M for input on a Flash-class model:

```
input  CO2e:  150_000 / 1_000_000 × 0.5 g  ≈ 0.08 g
output CO2e:    3_000 / 1_000_000 × 5   g  ≈ 0.02 g
total per run: ~0.1 g CO2e   (rough order of magnitude)
```

Compared to the displaced human work — ~7.5 hours of analyst time
on the same 15 cells — even back-of-envelope CO2e accounting
favours the pipeline by orders of magnitude (laptop + monitor +
heating + commute ≈ several hundred grams per analyst-day on most
estimates). The honest framing is: the LLM inference cost is small
enough relative to the hardware-and-life-support costs of human
work that it is not a credible objection to the project under
current grid mixes. This calculus could change if the model tier
shifted to a much larger reasoning-class model, or if the displaced
work were itself much less carbon-intensive than analyst hours.

### 6.4 Dual-use considerations

The pipeline is purely an *extraction* tool — it surfaces values
that are already publicly disclosed by the issuer in their own
sustainability report. Three plausible uses:

- **Audit corporate disclosures for inconsistency.** Comparing
  the same company's reported figures across years, or across
  reports (annual report vs sustainability report vs CDP
  submission). Positive use; the auditability features (§2.d)
  directly support it.
- **Benchmark competitor disclosures.** Comparing peers within a
  sector. Neutral-to-positive; this is what investors,
  regulators, and journalists already do by hand.
- **Generate compliance reports automatically.** A misuse mode if
  the *generation* layer is added downstream. The current scope
  is extraction-only and the system prompt explicitly requires a
  `source_snippet` quoting the input PDF. Generation-from-thin-air
  is not architecturally available in this project. A future
  extension that *fills in missing values* (e.g. predicting Scope
  3 for a company that does not disclose) would cross this line
  and would need an explicit guardrail design.

### 6.5 Data licensing

The evaluation PDFs are publicly published sustainability reports
downloaded from the issuers' investor-relations sites. They are
**not** redistributed in this repository: `data/reports/` is
gitignored. Only five hand-labelled gold rows
(`data/labels/gold_labels.csv`) are committed. The repository
itself is a private coursework deliverable; if it were ever made
public, the licensing of derivatives — cached embeddings, prompt
logs, the gold labels themselves — would need explicit review,
particularly because the gold labels are a small but copyrightable
selection from each issuer's report.

### 6.6 Non-replacement framing

The pipeline is designed as a *first pass*. Analyst review remains
required for any cell that flows into regulatory filings,
investment decisions, or public benchmarking. Concretely:

- 0.88 single-extractor F1 means roughly 12% of single-pipeline
  extractions need correction.
- 0.96 best-of-either F1 still leaves ~4% needing review — and the
  remaining errors are concentrated in the cells where the
  reporting boundary is genuinely ambiguous, which is precisely
  where human judgement is most needed.

The intended deployment posture is: pipeline produces normalised
candidate values with auditable source snippets; analyst reviews
the candidates and either accepts, corrects, or escalates. The
analyst's labour shifts from transcription to adjudication —
fewer cells, harder calls. That shift is the project's real
contribution.

## 7. See also

- [`README.md`](../README.md) — install, usage, headline numbers.
- [`CLAUDE.md`](../CLAUDE.md) — architecture summary and the
  load-bearing invariants you need to know before changing code.
- [`docs/FINDINGS.md`](FINDINGS.md) — full iteration history.
  Particularly:
  - §10 — error analysis split into normalisation, retrieval,
    extraction.
  - §11 — LLM reproducibility (per-KPI prompt logs, the artefacts
    behind §2.d).
  - §12 — `gemini-2.5-flash-lite` vs `gemini-2.5-flash` cell-by-cell
    comparison, the source for the +4 TP delta cited in §1 and §5.
- [`docs/API.md`](API.md) — Python module API reference.
