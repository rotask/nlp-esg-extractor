# NLP ESG KPI Extraction — Design

**Date:** 2026-04-23
**Project:** Team B — NLP track
**Milestone:** Week 7 deliverable

## 1. Goal

Build an NLP pipeline that extracts specific numerical sustainability KPIs from corporate sustainability PDFs and transforms unstructured text into a structured, comparable table across companies.

The question the pipeline answers: *"What sustainability metrics does each company report, and how do they compare?"*

## 2. Scope

### In scope

- 10–15 PDF sustainability reports across 5 companies (report count is config-driven; final number may differ).
- Exactly three KPIs:
  1. **Scope 1 emissions** (canonical unit: tCO₂e)
  2. **Total energy consumption from renewable sources** — absolute quantity, *not* a percentage (canonical unit: MWh)
  3. **Water consumption** (canonical unit: m³)
- Current-year value per report only. No historical trends.
- Two extractors run side-by-side and compared:
  - **Baseline:** semantic retrieval (ClimateBERT embeddings) + regex/pattern matching, table-first with sentence fallback.
  - **LLM:** Anthropic Claude with structured JSON output, consuming the same retrieved context.
- Evaluation on 5 hand-labeled reports using precision / recall / F1 per extractor per KPI, plus coverage (fraction of cells extracted automatically vs. left to manual fallback).

### Out of scope

- No web UI or dashboard.
- No CI pipeline, Docker image, or production deployment.
- No multi-year trends.
- No automated report fetching — PDFs are placed manually under `data/reports/`.
- No OCR for scanned PDFs. If pdfplumber returns little or no text, the report is logged and skipped.
- No automatic summation of subsidiary breakdowns. If a report only gives regional Scope 1 figures with no consolidated total, the extractor emits *not reported*.

## 3. Rubric: what "correct" means

For each (report, KPI) cell, a prediction is correct iff **all** of:

- Unit matches the canonical unit **after normalization** (e.g., 1.2 GWh ≡ 1200 MWh).
- Reporting year matches the gold label.
- `|predicted − gold| / gold ≤ ε`, where `ε = 0.01` (tolerates rounding differences between prose and tables).

`value = None` (i.e., "not reported") is a distinct prediction class:

- Predicting `None` when gold is `None` is a true positive for that class.
- Predicting a value when gold is `None` is a false positive (penalises hallucinations on the LLM side).
- Predicting `None` when gold has a value is a false negative.

Metrics reported: precision, recall, F1 per extractor per KPI, plus coverage (fraction of cells extracted automatically vs. left to manual fallback).

## 4. Architecture

### 4.1 Module layout

```
NLP_ESG_Project/
├── data/
│   ├── reports/           # raw PDFs, named {company}_{year}.pdf
│   ├── labels/            # gold_labels.csv + labels_README.md
│   └── cache/             # pickled parsed PDFs + embeddings; LLM response cache
├── src/
│   ├── config.py          # KPI registry, report list, paths, model names
│   ├── ingest.py          # PDF -> pages + tables (pdfplumber)
│   ├── retrieval.py       # sentence + table-header embeddings, semantic search
│   ├── normalize.py       # unit conversion, number parsing, magnitude words
│   ├── extractors/
│   │   ├── base.py        # abstract Extractor (one method: extract)
│   │   ├── baseline.py    # table-first + sentence fallback + regex
│   │   └── llm.py         # Claude structured JSON extractor
│   ├── evaluate.py        # P / R / F1 / coverage per extractor per KPI
│   └── compare.py         # build the companies x KPIs comparison table
├── notebooks/
│   └── demo.ipynb         # imports src/, shows table + metrics only
├── tests/
│   ├── fixtures/          # canned ParsedReport dicts, canned LLM responses
│   ├── test_ingest.py
│   ├── test_normalize.py
│   ├── test_baseline.py
│   ├── test_llm.py
│   └── test_evaluate.py
├── pyproject.toml
├── .env.example
├── .gitignore
└── README.md
```

### 4.2 Core data contract

A single dataclass flows from extractor → evaluator → comparison table:

```python
@dataclass
class KPIExtraction:
    company: str
    report_year: int                # year the PDF covers (parsed from filename)
    kpi: str                        # "scope_1_emissions" | "renewable_energy" | "water_consumption"
    value: float | None             # None means "not reported"
    unit: str | None                # canonical unit after normalization
    reporting_year: int | None      # year the value itself refers to
    source_snippet: str | None      # table cell text or sentence — for audit
    source_page: int | None
    confidence: float | None        # cosine sim (baseline) or LLM self-reported (llm)
    extractor: str                  # "baseline" | "llm"
    flags: list[str]                # see Section 6
```

Gold labels use the same shape minus `confidence`, `extractor`, `flags`.

### 4.3 KPI registry (`config.py`)

```python
KPIS = {
    "scope_1_emissions": {
        "query": "Scope 1 direct greenhouse gas emissions",
        "unit_family": ["tCO2e", "ktCO2e", "MtCO2e", "t CO2-eq", "t CO2e"],
        "canonical_unit": "tCO2e",
        "plausible_range": (1e2, 1e9),
    },
    "renewable_energy": {
        "query": "Total energy consumption from renewable sources",
        "unit_family": ["MWh", "GWh", "TWh", "GJ", "TJ", "PJ"],
        "canonical_unit": "MWh",
        "plausible_range": (1e2, 1e9),
    },
    "water_consumption": {
        "query": "Total water consumption withdrawal",
        "unit_family": ["m3", "m³", "ML", "megaliters", "kL", "thousand m3"],
        "canonical_unit": "m3",
        "plausible_range": (1e1, 1e10),
    },
}
```

`unit_family` does double duty: it's the regex unit alternation *and* the allow-list used to reject wrong-unit candidates.

### 4.4 Report types

Two shapes flow through the pipeline:

```python
ParsedReport = {
    "company": str,
    "report_year": int,
    "pages": [{"page_num": int, "text": str}],
    "tables": [{"page_num": int, "headers": list[str], "rows": list[list[str]]}],
}

IndexedReport = ParsedReport + {
    "sentences": [{"page_num": int, "text": str, "embedding": np.ndarray}],
    "table_headers": [{"table_idx": int, "header_string": str, "embedding": np.ndarray}],
}
```

`ingest.py` produces `ParsedReport`. `retrieval.py` adds the embedding fields, producing `IndexedReport`. Both shapes are cached to disk.

### 4.5 Extractor interface

```python
class Extractor(ABC):
    @abstractmethod
    def extract(self, report: IndexedReport, kpi_key: str) -> KPIExtraction:
        ...
```

Both `BaselineExtractor` and `LLMExtractor` implement this. Downstream evaluation and comparison code does not know which one produced a given row.

## 5. Data flow

### 5.1 Ingest (`ingest.py`)

`pdfplumber.open(pdf)` produces a `ParsedReport` (shape defined in §4.4). `company` and `report_year` are parsed from the PDF filename. Cached to `data/cache/{company}_{year}.pkl`, invalidating on PDF mtime change. PDF parse failures log at `ERROR` and the report is skipped.

### 5.2 Index (`retrieval.py`)

Per report, one-time:

- Sentence-split page text (`nltk.sent_tokenize` or a regex splitter).
- Embed each sentence using **ClimateBERT** (`climatebert/distilroberta-base-climate-f`) wrapped with mean-pooling via `sentence_transformers.models.Transformer + Pooling`. MiniLM (`all-MiniLM-L6-v2`) stays in config as a toggleable fallback for the embedding-model comparison cell in the notebook.
- Embed each table header row as a single concatenated header string (e.g., `"GHG emissions | Scope 1 | 2024 | tCO2e"`).
- Cache all embeddings alongside the parsed report.

### 5.3 Baseline extract (`extractors/baseline.py`)

For each `(report, kpi)`:

1. **Table-first search**
   - Cosine similarity between the KPI query embedding and each table-header embedding.
   - Tables with sim ≥ `τ_table = 0.55` are considered candidates.
   - Within each candidate, locate the column header matching the report year (or the most recent year if ambiguous), take the numeric cell, and pull the unit from the column/row header where the unit is typically declared.
   - Ties broken by `sim × structural_score`, where `structural_score = 1.0` if the table has at least one header cell that parses as a 4-digit year matching `report_year`, and `0.5` otherwise. Keeps the scoring deterministic and trivial to unit-test.
   - Validate: unit ∈ `kpi.unit_family` **and** value ∈ `kpi.plausible_range`. If valid, accept; `source_snippet = "table@page N: <header> | <cell>"`.

2. **Sentence fallback** (only if step 1 yields nothing)
   - Retrieve top-5 sentences by cosine similarity to the KPI query.
   - For each sentence, apply:
     ```
     NUMBER_RE = r"([-+]?\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?|\d+\.\d+)"
     UNIT_ALT  = "|".join(re.escape(u) for u in kpi.unit_family)
     PATTERN   = rf"{NUMBER_RE}\s*({UNIT_ALT})"
     ```
   - Candidate score = `cosine_sim + year_match_bonus` (bonus applies if the sentence mentions `report_year`).
   - Discard if value ∉ plausible range or unit ∉ family.
   - Pick the best candidate; if none passes, emit `value=None` (not reported).

3. **Normalize** (`normalize.py`) the value to the canonical unit.

### 5.4 LLM extract (`extractors/llm.py`)

Uses the **same retrieval step** to build context: top table pages plus top-k sentences, concatenated with page-number markers. This guarantees the comparison isolates extraction differences, not retrieval advantages.

The context is sent to Claude (Anthropic SDK) with a strict JSON output schema:

```json
{
  "value": "number or null",
  "unit": "string or null",
  "reporting_year": "int or null",
  "source_snippet": "string or null",
  "confidence": "number 0..1"
}
```

Prompt structure:

- **System prompt** (cached via `cache_control`): task description, JSON schema, instructions for "not reported" cases, unit handling rules.
- **User message**: KPI being extracted + the retrieved context for this specific report.

`temperature=0`. One retry on JSON parse failure with a stricter follow-up prompt. API errors retried with exponential backoff up to 3 times before emitting `value=None, flags=["api_error"]`.

### 5.5 Compare and evaluate

- The pipeline extracts from every report (10–15 total). When a company has multiple reports across years, `compare.py` selects the **most recent report per company** for the companies × KPIs comparison table. All reports (including older ones) are retained in the raw extractions DataFrame for auditing and for use by `evaluate.py`.
- `compare.py` pivots the selected extractions into a `companies × KPIs` DataFrame (one per extractor, plus a merged view).
- `evaluate.py` compares predictions against gold labels using the rubric in Section 3. The 5 hand-labeled reports for the gold set may include older reports; the eval scores every (report, KPI) pair in the gold set regardless of whether that report was picked for the comparison table.

## 6. Error handling

### Parsing and IO

- PDF open or parse failure: log `ERROR`, skip the report. The eval treats the missing report as coverage loss, not as "all KPIs wrong."
- Table extraction returns nothing: log `WARNING` and tag every extraction from that report with `flags=["no_tables_extracted"]`.

### Baseline extractor

- No table match and no sentence passes filters: emit `value=None`. The pipeline does not guess.
- Multiple strong candidates: pick the highest combined score; log alternatives in `flags=[f"alternatives={n}"]`.
- Unit not in KPI family: reject the candidate.
- Value out of plausible range: reject and flag `out_of_range` (prevents picking up years as values).
- Only subsidiary breakdowns, no consolidated total: emit `value=None` with `flags=["only_breakdowns_found"]`. No automatic summation.

### LLM extractor

- JSON parse failure: retry once with stricter instruction. If the retry fails, emit `value=None, flags=["parse_failed"]`. No further retries.
- Model returns a value but no unit: reject, emit `value=None, flags=["llm_missing_unit"]`.
- Model returns a value out of plausible range: reject, emit `value=None, flags=["out_of_range"]`.
- API errors: exponential backoff up to 3 retries, then `value=None, flags=["api_error"]`.

### Normalization

- EU decimal comma vs. thousands separator: a comma followed by exactly 3 digits is treated as a thousands separator; otherwise as a decimal. Unit tests cover both conventions.
- Magnitude words (`thousand`, `million`, `billion`): multiplier lookup applied after number parsing, e.g., `"1.2 million m³"` → `(1_200_000, "m3")`.
- Unicode variants of units (`m³` vs `m3` vs `cubic metres`) all normalize to `m3`.
- Energy unit conversions (`1 GJ = 0.2778 MWh`, etc.) round-trip tested.

### Determinism

- Baseline is fully deterministic.
- LLM is called with `temperature=0` but is not bit-exact across runs. Raw LLM responses are cached to `data/cache/llm/{sha256_of_prompt}.json` for reproducibility.

## 7. Testing strategy

The full `pytest` suite (excluding integration) must run in under 10 seconds with no network, no model loading, and no real PDFs.

### Unit tests

- `normalize.py`: unit conversions in both directions; thousand vs. decimal comma; magnitude words.
- `evaluate.py`: metric correctness on synthetic predictions vs. gold; ε-tolerance boundary cases; "not reported" handled as its own class.
- `extractors/baseline.py`: uses **canned `ParsedReport` fixtures** from `tests/fixtures/`. No real PDF parsing inside these tests.
  - Table with a clear KPI row returns the correct value and unit.
  - No table present, narrative-only sentence returns the correct value via fallback.
  - Only subsidiary breakdowns present returns `None` with the right flag.
  - Candidate with unit outside the family is rejected.
- `extractors/llm.py`: patches `anthropic.Anthropic.messages.create` to return canned responses.
  - Valid JSON is parsed correctly.
  - Invalid JSON triggers exactly one retry; second failure returns `None` with `parse_failed`.
  - Missing unit rejected with `llm_missing_unit`.

### Test fixtures

- One small synthetic PDF, either generated with `reportlab` in a conftest or pre-committed, used only by `test_ingest.py` to verify pdfplumber output shape (not accuracy).
- Canned `ParsedReport` fixtures as plain Python dicts.
- Canned Anthropic responses as JSON strings loaded via a pytest fixture.

### Integration tests (skippable)

- `test_integration_llm.py`: one real Anthropic call on a short passage. Skipped unless `ANTHROPIC_API_KEY` is set **and** `RUN_INTEGRATION=1`.
- `test_integration_real_pdf.py`: parses one real report and runs the full pipeline. Skipped if no reports are present.

## 8. Deliverables

### Notebook (`notebooks/demo.ipynb`)

Thin and linear. Each cell does one thing:

1. Imports from `src/`.
2. Loads the report list from config.
3. Runs both extractors over all reports (reads cache where available).
4. Prints the `companies × KPIs` comparison table (one per extractor, plus a merged view).
5. Loads gold labels and computes P / R / F1 / coverage per extractor per KPI.
6. Re-runs the eval with MiniLM instead of ClimateBERT, for the embedding-model comparison in the writeup.
7. Qualitative examples: one case where the baseline wins, one where the LLM wins, one where both fail.

No heavy logic in the notebook. If a cell grows past ~15 lines, the logic moves into `src/`.

### Week 7 artefacts

- `companies × KPIs` comparison table (CSV + rendered markdown).
- Metrics table: precision, recall, F1, coverage per extractor per KPI.
- `data/labels/gold_labels.csv` (5 reports × 3 KPIs, hand-labeled) plus `labels_README.md` documenting labeling conventions.
- Short writeup (inline in the notebook): where baseline wins, where LLM wins, where both fail.

## 9. Environment and reproducibility

### Dependencies (`pyproject.toml`)

```
pdfplumber>=0.11,<0.12
sentence-transformers>=2.7,<3
transformers>=4.40,<5
torch>=2.2
anthropic>=0.34,<1
pandas>=2.2
numpy>=1.26
pytest>=8
python-dotenv>=1
```

Python 3.11+. CPU-only — no GPU required at this scale.

### Secrets

- `.env` holds `ANTHROPIC_API_KEY`. Listed in `.gitignore`. A checked-in `.env.example` documents the expected keys.
- Optional `EMBEDDING_MODEL=climatebert|minilm` toggles models without code edits.

### Caching

- `data/cache/{company}_{year}.pkl`: parsed PDFs + embeddings, keyed on PDF path + mtime.
- `data/cache/llm/{sha256_of_prompt}.json`: raw LLM responses, keyed on prompt hash. Enables deterministic replay of the eval after API credits are exhausted.
- Both caches opt-out via `--no-cache` CLI flag; on by default.

## 10. Open questions

None at time of writing.
