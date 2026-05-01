# nlp_esg — Python API reference

Programmatic interface for the two-extractor ESG KPI pipeline. Every public
symbol lives under the `nlp_esg.*` namespace; the package root re-exports
nothing, so import directly from the relevant module
(`from nlp_esg.ingest import parse_pdf`).

For installation, the CLI invocation, and the headline P/R/F1 numbers, see
[`README.md`](../README.md). For architectural background and the iteration
history, see [`CLAUDE.md`](../CLAUDE.md) and [`docs/FINDINGS.md`](FINDINGS.md).

## Quick start

Parse one PDF, build the embedding index, run the deterministic baseline
across all three KPIs, and print the results:

```python
from pathlib import Path

from nlp_esg.config import KPI_KEYS
from nlp_esg.extractors.baseline import BaselineExtractor
from nlp_esg.ingest import parse_pdf
from nlp_esg.retrieval import build_index

# 1. Parse a PDF — Docling first, pdfplumber fallback. Cached on disk.
parsed = parse_pdf(Path("data/reports/iberdrola_2025.pdf"))

# 2. Add per-sentence + per-table-header embeddings (ClimateBERT). Cached.
indexed = build_index(parsed)

# 3. Run the deterministic baseline across the three KPIs.
extractor = BaselineExtractor()
for kpi_key in KPI_KEYS:
    result = extractor.extract(indexed, kpi_key)
    print(f"{kpi_key}: {result.value} {result.unit}  ({result.source_snippet})")
```

The LLM extractor follows the same shape; see
[`nlp_esg.extractors.llm.LLMExtractor`](#nlp_esgextractorsllmllmextractor)
below.

## Data types

### `nlp_esg.types.KPIExtraction`

Dataclass — the canonical row produced by both extractors and serialised to
`extractions.csv`.

| Field            | Type            | Notes                                                                |
|------------------|-----------------|----------------------------------------------------------------------|
| `company`        | `str`           | Lowercase company slug parsed from the PDF filename.                 |
| `report_year`    | `int`           | Year parsed from the PDF filename (e.g. `iberdrola_2025.pdf` -> 2025). |
| `kpi`            | `str`           | One of the keys in `nlp_esg.config.KPI_KEYS`.                        |
| `value`          | `float \| None` | Canonicalised value, or `None` for "not reported".                   |
| `unit`           | `str \| None`   | Canonical unit (`tCO2e`, `MWh`, `m3`), or `None`.                    |
| `reporting_year` | `int \| None`   | Year the value refers to (overwritten with `report_year` post-extract). |
| `source_snippet` | `str \| None`   | Short text fragment supporting the value.                            |
| `source_page`    | `int \| None`   | 1-based page index inside the parsed PDF (not the printed page).     |
| `confidence`     | `float \| None` | Extractor-defined score in `[0, 1]` (LLM only) or unbounded (baseline). |
| `extractor`      | `str`           | `"baseline"` or `"llm"`.                                             |
| `flags`          | `list[str]`     | Diagnostic tags (e.g. `out_of_range`, `unit_unknown`, `api_error`).  |
| `run_tag`        | `str \| None`   | Set by `pipeline.main` after extraction; `None` otherwise.           |

### `ParsedReport` (TypedDict, defined in `nlp_esg.ingest`)

| Key           | Type                | Notes                                                          |
|---------------|---------------------|----------------------------------------------------------------|
| `company`     | `str`               | Lowercase company slug from the filename.                      |
| `report_year` | `int`               | Filename year.                                                 |
| `parser`      | `str`               | `"docling"` or `"pdfplumber"` — which path produced this report. |
| `pages`       | `list[Page]`        | Each `{"page_num": int, "text": str}`.                         |
| `tables`      | `list[TableEntry]`  | Each `{"page_num": int, "headers": list[str], "rows": list[list[str]]}`. |

### `IndexedReport` (TypedDict, defined in `nlp_esg.retrieval`)

| Key             | Type                    | Notes                                                       |
|-----------------|-------------------------|-------------------------------------------------------------|
| `company`       | `str`                   | Carried through from the `ParsedReport`.                    |
| `report_year`   | `int`                   |                                                             |
| `pages`         | `list[Page]`            |                                                             |
| `tables`        | `list[TableEntry]`      |                                                             |
| `sentences`     | `list[Sentence]`        | `{"page_num": int, "text": str, "embedding": np.ndarray}`.  |
| `table_headers` | `list[TableHeaderEmb]`  | `{"table_idx": int, "header_string": str, "embedding": np.ndarray}`. |

`build_index` includes the first 5 row labels alongside the headers in
`header_string`, because some tables (e.g. Eni-style) put the KPI label in
`row[0]` and have year-only headers like `["", "2024", "2023"]`.

## `nlp_esg.config`

Project-wide constants and the KPI registry.

### `KPIS: dict[str, dict]`

Top-level keys are KPI identifiers (`scope_1_emissions`,
`total_energy_consumption`, `water_consumption`). Each value is:

```python
{
    "query": str,                # Single-phrase query (used by table-first search).
    "queries": list[str],        # 2-4 phrasings used by hybrid retrieval / RRF.
    "unit_family": list[str],    # Acceptable raw units (canonicalisable).
    "canonical_unit": str,       # Unit values are converted to.
    "plausible_range": tuple[float, float],  # (lo, hi) inclusive bounds.
    "negative_tokens": list[str],  # Lines / row labels containing any of these are rejected.
}
```

Per-KPI summary:

| KPI                          | Canonical unit | Plausible range  | Notable negative tokens                      |
|------------------------------|----------------|------------------|----------------------------------------------|
| `scope_1_emissions`          | `tCO2e`        | `(1e2, 1e9)`     | `scope 2`, `scope 3`, `methane`, `intensity`, `per `, `fugitive` |
| `total_energy_consumption`   | `MWh`          | `(1e2, 1e9)`     | `renewable`, `produced`, `production`, `intensity`, `discontinued` |
| `water_consumption`          | `m3`           | `(1e1, 1e10)`    | `withdrawal`, `withdrawn`, `discharge`, `recycled`, `reclaimed`, `intake` |

### Other module attributes

| Symbol                  | Type   | Meaning                                                              |
|-------------------------|--------|----------------------------------------------------------------------|
| `KPI_KEYS`              | `list[str]` | `list(KPIS.keys())` — iteration-order canonical KPI list.       |
| `EMBEDDING_MODEL_NAME`  | `str`  | From `EMBEDDING_MODEL` env var; default `"climatebert"`.             |
| `ANTHROPIC_MODEL`       | `str`  | From `ANTHROPIC_MODEL` env var; default `"claude-sonnet-4-6"`.       |
| `EPSILON`               | `float`| `0.01` — relative tolerance for value correctness.                   |
| `TAU_TABLE`             | `float`| `0.55` — cosine threshold for table-header match.                    |
| `TOP_K_SENTENCES`       | `int`  | `5`.                                                                 |
| `ROOT`                  | `Path` | Repo root.                                                           |
| `DATA_DIR`              | `Path` | `<root>/data`.                                                       |
| `REPORTS_DIR`           | `Path` | `<root>/data/reports` — drop PDFs here.                              |
| `LABELS_DIR`            | `Path` | `<root>/data/labels` — gold CSV.                                     |
| `CACHE_DIR`             | `Path` | `<root>/data/cache` — parser + index + LLM caches.                   |
| `RUNS_DIR`              | `Path` | `<root>/data/runs` — per-run artefacts.                              |

## `nlp_esg.ingest`

```python
def parse_pdf(path: Path, use_cache: bool = True) -> ParsedReport
```

Dispatcher: tries `parse_with_docling(path)` first and falls back to
in-file `_parse_with_pdfplumber` on `None` / empty / failed quality check.
Returns the `ParsedReport` TypedDict described above.

Cache filenames include the parser tag so v1 (pdfplumber) and v2 (Docling)
caches coexist without collision:

```
data/cache/{company}_{year}_pdfplumber.pkl
data/cache/{company}_{year}_docling.pkl
```

The cache is invalidated by file mtime: a cache pickle is reused only if
its mtime is `>=` the source PDF's mtime.

Filename convention is enforced by a regex: `^{company}_{year}.pdf$` where
`{company}` matches `[a-z0-9_\-]+` (case-insensitive) and `{year}` is four
digits. Anything else raises `ValueError`.

Environment hook: setting `NLP_ESG_DISABLE_DOCLING=1` (or `true`/`yes`/`on`)
short-circuits the Docling attempt entirely. Required on machines where
Docling's C++ layout model SIGSEGVs on long PDFs.

## `nlp_esg.ingest_docling`

```python
def parse_with_docling(path: Path) -> ParsedReport | None
```

Returns `None` (so the caller falls back to pdfplumber) when:

- `NLP_ESG_DISABLE_DOCLING` is truthy.
- `docling` is not importable.
- The PDF is larger than 15 MB (Docling's layout model OOMs on long PDFs).
- The filename does not match `{company}_{year}.pdf`.
- Docling raises during `convert()`.
- Docling returns 0 pages.
- Fewer than 50% of returned pages have substantive (`>= 100` chars) text.

Otherwise returns a `ParsedReport` with `"parser": "docling"`.

## `nlp_esg.normalize`

Unit canonicalisation, value extraction, and PDF text fix-ups.

| Function                                        | Returns                       | Behaviour                                                          |
|-------------------------------------------------|-------------------------------|--------------------------------------------------------------------|
| `parse_number(text: str) -> float`              | `float`                       | Parses thousands-comma, EU-decimal, plain. Raises `ValueError`.    |
| `canonicalize_unit(unit: str) -> str`           | canonical alias               | Strips trailing punctuation; raises `ValueError` on unknown unit.  |
| `to_canonical_value(value, unit, canonical) -> float` | `float`                 | Cross-unit conversion. Raises `ValueError` for unsupported pair.   |
| `parse_value(text, kpi_unit_family) -> tuple[float, str] \| None` | `(value, canonical_unit)` | Finds the first `(number, unit)` pair in free text whose unit canonicalises into the family. Returns `None` on miss. |
| `normalize_co2(text: str) -> str`               | `str`                         | Pre-pass that fixes `CO 2 e`, `COe2`, multi-line subscript artefacts. |

### Canonical-unit conversion table

`to_canonical_value` first calls `canonicalize_unit` on each side, then
applies the multiplier from this table. Identity entries are omitted.

| From    | To       | Multiplier         | Notes                                  |
|---------|----------|--------------------|----------------------------------------|
| `kWh`   | `MWh`    | `1e-3`             |                                        |
| `GWh`   | `MWh`    | `1e3`              |                                        |
| `TWh`   | `MWh`    | `1e6`              |                                        |
| `GJ`    | `MWh`    | `1 / 3.6`          | 1 GJ = 0.2778 MWh                      |
| `TJ`    | `MWh`    | `1e3 / 3.6`        |                                        |
| `PJ`    | `MWh`    | `1e6 / 3.6`        |                                        |
| `ktCO2e`| `tCO2e`  | `1e3`              |                                        |
| `MtCO2e`| `tCO2e`  | `1e6`              |                                        |
| `kL`    | `m3`     | `1.0`              | 1 kilolitre = 1 cubic metre.           |
| `ML`    | `m3`     | `1e3`              | 1 megalitre = 1000 m3.                 |
| `Mm3`   | `m3`     | `1e6`              | "Mm3" / "million m3" / "Mm³".          |

Aliases recognised by `canonicalize_unit` (case-insensitive, lookup is on
the lowercased input with trailing `.,;:` stripped):

- Energy: `kwh`, `mwh`, `gwh`, `twh`, `gj`, `tj`, `pj`.
- Emissions: `tco2e`, `t co2e`, `t co2-eq`, `tco2eq`, `tonnes co2e`,
  `tonnes co2-eq`, `tonnes co2eq` (and `kt`/`Mt` prefixes thereof).
- Water: `m3`, `m³`, `cubic metres`, `cubic meters`, `ml`, `megalitres`,
  `megaliters`, `kl`, `thousand m3` (-> `ML`), `mm3`, `mm³`.

Example:

```python
from nlp_esg.normalize import parse_value, to_canonical_value

parse_value("Total Scope 1 emissions: 12.4 MtCO2e", ["tCO2e", "MtCO2e"])
# -> (12.4, "MtCO2e")

to_canonical_value(12.4, "MtCO2e", "tCO2e")
# -> 12_400_000.0
```

## `nlp_esg.retrieval`

Hybrid page-level retrieval shared by both extractors.

### Functions

```python
def embed_texts(texts: list[str], model_name: str | None = None) -> np.ndarray
```

Embed a list of strings. `model_name` falls back to
`config.EMBEDDING_MODEL_NAME` (set via the `EMBEDDING_MODEL` env var).
Supported models: `"climatebert"` (default) and `"minilm"`. Returns a
`(len(texts), dim)` `float32` array, or `(0, dim)` for empty input.

```python
def build_index(
    report: ParsedReport,
    model_name: str | None = None,
    use_cache: bool = True,
) -> IndexedReport
```

Adds per-sentence and per-table-header embeddings. The forward pass is
slow on CPU (5–15 minutes per long report on ClimateBERT), so this is
cached to:

```
data/cache/{company}_{year}_{parser}_indexed_{model}.pkl
```

The cache validates on `(company, report_year, len(pages))`; a mismatch
triggers a rebuild. Invalidation happens implicitly when the parser tag or
model name changes (different cache key).

```python
def rank_pages_cosine(
    report: IndexedReport,
    query_emb: np.ndarray,
    unit_tokens: list[str] | None = None,
) -> list[tuple[int, float]]
```

Score each page by max sentence/table-header cosine similarity to
`query_emb`. Optional `+0.1` bonus for pages whose normalised text contains
any KPI unit token (case-insensitive). Returns `[(page_num, score), ...]`
sorted descending.

```python
def rank_pages_hybrid(
    report: IndexedReport,
    queries: list[str],
    alpha: float = 0.5,
    rrf_k: int = 60,
) -> list[tuple[int, float]]
```

`alpha * cosine_rrf + (1 - alpha) * bm25`, both min-max normalised into
`[0, 1]`. The cosine side first runs `rank_pages_cosine` per query phrasing
then fuses via reciprocal-rank fusion (`rrf_k=60` is the customary RRF
constant). The BM25 side tokenises every page (lowercase alphanumerics) and
scores against a single combined query string. Returns
`[(page_num, fused_score), ...]` sorted descending.

```python
def rank_pages_rrf(
    report: IndexedReport,
    queries: list[str],
    unit_tokens: list[str] | None = None,
    k: int = 60,
) -> list[tuple[int, float]]
```

The cosine-only RRF helper used inside `rank_pages_hybrid`. Useful when you
want multi-query fusion without the BM25 component.

Other helpers exposed by the module: `cosine_sim(a, b) -> float`,
`top_k(query, corpus, k) -> list[int]`, and `split_sentences(text) -> list[str]`.

## `nlp_esg.extractors`

Both concrete extractors share a common interface and cooperate with the
`pipeline` orchestration loop.

### `nlp_esg.extractors.base.Extractor` (ABC)

```python
class Extractor(ABC):
    name: str = "base"

    @abstractmethod
    def extract(self, report: Any, kpi_key: str) -> KPIExtraction: ...
```

Subclasses set `name` (used to slice extractions per-extractor in
`compare.build_comparison_table` and `pipeline.run_evaluation`).

### `nlp_esg.extractors.baseline.BaselineExtractor`

```python
class BaselineExtractor(Extractor):
    name = "baseline"

    def extract(self, report: IndexedReport, kpi_key: str) -> KPIExtraction
```

Two-stage pipeline:

1. **Table-first** — score every table whose header embedding has cosine
   similarity `>= TAU_TABLE` to the KPI query embedding, multiplied by a
   structural bonus when the table has the report year in a header. For
   each candidate table find the year column, then the best-matching
   non-negative row by phrase + token overlap. Infer the unit from the
   value cell, an explicit `Unit` column, the value column header, the
   row label, or the table headers (in that order). Reject anything
   outside `plausible_range` or matching a `negative_token`.
2. **Page-level line scanner** (fallback) — `rank_pages_hybrid` over the
   KPI's `queries`, scan each line on the top 25 pages for KPI keywords
   plus a `(value, unit)` parse, apply year-column awareness via
   `_pick_year_column_value` (preserves magnitude when picking from the
   most-recent year column, taking the **last N** numbers parsed from
   the data line). Tie-break candidates with similar keyword score by
   preferring the larger canonical value (correct for
   consolidated-vs-segment ambiguity).

The constructor takes no arguments. The extractor is stateless across
calls.

### `nlp_esg.extractors.llm.LLMExtractor`

```python
class LLMExtractor(Extractor):
    name = "llm"

    def __init__(
        self,
        model: str | None = None,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        provider: str | None = None,
        min_call_interval_s: float | None = None,
        prompt_log_dir: Path | None = None,
    )

    def extract(self, report: IndexedReport, kpi_key: str) -> KPIExtraction
```

#### Constructor parameters

| Parameter             | Default (when `None`)                                          | Notes                                                                 |
|-----------------------|----------------------------------------------------------------|-----------------------------------------------------------------------|
| `provider`            | `os.environ["LLM_PROVIDER"]`, then `"anthropic"`               | One of `"anthropic"` or `"gemini"`. Raises `ValueError` otherwise.    |
| `model`               | `os.environ["GEMINI_MODEL"]` (Gemini) or `os.environ["ANTHROPIC_MODEL"]` then `config.ANTHROPIC_MODEL` | Provider-specific default (`"gemini-2.0-flash"` / `"claude-sonnet-4-6"`). |
| `max_retries`         | `3`                                                            | Total attempts per call.                                              |
| `retry_base_delay`    | `1.0`                                                          | Seconds; doubled per attempt for non-429 errors.                      |
| `min_call_interval_s` | `6.5` (Gemini) / `0.0` (Anthropic)                             | Throttle to keep Gemini under the free-tier 10 RPM cap.               |
| `prompt_log_dir`      | `None` — no logging                                            | Directory to write per-(company, kpi) prompt logs.                    |

#### Environment variables consulted

| Variable               | Used when                                  | Purpose                                            |
|------------------------|--------------------------------------------|----------------------------------------------------|
| `LLM_PROVIDER`         | `provider=None`                            | Selects backend.                                   |
| `ANTHROPIC_API_KEY`    | provider == anthropic, on first API call   | Read by the `anthropic` SDK from the env directly. |
| `ANTHROPIC_MODEL`      | provider == anthropic and `model=None`     | Anthropic model id.                                |
| `GEMINI_API_KEY`       | provider == gemini, on first API call      | Primary Gemini key.                                |
| `GOOGLE_API_KEY`       | provider == gemini and `GEMINI_API_KEY` unset | Fallback Gemini key.                            |
| `GEMINI_MODEL`         | provider == gemini and `model=None`        | Gemini model id.                                   |

#### Cache key

Each call hashes `(model, kpi_key, system_prompt, user_prompt)` with
SHA256 and stores the tool-call response at:

```
data/cache/llm/<sha256>.json
```

The system prompt is part of the key — prompt-rule edits invalidate stale
responses. Old caches under previous keys remain on disk but are not
served.

#### `extract` flow

1. `_build_context` runs `rank_pages_hybrid` on the KPI's `queries`,
   takes the top 16 pages (cap 4000 chars per page), and appends any
   tables whose `page_num` is in the top-16 set.
2. Compose the user prompt with the KPI query, the acceptable unit list,
   and the document excerpt.
3. Look up the cache. On hit, skip the API call.
4. On miss: dispatch to `_call_anthropic_once` (forced
   `tool_choice` = `record_kpi`, ephemeral cache_control on the system
   prompt) or `_call_gemini_once` (`function_calling_config.mode = "ANY"`,
   `thinking_budget = 0`).
5. Convert the tool input into a `KPIExtraction`. Out-of-range,
   unknown-unit, missing-unit, and api-error cases collapse to a
   "not reported" row with a diagnostic flag (`out_of_range`,
   `unit_unknown`, `llm_missing_unit`, `api_error`).

#### `prompt_log_dir`

When set, every call (including cache hits) writes
`<prompt_log_dir>/{company}_{year}_{kpi_key}.json` with this schema:

```python
{
  "company": str,
  "report_year": int,
  "kpi": str,
  "provider": "anthropic" | "gemini",
  "model": str,
  "from_cache": bool,
  "retrieved_pages": list[int],   # top-16 page nums in rank order
  "system_prompt": str,
  "user_prompt": str,
  "tool_response": dict | None,   # None on api_error
}
```

Used downstream by the comparison notebook and the FINDINGS analysis
(see [`docs/FINDINGS.md`](FINDINGS.md) §11).

## `nlp_esg.evaluate`

```python
def is_correct(pred: KPIExtraction, gold: dict[str, Any]) -> bool

def evaluate(
    preds: list[KPIExtraction],
    golds: list[dict[str, Any]],
    extractor: str,
    kpi: str,
) -> dict[str, float]
```

A prediction is **correct** when:

- Both sides are `None` (true negative), OR
- `pred.unit == gold["unit"]`, AND
- `pred.reporting_year == gold["reporting_year"]`, AND
- `abs(pred.value - gold["value"]) / abs(gold["value"]) <= EPSILON` (1%
  relative tolerance from `config.EPSILON`).

Predictions and golds are paired on `(company, report_year)`. The returned
dict has keys: `extractor`, `kpi`, `tp`, `fp`, `fn`, `tn`, `precision`,
`recall`, `f1`, `coverage`. `coverage` is the fraction of matched
prediction–gold pairs where the prediction is non-null.

## `nlp_esg.compare`

```python
def build_comparison_table(
    extractions: Iterable[KPIExtraction], extractor: str
) -> pandas.DataFrame
```

Pivots a list of extractions into a `companies x KPI_KEYS` DataFrame for
the given `extractor`. When a company appears multiple times, the most
recent `report_year` wins. Cell values are the canonical numeric values
from each `KPIExtraction.value` (or `NaN`).

```python
def build_run_comparison(
    runs: list[str], runs_dir: Path = RUNS_DIR
) -> pandas.DataFrame
```

Joins per-run `extractions.csv` files on `(company, report_year, kpi)`,
emitting one column per `(run_tag, extractor)` pair, e.g.
`v9_magnitude_tiebreak_baseline_value`. Useful for v1 vs v2 deltas.

Example:

```python
from nlp_esg.compare import build_run_comparison

df = build_run_comparison(["v8", "v9_magnitude_tiebreak"])
print(df.head())
```

## `nlp_esg.pipeline`

Top-level orchestration. The CLI (`python -m nlp_esg.pipeline`) calls
`main(run_tag=...)`; the helpers below are also usable directly from a
notebook or another script.

```python
def load_indexed_reports(reports_dir: Path = REPORTS_DIR) -> list[IndexedReport]
```

Globs `*.pdf` under `reports_dir`, parses each, and builds an indexed
report. Errors per-PDF are logged but do not halt the loop.

```python
def run_extraction(
    indexed_reports: Iterable[IndexedReport],
    include_llm: bool = True,
    prompt_log_dir: Path | None = None,
) -> list[KPIExtraction]
```

Runs the baseline (and optionally the LLM extractor) across all KPIs for
every report. Returns one row per `(report, extractor, kpi)`.

```python
def load_gold_labels(path: Path | None = None) -> list[dict]
```

Reads `data/labels/gold_labels.csv` (or a custom path). Returns a list of
dicts; integer columns (`report_year`, `reporting_year`) are coerced from
floats and `NaN` is normalised to `None`. Returns `[]` when the file is
absent.

```python
def run_evaluation(
    extractions: list[KPIExtraction], golds: list[dict]
) -> pandas.DataFrame
```

For every (extractor, kpi) slice present in `extractions`, calls
`evaluate.evaluate` and returns the rows as a DataFrame.

```python
def main(run_tag: str = "v2_docling") -> None
```

Loads `.env` (via `python-dotenv` if installed), loads + indexes reports,
runs both extractors, prints the comparison + metrics tables, and persists
artefacts under `data/runs/<run_tag>/`.

## CLI

```bash
python -m nlp_esg.pipeline --run-tag v_my_run
NLP_ESG_DISABLE_DOCLING=1 python -m nlp_esg.pipeline --run-tag v_my_run
```

Outputs under `data/runs/<run_tag>/`:

| Artefact                                  | Producer                       |
|-------------------------------------------|--------------------------------|
| `extractions.csv`                         | `_persist_run` -> `dataclasses.asdict` per `KPIExtraction`. |
| `metrics.csv`                             | `run_evaluation` (only when gold labels are present). |
| `llm_prompts/<company>_<year>_<kpi>.json` | `LLMExtractor._write_prompt_log` (only when `prompt_log_dir` is set; the CLI sets it). |

## Environment variables (consolidated)

| Variable                  | Read in module     | Default                  | Purpose                                                          |
|---------------------------|--------------------|--------------------------|------------------------------------------------------------------|
| `LLM_PROVIDER`            | `extractors.llm`   | `"anthropic"`            | Selects LLM backend (`anthropic` or `gemini`).                   |
| `ANTHROPIC_API_KEY`       | `anthropic` SDK    | (none, required)         | Anthropic credentials when provider == anthropic.                |
| `ANTHROPIC_MODEL`         | `config`, `extractors.llm` | `"claude-sonnet-4-6"` | Anthropic model id.                                              |
| `GEMINI_API_KEY`          | `extractors.llm`   | (falls back to `GOOGLE_API_KEY`) | Gemini credentials when provider == gemini.            |
| `GOOGLE_API_KEY`          | `extractors.llm`   | (none)                   | Fallback name for Gemini credentials.                            |
| `GEMINI_MODEL`            | `extractors.llm`   | `"gemini-2.0-flash"`     | Gemini model id (recommend `gemini-2.5-flash`).                  |
| `EMBEDDING_MODEL`         | `config`           | `"climatebert"`          | Embedder name (`climatebert` or `minilm`).                       |
| `NLP_ESG_DISABLE_DOCLING` | `ingest_docling`   | unset                    | Truthy to skip Docling and force pdfplumber. Set in shell, not `.env`. |

A copy-pasteable `.env.example` lives at the project root.

## See also

- [`README.md`](../README.md) — install, run, headline numbers.
- [`CLAUDE.md`](../CLAUDE.md) — architecture and gotchas.
- [`docs/FINDINGS.md`](FINDINGS.md) — iteration history, error analysis,
  baseline vs LLM comparison, model comparison.
