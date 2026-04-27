# NLP ESG KPI Extraction

A two-extractor pipeline that pulls three numerical KPIs out of corporate
sustainability PDFs and produces a comparison table with precision/recall/F1
against hand-labelled gold:

- **Scope 1 emissions** (canonical unit: `tCO2e`)
- **Total energy consumption** (canonical unit: `MWh`)
- **Water consumption** (canonical unit: `m3`)

A deterministic baseline (table-first regex with negative-token filtering and
year-column awareness) and an LLM extractor (Anthropic Claude or Google Gemini
via tool-use) share the same retrieval layer (hybrid BM25 + ClimateBERT
embeddings, reciprocal-rank-fused across multiple KPI query phrasings). Both
emit a normalised `KPIExtraction` row, the pipeline writes a side-by-side
comparison CSV, and `evaluate.py` computes per-extractor P/R/F1 against the
gold labels in `data/labels/gold_labels.csv`. This is a coursework deliverable
for Team B.

## Headline results

Five reports x three KPIs = 15 cells of gold. Best per-extractor and combined
results from the committed reproducibility runs:

| Run                                | Extractor      | TP    | F1   | Run dir                                  |
|------------------------------------|----------------|-------|------|------------------------------------------|
| Deterministic baseline             | baseline       | 12/15 | 0.88 | `data/runs/v9_magnitude_tiebreak/`       |
| LLM (`gemini-2.5-flash`)           | llm            | 12/15 | 0.88 | `data/runs/v_gemini_25flash_post_quota/` |
| LLM (`gemini-2.5-flash-lite`)      | llm            | 8/15  | 0.66 | `data/runs/v_gemini_post_quota/`         |
| **best-of-either** (baseline ∪ flash) | -          | **14/15** | **0.96** | union of the first two                |

The two extractors recover different cells — together they cover 14/15 of the
gold corpus. The single unrecovered cell is Shell's water consumption (gold
26 M m³); see `docs/FINDINGS.md` §12 for the failure-mode analysis.

## Installation

Requires Python 3.11 or newer.

```bash
git clone <this repo>
cd NLP_ESG_Project
pip install -e ".[dev]"
```

This installs the runtime dependencies (pdfplumber, sentence-transformers,
torch, anthropic, google-genai, docling, rank-bm25, pandas) plus pytest and
nbformat for the dev tools.

PDFs and run outputs are git-ignored (see `.gitignore`). Drop your own report
PDFs into `data/reports/` named `{company}_{year}.pdf` (e.g. `shell_2024.pdf`).
Five hand-labelled gold rows live in `data/labels/gold_labels.csv` and are
checked into the repo.

## Configuration

Copy `.env.example` to `.env` and set the keys for the provider you intend to
use. The variables actually consulted by the code:

| Variable               | Default              | Required when                                  |
|------------------------|----------------------|------------------------------------------------|
| `LLM_PROVIDER`         | `anthropic`          | always (one of `anthropic`, `gemini`)          |
| `ANTHROPIC_API_KEY`    | -                    | `LLM_PROVIDER=anthropic`                       |
| `ANTHROPIC_MODEL`      | `claude-sonnet-4-6`  | optional override                              |
| `GEMINI_API_KEY`       | -                    | `LLM_PROVIDER=gemini` (`GOOGLE_API_KEY` ok too)|
| `GEMINI_MODEL`         | `gemini-2.0-flash`   | optional override (recommend `gemini-2.5-flash`)|
| `EMBEDDING_MODEL`      | `climatebert`        | optional (`minilm` is the alternative)         |

`NLP_ESG_DISABLE_DOCLING=1` is a **process-level env var** (not loaded from
`.env`). Setting it short-circuits the Docling-first ingest path and falls
back to pdfplumber-only. Required on machines where Docling's C++ layout
model SIGSEGVs on long PDFs (it does on the dev machine for this corpus).

## Running the pipeline

The full pipeline — parse PDFs, build the embedding index, run both
extractors, evaluate, persist artifacts — is one command:

```bash
NLP_ESG_DISABLE_DOCLING=1 LLM_PROVIDER=gemini \
  python -m nlp_esg.pipeline --run-tag my_run
```

(Drop `LLM_PROVIDER=gemini` to use Anthropic; drop the docling env var if
your machine handles Docling fine.)

Output goes under `data/runs/my_run/`:

- `extractions.csv` — one row per `(company, kpi, extractor)`. Columns
  include `value`, `unit`, `reporting_year`, `source_snippet`, `flags`,
  `confidence`.
- `metrics.csv` — per-extractor x per-KPI precision/recall/F1/coverage.
- `llm_prompts/{company}_{year}_{kpi}.json` — full prompt + retrieved
  pages + tool response, written when the LLM extractor runs. Useful for
  reproducing or debugging individual cells.

The pipeline also prints the side-by-side comparison tables and the
evaluation summary to stdout.

The first run is slow (~5–15 minutes per long PDF on CPU) because
ClimateBERT embeddings have to be computed for every sentence and table
header. Both the parsed `ParsedReport` (under
`data/cache/{company}_{year}_{parser}.pkl`) and the indexed report (under
`data/cache/{company}_{year}_{parser}_indexed_{model}.pkl`) are cached on
disk, so subsequent runs are seconds. The LLM responses are also cached
keyed on `sha256(model | kpi | system_prompt | user_prompt)`, so prompt
edits invalidate the cache automatically.

## Reproducing the published results

Three runs are committed to git as canonical reproducibility artifacts:

| Run dir                              | Reproducing command                                                                                |
|--------------------------------------|----------------------------------------------------------------------------------------------------|
| `data/runs/v9_magnitude_tiebreak/`   | `NLP_ESG_DISABLE_DOCLING=1 python -m nlp_esg.pipeline --run-tag v9_magnitude_tiebreak` (look at the **baseline** rows in metrics.csv — this run pre-dates the working LLM path, so the LLM rows are all zeros) |
| `data/runs/v_gemini_25flash_post_quota/` | `NLP_ESG_DISABLE_DOCLING=1 LLM_PROVIDER=gemini GEMINI_MODEL=gemini-2.5-flash python -m nlp_esg.pipeline --run-tag v_gemini_25flash_post_quota` |
| `data/runs/v_gemini_post_quota/`     | `NLP_ESG_DISABLE_DOCLING=1 LLM_PROVIDER=gemini GEMINI_MODEL=gemini-2.5-flash-lite python -m nlp_esg.pipeline --run-tag v_gemini_post_quota` |

The headline numbers in `metrics.csv` are temperature-zero and provider-side
caching is not relied upon, so re-running the same command should yield the
same TP/FP/FN counts modulo Gemini API non-determinism on borderline cases.

## Models

No model is fine-tuned. All weights are loaded as-is from official providers.

- **Embeddings** — `climatebert/distilroberta-base-climate-f` (HuggingFace,
  768-dim, ~82 M parameters), loaded by `sentence-transformers` with mean-token
  pooling. Cached locally after first use. The MiniLM alternative
  (`sentence-transformers/all-MiniLM-L6-v2`, 384-dim) is wired up via
  `EMBEDDING_MODEL=minilm` and benchmarked in `docs/FINDINGS.md`.
- **LLM** — Anthropic `claude-sonnet-4-6` or Google `gemini-2.5-flash` /
  `gemini-2.5-flash-lite`, selected via `LLM_PROVIDER`. Both go through their
  official SDK with strict tool-use schemas (`record_kpi`), `temperature=0`,
  and a system prompt encoding the disambiguation rules (ESRS-aligned
  consolidation, withdrawal-vs-consumption, magnitude prefix multiplication).
- **Lexical retrieval** — `rank-bm25` (the standard BM25Okapi
  implementation). Deterministic, no training.

## Testing

```bash
# Non-integration suite, ~30 s on CPU.
pytest -q --ignore=tests/test_integration_llm.py --ignore=tests/test_integration_real_pdf.py

# Single test, useful for development.
pytest tests/test_normalize.py::test_lone_subscript_2_on_own_line -v

# Opt-in real-API + real-PDF tests (require keys, ~minutes).
RUN_INTEGRATION=1 pytest
```

The non-integration suite (122 tests as of this commit) must stay under ~30 s
on CPU and must not load embeddings or hit the API. Integration tests are
opt-in to keep CI cheap.

## Project structure

```
.
├── src/nlp_esg/
│   ├── pipeline.py           # `python -m nlp_esg.pipeline` entry point
│   ├── ingest.py             # PDF parser dispatch + pdfplumber fallback
│   ├── ingest_docling.py     # Docling parser path (skippable)
│   ├── retrieval.py          # ClimateBERT/MiniLM index + BM25 + hybrid ranking
│   ├── normalize.py          # value/unit parser, CO2 fixup, magnitude logic
│   ├── extractors/
│   │   ├── baseline.py       # deterministic table-first + line-scan fallback
│   │   └── llm.py            # Anthropic / Gemini tool-use extractor
│   ├── compare.py            # builds the comparison table
│   ├── evaluate.py           # per-extractor P/R/F1 vs gold
│   └── config.py             # KPI registry + env-var defaults
├── tests/                    # 122 unit tests + 2 opt-in integration files
├── data/
│   ├── reports/              # drop your PDFs here (git-ignored)
│   ├── labels/gold_labels.csv# 5 hand-labelled reports x 3 KPIs (committed)
│   ├── cache/                # parsed-report + index caches (git-ignored)
│   └── runs/                 # output dir; three reproducibility runs committed
├── docs/                     # architecture, findings, API, sustainability
├── notebooks/                # demo notebook
├── pyproject.toml
└── .env.example
```

## CLI / API surface

The primary interface is the CLI:

```bash
python -m nlp_esg.pipeline --run-tag <tag>
```

For programmatic use (importing `parse_pdf`, `build_index`, the extractor
classes, etc.), see `docs/API.md`.

A demo notebook at `notebooks/demo.ipynb` walks the same pipeline
interactively and includes the MiniLM-vs-ClimateBERT comparison cell:

```bash
jupyter notebook notebooks/demo.ipynb
```

## Documentation

- `CLAUDE.md` — architecture, dispatcher logic, cache-key invariants, the
  gotchas you need to know before changing the code.
- `docs/FINDINGS.md` — full iteration history. What was tried, what failed,
  why we stopped chasing each blind alley. Read before making structural
  changes.
- `docs/API.md` — module-level Python API reference.
- `docs/SUSTAINABILITY.md` — impact, SDG alignment, scalability, ethical
  considerations.

## Acknowledgement

Coursework deliverable, Team B. No LICENSE file — treat as
all-rights-reserved unless told otherwise by the course staff.
