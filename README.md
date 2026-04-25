# NLP ESG KPI Extraction

Extracts Scope 1 emissions, renewable energy consumption, and water consumption
from corporate sustainability PDFs. Compares a retrieval + regex baseline with
an Anthropic Claude structured-output extractor.

## Setup

1. Python 3.11+.
2. `pip install -e ".[dev]"`
3. Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY`.
4. Drop PDFs into `data/reports/` named `{company}_{year}.pdf`.

## Run

```bash
# Parse PDFs, extract, compare, evaluate — prints tables to stdout.
python -m nlp_esg.pipeline
```

## Demo notebook

```bash
jupyter notebook notebooks/demo.ipynb
```

Runs the same pipeline interactively and includes the MiniLM vs. ClimateBERT
comparison and qualitative-examples cells.

## Test

```bash
pytest              # unit tests only
RUN_INTEGRATION=1 pytest   # also runs LLM + real-PDF integration tests
```
