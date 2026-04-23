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
python -m nlp_esg.pipeline
```

Open `notebooks/demo.ipynb` to see the comparison table and eval metrics.

## Test

```bash
pytest
```
