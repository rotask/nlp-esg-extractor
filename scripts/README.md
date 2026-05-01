# `scripts/` — runnable entry points

Two ways to drive the pipeline beyond `python -m nlp_esg.pipeline`. Run
both **from the repository root** so relative `data/...` paths resolve.

## `run_baseline_only.py` — deterministic baseline only

Runs the rule-based baseline extractor against the cached IndexedReports
for every PDF in `data/reports/`, evaluates against
`data/labels/gold_labels.csv`, and writes `extractions.csv` +
`metrics.csv` under `data/runs/<run_tag>/`.

```bash
# from the repo root
python scripts/run_baseline_only.py
python scripts/run_baseline_only.py --run-tag my_baseline_run
```

No API keys, no LLM calls, no Gemini quota required. Use this when:
- You want the rule-based numbers in isolation.
- The LLM provider's daily quota is exhausted.
- You're iterating on baseline rules and want fast feedback.

First invocation parses + indexes any PDF that doesn't yet have a cache
entry under `data/cache/` (slow on a fresh machine — see top-level
`README.md` §Setup). Subsequent invocations reuse the cache and finish
in seconds.

## `run_docling_full.py` — full pipeline, Docling-first

Like `python -m nlp_esg.pipeline` but bypasses the dispatcher's
pdfplumber-cache fallback so we always exercise Docling, and logs
per-PDF parse + index timings to `data/runs/<run_tag>/parse_timings.csv`.

```bash
# from the repo root
LLM_PROVIDER=gemini GEMINI_MODEL=gemini-2.5-flash \
    python scripts/run_docling_full.py --run-tag my_full_run
```

This is the canonical reproducer for the **best-of-either 15/15**
headline. See top-level `README.md` §"Reproducing the headline numbers".
