"""One-off: parse+index from cache, run baseline only, evaluate, persist."""
from __future__ import annotations
import dataclasses
import logging
import pickle
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
sys.path.insert(0, "src")

from nlp_esg.config import CACHE_DIR, KPI_KEYS, REPORTS_DIR, RUNS_DIR
from nlp_esg.evaluate import evaluate
from nlp_esg.extractors.baseline import BaselineExtractor
from nlp_esg.pipeline import load_gold_labels
from nlp_esg.retrieval import build_index

run_tag = "v_docling_baseline_only"
out_dir = RUNS_DIR / run_tag
out_dir.mkdir(parents=True, exist_ok=True)

ext = BaselineExtractor()
extractions = []
for pdf in sorted(REPORTS_DIR.glob("*.pdf")):
    company, year = pdf.stem.split("_")
    cache = CACHE_DIR / f"{company}_{year}_docling.pkl"
    if not cache.exists():
        print(f"NO CACHE for {company}/{year}")
        continue
    with cache.open("rb") as f:
        parsed = pickle.load(f)
    indexed = build_index(parsed)
    for kpi_key in KPI_KEYS:
        e = ext.extract(indexed, kpi_key)
        e.run_tag = run_tag
        extractions.append(e)

# Eval
golds = load_gold_labels()
rows = []
for kpi in KPI_KEYS:
    preds = [e for e in extractions if e.extractor == "baseline" and e.kpi == kpi]
    kpi_golds = [g for g in golds if g["kpi"] == kpi]
    rows.append(evaluate(preds, kpi_golds, extractor="baseline", kpi=kpi))
metrics = pd.DataFrame(rows)
print("\n=== Baseline-only metrics ===")
print(metrics.to_string())
total_tp = metrics["tp"].sum()
total = metrics["tp"].sum() + metrics["fp"].sum() + metrics["fn"].sum()
print(f"\nBaseline total: {total_tp}/15 (TP); F1 macro = {metrics['f1'].mean():.3f}")

# Persist
ext_rows = [dataclasses.asdict(e) for e in extractions]
pd.DataFrame(ext_rows).to_csv(out_dir / "extractions.csv", index=False)
metrics.to_csv(out_dir / "metrics.csv", index=False)
print(f"\nPersisted to {out_dir}/")
