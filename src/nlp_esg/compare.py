"""Side-by-side comparison DataFrames consumed by `pipeline.main()`.

`build_comparison_table` pivots a list of `KPIExtraction` rows into a
companies x KPIs grid (most-recent-year wins per company), and
`build_run_comparison` joins multiple persisted runs on
(company, report_year, kpi) so v1 vs v2 deltas can be inspected.
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable

import pandas as pd

from nlp_esg.config import KPI_KEYS, RUNS_DIR
from nlp_esg.types import KPIExtraction


def build_comparison_table(
    extractions: Iterable[KPIExtraction], extractor: str
) -> pd.DataFrame:
    """
    Pivot extractions into a companies x KPIs DataFrame. For companies with multiple
    reports, the most recent report year is used.
    """
    rows = [e for e in extractions if e.extractor == extractor]
    if not rows:
        return pd.DataFrame(columns=KPI_KEYS)

    # Pick the most recent report year per (company, kpi).
    latest: dict[tuple[str, str], KPIExtraction] = {}
    for e in rows:
        key = (e.company, e.kpi)
        if key not in latest or e.report_year > latest[key].report_year:
            latest[key] = e

    # Build the dataframe: rows = companies, cols = KPI keys.
    companies = sorted({c for (c, _) in latest})
    df = pd.DataFrame(index=companies, columns=KPI_KEYS, dtype=object)
    for (company, kpi), e in latest.items():
        df.loc[company, kpi] = e.value
    return df


def build_run_comparison(
    runs: list[str], runs_dir: Path = RUNS_DIR
) -> pd.DataFrame:
    """Join per-run extractions on (company, report_year, kpi).

    One value column per (run_tag, extractor) pair, e.g.
    'v1_pdfplumber_baseline_value', 'v2_docling_llm_value'.
    """
    frames = []
    for run in runs:
        path = runs_dir / run / "extractions.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        wide = df.pivot_table(
            index=["company", "report_year", "kpi"],
            columns="extractor",
            values="value",
            aggfunc="first",
        ).reset_index()
        wide.columns = [
            f"{run}_{c}_value" if c not in ("company", "report_year", "kpi") else c
            for c in wide.columns
        ]
        frames.append(wide)
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on=["company", "report_year", "kpi"], how="outer")
    return out
