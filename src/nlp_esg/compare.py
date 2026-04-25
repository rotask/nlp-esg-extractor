from __future__ import annotations
from typing import Iterable

import pandas as pd

from nlp_esg.config import KPI_KEYS
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
