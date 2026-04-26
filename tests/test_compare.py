import pandas as pd
import pytest
from nlp_esg.compare import build_comparison_table
from nlp_esg.types import KPIExtraction


def _e(company, year, kpi, value, extractor="baseline"):
    return KPIExtraction(
        company=company, report_year=year, kpi=kpi,
        value=value, unit=("tCO2e" if value is not None else None),
        reporting_year=(year if value is not None else None),
        source_snippet=None, source_page=None, confidence=1.0,
        extractor=extractor, flags=[],
    )


def test_most_recent_per_company_selected():
    extractions = [
        _e("acme", 2022, "scope_1_emissions", 100),
        _e("acme", 2024, "scope_1_emissions", 200),   # most recent
        _e("acme", 2023, "scope_1_emissions", 150),
        _e("globex", 2024, "scope_1_emissions", 500),
    ]
    df = build_comparison_table(extractions, extractor="baseline")
    assert df.loc["acme", "scope_1_emissions"] == 200
    assert df.loc["globex", "scope_1_emissions"] == 500


def test_not_reported_as_na():
    extractions = [_e("acme", 2024, "water_consumption", None)]
    df = build_comparison_table(extractions, extractor="baseline")
    assert pd.isna(df.loc["acme", "water_consumption"])


def test_only_selected_extractor_used():
    extractions = [
        _e("acme", 2024, "scope_1_emissions", 100, extractor="baseline"),
        _e("acme", 2024, "scope_1_emissions", 200, extractor="llm"),
    ]
    df_baseline = build_comparison_table(extractions, extractor="baseline")
    df_llm = build_comparison_table(extractions, extractor="llm")
    assert df_baseline.loc["acme", "scope_1_emissions"] == 100
    assert df_llm.loc["acme", "scope_1_emissions"] == 200


def test_all_kpi_columns_present():
    extractions = [_e("acme", 2024, "scope_1_emissions", 100)]
    df = build_comparison_table(extractions, extractor="baseline")
    assert set(df.columns) >= {"scope_1_emissions", "total_energy_consumption", "water_consumption"}


def test_build_run_comparison_joins_two_runs(tmp_path):
    from nlp_esg.compare import build_run_comparison

    (tmp_path / "v1").mkdir()
    (tmp_path / "v2").mkdir()
    pd.DataFrame([
        {"company": "bp", "report_year": 2024, "kpi": "scope_1_emissions",
         "extractor": "llm", "value": 33.7e6, "unit": "tCO2e",
         "reporting_year": 2024, "run_tag": "v1"},
    ]).to_csv(tmp_path / "v1" / "extractions.csv", index=False)
    pd.DataFrame([
        {"company": "bp", "report_year": 2024, "kpi": "scope_1_emissions",
         "extractor": "llm", "value": 33.7e6, "unit": "tCO2e",
         "reporting_year": 2024, "run_tag": "v2"},
    ]).to_csv(tmp_path / "v2" / "extractions.csv", index=False)

    df = build_run_comparison(["v1", "v2"], runs_dir=tmp_path)
    assert ("bp", "scope_1_emissions") in {(r["company"], r["kpi"]) for _, r in df.iterrows()}
    cols = set(df.columns)
    assert "v1_llm_value" in cols
    assert "v2_llm_value" in cols
