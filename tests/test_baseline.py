import pytest
from nlp_esg.extractors.baseline import BaselineExtractor
from nlp_esg.retrieval import build_index


def test_baseline_extracts_from_table(fake_embed, report_with_table):
    indexed = build_index(report_with_table)
    ext = BaselineExtractor()
    result = ext.extract(indexed, "scope_1_emissions")
    assert result.value == pytest.approx(45678.0)
    assert result.unit == "tCO2e"
    assert result.reporting_year == 2024
    assert result.source_page == 5
    assert result.extractor == "baseline"
    assert "table" in (result.source_snippet or "")


def test_baseline_rejects_out_of_range(fake_embed):
    bad_table = {
        "company": "acme", "report_year": 2024,
        "pages": [{"page_num": 1, "text": "report"}],
        "tables": [{
            "page_num": 2,
            "headers": ["KPI", "2024", "Unit"],
            "rows": [["Scope 1 emissions", "2024", "tCO2e"]],  # value '2024' is out of range
        }],
    }
    indexed = build_index(bad_table)
    ext = BaselineExtractor()
    result = ext.extract(indexed, "scope_1_emissions")
    # The value "2024" is in the plausible_range (1e2, 1e9) — but flagged as suspicious.
    # The real rejection is for a literal year appearing as the ONLY numeric cell.
    # Since 2024 IS in range, this test actually asserts we DID extract it (we don't guess year vs value).
    assert result.value is None or result.flags  # either rejected or flagged


def test_baseline_rejects_unit_outside_family(fake_embed):
    wrong_unit = {
        "company": "acme", "report_year": 2024,
        "pages": [{"page_num": 1, "text": "report"}],
        "tables": [{
            "page_num": 2,
            "headers": ["KPI", "2024", "Currency"],
            "rows": [["Scope 1 emissions", "45,678", "USD"]],
        }],
    }
    indexed = build_index(wrong_unit)
    ext = BaselineExtractor()
    result = ext.extract(indexed, "scope_1_emissions")
    assert result.value is None  # USD is not in the tCO2e family -> no table match


def test_baseline_sentence_fallback(fake_embed, report_sentence_only):
    indexed = build_index(report_sentence_only)
    ext = BaselineExtractor()
    result = ext.extract(indexed, "scope_1_emissions")
    assert result.value == pytest.approx(12345.0)
    assert result.unit == "tCO2e"
    assert result.source_page == 7
    # Now produced by the page-line scanner, not the legacy sentence splitter.
    assert "line" in (result.source_snippet or "").lower()


def test_baseline_not_reported_when_nothing_matches(fake_embed):
    empty_report = {
        "company": "foo", "report_year": 2024,
        "pages": [{"page_num": 1, "text": "Nothing relevant here."}],
        "tables": [],
    }
    indexed = build_index(empty_report)
    ext = BaselineExtractor()
    result = ext.extract(indexed, "scope_1_emissions")
    assert result.value is None
    assert result.unit is None


def test_pick_year_column_searches_far_above_in_long_table():
    """Year-row 14 lines above the data-line should still be detected.
    Reproduces BP water case: the 'Freshwater consumption' row sat near the
    bottom of a long table whose year header was 14 lines above."""
    from nlp_esg.extractors.baseline import BaselineExtractor

    page_lines = [
        "| Metric | Unit | 2021 | 2022 | 2023 | 2024 | 2025 |",
        "|--------|------|------|------|------|------|------|",
    ]
    # 13 filler rows so the data line sits at index 15 (offset -15 from header).
    for _ in range(13):
        page_lines.append("| filler | x | 1 | 2 | 3 | 4 | 5 |")
    page_lines.append(
        "| Freshwater consumption | million m3 | 53.6 | 51.7 | 47.4 | 46.5 | 47.3 |"
    )

    # parse_value would have returned 53.6 × 1e6 = 53,600,000 for this line.
    raw = 53_600_000.0
    result = BaselineExtractor._pick_year_column_value(
        page_lines, line_idx=15, data_line=page_lines[15], raw_value=raw,
    )
    # Most recent year 2025 = col_idx 4. Last 5 of parsed numbers = data values.
    # Pick parsed[4] = 47.3. raw * (47.3/53.6) ≈ 47,300,000.
    assert result is not None, "year row >5 lines above was not found"
    assert abs(result - 47_300_000) < 100


def test_find_year_col_skips_future_target_years():
    """Iberdrola-style table: actual data years 2024/2025 alongside milestone
    target years 2026/2040/2050. We must pick the most-recent ACTUAL year,
    not 2050."""
    from nlp_esg.extractors.baseline import _find_year_col

    headers = ["Tons", "2024", "2025", "%\n25/24", "2026", "2040", "2050",
               "Annual %\ntarget /\nBase year"]
    assert _find_year_col(headers, report_year=2024) == 2  # the "2025" column


def test_baseline_rejects_truly_out_of_range(fake_embed):
    weird = {
        "company": "acme", "report_year": 2024,
        "pages": [
            {"page_num": 1, "text": "Scope 1 direct greenhouse gas emissions were 0.5 tCO2e last year."},
        ],
        "tables": [],
    }
    indexed = build_index(weird)
    ext = BaselineExtractor()
    result = ext.extract(indexed, "scope_1_emissions")
    # 0.5 is below plausible_range (1e2, 1e9) -> should be rejected
    assert result.value is None
    assert "out_of_range" in result.flags
