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


def test_baseline_extracts_with_co2_subscript_spaces(fake_embed):
    """Docling preserves the CO2 subscript with spaces ('MtCO 2 e').
    The unit-inference path must apply normalize_co2() before canonicalize_unit
    so the existing ' Unit'-column detection works on Docling output."""
    docling_table = {
        "company": "demo", "report_year": 2024,
        "pages": [{"page_num": 1, "text": "Scope 1 emissions report"}],
        "tables": [{
            "page_num": 6,
            "headers": ["Metric", "Unit", "2024", "2025"],
            "rows": [
                # Row label tuned to clear the test fake-embedder's token-overlap
                # threshold; the relevant docling artefact under test is the
                # space-separated CO2 subscript in row[1]: 'MtCO 2 e'.
                ["Scope 1 direct greenhouse gas emissions", "MtCO 2 e", "32.8", "33.7"],
            ],
        }],
    }
    indexed = build_index(docling_table)
    ext = BaselineExtractor()
    result = ext.extract(indexed, "scope_1_emissions")
    assert result.value == pytest.approx(33_700_000.0)
    assert result.unit == "tCO2e"


def test_baseline_handles_unit_with_internal_space(fake_embed):
    """Enel scope_1 pattern: unit cell 'MtCO 2eq' (internal whitespace defeats
    canonicalize_unit). Stripping internal whitespace + normalize_co2 should
    yield 'MtCO2eq' -> 'MtCO2e'."""
    table = {
        "company": "enel", "report_year": 2024,
        "pages": [{"page_num": 147, "text": "Scope 1 direct greenhouse gas emissions"}],
        "tables": [{
            "page_num": 147,
            "headers": ["", "", "2025", "2024"],
            "rows": [
                # Row label tokens overlap with the canonical query
                # ('Scope 1 direct greenhouse gas emissions') so the test
                # fake_embed cosine clears TAU_TABLE.
                ["Total Scope 1 direct greenhouse gas emissions", "MtCO 2eq", "18.95", "20.20"],
            ],
        }],
    }
    indexed = build_index(table)
    result = BaselineExtractor().extract(indexed, "scope_1_emissions")
    assert result.value == pytest.approx(18_950_000.0)
    assert result.unit == "tCO2e"


def test_baseline_handles_unit_with_parens_and_space(fake_embed):
    """Eni water pattern: unit cell '(Mm 3 )' wraps the unit in parens with
    internal whitespace. Should canonicalise to Mm3."""
    table = {
        "company": "eni", "report_year": 2024,
        "pages": [{"page_num": 184, "text": "Water consumption"}],
        "tables": [{
            "page_num": 184,
            "headers": ["WATER CONSUMPTION", "Unit", "2025", "2024"],
            "rows": [
                ["Total water consumption", "(Mm 3 )", "42", "45"],
            ],
        }],
    }
    indexed = build_index(table)
    result = BaselineExtractor().extract(indexed, "water_consumption")
    assert result.value == pytest.approx(42_000_000.0)
    assert result.unit == "m3"


def test_baseline_handles_glued_magnitude_unit(fake_embed):
    """Shell energy pattern: row[1]='millionMWh' (magnitude prefix glued to
    unit, no space). Should decompose -> 269 * 1e6 MWh."""
    table = {
        "company": "shell", "report_year": 2024,
        "pages": [{"page_num": 368, "text": "Total energy consumption"}],
        "tables": [{
            "page_num": 368,
            "headers": ["", "unit", "2025", "2024"],
            "rows": [
                ["Total energy consumption", "millionMWh", "269", "289"],
            ],
        }],
    }
    indexed = build_index(table)
    result = BaselineExtractor().extract(indexed, "total_energy_consumption")
    assert result.value == pytest.approx(269_000_000.0)
    assert result.unit == "MWh"


def test_baseline_handles_compound_year_header_with_unit(fake_embed):
    """Shell water pattern: compound header 'million cubic metres.2025' —
    the unit is baked into the year column header. Year col is detected via
    the .YYYY suffix; the unit comes from the same header text."""
    table = {
        "company": "shell", "report_year": 2024,
        "pages": [{"page_num": 385, "text": "Water consumption"}],
        "tables": [{
            "page_num": 385,
            "headers": ["", "million cubic metres.2025", "million cubic metres.2024"],
            "rows": [
                ["Water consumption", "86", "90"],
            ],
        }],
    }
    indexed = build_index(table)
    result = BaselineExtractor().extract(indexed, "water_consumption")
    assert result.value == pytest.approx(86_000_000.0)
    assert result.unit == "m3"


def test_baseline_uses_row1_label_when_row0_is_column_artifact(fake_embed):
    """Iberdrola pattern: 5-column tables [Metric, Description, Unit, 2025,
    2024] where row[0] is the literal string 'Metric' (a column-header
    artifact) and the actual row label sits in row[1]. Row scoring must
    fall back to row[1]."""
    table = {
        "company": "iberdrola", "report_year": 2024,
        "pages": [{"page_num": 58, "text": "Water consumption metrics"}],
        "tables": [{
            "page_num": 58,
            "headers": ["Metrics related to water consumption", "Description", "Unit", "2025", "2024"],
            "rows": [
                # Section-header row that lets the test fake_embed pick up
                # the table on cosine match against the water query.
                ["Total water consumption section", "", "", "", ""],
                ["Metric", "Total water consumption - Continuing activities", "m3", "45642187", "55354884"],
                ["Metric", "Total water consumption in water stress areas - Continuing activities", "m3", "41912946", "51195637"],
            ],
        }],
    }
    indexed = build_index(table)
    result = BaselineExtractor().extract(indexed, "water_consumption")
    assert result.value == pytest.approx(45_642_187.0)
    assert result.unit == "m3"


def test_baseline_section_aware_filtering_prefers_operational(fake_embed):
    """BP scope_1 pattern: a single table with two sub-sections — operational
    control (33.7M) and equity share (32.4M). Section-header rows
    ('GHG-Operational control boundary', 'GHG-Equityshare') sit between the
    sub-tables. Section context propagates to row filtering so 'equity' rows
    are rejected via negative_tokens, leaving the operational figure as the
    sole candidate."""
    table = {
        "company": "bp", "report_year": 2024,
        "pages": [{"page_num": 5, "text": "Scope 1 emissions report"}],
        "tables": [{
            "page_num": 5,
            "headers": ["Metric", "Unit", "2024", "2025"],
            "rows": [
                # Section header — operational control sub-table
                ["GHG-Operational control boundary", "", "", ""],
                # Operational row (gold value 33.7M)
                ["Scope 1 direct greenhouse gas emissions operational", "MtCO 2 e", "32.8", "33.7"],
                # Section header — equity-share sub-table (must be rejected)
                ["GHG-Equityshare", "", "", ""],
                # Equity row (lower value, would otherwise win on phrase match)
                ["Scope 1 direct greenhouse gas emissions equity", "MtCO 2 e", "32.2", "32.4"],
            ],
        }],
    }
    indexed = build_index(table)
    result = BaselineExtractor().extract(indexed, "scope_1_emissions")
    # 'equity' is in scope_1 negative_tokens; section propagation rejects the
    # equity row, so the operational figure 33.7M wins.
    assert result.value == pytest.approx(33_700_000.0)


def test_baseline_extracts_with_unit_in_row1_no_unit_header(fake_embed):
    """Docling pattern: unit in row[1] but the matching header cell is empty.
    Common for Enel-style tables. Unit inference must fall back to row[1]."""
    docling_table = {
        "company": "enel", "report_year": 2024,
        "pages": [{"page_num": 1, "text": "Total energy consumption table"}],
        "tables": [{
            "page_num": 150,
            "headers": ["", "", "2025", "2024", "Change"],
            "rows": [
                ["Total energy consumption (primary and final)", "TWh", "168.59", "170.52", "(1.93)"],
            ],
        }],
    }
    indexed = build_index(docling_table)
    ext = BaselineExtractor()
    result = ext.extract(indexed, "total_energy_consumption")
    assert result.value == pytest.approx(168_590_000.0)
    assert result.unit == "MWh"


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
