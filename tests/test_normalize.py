import pytest
from nlp_esg.normalize import (
    canonicalize_unit,
    normalize_co2,
    parse_number,
    parse_value,
    to_canonical_value,
)


def test_plain_integer():
    assert parse_number("1234") == 1234.0


def test_plain_decimal():
    assert parse_number("1234.5") == 1234.5


def test_thousands_comma():
    # comma followed by exactly three digits -> thousands separator
    assert parse_number("1,234") == 1234.0
    assert parse_number("1,234,567") == 1234567.0


def test_thousands_comma_with_decimal():
    assert parse_number("1,234.5") == 1234.5
    assert parse_number("1,234,567.89") == 1234567.89


def test_eu_decimal_comma():
    # comma not followed by exactly three digits -> decimal comma
    assert parse_number("12,5") == 12.5
    assert parse_number("0,75") == 0.75


def test_space_as_thousands_separator():
    assert parse_number("1 234 567") == 1234567.0


def test_negative_number():
    assert parse_number("-42.5") == -42.5


def test_rejects_non_numeric():
    with pytest.raises(ValueError):
        parse_number("abc")


def test_three_digit_comma_is_thousands():
    # Locks in the heuristic: 3-digit comma groups always thousands.
    assert parse_number("12,345") == 12345.0


def test_two_digit_comma_is_eu_decimal():
    assert parse_number("1,23") == 1.23


def test_negative_with_thousands_and_decimal():
    assert parse_number("-1,234.5") == -1234.5


def test_canonicalize_unit_energy():
    assert canonicalize_unit("GWh") == "GWh"
    assert canonicalize_unit("gwh") == "GWh"
    assert canonicalize_unit("MWh") == "MWh"


def test_canonicalize_unit_emissions():
    assert canonicalize_unit("tCO2e") == "tCO2e"
    assert canonicalize_unit("t CO2-eq") == "tCO2e"
    assert canonicalize_unit("tonnes CO2e") == "tCO2e"
    assert canonicalize_unit("ktCO2e") == "ktCO2e"


def test_canonicalize_unit_water():
    assert canonicalize_unit("m³") == "m3"
    assert canonicalize_unit("cubic metres") == "m3"
    assert canonicalize_unit("ML") == "ML"
    assert canonicalize_unit("megalitres") == "ML"


def test_to_canonical_value_energy_gwh_to_mwh():
    assert to_canonical_value(1.2, "GWh", canonical="MWh") == pytest.approx(1200.0)


def test_to_canonical_value_energy_gj_to_mwh():
    # 1 GJ = 0.27777... MWh
    assert to_canonical_value(3600, "GJ", canonical="MWh") == pytest.approx(1000.0, rel=1e-3)


def test_to_canonical_value_energy_kwh_to_mwh():
    assert to_canonical_value(1_000_000, "kWh", canonical="MWh") == pytest.approx(1000.0)


def test_to_canonical_value_emissions_kt_to_t():
    assert to_canonical_value(1.5, "ktCO2e", canonical="tCO2e") == pytest.approx(1500.0)


def test_to_canonical_value_water_ml_to_m3():
    # 1 megalitre = 1,000 m³
    assert to_canonical_value(5, "ML", canonical="m3") == pytest.approx(5000.0)


def test_to_canonical_value_same_unit_unchanged():
    assert to_canonical_value(42.0, "MWh", canonical="MWh") == 42.0


def test_to_canonical_value_unknown_unit_raises():
    with pytest.raises(ValueError):
        to_canonical_value(1.0, "fathoms", canonical="m3")


def test_to_canonical_value_accepts_alias_canonical_arg():
    # canonical passed as alias form should work the same as canonical form
    assert to_canonical_value(1.0, "GWh", "mwh") == pytest.approx(1000.0)


def test_parse_value_simple():
    assert parse_value("1,234 tCO2e", kpi_unit_family=["tCO2e"]) == (1234.0, "tCO2e")


def test_parse_value_with_magnitude_word():
    # "1.2 million m³" -> (1_200_000, "m3")
    assert parse_value("1.2 million m³", kpi_unit_family=["m3", "m³"]) == (1_200_000.0, "m3")


def test_parse_value_thousand_magnitude():
    assert parse_value("45 thousand tCO2e", kpi_unit_family=["tCO2e"]) == (45_000.0, "tCO2e")


def test_parse_value_billion_magnitude():
    assert parse_value("2.5 billion kWh", kpi_unit_family=["kWh", "MWh"]) == (2.5e9, "kWh")


def test_parse_value_eu_comma_decimal():
    # "12,5 GWh" -> (12.5, "GWh")
    assert parse_value("12,5 GWh", kpi_unit_family=["GWh"]) == (12.5, "GWh")


def test_parse_value_rejects_unit_outside_family():
    assert parse_value("1000 USD", kpi_unit_family=["m3", "ML"]) is None


def test_parse_value_no_number_returns_none():
    assert parse_value("no data available", kpi_unit_family=["tCO2e"]) is None


def test_parse_value_picks_first_match():
    # two candidates in one string; the first matching-unit candidate wins
    assert parse_value(
        "Scope 1 was 12,345 tCO2e, Scope 2 was 6,789 tCO2e",
        kpi_unit_family=["tCO2e"],
    ) == (12345.0, "tCO2e")


def test_parse_value_rejects_unit_substring_match():
    # "5 mlg" must NOT match "5 ml" and be treated as 5 ML water
    assert parse_value("5 mlg of something", kpi_unit_family=["m3", "ML"]) is None


def test_lone_subscript_2_on_own_line():
    """BP pattern: 'MtCOe ... \\n2 \\n' — subscript-2 alone on its own line."""
    raw = "Scope 1 (direct) greenhouse gas MtCOe 33.2 30.4 31.1 32.8 33.7\n2\nemissions k l GHG"
    out = normalize_co2(raw)
    assert "MtCO2e" in out
    # The pre-fix unit form must be gone.
    assert "MtCOe " not in out


def test_lone_subscript_2_does_not_corrupt_normal_text():
    """A '2' on its own line that is NOT preceded by a CO unit must not change."""
    raw = "Methodology section.\n2\nThis paragraph discusses scope 1."
    out = normalize_co2(raw)
    assert out == raw


def test_parse_value_magnitude_before_unit_before_number():
    """Shell pattern: 'million MWh 269' -> 269 * 1e6 MWh."""
    out = parse_value("Total energy consumption [A] million MWh 269 289",
                      kpi_unit_family=["MWh", "GWh"])
    assert out == (269_000_000.0, "MWh")


def test_parse_value_unit_in_parens_then_number():
    """Eni pattern: '(MWh) 84,399,860 ...' — unit wrapped in parens."""
    out = parse_value(
        "Total energy consumption(b) (MWh) 84,399,860 31,146,286 92,738,602",
        kpi_unit_family=["MWh"],
    )
    assert out == (84_399_860.0, "MWh")


def test_parse_value_millionm3_token_split():
    """BP pattern: 'millionm 3' -> split into 'million' + 'm3' for parsing."""
    out = parse_value(
        "Freshwater consumption | millionm 3 | 53.6 | 51.7 | 47.4 | 46.5 | 47.3",
        kpi_unit_family=["m3"],
    )
    # First number after the unit-magnitude token is 53.6 ⇒ 53.6 × 1e6
    assert out == (53_600_000.0, "m3")


def test_parse_value_dot_zero_zero_zero_thousand():
    """Enel pattern: '.000 m3 32,141' represents 'thousand m3' = 32,141 × 1000."""
    out = parse_value(
        "Total water consumption .000 m3 32,141 30,881 1,260 4.1%",
        kpi_unit_family=["m3", "ML"],
    )
    # thousand m3 32,141 -> 32,141 × 1000 m3 = 32,141,000 m3
    assert out == (32_141_000.0, "m3")
