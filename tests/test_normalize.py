import pytest
from nlp_esg.normalize import canonicalize_unit, parse_number, to_canonical_value


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
