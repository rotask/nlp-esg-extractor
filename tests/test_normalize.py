import pytest
from nlp_esg.normalize import parse_number


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
