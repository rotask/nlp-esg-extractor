from __future__ import annotations
import re


_THOUSANDS_COMMA = re.compile(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$")
_SPACE_THOUSANDS = re.compile(r"^-?\d{1,3}( \d{3})+(\.\d+)?$")
_EU_DECIMAL = re.compile(r"^-?\d+,\d{1,2}$")  # comma + 1 or 2 trailing digits -> decimal
_PLAIN = re.compile(r"^-?\d+(\.\d+)?$")


def parse_number(text: str) -> float:
    """Parse a human-written number.

    Convention: a comma followed by exactly three digits is always treated
    as a thousands separator (so '12,345' -> 12345.0, not 12.345).
    Raises ValueError if unrecognized.
    """
    s = text.strip()
    if _THOUSANDS_COMMA.match(s):
        return float(s.replace(",", ""))
    if _SPACE_THOUSANDS.match(s):
        return float(s.replace(" ", ""))
    if _EU_DECIMAL.match(s):
        return float(s.replace(",", "."))
    if _PLAIN.match(s):
        return float(s)
    raise ValueError(f"Cannot parse number: {text!r}")
