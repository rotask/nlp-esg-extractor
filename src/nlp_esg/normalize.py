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


# Canonical unit aliases. Keys are lowercased lookups.
_UNIT_ALIASES: dict[str, str] = {
    # energy
    "kwh": "kWh", "mwh": "MWh", "gwh": "GWh", "twh": "TWh",
    "gj": "GJ", "tj": "TJ", "pj": "PJ",
    # emissions
    "tco2e": "tCO2e", "t co2e": "tCO2e", "t co2-eq": "tCO2e",
    "tonnes co2e": "tCO2e", "tonnes co2-eq": "tCO2e",
    "ktco2e": "ktCO2e", "kt co2e": "ktCO2e",
    "mtco2e": "MtCO2e", "mt co2e": "MtCO2e",
    # water
    "m3": "m3", "m³": "m3", "cubic metres": "m3", "cubic meters": "m3",
    "ml": "ML", "megalitres": "ML", "megaliters": "ML",
    "kl": "kL", "thousand m3": "kL",  # 1 thousand m3 == 1 kL? No, 1 thousand m3 = 1 ML.
    # Correction below handles "thousand m3" properly.
}

# Overwrite the ambiguous entry: "thousand m3" == 1000 m3 == 1 ML (per ESG reporting convention)
_UNIT_ALIASES["thousand m3"] = "ML"


def canonicalize_unit(unit: str) -> str:
    """Map a written unit to its canonical form. Raises ValueError if unknown."""
    key = unit.strip().lower()
    if key in _UNIT_ALIASES:
        return _UNIT_ALIASES[key]
    raise ValueError(f"Unknown unit: {unit!r}")


# Conversion factors. Key = (from_unit, to_unit), value = multiplier.
# Only canonical -> canonical conversions are defined here; canonicalize_unit maps aliases first.
_CONVERSIONS: dict[tuple[str, str], float] = {
    # energy -> MWh
    ("kWh", "MWh"): 1e-3,
    ("MWh", "MWh"): 1.0,
    ("GWh", "MWh"): 1e3,
    ("TWh", "MWh"): 1e6,
    ("GJ", "MWh"): 1.0 / 3.6,
    ("TJ", "MWh"): 1e3 / 3.6,
    ("PJ", "MWh"): 1e6 / 3.6,
    # emissions -> tCO2e
    ("tCO2e", "tCO2e"): 1.0,
    ("ktCO2e", "tCO2e"): 1e3,
    ("MtCO2e", "tCO2e"): 1e6,
    # water -> m3
    ("m3", "m3"): 1.0,
    ("kL", "m3"): 1.0,       # 1 kL = 1 m3
    ("ML", "m3"): 1e3,       # 1 megalitre = 1000 m3
}


def to_canonical_value(value: float, unit: str, canonical: str) -> float:
    """Convert value from `unit` into `canonical`. Raises ValueError on unknown units."""
    unit_c = canonicalize_unit(unit)
    canonical_c = canonicalize_unit(canonical) if canonical.lower() in _UNIT_ALIASES else canonical
    factor = _CONVERSIONS.get((unit_c, canonical_c))
    if factor is None:
        raise ValueError(f"No conversion from {unit_c!r} to {canonical_c!r}")
    return value * factor
