"""Unit canonicalisation and value extraction from raw PDF text.

`canonicalize_unit` maps written units to a canonical form and
`to_canonical_value` performs the cross-unit conversion (e.g. GWh ->
MWh, ML -> m3). `parse_value` finds the first (number, unit) pair
in free text matching a KPI's allowed unit family, handling forward,
magnitude-reverse, and plain-reverse shapes plus magnitude words
("1.2 million m3"). `normalize_co2` is a pre-pass that fixes
pdfplumber's columnar artefacts ("CO 2 e", "MtCOe ... \\n2") so the
embedder and the regexes see contiguous tokens.
"""
from __future__ import annotations
import re


# ---------------------------------------------------------------------------
# CO₂ text normalisation
# PDF renderers often split "CO₂e" into fragments like "CO 2 e", "COe\n2",
# or "MtCO [numbers]\n2eq" (subscript on a separate line).
# ---------------------------------------------------------------------------
_CO2_INLINE = re.compile(r"CO\s+2\s+eq?", re.IGNORECASE)        # "CO 2 e", "CO 2 eq"
_CO2_SUFFIX_AFTER = re.compile(r"COe\s*q?\s*2", re.IGNORECASE)  # "COe2", "COeq 2"
# "MtCO [anything on same line] \n 2eq?" — subscript digit on next line.
# The (?<![A-Za-z]) lookbehind prevents the regex from matching 'co' inside
# words like 'Scope' — without it, the greedy [^\n]* would eat the rest of
# the line and inject '2eq' inside the word, corrupting both ends.
_CO2_NEXT_LINE = re.compile(
    r"(?<![A-Za-z])((?:M[tT]|[kKgG][tT]|[tT])?CO)([^\n]*)\n\s*(2eq?)",
    re.IGNORECASE,
)
# BP pattern: "MtCOe [numbers]\n2\n" — the unit token already contains the trailing
# "e" and the subscript "2" is alone on its own line.
_CO2_LONE_SUBSCRIPT = re.compile(
    r"(?<![A-Za-z])(?P<unit>(?:M[tT]|[kKgG][tT]|[tT])?CO)e([^\n]*)\n\s*2(?=\s*\n)"
)


def normalize_co2(text: str) -> str:
    """Fix common PDF artefacts where CO₂e is split across tokens or lines."""
    def _repl_inline(m: re.Match) -> str:
        return "CO2eq" if "q" in m.group(0).lower() else "CO2e"

    text = _CO2_INLINE.sub(_repl_inline, text)
    text = _CO2_SUFFIX_AFTER.sub("CO2e", text)

    def _repl_next_line(m: re.Match) -> str:
        prefix, middle, suffix = m.group(1), m.group(2), m.group(3)
        unit_tag = "2eq" if "q" in suffix.lower() else "2e"
        return prefix + unit_tag + middle

    text = _CO2_NEXT_LINE.sub(_repl_next_line, text)
    text = _CO2_LONE_SUBSCRIPT.sub(r"\g<unit>2e\2", text)
    return text


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
    "tco2eq": "tCO2e", "t co2eq": "tCO2e",
    "tonnes co2e": "tCO2e", "tonnes co2-eq": "tCO2e", "tonnes co2eq": "tCO2e",
    "ktco2e": "ktCO2e", "kt co2e": "ktCO2e", "ktco2eq": "ktCO2e",
    "mtco2e": "MtCO2e", "mt co2e": "MtCO2e", "mtco2eq": "MtCO2e",
    # water
    "m3": "m3", "m³": "m3", "cubic metres": "m3", "cubic meters": "m3",
    "ml": "ML", "megalitres": "ML", "megaliters": "ML",
    "kl": "kL", "thousand m3": "kL",  # placeholder - overwritten below to ML (intentional two-step)
    "mm3": "Mm3", "mm³": "Mm3",        # million m3 (e.g. Eni 'Mm3' header)
}

# Per ESG reporting convention, "thousand m3" == 1000 m3 == 1 ML. The two-step
# assignment (placeholder above, override here) is deliberate per the Task 4 plan.
_UNIT_ALIASES["thousand m3"] = "ML"


def canonicalize_unit(unit: str) -> str:
    """Map a written unit to its canonical form. Raises ValueError if unknown.

    Strips trailing punctuation (`.,;:`) — LLMs sometimes return units like
    'MtCO2eq.' or 'tCO2e,' picked up from in-line text where punctuation
    follows the unit token.
    """
    key = unit.strip().rstrip(".,;:").lower()
    if key in _UNIT_ALIASES:
        return _UNIT_ALIASES[key]
    raise ValueError(f"Unknown unit: {unit!r}")


_YEAR_SUFFIX_RE = re.compile(r"\.\s*(?:19|20)\d{2}\s*$")


def canonicalize_unit_robust(s: str) -> tuple[float, str] | None:
    """Robust unit resolution for Docling-style headers and cells.

    Returns `(multiplier, canonical_unit)` on success, or `None` if the string
    cannot be resolved. The multiplier accounts for magnitude prefixes
    ('million', 'thousand', 'billion') so callers can apply it to the parsed
    value: `canonical_value = parsed_value * multiplier * conversion_factor`.

    Handles patterns the strict `canonicalize_unit` rejects:
      - Outer whitespace, parens, brackets (Eni 'WATER (Mm 3 )')
      - Internal whitespace (Enel 'MtCO 2eq')
      - Trailing `.YYYY` (Docling compound header 'million cubic metres.2025')
      - Glued magnitude+unit (Shell 'millionMWh' -> 1e6 * MWh)
      - Loose magnitude prefix ('million m3', 'million cubic metres')
      - CO2 subscript artefacts (delegates to `normalize_co2`)
    """
    if not s:
        return None
    text = normalize_co2(s)
    text = text.strip(" \t\n()[]{}*")
    if not text:
        return None
    text = _YEAR_SUFFIX_RE.sub("", text).strip()
    if not text:
        return None

    # Try direct + internal-whitespace-stripped variants.
    for variant in (text, re.sub(r"\s+", "", text)):
        try:
            return (1.0, canonicalize_unit(variant))
        except ValueError:
            pass

    # Try peeling a magnitude prefix off (loose: "million MWh", tight:
    # "millionMWh"). Iterate longest-first so 'million' beats 'mill' if we
    # ever add both.
    text_lc = text.lower()
    text_lc_nospace = re.sub(r"\s+", "", text_lc)
    for mag_word in sorted(_MAGNITUDE, key=len, reverse=True):
        mag_factor = _MAGNITUDE[mag_word]
        for candidate in (text_lc, text_lc_nospace):
            if candidate.startswith(mag_word) and candidate != mag_word:
                rest = candidate[len(mag_word):].strip(" \t().,")
                if not rest:
                    continue
                for variant in (rest, re.sub(r"\s+", "", rest)):
                    try:
                        return (mag_factor, canonicalize_unit(variant))
                    except ValueError:
                        continue
    return None


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
    ("Mm3", "m3"): 1e6,      # 1 megam3 (million m3) = 1,000,000 m3
}


def to_canonical_value(value: float, unit: str, canonical: str) -> float:
    """Convert value from `unit` into `canonical`. Raises ValueError on unknown units."""
    unit_c = canonicalize_unit(unit)
    try:
        canonical_c = canonicalize_unit(canonical)
    except ValueError:
        canonical_c = canonical
    if unit_c == canonical_c:
        return value
    factor = _CONVERSIONS.get((unit_c, canonical_c))
    if factor is None:
        raise ValueError(f"No conversion from {unit_c!r} to {canonical_c!r}")
    return value * factor


_MAGNITUDE = {"thousand": 1e3, "million": 1e6, "billion": 1e9}
_NUMBER_RE = r"[-+]?\d{1,3}(?:[,\s]\d{3})+(?:\.\d+)?|[-+]?\d+(?:[.,]\d+)?"

# When scanning free text we must NOT treat space-as-thousands greedily,
# because pdfplumber-extracted year columns look like "269 289" and would be
# misread as the single number 269,289. Comma-thousands stays allowed.
_NUMBER_IN_TEXT_RE = r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-+]?\d+(?:[.,]\d+)?"

# Separator allowed between a unit token and its associated value when the unit
# precedes the number (tabular flattened text often uses "|", "(", ")", ".", or
# whitespace as separators).
_UNIT_VALUE_SEP = r"[^A-Za-z0-9]{1,8}"

# PDF rendering quirks where a unit + magnitude lose their separator.
# Applied as a pre-pass before parse_value's regexes run.
_UNIT_RENDER_FIXUPS = [
    # "millionm 3" / "millionm3" / "thousandm 3" / "billionm 3" -> split magnitude from m3
    (re.compile(r"\b(thousand|million|billion)m\s*3\b", re.IGNORECASE),
     lambda m: f"{m.group(1).lower()} m3"),
    # ".000 m3" / ".000m3" -> "thousand m3" (the dot-zero-zero-zero is the way some
    # ESG datasheets render the "in thousands" column annotation)
    (re.compile(r"\.000\s*m\s*3\b", re.IGNORECASE), lambda m: "thousand m3"),
    # "m 3" with whitespace -> "m3"  (catches lone "m\s+3" not preceded by a magnitude word)
    (re.compile(r"\bm\s+3\b"), lambda m: "m3"),
]


def _normalize_for_parse(text: str) -> str:
    """Apply unit-rendering fixups so parse_value's regexes see canonical-ish tokens."""
    for pat, repl in _UNIT_RENDER_FIXUPS:
        text = pat.sub(repl, text)
    return text


def parse_value(
    text: str, kpi_unit_family: list[str]
) -> tuple[float, str] | None:
    """
    Find the first (number, unit) pair in `text` whose unit canonicalizes to one
    of the KPI's allowed units. Returns (value, canonical_unit) or None.
    Handles magnitude words ("1.2 million m³") by multiplying in, and the common
    PDF tabular shapes "(MWh) 84,399,860" and "million MWh 269".
    """
    text = _normalize_for_parse(text)

    accepted_canonicals = set()
    for u in kpi_unit_family:
        try:
            accepted_canonicals.add(canonicalize_unit(u))
        except ValueError:
            continue

    unit_alt = "|".join(
        sorted((re.escape(u) for u in _UNIT_ALIASES), key=len, reverse=True)
    )
    mag_alt = "|".join(_MAGNITUDE)

    # Forward: number (magnitude?) unit  e.g. "1.2 million m³", "47.3 m3"
    forward = re.compile(
        rf"({_NUMBER_IN_TEXT_RE})\s*(?:({mag_alt})\s*)?({unit_alt})(?![A-Za-z0-9])",
        re.IGNORECASE,
    )
    for m in forward.finditer(text):
        raw_num, magnitude, raw_unit = m.group(1), m.group(2), m.group(3)
        try:
            value = parse_number(raw_num)
        except ValueError:
            continue
        if magnitude:
            value *= _MAGNITUDE[magnitude.lower()]
        try:
            canonical = canonicalize_unit(raw_unit)
        except ValueError:
            continue
        if canonical in accepted_canonicals:
            return value, canonical

    # Reverse with magnitude: magnitude unit number  e.g. "million MWh 269"
    mag_rev = re.compile(
        rf"({mag_alt})\s+({unit_alt}){_UNIT_VALUE_SEP}({_NUMBER_IN_TEXT_RE})",
        re.IGNORECASE,
    )
    for m in mag_rev.finditer(text):
        magnitude, raw_unit, raw_num = m.group(1), m.group(2), m.group(3)
        try:
            canonical = canonicalize_unit(raw_unit)
        except ValueError:
            continue
        if canonical not in accepted_canonicals:
            continue
        try:
            value = parse_number(raw_num)
        except ValueError:
            continue
        return value * _MAGNITUDE[magnitude.lower()], canonical

    # Reverse plain: unit number  e.g. "MWh 269", "(MWh) 84,399,860"
    rev = re.compile(
        rf"({unit_alt}){_UNIT_VALUE_SEP}({_NUMBER_IN_TEXT_RE})",
        re.IGNORECASE,
    )
    for m in rev.finditer(text):
        raw_unit, raw_num = m.group(1), m.group(2)
        try:
            canonical = canonicalize_unit(raw_unit)
        except ValueError:
            continue
        if canonical not in accepted_canonicals:
            continue
        try:
            value = parse_number(raw_num)
        except ValueError:
            continue
        return value, canonical

    return None
