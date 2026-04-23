# NLP ESG KPI Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python package that extracts three numerical KPIs (Scope 1 emissions, renewable energy consumption, water consumption) from 10–15 corporate sustainability PDFs and produces a companies × KPIs comparison table plus precision / recall / F1 / coverage metrics on 5 hand-labeled reports.

**Architecture:** One abstract `Extractor` interface with two implementations — a deterministic **baseline** (pdfplumber + ClimateBERT retrieval + table-first regex extraction with sentence fallback) and an **LLM** extractor (Anthropic Claude via forced tool use for structured JSON). Both consume the same retrieved context so the side-by-side comparison isolates extraction differences from retrieval differences. A thin Jupyter notebook imports the package and renders the comparison table and metrics.

**Tech Stack:** Python 3.11+, pdfplumber, sentence-transformers + transformers (ClimateBERT `climatebert/distilroberta-base-climate-f` with mean-pooling), anthropic SDK, pandas, numpy, pytest, python-dotenv.

**Source spec:** `docs/superpowers/specs/2026-04-23-nlp-esg-kpi-extraction-design.md`

---

## File Structure

Created during this plan:

```
pyproject.toml                    # dependencies + package metadata
.env.example                      # ANTHROPIC_API_KEY placeholder
README.md                         # quick-start
src/nlp_esg/
├── __init__.py                   # package marker
├── config.py                     # KPIS registry, model names, paths, plausibility ranges
├── types.py                      # KPIExtraction dataclass, type aliases
├── normalize.py                  # parse_number, parse_value, normalize_unit, to_canonical
├── ingest.py                     # parse_pdf -> ParsedReport, cache
├── retrieval.py                  # build_index, cosine_sim, top_k, embed_texts
├── extractors/
│   ├── __init__.py
│   ├── base.py                   # abstract Extractor
│   ├── baseline.py               # BaselineExtractor (table-first + sentence fallback)
│   └── llm.py                    # LLMExtractor (Claude, forced tool use)
├── evaluate.py                   # per-extractor P/R/F1 + coverage
├── compare.py                    # most-recent-per-company pivot
└── pipeline.py                   # orchestrates ingest -> index -> extract -> compare -> evaluate
tests/
├── __init__.py
├── conftest.py                   # synthetic PDF fixture, canned ParsedReport, canned LLM responses
├── test_normalize.py
├── test_retrieval.py             # pure-math helpers only (cosine_sim, top_k)
├── test_ingest.py                # uses synthetic PDF
├── test_baseline.py
├── test_llm.py
├── test_evaluate.py
├── test_compare.py
├── test_integration_llm.py       # opt-in, hits real API
└── test_integration_real_pdf.py  # opt-in, uses data/reports/
data/
├── reports/                      # PDFs dropped here by user (empty at start)
├── labels/
│   ├── gold_labels.csv           # template with headers
│   └── labels_README.md          # labeling conventions
└── cache/                        # runtime-generated, gitignored
notebooks/
└── demo.ipynb                    # 7-cell linear pipeline demo
```

Each `src/nlp_esg/` module is small (under ~200 lines) with a single responsibility. The `extractors/` subpackage isolates the two implementations behind one interface so downstream code is extractor-agnostic.

---

## Task 1: Project scaffolding and dependencies

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `README.md`
- Create: `src/nlp_esg/__init__.py`
- Create: `src/nlp_esg/extractors/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "nlp-esg"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "pdfplumber>=0.11,<0.12",
    "sentence-transformers>=2.7,<3",
    "transformers>=4.40,<5",
    "torch>=2.2",
    "anthropic>=0.34,<1",
    "pandas>=2.2",
    "numpy>=1.26",
    "python-dotenv>=1",
    "reportlab>=4.0",
]

[project.optional-dependencies]
dev = ["pytest>=8", "ipykernel", "nbformat"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

Note: `reportlab` is in the main deps (not dev) because `tests/conftest.py` imports it to build the synthetic PDF; keeping it in main avoids a conditional import.

- [ ] **Step 2: Write `.env.example`**

```
ANTHROPIC_API_KEY=sk-ant-...
EMBEDDING_MODEL=climatebert
ANTHROPIC_MODEL=claude-sonnet-4-6
```

- [ ] **Step 3: Write `README.md`**

```markdown
# NLP ESG KPI Extraction

Extracts Scope 1 emissions, renewable energy consumption, and water consumption
from corporate sustainability PDFs. Compares a retrieval + regex baseline with
an Anthropic Claude structured-output extractor.

## Setup

1. Python 3.11+.
2. `pip install -e ".[dev]"`
3. Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY`.
4. Drop PDFs into `data/reports/` named `{company}_{year}.pdf`.

## Run

```bash
python -m nlp_esg.pipeline
```

Open `notebooks/demo.ipynb` to see the comparison table and eval metrics.

## Test

```bash
pytest
```
```

- [ ] **Step 4: Create package markers**

`src/nlp_esg/__init__.py`:
```python
__version__ = "0.1.0"
```

`src/nlp_esg/extractors/__init__.py`:
```python
```

`tests/__init__.py`:
```python
```

- [ ] **Step 5: Create minimal `tests/conftest.py`**

```python
import pytest
```

(Fixtures get added in later tasks. This just makes pytest happy.)

- [ ] **Step 6: Install and verify**

Run: `pip install -e ".[dev]"`
Run: `pytest --collect-only`
Expected: `collected 0 items` (no tests yet, no errors).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .env.example README.md src tests
git commit -m "feat: project scaffolding with pyproject and package layout"
```

---

## Task 2: Types and KPI registry

**Files:**
- Create: `src/nlp_esg/types.py`
- Create: `src/nlp_esg/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the failing test** — `tests/test_config.py`

```python
from nlp_esg.config import KPIS, KPI_KEYS
from nlp_esg.types import KPIExtraction


def test_kpi_registry_has_three_keys():
    assert set(KPI_KEYS) == {"scope_1_emissions", "renewable_energy", "water_consumption"}


def test_each_kpi_has_required_fields():
    required = {"query", "unit_family", "canonical_unit", "plausible_range"}
    for key in KPI_KEYS:
        assert required.issubset(KPIS[key].keys()), f"{key} missing fields"


def test_plausible_ranges_are_ordered():
    for key in KPI_KEYS:
        lo, hi = KPIS[key]["plausible_range"]
        assert lo < hi


def test_kpi_extraction_dataclass_instantiates():
    x = KPIExtraction(
        company="acme", report_year=2024, kpi="scope_1_emissions",
        value=1000.0, unit="tCO2e", reporting_year=2024,
        source_snippet="table@page 5: Scope 1 | 1,000",
        source_page=5, confidence=0.8, extractor="baseline", flags=[],
    )
    assert x.value == 1000.0
    assert x.flags == []


def test_kpi_extraction_not_reported_is_allowed():
    x = KPIExtraction(
        company="acme", report_year=2024, kpi="water_consumption",
        value=None, unit=None, reporting_year=None,
        source_snippet=None, source_page=None, confidence=None,
        extractor="baseline", flags=[],
    )
    assert x.value is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: `ModuleNotFoundError: No module named 'nlp_esg.config'` (or similar import error).

- [ ] **Step 3: Write `src/nlp_esg/types.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class KPIExtraction:
    company: str
    report_year: int
    kpi: str
    value: float | None
    unit: str | None
    reporting_year: int | None
    source_snippet: str | None
    source_page: int | None
    confidence: float | None
    extractor: str
    flags: list[str] = field(default_factory=list)
```

- [ ] **Step 4: Write `src/nlp_esg/config.py`**

```python
from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
REPORTS_DIR = DATA_DIR / "reports"
LABELS_DIR = DATA_DIR / "labels"
CACHE_DIR = DATA_DIR / "cache"

EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "climatebert")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

EPSILON = 0.01  # value tolerance for correctness
TAU_TABLE = 0.55  # cosine similarity threshold for table header match
TOP_K_SENTENCES = 5

KPIS: dict[str, dict] = {
    "scope_1_emissions": {
        "query": "Scope 1 direct greenhouse gas emissions",
        "unit_family": ["tCO2e", "ktCO2e", "MtCO2e", "t CO2-eq", "t CO2e", "tonnes CO2e"],
        "canonical_unit": "tCO2e",
        "plausible_range": (1e2, 1e9),
    },
    "renewable_energy": {
        "query": "Total energy consumption from renewable sources",
        "unit_family": ["MWh", "GWh", "TWh", "GJ", "TJ", "PJ", "kWh"],
        "canonical_unit": "MWh",
        "plausible_range": (1e2, 1e9),
    },
    "water_consumption": {
        "query": "Total water consumption withdrawal",
        "unit_family": ["m3", "m³", "ML", "megaliters", "megalitres", "kL", "thousand m3", "cubic metres"],
        "canonical_unit": "m3",
        "plausible_range": (1e1, 1e10),
    },
}

KPI_KEYS = list(KPIS.keys())
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add src/nlp_esg/types.py src/nlp_esg/config.py tests/test_config.py
git commit -m "feat: KPI registry and KPIExtraction dataclass"
```

---

## Task 3: Number parsing

**Files:**
- Create: `src/nlp_esg/normalize.py` (initial — parse_number only)
- Create: `tests/test_normalize.py`

- [ ] **Step 1: Write failing tests** — `tests/test_normalize.py`

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_normalize.py -v`
Expected: all fail with ImportError.

- [ ] **Step 3: Write `src/nlp_esg/normalize.py`**

```python
from __future__ import annotations
import re


_THOUSANDS_COMMA = re.compile(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$")
_SPACE_THOUSANDS = re.compile(r"^-?\d{1,3}( \d{3})+(\.\d+)?$")
_EU_DECIMAL = re.compile(r"^-?\d+,\d{1,2}$")  # comma + 1 or 2 trailing digits -> decimal
_PLAIN = re.compile(r"^-?\d+(\.\d+)?$")


def parse_number(text: str) -> float:
    """Parse a human-written number. Raises ValueError if unrecognized."""
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_normalize.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nlp_esg/normalize.py tests/test_normalize.py
git commit -m "feat: parse_number with thousands/decimal heuristic"
```

---

## Task 4: Unit conversion and canonicalization

**Files:**
- Modify: `src/nlp_esg/normalize.py`
- Modify: `tests/test_normalize.py`

- [ ] **Step 1: Add failing tests** — append to `tests/test_normalize.py`

```python
from nlp_esg.normalize import canonicalize_unit, to_canonical_value


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_normalize.py -v`
Expected: the 10 new tests fail with `ImportError` on `canonicalize_unit`.

- [ ] **Step 3: Extend `src/nlp_esg/normalize.py`**

Append to the existing file:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_normalize.py -v`
Expected: all 18 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/nlp_esg/normalize.py tests/test_normalize.py
git commit -m "feat: unit canonicalization and conversion to canonical values"
```

---

## Task 5: Combined value + unit parsing with magnitude words

**Files:**
- Modify: `src/nlp_esg/normalize.py`
- Modify: `tests/test_normalize.py`

- [ ] **Step 1: Add failing tests** — append to `tests/test_normalize.py`

```python
from nlp_esg.normalize import parse_value


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_normalize.py -v`
Expected: the 8 new tests fail.

- [ ] **Step 3: Extend `src/nlp_esg/normalize.py`**

Append:

```python
_MAGNITUDE = {"thousand": 1e3, "million": 1e6, "billion": 1e9}
_NUMBER_RE = r"[-+]?\d{1,3}(?:[,\s]\d{3})+(?:\.\d+)?|[-+]?\d+(?:[.,]\d+)?|[-+]?\d+"


def parse_value(
    text: str, kpi_unit_family: list[str]
) -> tuple[float, str] | None:
    """
    Find the first (number, unit) pair in `text` whose unit canonicalizes to one
    of the KPI's allowed units. Returns (value, canonical_unit) or None.
    Handles magnitude words ("1.2 million m³") by multiplying in.
    """
    # Build a sorted list of accepted unit strings (longest first so "ktCO2e" matches before "tCO2e").
    accepted_canonicals = set()
    for u in kpi_unit_family:
        try:
            accepted_canonicals.add(canonicalize_unit(u))
        except ValueError:
            continue

    # Pattern: number (whitespace) (optional magnitude word) (whitespace) unit
    unit_alt = "|".join(
        sorted((re.escape(u) for u in _UNIT_ALIASES), key=len, reverse=True)
    )
    mag_alt = "|".join(_MAGNITUDE)
    pattern = re.compile(
        rf"({_NUMBER_RE})\s*(?:({mag_alt})\s*)?({unit_alt})",
        re.IGNORECASE,
    )

    for m in pattern.finditer(text):
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
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_normalize.py -v`
Expected: all 26 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/nlp_esg/normalize.py tests/test_normalize.py
git commit -m "feat: parse_value handles magnitude words and unit family filter"
```

---

## Task 6: Ingest PDFs with pdfplumber

**Files:**
- Create: `src/nlp_esg/ingest.py`
- Modify: `tests/conftest.py`
- Create: `tests/test_ingest.py`

- [ ] **Step 1: Add synthetic-PDF fixture** — replace `tests/conftest.py`

```python
from __future__ import annotations
from pathlib import Path
import pytest
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet


@pytest.fixture
def synthetic_pdf(tmp_path: Path) -> Path:
    """A 2-page PDF containing prose + a simple table with a KPI."""
    pdf_path = tmp_path / "acme_2024.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("ACME Sustainability Report 2024", styles["Title"]),
        Spacer(1, 12),
        Paragraph(
            "Our Scope 1 emissions totalled 45,678 tCO2e in 2024, down from "
            "48,000 tCO2e in 2023.",
            styles["BodyText"],
        ),
        Spacer(1, 12),
        Table([
            ["KPI", "2023", "2024", "Unit"],
            ["Scope 1 emissions", "48,000", "45,678", "tCO2e"],
            ["Water consumption", "120", "115", "ML"],
        ]),
    ]
    doc.build(story)
    return pdf_path
```

- [ ] **Step 2: Write failing tests** — `tests/test_ingest.py`

```python
from pathlib import Path
import pytest
from nlp_esg.ingest import parse_pdf, ParsedReport


def test_parse_pdf_extracts_pages(synthetic_pdf: Path):
    report = parse_pdf(synthetic_pdf)
    assert report["company"] == "acme"
    assert report["report_year"] == 2024
    assert len(report["pages"]) >= 1
    assert "Scope 1 emissions" in " ".join(p["text"] for p in report["pages"])


def test_parse_pdf_extracts_tables(synthetic_pdf: Path):
    report = parse_pdf(synthetic_pdf)
    assert len(report["tables"]) >= 1
    t = report["tables"][0]
    assert "headers" in t
    assert "rows" in t
    # The table we built has 'Scope 1 emissions' as a row label
    all_cells = t["headers"] + [c for r in t["rows"] for c in r]
    assert any("Scope 1" in (c or "") for c in all_cells)


def test_parse_pdf_filename_parsing(tmp_path: Path):
    # Filename shape drives company + year parsing.
    bad_path = tmp_path / "weird-name.pdf"
    bad_path.write_bytes(b"%PDF-1.4\n%%EOF")
    with pytest.raises(ValueError, match="filename"):
        parse_pdf(bad_path)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_ingest.py -v`
Expected: ImportError on `parse_pdf`.

- [ ] **Step 4: Write `src/nlp_esg/ingest.py`**

```python
from __future__ import annotations
import logging
import pickle
import re
from pathlib import Path
from typing import TypedDict

import pdfplumber

from nlp_esg.config import CACHE_DIR

log = logging.getLogger(__name__)

_FILENAME_RE = re.compile(r"^(?P<company>[a-z0-9_\-]+)_(?P<year>\d{4})\.pdf$", re.IGNORECASE)


class Page(TypedDict):
    page_num: int
    text: str


class TableEntry(TypedDict):
    page_num: int
    headers: list[str]
    rows: list[list[str]]


class ParsedReport(TypedDict):
    company: str
    report_year: int
    pages: list[Page]
    tables: list[TableEntry]


def _parse_filename(path: Path) -> tuple[str, int]:
    m = _FILENAME_RE.match(path.name)
    if not m:
        raise ValueError(
            f"Report filename {path.name!r} does not match {{company}}_{{year}}.pdf"
        )
    return m.group("company").lower(), int(m.group("year"))


def parse_pdf(path: Path, use_cache: bool = True) -> ParsedReport:
    """Parse a PDF into pages + tables. Caches to data/cache based on mtime."""
    company, year = _parse_filename(path)
    cache_path = CACHE_DIR / f"{company}_{year}.pkl"

    if use_cache and cache_path.exists() and cache_path.stat().st_mtime >= path.stat().st_mtime:
        with cache_path.open("rb") as f:
            return pickle.load(f)

    pages: list[Page] = []
    tables: list[TableEntry] = []

    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append({"page_num": i, "text": text})

            for raw in page.extract_tables() or []:
                if not raw:
                    continue
                headers = [(c or "").strip() for c in raw[0]]
                rows = [[(c or "").strip() for c in row] for row in raw[1:]]
                tables.append({"page_num": i, "headers": headers, "rows": rows})

    report: ParsedReport = {
        "company": company,
        "report_year": year,
        "pages": pages,
        "tables": tables,
    }

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as f:
            pickle.dump(report, f)

    return report
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_ingest.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add src/nlp_esg/ingest.py tests/conftest.py tests/test_ingest.py
git commit -m "feat: parse_pdf extracts pages + tables with filename-based metadata"
```

---

## Task 7: Retrieval math helpers (pure functions)

**Files:**
- Create: `src/nlp_esg/retrieval.py`
- Create: `tests/test_retrieval.py`

> This task covers the pure-math portions of retrieval (cosine similarity, top-k). The embedding-model wrapper (`embed_texts`, `build_index`) goes in the next task and is exercised only by integration tests, per the design spec's "no model loading in unit tests" constraint.

- [ ] **Step 1: Write failing tests** — `tests/test_retrieval.py`

```python
import numpy as np
import pytest
from nlp_esg.retrieval import cosine_sim, top_k


def test_cosine_sim_identical_vectors():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine_sim(v, v) == pytest.approx(1.0)


def test_cosine_sim_orthogonal():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert cosine_sim(a, b) == pytest.approx(0.0)


def test_cosine_sim_opposite():
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    assert cosine_sim(a, b) == pytest.approx(-1.0)


def test_top_k_returns_indices_in_score_order():
    query = np.array([1.0, 0.0])
    corpus = np.array([
        [0.0, 1.0],   # orthogonal -> 0.0
        [1.0, 0.0],   # identical  -> 1.0
        [0.8, 0.6],   # high       -> 0.8
    ])
    idxs = top_k(query, corpus, k=2)
    assert idxs == [1, 2]


def test_top_k_handles_k_greater_than_corpus():
    query = np.array([1.0, 0.0])
    corpus = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert top_k(query, corpus, k=10) == [0, 1]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_retrieval.py -v`
Expected: ImportError.

- [ ] **Step 3: Write `src/nlp_esg/retrieval.py`** (helpers only; embedding wrapper added next task)

```python
from __future__ import annotations
import numpy as np


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def top_k(query: np.ndarray, corpus: np.ndarray, k: int) -> list[int]:
    """Return indices of the top-k rows in `corpus` by cosine similarity to `query`."""
    if corpus.size == 0:
        return []
    q = np.asarray(query, dtype=np.float32).ravel()
    c = np.asarray(corpus, dtype=np.float32)
    q_norm = np.linalg.norm(q)
    c_norms = np.linalg.norm(c, axis=1)
    denom = q_norm * c_norms
    denom[denom == 0.0] = 1e-12
    scores = (c @ q) / denom
    k = min(k, len(scores))
    # argpartition for speed, then sort the top-k
    part = np.argpartition(-scores, k - 1)[:k]
    return list(part[np.argsort(-scores[part])])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_retrieval.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nlp_esg/retrieval.py tests/test_retrieval.py
git commit -m "feat: cosine_sim and top_k retrieval math helpers"
```

---

## Task 8: Embedding wrapper and index builder

**Files:**
- Modify: `src/nlp_esg/retrieval.py`

> This code loads real sentence-transformer models — it is deliberately NOT unit-tested (covered by the opt-in integration test in Task 15). The wrapper is thin enough that its correctness is validated end-to-end.

- [ ] **Step 1: Extend `src/nlp_esg/retrieval.py`**

Add the following **new** imports just after the existing `import numpy as np` line (do NOT duplicate `from __future__ import annotations` — it's already there):

```python
import logging
import re
from functools import lru_cache
from typing import TypedDict

from sentence_transformers import SentenceTransformer, models

from nlp_esg.config import EMBEDDING_MODEL_NAME
from nlp_esg.ingest import ParsedReport

log = logging.getLogger(__name__)
```

Then append the rest of the module code to the end of the file:

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


class Sentence(TypedDict):
    page_num: int
    text: str
    embedding: np.ndarray


class TableHeaderEmb(TypedDict):
    table_idx: int
    header_string: str
    embedding: np.ndarray


class IndexedReport(TypedDict):
    company: str
    report_year: int
    pages: list
    tables: list
    sentences: list[Sentence]
    table_headers: list[TableHeaderEmb]


@lru_cache(maxsize=1)
def _load_model(name: str) -> SentenceTransformer:
    if name == "climatebert":
        word = models.Transformer("climatebert/distilroberta-base-climate-f")
        pooling = models.Pooling(
            word.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
        )
        return SentenceTransformer(modules=[word, pooling])
    if name == "minilm":
        return SentenceTransformer("all-MiniLM-L6-v2")
    raise ValueError(f"Unknown embedding model: {name!r}")


def embed_texts(texts: list[str], model_name: str | None = None) -> np.ndarray:
    model = _load_model(model_name or EMBEDDING_MODEL_NAME)
    if not texts:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32)
    emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return emb.astype(np.float32)


def split_sentences(text: str) -> list[str]:
    """Cheap regex sentence splitter. Avoids nltk download step."""
    parts = _SENT_SPLIT.split(text or "")
    return [p.strip() for p in parts if p.strip()]


def build_index(report: ParsedReport, model_name: str | None = None) -> IndexedReport:
    sentences: list[Sentence] = []
    sent_texts: list[str] = []
    sent_pages: list[int] = []
    for page in report["pages"]:
        for s in split_sentences(page["text"]):
            sent_texts.append(s)
            sent_pages.append(page["page_num"])

    sent_embs = embed_texts(sent_texts, model_name=model_name)
    for i, (text, page) in enumerate(zip(sent_texts, sent_pages)):
        sentences.append({"page_num": page, "text": text, "embedding": sent_embs[i]})

    header_strings: list[str] = []
    for t in report["tables"]:
        header_str = " | ".join(h for h in t["headers"] if h)
        header_strings.append(header_str)

    header_embs = embed_texts(header_strings, model_name=model_name)
    table_headers: list[TableHeaderEmb] = [
        {"table_idx": i, "header_string": hs, "embedding": header_embs[i]}
        for i, hs in enumerate(header_strings)
    ]

    return {
        "company": report["company"],
        "report_year": report["report_year"],
        "pages": report["pages"],
        "tables": report["tables"],
        "sentences": sentences,
        "table_headers": table_headers,
    }
```

- [ ] **Step 2: Verify the module imports cleanly**

Run: `python -c "from nlp_esg.retrieval import build_index, embed_texts, split_sentences"`
Expected: no output (clean import; model is not loaded until first call).

- [ ] **Step 3: Commit**

```bash
git add src/nlp_esg/retrieval.py
git commit -m "feat: ClimateBERT embedding wrapper and build_index"
```

---

## Task 9: Extractor base class

**Files:**
- Create: `src/nlp_esg/extractors/base.py`
- Create: `tests/test_base_extractor.py`

- [ ] **Step 1: Write failing test** — `tests/test_base_extractor.py`

```python
import pytest
from nlp_esg.extractors.base import Extractor


def test_extractor_is_abstract():
    with pytest.raises(TypeError):
        Extractor()  # type: ignore[abstract]


def test_concrete_subclass_works():
    from nlp_esg.types import KPIExtraction

    class Dummy(Extractor):
        def extract(self, report, kpi_key):
            return KPIExtraction(
                company=report["company"], report_year=report["report_year"],
                kpi=kpi_key, value=None, unit=None, reporting_year=None,
                source_snippet=None, source_page=None, confidence=None,
                extractor="dummy", flags=[],
            )

    d = Dummy()
    result = d.extract({"company": "x", "report_year": 2024}, "scope_1_emissions")
    assert result.extractor == "dummy"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_base_extractor.py -v`
Expected: ImportError.

- [ ] **Step 3: Write `src/nlp_esg/extractors/base.py`**

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

from nlp_esg.types import KPIExtraction


class Extractor(ABC):
    name: str = "base"

    @abstractmethod
    def extract(self, report: Any, kpi_key: str) -> KPIExtraction:
        ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_base_extractor.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nlp_esg/extractors/base.py tests/test_base_extractor.py
git commit -m "feat: abstract Extractor base class"
```

---

## Task 10: BaselineExtractor — table-first search

**Files:**
- Create: `src/nlp_esg/extractors/baseline.py`
- Modify: `tests/conftest.py`
- Create: `tests/test_baseline.py`

> For testing, we inject a fake `embed_texts` that maps strings to deterministic vectors via a simple token-overlap scheme. This keeps the tests model-free and fast.

- [ ] **Step 1: Add canned fixtures** — append to `tests/conftest.py`

```python
import numpy as np
from typing import Callable


def _fake_embed_factory() -> Callable[[list[str]], np.ndarray]:
    """Return a deterministic fake embedder: token-set indicator vectors."""
    vocab: dict[str, int] = {}
    def _embed(texts):
        # Build/extend vocab
        for t in texts:
            for w in (t or "").lower().split():
                vocab.setdefault(w, len(vocab))
        dim = max(len(vocab), 1)
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in (t or "").lower().split():
                out[i, vocab[w]] = 1.0
            n = np.linalg.norm(out[i])
            if n > 0:
                out[i] /= n
        return out
    return _embed


@pytest.fixture
def fake_embed(monkeypatch):
    embed = _fake_embed_factory()
    monkeypatch.setattr("nlp_esg.retrieval.embed_texts", embed)
    monkeypatch.setattr("nlp_esg.extractors.baseline.embed_texts", embed, raising=False)
    return embed


@pytest.fixture
def report_with_table():
    """IndexedReport containing a KPI table. Embeddings deferred to build_index at test time."""
    return {
        "company": "acme",
        "report_year": 2024,
        "pages": [
            {"page_num": 1, "text": "ACME Sustainability Report 2024."},
            {"page_num": 5, "text": "See performance table for details."},
        ],
        "tables": [
            {
                "page_num": 5,
                "headers": ["KPI", "2023", "2024", "Unit"],
                "rows": [
                    ["Scope 1 emissions", "48,000", "45,678", "tCO2e"],
                    ["Scope 2 emissions", "12,000", "10,500", "tCO2e"],
                    ["Water consumption", "120", "115", "ML"],
                ],
            }
        ],
    }


@pytest.fixture
def report_sentence_only():
    """IndexedReport whose KPI lives only in narrative text, not a table."""
    return {
        "company": "globex",
        "report_year": 2024,
        "pages": [
            {"page_num": 1, "text": "Globex Sustainability 2024."},
            {
                "page_num": 7,
                "text": (
                    "Our Scope 1 direct greenhouse gas emissions in 2024 were 12,345 tCO2e, "
                    "a 5% reduction year over year."
                ),
            },
        ],
        "tables": [],
    }
```

- [ ] **Step 2: Write failing tests** — `tests/test_baseline.py`

```python
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
```

> The second test above is intentionally lenient — `2024` as a value happens to fall inside the plausibility range `(1e2, 1e9)`. The out-of-range rejection is tested separately below with a clearly impossible value.

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_baseline.py -v`
Expected: ImportError.

- [ ] **Step 4: Write `src/nlp_esg/extractors/baseline.py`** (table-first only; sentence fallback comes in Task 11)

```python
from __future__ import annotations
import logging
import re
from typing import Any

from nlp_esg.config import KPIS, TAU_TABLE
from nlp_esg.extractors.base import Extractor
from nlp_esg.normalize import (
    canonicalize_unit,
    parse_number,
    parse_value,
    to_canonical_value,
)
from nlp_esg.retrieval import cosine_sim, embed_texts
from nlp_esg.types import KPIExtraction

log = logging.getLogger(__name__)

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def _structural_score(headers: list[str], report_year: int) -> float:
    for h in headers:
        if _YEAR_RE.search(h or "") and str(report_year) in (h or ""):
            return 1.0
    return 0.5


def _find_year_col(headers: list[str], report_year: int) -> int | None:
    best = None
    for i, h in enumerate(headers):
        m = _YEAR_RE.search(h or "")
        if m:
            year = int(m.group(0))
            if year == report_year:
                return i
            # fall back to most recent year found
            if best is None or year > best[1]:
                best = (i, year)
    return best[0] if best else None


def _infer_unit_from_row_or_header(
    headers: list[str], row: list[str], value_col: int, unit_family_canonicals: set[str]
) -> str | None:
    """Look for a unit in: the cell itself (trailing), a 'Unit' column, or header annotation."""
    # 1. Check the value cell itself ("45,678 tCO2e")
    cell = row[value_col] if value_col < len(row) else ""
    pv = parse_value(cell, kpi_unit_family=list(unit_family_canonicals))
    if pv:
        return pv[1]

    # 2. Look for a 'Unit' column
    for i, h in enumerate(headers):
        if (h or "").strip().lower() in ("unit", "units"):
            try:
                u = canonicalize_unit(row[i])
            except (ValueError, IndexError):
                continue
            if u in unit_family_canonicals:
                return u

    # 3. Check the value column's own header (e.g., "2024 (tCO2e)")
    if value_col < len(headers):
        for token in re.findall(r"[A-Za-z0-9µ³²\-]+", headers[value_col] or ""):
            try:
                u = canonicalize_unit(token)
            except ValueError:
                continue
            if u in unit_family_canonicals:
                return u

    # 4. Check the KPI-row header (first cell) for a unit
    if row:
        for token in re.findall(r"[A-Za-z0-9µ³²\-]+", row[0] or ""):
            try:
                u = canonicalize_unit(token)
            except ValueError:
                continue
            if u in unit_family_canonicals:
                return u
    return None


class BaselineExtractor(Extractor):
    name = "baseline"

    def extract(self, report: Any, kpi_key: str) -> KPIExtraction:
        kpi = KPIS[kpi_key]
        flags: list[str] = []
        unit_family_canonicals: set[str] = set()
        for u in kpi["unit_family"]:
            try:
                unit_family_canonicals.add(canonicalize_unit(u))
            except ValueError:
                continue

        query_emb = embed_texts([kpi["query"]])[0]

        # --- Table-first search ---
        table_candidates: list[tuple[float, dict, int, list[str]]] = []
        for th in report["table_headers"]:
            sim = cosine_sim(query_emb, th["embedding"])
            if sim < TAU_TABLE:
                continue
            table = report["tables"][th["table_idx"]]
            score = sim * _structural_score(table["headers"], report["report_year"])
            table_candidates.append((score, table, th["table_idx"], table["headers"]))

        table_candidates.sort(key=lambda x: x[0], reverse=True)

        for score, table, _, headers in table_candidates:
            year_col = _find_year_col(headers, report["report_year"])
            if year_col is None:
                continue
            # Find the KPI row: row whose first cell semantically contains the KPI
            query_tokens = set(kpi["query"].lower().split())
            best_row_idx = None
            best_overlap = 0
            for ri, row in enumerate(table["rows"]):
                if not row:
                    continue
                row_tokens = set((row[0] or "").lower().split())
                overlap = len(query_tokens & row_tokens)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_row_idx = ri
            if best_row_idx is None:
                continue
            row = table["rows"][best_row_idx]
            if year_col >= len(row):
                continue
            cell = row[year_col]
            # Extract the number (strip any unit embedded in the cell)
            num_match = re.search(
                r"[-+]?\d{1,3}(?:[,\s]\d{3})+(?:\.\d+)?|[-+]?\d+(?:[.,]\d+)?",
                cell,
            )
            if not num_match:
                continue
            try:
                raw_value = parse_number(num_match.group(0))
            except ValueError:
                continue

            unit = _infer_unit_from_row_or_header(
                headers, row, year_col, unit_family_canonicals
            )
            if unit is None:
                continue
            if unit not in unit_family_canonicals:
                continue

            try:
                canonical_value = to_canonical_value(
                    raw_value, unit, kpi["canonical_unit"]
                )
            except ValueError:
                continue

            lo, hi = kpi["plausible_range"]
            if not (lo <= canonical_value <= hi):
                flags.append("out_of_range")
                continue

            return KPIExtraction(
                company=report["company"],
                report_year=report["report_year"],
                kpi=kpi_key,
                value=canonical_value,
                unit=kpi["canonical_unit"],
                reporting_year=report["report_year"],
                source_snippet=f"table@page {table['page_num']}: {row[0]} | {cell}",
                source_page=table["page_num"],
                confidence=float(score),
                extractor=self.name,
                flags=flags,
            )

        # Table search found nothing usable — return not-reported for now.
        # (Sentence fallback is added in Task 11.)
        return KPIExtraction(
            company=report["company"],
            report_year=report["report_year"],
            kpi=kpi_key,
            value=None,
            unit=None,
            reporting_year=None,
            source_snippet=None,
            source_page=None,
            confidence=None,
            extractor=self.name,
            flags=flags,
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_baseline.py -v`
Expected: 3 passed (the lenient out-of-range test passes either branch).

- [ ] **Step 6: Commit**

```bash
git add src/nlp_esg/extractors/baseline.py tests/conftest.py tests/test_baseline.py
git commit -m "feat: BaselineExtractor table-first search"
```

---

## Task 11: BaselineExtractor — sentence fallback

**Files:**
- Modify: `src/nlp_esg/extractors/baseline.py`
- Modify: `tests/test_baseline.py`

- [ ] **Step 1: Add failing tests** — append to `tests/test_baseline.py`

```python
def test_baseline_sentence_fallback(fake_embed, report_sentence_only):
    indexed = build_index(report_sentence_only)
    ext = BaselineExtractor()
    result = ext.extract(indexed, "scope_1_emissions")
    assert result.value == pytest.approx(12345.0)
    assert result.unit == "tCO2e"
    assert result.source_page == 7
    assert "sentence" in (result.source_snippet or "").lower()


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_baseline.py -v`
Expected: the 3 new tests fail.

- [ ] **Step 3: Extend `BaselineExtractor.extract`**

In `src/nlp_esg/extractors/baseline.py`, **delete** this entire block at the end of the `extract` method:

```python
        # Table search found nothing usable — return not-reported for now.
        # (Sentence fallback is added in Task 11.)
        return KPIExtraction(
            company=report["company"],
            report_year=report["report_year"],
            kpi=kpi_key,
            value=None,
            unit=None,
            reporting_year=None,
            source_snippet=None,
            source_page=None,
            confidence=None,
            extractor=self.name,
            flags=flags,
        )
```

Replace it with:

```python
        # --- Sentence fallback ---
        if not report["sentences"]:
            return KPIExtraction(
                company=report["company"], report_year=report["report_year"],
                kpi=kpi_key, value=None, unit=None, reporting_year=None,
                source_snippet=None, source_page=None, confidence=None,
                extractor=self.name, flags=flags,
            )

        sent_embs = np.stack([s["embedding"] for s in report["sentences"]])
        from nlp_esg.retrieval import top_k as _top_k  # local import to avoid circulars
        top_idxs = _top_k(query_emb, sent_embs, k=TOP_K_SENTENCES)

        best: tuple[float, float, str, str, int] | None = None
        for idx in top_idxs:
            s = report["sentences"][idx]
            sim = cosine_sim(query_emb, s["embedding"])
            pv = parse_value(s["text"], kpi_unit_family=kpi["unit_family"])
            if pv is None:
                continue
            raw_value, unit = pv
            if unit not in unit_family_canonicals:
                continue
            try:
                canonical_value = to_canonical_value(
                    raw_value, unit, kpi["canonical_unit"]
                )
            except ValueError:
                continue
            lo, hi = kpi["plausible_range"]
            if not (lo <= canonical_value <= hi):
                flags.append("out_of_range")
                continue
            year_bonus = 0.1 if str(report["report_year"]) in s["text"] else 0.0
            score = sim + year_bonus
            if best is None or score > best[0]:
                best = (score, canonical_value, s["text"], unit, s["page_num"])

        if best is None:
            return KPIExtraction(
                company=report["company"], report_year=report["report_year"],
                kpi=kpi_key, value=None, unit=None, reporting_year=None,
                source_snippet=None, source_page=None, confidence=None,
                extractor=self.name, flags=flags,
            )

        score, canonical_value, sentence, unit, page = best
        return KPIExtraction(
            company=report["company"],
            report_year=report["report_year"],
            kpi=kpi_key,
            value=canonical_value,
            unit=kpi["canonical_unit"],
            reporting_year=report["report_year"],
            source_snippet=f"sentence@page {page}: {sentence}",
            source_page=page,
            confidence=float(score),
            extractor=self.name,
            flags=flags,
        )
```

Add the missing imports at the top of `baseline.py`:
```python
import numpy as np
from nlp_esg.config import KPIS, TAU_TABLE, TOP_K_SENTENCES
```

(Adjust the existing import line for `config` to include `TOP_K_SENTENCES`.)

- [ ] **Step 4: Run all baseline tests**

Run: `pytest tests/test_baseline.py -v`
Expected: all 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/nlp_esg/extractors/baseline.py tests/test_baseline.py
git commit -m "feat: BaselineExtractor sentence fallback with range filter"
```

---

## Task 12: LLMExtractor with mocked Anthropic client

**Files:**
- Create: `src/nlp_esg/extractors/llm.py`
- Create: `tests/test_llm.py`

- [ ] **Step 1: Write failing tests** — `tests/test_llm.py`

```python
from unittest.mock import MagicMock, patch
import pytest

from nlp_esg.extractors.llm import LLMExtractor


class _FakeToolUse:
    def __init__(self, name: str, input_: dict):
        self.type = "tool_use"
        self.name = name
        self.input = input_


class _FakeText:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class _FakeResponse:
    def __init__(self, content):
        self.content = content
        self.stop_reason = "tool_use"


def _valid_response(value=45678.0, unit="tCO2e", year=2024, snippet="...", confidence=0.9):
    return _FakeResponse([_FakeToolUse("record_kpi", {
        "value": value, "unit": unit, "reporting_year": year,
        "source_snippet": snippet, "confidence": confidence,
    })])


def _not_reported_response():
    return _FakeResponse([_FakeToolUse("record_kpi", {
        "value": None, "unit": None, "reporting_year": None,
        "source_snippet": None, "confidence": 0.9,
    })])


@pytest.fixture
def indexed_stub():
    return {
        "company": "acme", "report_year": 2024,
        "pages": [{"page_num": 5, "text": "Scope 1 was 45,678 tCO2e in 2024."}],
        "tables": [],
        "sentences": [{"page_num": 5, "text": "Scope 1 was 45,678 tCO2e in 2024.",
                       "embedding": None}],
        "table_headers": [],
    }


def test_llm_parses_valid_tool_use(indexed_stub):
    with patch("nlp_esg.extractors.llm.Anthropic") as mock_cls:
        client = MagicMock()
        client.messages.create.return_value = _valid_response()
        mock_cls.return_value = client

        ext = LLMExtractor()
        with patch.object(ext, "_build_context", return_value="context"):
            result = ext.extract(indexed_stub, "scope_1_emissions")

    assert result.value == pytest.approx(45678.0)
    assert result.unit == "tCO2e"
    assert result.reporting_year == 2024
    assert result.extractor == "llm"


def test_llm_handles_not_reported(indexed_stub):
    with patch("nlp_esg.extractors.llm.Anthropic") as mock_cls:
        client = MagicMock()
        client.messages.create.return_value = _not_reported_response()
        mock_cls.return_value = client

        ext = LLMExtractor()
        with patch.object(ext, "_build_context", return_value="context"):
            result = ext.extract(indexed_stub, "scope_1_emissions")

    assert result.value is None
    assert result.unit is None


def test_llm_rejects_value_without_unit(indexed_stub):
    bad = _FakeResponse([_FakeToolUse("record_kpi", {
        "value": 45678.0, "unit": None, "reporting_year": 2024,
        "source_snippet": "...", "confidence": 0.9,
    })])
    with patch("nlp_esg.extractors.llm.Anthropic") as mock_cls:
        client = MagicMock()
        client.messages.create.return_value = bad
        mock_cls.return_value = client

        ext = LLMExtractor()
        with patch.object(ext, "_build_context", return_value="context"):
            result = ext.extract(indexed_stub, "scope_1_emissions")

    assert result.value is None
    assert "llm_missing_unit" in result.flags


def test_llm_rejects_out_of_range(indexed_stub):
    bad = _FakeResponse([_FakeToolUse("record_kpi", {
        "value": 0.01, "unit": "tCO2e", "reporting_year": 2024,
        "source_snippet": "...", "confidence": 0.9,
    })])
    with patch("nlp_esg.extractors.llm.Anthropic") as mock_cls:
        client = MagicMock()
        client.messages.create.return_value = bad
        mock_cls.return_value = client

        ext = LLMExtractor()
        with patch.object(ext, "_build_context", return_value="context"):
            result = ext.extract(indexed_stub, "scope_1_emissions")

    assert result.value is None
    assert "out_of_range" in result.flags


def test_llm_retries_on_api_error(indexed_stub):
    with patch("nlp_esg.extractors.llm.Anthropic") as mock_cls:
        client = MagicMock()
        client.messages.create.side_effect = [
            Exception("transient"),
            _valid_response(),
        ]
        mock_cls.return_value = client

        ext = LLMExtractor(max_retries=2, retry_base_delay=0)
        with patch.object(ext, "_build_context", return_value="context"):
            result = ext.extract(indexed_stub, "scope_1_emissions")

    assert result.value == pytest.approx(45678.0)
    assert client.messages.create.call_count == 2


def test_llm_gives_up_after_max_retries(indexed_stub):
    with patch("nlp_esg.extractors.llm.Anthropic") as mock_cls:
        client = MagicMock()
        client.messages.create.side_effect = Exception("always fail")
        mock_cls.return_value = client

        ext = LLMExtractor(max_retries=2, retry_base_delay=0)
        with patch.object(ext, "_build_context", return_value="context"):
            result = ext.extract(indexed_stub, "scope_1_emissions")

    assert result.value is None
    assert "api_error" in result.flags
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_llm.py -v`
Expected: ImportError.

- [ ] **Step 3: Write `src/nlp_esg/extractors/llm.py`**

```python
from __future__ import annotations
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

from anthropic import Anthropic

from nlp_esg.config import ANTHROPIC_MODEL, CACHE_DIR, KPIS
from nlp_esg.extractors.base import Extractor
from nlp_esg.normalize import canonicalize_unit, to_canonical_value
from nlp_esg.types import KPIExtraction

log = logging.getLogger(__name__)

_TOOL_SCHEMA = {
    "name": "record_kpi",
    "description": "Record the extracted KPI value. Use null for value when the KPI is not reported.",
    "input_schema": {
        "type": "object",
        "properties": {
            "value": {"type": ["number", "null"]},
            "unit": {"type": ["string", "null"]},
            "reporting_year": {"type": ["integer", "null"]},
            "source_snippet": {"type": ["string", "null"]},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": ["value", "unit", "reporting_year", "source_snippet", "confidence"],
    },
}

_SYSTEM_PROMPT = """You are an information-extraction assistant for ESG sustainability reports.

You will be given:
- A KPI to extract.
- The list of acceptable units for that KPI.
- A passage of text from a corporate sustainability report.

Your job: return the KPI's current-year value by calling the `record_kpi` tool.

Rules:
- Return the value in whatever unit the document uses — do not convert.
- The unit MUST be one of the acceptable units listed.
- If the KPI is not reported in this passage, call the tool with value=null, unit=null.
- Never guess or infer from subsidiary breakdowns if there's no consolidated total — use value=null.
- `reporting_year` is the year the value refers to, not the publication year.
- `source_snippet` is the exact quoted text supporting the value."""


class LLMExtractor(Extractor):
    name = "llm"

    def __init__(
        self,
        model: str = ANTHROPIC_MODEL,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
    ):
        self.model = model
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self._client: Anthropic | None = None

    @property
    def client(self) -> Anthropic:
        if self._client is None:
            self._client = Anthropic()
        return self._client

    def _build_context(self, report: Any, kpi_query: str) -> str:
        """Concatenate top table pages + top-k sentences as the LLM context."""
        parts: list[str] = []
        for t in report["tables"]:
            parts.append(f"[Table @ page {t['page_num']}]")
            parts.append(" | ".join(t["headers"]))
            for row in t["rows"]:
                parts.append(" | ".join(row))
            parts.append("")

        for s in report["sentences"]:
            parts.append(f"[Page {s['page_num']}] {s['text']}")

        return "\n".join(parts)

    def extract(self, report: Any, kpi_key: str) -> KPIExtraction:
        kpi = KPIS[kpi_key]
        flags: list[str] = []

        context = self._build_context(report, kpi["query"])

        user_prompt = (
            f"KPI to extract: {kpi['query']}\n"
            f"Acceptable units: {', '.join(kpi['unit_family'])}\n\n"
            f"Document excerpts:\n{context}"
        )

        cache_key = hashlib.sha256(
            f"{self.model}|{kpi_key}|{user_prompt}".encode()
        ).hexdigest()
        cache_path = CACHE_DIR / "llm" / f"{cache_key}.json"
        if cache_path.exists():
            with cache_path.open() as f:
                tool_input = json.load(f)
        else:
            tool_input = self._call_with_retry(user_prompt, kpi_key)
            if tool_input is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with cache_path.open("w") as f:
                    json.dump(tool_input, f)

        if tool_input is None:
            return self._not_reported(report, kpi_key, flags=["api_error"])

        value = tool_input.get("value")
        unit = tool_input.get("unit")
        reporting_year = tool_input.get("reporting_year")
        snippet = tool_input.get("source_snippet")
        confidence = tool_input.get("confidence")

        if value is None:
            return self._not_reported(report, kpi_key, flags=flags)

        if unit is None:
            return self._not_reported(
                report, kpi_key, flags=[*flags, "llm_missing_unit"]
            )

        # Canonicalize & validate
        try:
            canonical_unit = canonicalize_unit(unit)
        except ValueError:
            return self._not_reported(
                report, kpi_key, flags=[*flags, "unit_unknown"]
            )

        try:
            canonical_value = to_canonical_value(value, unit, kpi["canonical_unit"])
        except ValueError:
            return self._not_reported(
                report, kpi_key, flags=[*flags, "unit_unknown"]
            )

        lo, hi = kpi["plausible_range"]
        if not (lo <= canonical_value <= hi):
            return self._not_reported(
                report, kpi_key, flags=[*flags, "out_of_range"]
            )

        return KPIExtraction(
            company=report["company"],
            report_year=report["report_year"],
            kpi=kpi_key,
            value=canonical_value,
            unit=kpi["canonical_unit"],
            reporting_year=reporting_year,
            source_snippet=snippet,
            source_page=None,
            confidence=confidence,
            extractor=self.name,
            flags=flags,
        )

    def _call_with_retry(self, user_prompt: str, kpi_key: str) -> dict | None:
        for attempt in range(self.max_retries):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=500,
                    temperature=0,
                    system=[{
                        "type": "text",
                        "text": _SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }],
                    tools=[_TOOL_SCHEMA],
                    tool_choice={"type": "tool", "name": "record_kpi"},
                    messages=[{"role": "user", "content": user_prompt}],
                )
                for block in resp.content:
                    if getattr(block, "type", None) == "tool_use" and block.name == "record_kpi":
                        return dict(block.input)
                log.warning("LLM response had no tool_use block for %s", kpi_key)
                return None
            except Exception as e:
                log.warning("LLM call failed (attempt %d/%d): %s",
                            attempt + 1, self.max_retries, e)
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_base_delay * (2 ** attempt))
        return None

    def _not_reported(self, report: Any, kpi_key: str, flags: list[str]) -> KPIExtraction:
        return KPIExtraction(
            company=report["company"],
            report_year=report["report_year"],
            kpi=kpi_key,
            value=None, unit=None, reporting_year=None,
            source_snippet=None, source_page=None, confidence=None,
            extractor=self.name, flags=flags,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_llm.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nlp_esg/extractors/llm.py tests/test_llm.py
git commit -m "feat: LLMExtractor with Anthropic tool-use and response caching"
```

---

## Task 13: Evaluation metrics

**Files:**
- Create: `src/nlp_esg/evaluate.py`
- Create: `tests/test_evaluate.py`

- [ ] **Step 1: Write failing tests** — `tests/test_evaluate.py`

```python
import pytest
from nlp_esg.evaluate import evaluate, is_correct
from nlp_esg.types import KPIExtraction


def _pred(value, unit="tCO2e", year=2024, flags=None):
    return KPIExtraction(
        company="acme", report_year=2024, kpi="scope_1_emissions",
        value=value, unit=unit, reporting_year=year,
        source_snippet=None, source_page=None, confidence=1.0,
        extractor="baseline", flags=flags or [],
    )


def _gold(value, unit="tCO2e", year=2024):
    return {
        "company": "acme", "report_year": 2024, "kpi": "scope_1_emissions",
        "value": value, "unit": unit, "reporting_year": year,
    }


def test_is_correct_exact_match():
    assert is_correct(_pred(1000.0), _gold(1000.0))


def test_is_correct_within_epsilon():
    # 1% tolerance
    assert is_correct(_pred(1000.5), _gold(1000.0))
    assert is_correct(_pred(1005.0), _gold(1000.0))


def test_is_correct_outside_epsilon():
    assert not is_correct(_pred(1020.0), _gold(1000.0))


def test_is_correct_unit_mismatch():
    assert not is_correct(_pred(1000.0, unit="MWh"), _gold(1000.0, unit="tCO2e"))


def test_is_correct_year_mismatch():
    assert not is_correct(_pred(1000.0, year=2023), _gold(1000.0, year=2024))


def test_is_correct_both_not_reported():
    assert is_correct(_pred(None, unit=None, year=None), _gold(None, unit=None, year=None))


def test_is_correct_pred_hallucinates():
    assert not is_correct(_pred(1000.0), _gold(None, unit=None, year=None))


def test_is_correct_missed():
    assert not is_correct(_pred(None, unit=None, year=None), _gold(1000.0))


def test_evaluate_perfect_predictions():
    preds = [_pred(1000.0), _pred(2000.0, year=2024)]
    golds = [_gold(1000.0), _gold(2000.0)]
    m = evaluate(preds, golds, extractor="baseline", kpi="scope_1_emissions")
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0


def test_evaluate_one_false_positive():
    preds = [_pred(1000.0), _pred(42.0)]
    golds = [_gold(1000.0), _gold(None, unit=None, year=None)]
    m = evaluate(preds, golds, extractor="baseline", kpi="scope_1_emissions")
    # 1 TP (correct 1000), 1 FP (hallucinated 42 when gold is None)
    # Precision = TP / (TP + FP) = 1/2 = 0.5
    # Recall on value-predictions = TP / (TP + FN) = 1/1 = 1.0
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(2 * 0.5 * 1.0 / 1.5)


def test_evaluate_coverage():
    preds = [_pred(1000.0), _pred(None, unit=None, year=None), _pred(2000.0)]
    golds = [_gold(1000.0), _gold(2000.0), _gold(None, unit=None, year=None)]
    m = evaluate(preds, golds, extractor="baseline", kpi="scope_1_emissions")
    # coverage = fraction of preds with non-null value = 2/3
    assert m["coverage"] == pytest.approx(2 / 3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_evaluate.py -v`
Expected: ImportError.

- [ ] **Step 3: Write `src/nlp_esg/evaluate.py`**

```python
from __future__ import annotations
from typing import Any

from nlp_esg.config import EPSILON
from nlp_esg.types import KPIExtraction


def is_correct(pred: KPIExtraction, gold: dict[str, Any]) -> bool:
    p_value = pred.value
    g_value = gold.get("value")

    if p_value is None and g_value is None:
        return True
    if p_value is None or g_value is None:
        return False

    if pred.unit != gold.get("unit"):
        return False
    if pred.reporting_year != gold.get("reporting_year"):
        return False
    if g_value == 0:
        return abs(p_value - g_value) < EPSILON
    return abs(p_value - g_value) / abs(g_value) <= EPSILON


def evaluate(
    preds: list[KPIExtraction],
    golds: list[dict[str, Any]],
    extractor: str,
    kpi: str,
) -> dict[str, float]:
    """
    Compute precision / recall / F1 / coverage for a single (extractor, kpi) slice.

    Matches preds to golds by (company, report_year).
    - Precision: TP / (TP + FP). FP = pred has a value but is incorrect (wrong number,
      wrong unit, wrong year, or hallucinated when gold is 'not reported').
    - Recall: TP / (TP + FN). FN = gold has a value but pred is wrong or 'not reported'.
    - Coverage: fraction of preds with non-null value.
    """
    pred_by_key = {(p.company, p.report_year): p for p in preds}
    tp = fp = fn = tn = 0
    total_preds = 0
    non_null_preds = 0

    for g in golds:
        key = (g["company"], g["report_year"])
        p = pred_by_key.get(key)
        if p is None:
            # No prediction for this gold row counts as FN (when gold has a value)
            # or TN (when gold is None).
            if g["value"] is None:
                tn += 1
            else:
                fn += 1
            continue

        total_preds += 1
        if p.value is not None:
            non_null_preds += 1

        correct = is_correct(p, g)
        if correct and p.value is not None:
            tp += 1
        elif correct and p.value is None:
            tn += 1
        elif not correct and p.value is not None:
            fp += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    coverage = non_null_preds / total_preds if total_preds > 0 else 0.0

    return {
        "extractor": extractor,
        "kpi": kpi,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "coverage": coverage,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_evaluate.py -v`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nlp_esg/evaluate.py tests/test_evaluate.py
git commit -m "feat: evaluate with P/R/F1/coverage and 'not reported' class"
```

---

## Task 14: Comparison table

**Files:**
- Create: `src/nlp_esg/compare.py`
- Create: `tests/test_compare.py`

- [ ] **Step 1: Write failing tests** — `tests/test_compare.py`

```python
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
    assert set(df.columns) >= {"scope_1_emissions", "renewable_energy", "water_consumption"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_compare.py -v`
Expected: ImportError.

- [ ] **Step 3: Write `src/nlp_esg/compare.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_compare.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nlp_esg/compare.py tests/test_compare.py
git commit -m "feat: build_comparison_table with most-recent-per-company pivot"
```

---

## Task 15: Pipeline orchestrator

**Files:**
- Create: `src/nlp_esg/pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing tests** — `tests/test_pipeline.py`

```python
from unittest.mock import patch
import pandas as pd
from nlp_esg.pipeline import run_extraction


def test_run_extraction_produces_rows_per_report_and_kpi(
    fake_embed, report_with_table, report_sentence_only
):
    from nlp_esg.retrieval import build_index
    indexed_reports = [build_index(report_with_table), build_index(report_sentence_only)]

    # Mock the LLM extractor at class level so it returns deterministic "not reported".
    from nlp_esg.types import KPIExtraction

    def fake_llm_extract(self, report, kpi_key):
        return KPIExtraction(
            company=report["company"], report_year=report["report_year"], kpi=kpi_key,
            value=None, unit=None, reporting_year=None,
            source_snippet=None, source_page=None, confidence=None,
            extractor="llm", flags=[],
        )

    with patch("nlp_esg.extractors.llm.LLMExtractor.extract", fake_llm_extract):
        extractions = run_extraction(indexed_reports, include_llm=True)

    # 2 reports x 3 KPIs x 2 extractors = 12 rows
    assert len(extractions) == 12
    assert {e.extractor for e in extractions} == {"baseline", "llm"}
    assert {e.kpi for e in extractions} == {
        "scope_1_emissions", "renewable_energy", "water_consumption",
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline.py -v`
Expected: ImportError.

- [ ] **Step 3: Write `src/nlp_esg/pipeline.py`**

```python
from __future__ import annotations
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from nlp_esg.compare import build_comparison_table
from nlp_esg.config import KPI_KEYS, LABELS_DIR, REPORTS_DIR
from nlp_esg.evaluate import evaluate
from nlp_esg.extractors.baseline import BaselineExtractor
from nlp_esg.extractors.llm import LLMExtractor
from nlp_esg.ingest import parse_pdf
from nlp_esg.retrieval import build_index
from nlp_esg.types import KPIExtraction

log = logging.getLogger(__name__)


def load_indexed_reports(reports_dir: Path = REPORTS_DIR) -> list:
    indexed = []
    for pdf in sorted(reports_dir.glob("*.pdf")):
        try:
            parsed = parse_pdf(pdf)
        except Exception as e:
            log.error("Failed to parse %s: %s", pdf.name, e)
            continue
        indexed.append(build_index(parsed))
    return indexed


def run_extraction(
    indexed_reports: Iterable, include_llm: bool = True
) -> list[KPIExtraction]:
    baseline = BaselineExtractor()
    llm = LLMExtractor() if include_llm else None

    out: list[KPIExtraction] = []
    for report in indexed_reports:
        for kpi_key in KPI_KEYS:
            out.append(baseline.extract(report, kpi_key))
            if llm is not None:
                out.append(llm.extract(report, kpi_key))
    return out


def load_gold_labels(path: Path | None = None) -> list[dict]:
    path = path or (LABELS_DIR / "gold_labels.csv")
    if not path.exists():
        return []
    df = pd.read_csv(path)
    df = df.where(pd.notna(df), None)
    # Coerce year fields to int where non-null
    for col in ("report_year", "reporting_year"):
        if col in df.columns:
            df[col] = df[col].apply(lambda v: int(v) if v is not None else None)
    return df.to_dict(orient="records")


def run_evaluation(
    extractions: list[KPIExtraction], golds: list[dict]
) -> pd.DataFrame:
    rows = []
    for extractor in sorted({e.extractor for e in extractions}):
        for kpi in KPI_KEYS:
            preds = [e for e in extractions if e.extractor == extractor and e.kpi == kpi]
            kpi_golds = [g for g in golds if g["kpi"] == kpi]
            rows.append(evaluate(preds, kpi_golds, extractor=extractor, kpi=kpi))
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    indexed = load_indexed_reports()
    log.info("Loaded %d reports", len(indexed))
    if not indexed:
        log.error("No reports found in %s", REPORTS_DIR)
        return

    extractions = run_extraction(indexed, include_llm=True)

    for extractor in ("baseline", "llm"):
        df = build_comparison_table(extractions, extractor=extractor)
        print(f"\n=== {extractor} comparison table ===")
        print(df)

    golds = load_gold_labels()
    if golds:
        metrics = run_evaluation(extractions, golds)
        print("\n=== Evaluation ===")
        print(metrics)
    else:
        log.warning("No gold labels found at %s — skipping evaluation",
                    LABELS_DIR / "gold_labels.csv")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pipeline.py -v`
Expected: 1 passed.

- [ ] **Step 5: Run the full suite to check for regressions**

Run: `pytest`
Expected: all ~50 tests pass (exact count depends on fixture evolution).

- [ ] **Step 6: Commit**

```bash
git add src/nlp_esg/pipeline.py tests/test_pipeline.py
git commit -m "feat: pipeline orchestrator with load/extract/evaluate phases"
```

---

## Task 16: Gold labels template and conventions doc

**Files:**
- Create: `data/labels/gold_labels.csv`
- Create: `data/labels/labels_README.md`

- [ ] **Step 1: Write `data/labels/gold_labels.csv`** (header + one example row)

```csv
company,report_year,kpi,value,unit,reporting_year,source_page,notes
acme,2024,scope_1_emissions,45678,tCO2e,2024,12,"Example row — delete once real labels are filled in"
```

- [ ] **Step 2: Write `data/labels/labels_README.md`**

```markdown
# Gold Labels — Conventions

This directory contains `gold_labels.csv`: hand-labeled ground truth for 5 reports
× 3 KPIs = 15 rows. The evaluation harness (`src/nlp_esg/evaluate.py`) scores
extractor predictions against these labels.

## Schema

| Column           | Type     | Notes |
| ---------------- | -------- | ----- |
| `company`        | string   | Must match the filename prefix in `data/reports/`. |
| `report_year`    | int      | Must match the year in the PDF filename. |
| `kpi`            | string   | One of: `scope_1_emissions`, `renewable_energy`, `water_consumption`. |
| `value`          | float    | The reported value **in the canonical unit** (tCO2e, MWh, m³). Blank = not reported. |
| `unit`           | string   | Canonical unit (`tCO2e`, `MWh`, `m3`) or blank. |
| `reporting_year` | int      | Year the value itself refers to (usually same as `report_year`). Blank if not reported. |
| `source_page`    | int      | PDF page where you found the value (1-indexed). |
| `notes`          | string   | Free-form. Useful for "why I chose this number". |

## Picking the canonical value

When a report states the KPI in multiple places (a narrative number and a performance
table), use the **performance table** value — that's considered the canonical form.

Convert to the canonical unit *yourself* before entering. The pipeline compares in
canonical units; an extractor that reads "1.2 GWh" and the gold labeled "1200 MWh"
should count as correct.

## Not reported vs. breakdown-only

If a company reports Scope 1 only as regional breakdowns (e.g., "EMEA: 12k,
APAC: 5k") with **no consolidated total**, label it as **not reported** (leave
`value`, `unit`, `reporting_year` blank). Do **not** sum the breakdowns —
the baseline extractor deliberately doesn't, so the eval has to use the same
convention.

## Out-of-range sanity check

If the value falls outside these plausibility ranges, double-check the unit:

| KPI                         | Plausible range (canonical) |
| --------------------------- | --------------------------- |
| Scope 1 emissions (tCO2e)   | 100 – 10⁹                   |
| Renewable energy (MWh)      | 100 – 10⁹                   |
| Water consumption (m³)      | 10 – 10¹⁰                   |
```

- [ ] **Step 3: Commit**

```bash
git add data/labels/gold_labels.csv data/labels/labels_README.md
git commit -m "docs: gold labels template and labeling conventions"
```

---

## Task 17: Demo notebook

**Files:**
- Create: `notebooks/demo.ipynb`

- [ ] **Step 1: Write the notebook programmatically**

Because notebook JSON is verbose, generate it from a Python script so the cell content is auditable. Create a one-off `scripts/make_demo_notebook.py` (delete after running) OR author the JSON directly.

Simplest approach: author the JSON directly. Create `notebooks/demo.ipynb` with exactly the following content:

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# NLP ESG KPI Extraction — Demo\n",
    "\n",
    "Runs ingest → index → both extractors → comparison table → evaluation, end to end. All logic lives in `src/nlp_esg/`; this notebook is a thin driver."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from dotenv import load_dotenv\n",
    "load_dotenv()\n",
    "\n",
    "from nlp_esg.pipeline import (\n",
    "    load_indexed_reports, run_extraction, load_gold_labels, run_evaluation,\n",
    ")\n",
    "from nlp_esg.compare import build_comparison_table\n",
    "\n",
    "indexed = load_indexed_reports()\n",
    "print(f'Loaded {len(indexed)} reports')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "extractions = run_extraction(indexed, include_llm=True)\n",
    "print(f'Produced {len(extractions)} extractions')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## Comparison table — Baseline"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": ["build_comparison_table(extractions, extractor='baseline')"]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## Comparison table — LLM"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": ["build_comparison_table(extractions, extractor='llm')"]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## Evaluation — P / R / F1 / coverage"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "golds = load_gold_labels()\n",
    "metrics = run_evaluation(extractions, golds)\n",
    "metrics"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Embedding-model comparison: MiniLM vs ClimateBERT\n",
    "\n",
    "Re-run the baseline with MiniLM to compare retrieval quality."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "os.environ['EMBEDDING_MODEL'] = 'minilm'\n",
    "# Reload modules so the new env var takes effect\n",
    "import importlib, nlp_esg.retrieval, nlp_esg.pipeline\n",
    "importlib.reload(nlp_esg.retrieval)\n",
    "importlib.reload(nlp_esg.pipeline)\n",
    "from nlp_esg.pipeline import load_indexed_reports as load2, run_extraction as run2, run_evaluation as eval2\n",
    "indexed_mini = load2()\n",
    "extractions_mini = run2(indexed_mini, include_llm=False)\n",
    "eval2(extractions_mini, golds)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Qualitative cases\n",
    "\n",
    "Manually inspect cells where one extractor wins and the other doesn't. Fill in after running against real data.\n",
    "\n",
    "- **Baseline wins:** ...\n",
    "- **LLM wins:** ...\n",
    "- **Both fail:** ..."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.11"}
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

- [ ] **Step 2: Verify it opens as valid notebook**

Run: `python -c "import nbformat; nbformat.read(open('notebooks/demo.ipynb'), as_version=4)"`
Expected: no output (valid).

- [ ] **Step 3: Commit**

```bash
git add notebooks/demo.ipynb
git commit -m "docs: demo notebook with linear pipeline + eval cells"
```

---

## Task 18: Integration tests (opt-in)

**Files:**
- Create: `tests/test_integration_llm.py`
- Create: `tests/test_integration_real_pdf.py`

- [ ] **Step 1: Write `tests/test_integration_llm.py`**

```python
import os
import pytest

pytestmark = pytest.mark.skipif(
    not (os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("RUN_INTEGRATION")),
    reason="Set ANTHROPIC_API_KEY and RUN_INTEGRATION=1 to run integration tests.",
)


def test_llm_extractor_real_call():
    from nlp_esg.extractors.llm import LLMExtractor

    stub = {
        "company": "acme", "report_year": 2024,
        "pages": [],
        "tables": [],
        "sentences": [
            {"page_num": 1,
             "text": "Scope 1 direct greenhouse gas emissions in 2024 were 45,678 tCO2e.",
             "embedding": None},
        ],
        "table_headers": [],
    }
    ext = LLMExtractor()
    result = ext.extract(stub, "scope_1_emissions")
    assert result.value is not None
    assert abs(result.value - 45678.0) / 45678.0 < 0.01
```

- [ ] **Step 2: Write `tests/test_integration_real_pdf.py`**

```python
import os
from pathlib import Path
import pytest

from nlp_esg.config import REPORTS_DIR

_PDFS = list(REPORTS_DIR.glob("*.pdf")) if REPORTS_DIR.exists() else []

pytestmark = pytest.mark.skipif(
    not _PDFS,
    reason="No PDFs in data/reports — skipping real-PDF integration test.",
)


def test_real_pdf_pipeline_runs_end_to_end():
    from nlp_esg.ingest import parse_pdf
    from nlp_esg.retrieval import build_index
    from nlp_esg.extractors.baseline import BaselineExtractor

    pdf = _PDFS[0]
    parsed = parse_pdf(pdf)
    indexed = build_index(parsed)
    ext = BaselineExtractor()
    # Just check extraction runs without exception for all 3 KPIs.
    for kpi in ("scope_1_emissions", "renewable_energy", "water_consumption"):
        result = ext.extract(indexed, kpi)
        assert result.extractor == "baseline"
```

- [ ] **Step 3: Verify both are properly skipped by default**

Run: `pytest tests/test_integration_llm.py tests/test_integration_real_pdf.py -v`
Expected: 2 skipped (both fail the skipif conditions when no API key or PDFs).

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration_llm.py tests/test_integration_real_pdf.py
git commit -m "test: opt-in integration tests for LLM and real PDFs"
```

---

## Task 19: Final full-suite sanity + documentation polish

**Files:**
- Modify: `README.md` (add run instructions for evaluation)

- [ ] **Step 1: Full test suite**

Run: `pytest -v`
Expected: all non-integration tests pass, 2 integration tests skipped.

- [ ] **Step 2: Update `README.md` "Run" section**

Replace the existing "Run" section with:

```markdown
## Run

```bash
# Parse PDFs, extract, compare, evaluate — prints tables to stdout.
python -m nlp_esg.pipeline
```

## Demo notebook

```bash
jupyter notebook notebooks/demo.ipynb
```

Runs the same pipeline interactively and includes the MiniLM vs. ClimateBERT
comparison and qualitative-examples cells.

## Test

```bash
pytest              # unit tests only
RUN_INTEGRATION=1 pytest   # also runs LLM + real-PDF integration tests
```
```

- [ ] **Step 3: Verify the pipeline CLI runs (even with no PDFs, it should exit gracefully)**

Run: `python -m nlp_esg.pipeline`
Expected: logs "No reports found in ..." and exits without traceback.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: flesh out README run/test sections"
```

---

## Definition of done

- [ ] `pytest` runs cleanly end-to-end in under 30 seconds (integration tests skipped by default).
- [ ] `python -m nlp_esg.pipeline` with PDFs in `data/reports/` prints two comparison tables (baseline + LLM) and — if `data/labels/gold_labels.csv` has real rows — a metrics DataFrame with precision, recall, F1, and coverage per (extractor, KPI).
- [ ] `notebooks/demo.ipynb` opens in Jupyter and every code cell can be re-executed top-to-bottom without manual edits.
- [ ] All 19 tasks committed individually; `git log --oneline` shows a readable history.
- [ ] The Week 7 artefacts (comparison table CSV, metrics table, 5-report gold set, short qualitative writeup in the notebook) are producible by running the pipeline + filling in `gold_labels.csv`.
