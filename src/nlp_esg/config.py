"""KPI registry and project-wide constants.

Defines the three target KPIs (Scope 1 emissions, total energy
consumption, water consumption) with their query phrasings, allowed
unit family, canonical unit, plausible-range guard, and per-KPI
negative tokens used by the extractors. Also holds path constants
under `data/` and model defaults read from the `EMBEDDING_MODEL` and
`ANTHROPIC_MODEL` env vars.
"""
from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
REPORTS_DIR = DATA_DIR / "reports"
LABELS_DIR = DATA_DIR / "labels"
CACHE_DIR = DATA_DIR / "cache"
RUNS_DIR = DATA_DIR / "runs"

EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "climatebert")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

EPSILON = 0.01  # value tolerance for correctness
TAU_TABLE = 0.55  # cosine similarity threshold for table header match
TOP_K_SENTENCES = 5

KPIS: dict[str, dict] = {
    "scope_1_emissions": {
        "query": "Scope 1 direct greenhouse gas emissions",
        "queries": [
            "Total gross Scope 1 GHG emissions",
            "Scope 1 direct greenhouse gas emissions tCO2e",
            "Scope 1 (direct) emissions",
        ],
        "unit_family": ["tCO2e", "ktCO2e", "MtCO2e", "t CO2-eq", "t CO2e", "tonnes CO2e"],
        "canonical_unit": "tCO2e",
        "plausible_range": (1e2, 1e9),
        # Lines containing any of these tokens are rejected by the line-scanning
        # fallback — they identify metrics that are NOT total scope 1.
        "negative_tokens": ["scope 2", "scope 3", "scopes 1, 2", "methane", "intensity", "per ", "fugitive"],
    },
    "total_energy_consumption": {
        "query": "Total energy consumption",
        "queries": [
            "Total energy consumption MWh",
            "Energy consumption GWh",
            "Total energy consumed across operations",
        ],
        "unit_family": ["MWh", "GWh", "TWh", "GJ", "TJ", "PJ", "kWh"],
        "canonical_unit": "MWh",
        "plausible_range": (1e2, 1e9),
        "negative_tokens": ["renewable", "produced", "production", "intensity", "per ", "discontinued"],
    },
    "water_consumption": {
        "query": "Total water consumption withdrawal",
        "queries": [
            "Total freshwater consumption million m3",
            "Water consumption m3",
            "Total water consumed",
        ],
        "unit_family": ["m3", "m³", "ML", "megaliters", "megalitres", "kL", "thousand m3", "cubic metres", "Mm3", "Mm³"],
        "canonical_unit": "m3",
        "plausible_range": (1e1, 1e10),
        "negative_tokens": ["withdrawal", "withdrawn", "abstracted", "discharge", "discharged", "intake", "recycled", "reclaimed", " net ", "discontinued"],
    },
}

KPI_KEYS = list(KPIS.keys())
