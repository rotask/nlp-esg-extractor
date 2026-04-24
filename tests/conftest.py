from __future__ import annotations
from pathlib import Path
import pytest
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet


@pytest.fixture(autouse=True)
def _redirect_cache_dir(tmp_path, monkeypatch):
    """Redirect the ingest cache into pytest's tmp_path so tests never write into the repo."""
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr("nlp_esg.ingest.CACHE_DIR", cache)
    return cache


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
        Table(
            [
                ["KPI", "2023", "2024", "Unit"],
                ["Scope 1 emissions", "48,000", "45,678", "tCO2e"],
                ["Water consumption", "120", "115", "ML"],
            ],
            style=TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.black)]),
        ),
    ]
    doc.build(story)
    return pdf_path
