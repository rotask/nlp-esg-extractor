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
