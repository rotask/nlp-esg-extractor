from unittest.mock import MagicMock, patch

import pytest

from nlp_esg.ingest_docling import parse_with_docling


def _fake_docling_doc():
    """Build a minimal mock that mimics what parse_with_docling consumes."""
    table_item = MagicMock()
    table_item.label = "table"
    table_item.prov = [MagicMock(page_no=2)]
    df_mock = MagicMock()
    df_mock.columns = ["", "2024", "2023"]
    df_mock.values.tolist.return_value = [["Total Scope 1 GHG (MtCO2eq)", "33.7", "32.8"]]
    table_item.export_to_dataframe.return_value = df_mock

    doc = MagicMock()
    doc.num_pages.return_value = 3
    doc.export_to_markdown.side_effect = lambda page_no=None: (
        f"# Page {page_no}\n" + "Substantive paragraph of report content. " * 5
    )
    doc.iterate_items.return_value = [(table_item, 0)]
    return doc


def test_parse_with_docling_returns_parsed_report(tmp_path):
    pdf = tmp_path / "demo_2024.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    converter = MagicMock()
    converter.convert.return_value = MagicMock(document=_fake_docling_doc())
    with patch("nlp_esg.ingest_docling.DocumentConverter", return_value=converter):
        report = parse_with_docling(pdf)
    assert report is not None
    assert report["parser"] == "docling"
    assert report["company"] == "demo"
    assert report["report_year"] == 2024
    assert len(report["pages"]) == 3
    assert any(t["page_num"] == 2 for t in report["tables"])


def test_parse_with_docling_returns_none_on_exception(tmp_path):
    pdf = tmp_path / "broken_2024.pdf"
    pdf.write_bytes(b"not a pdf")
    converter = MagicMock()
    converter.convert.side_effect = RuntimeError("boom")
    with patch("nlp_esg.ingest_docling.DocumentConverter", return_value=converter):
        report = parse_with_docling(pdf)
    assert report is None


def test_parse_with_docling_returns_none_on_empty_pages(tmp_path):
    pdf = tmp_path / "empty_2024.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    doc = MagicMock()
    doc.num_pages.return_value = 0
    doc.iterate_items.return_value = []
    converter = MagicMock()
    converter.convert.return_value = MagicMock(document=doc)
    with patch("nlp_esg.ingest_docling.DocumentConverter", return_value=converter):
        report = parse_with_docling(pdf)
    assert report is None


def test_parse_with_docling_returns_none_when_majority_pages_empty(tmp_path):
    """If Docling OOMs partway through, most pages come back empty.
    The quality check should reject the result so the caller falls back."""
    pdf = tmp_path / "oom_2024.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    doc = MagicMock()
    doc.num_pages.return_value = 100
    doc.iterate_items.return_value = []
    # First 28 pages succeed; pages 29-100 return empty (the OOM pattern).
    def export(page_no=None):
        if page_no is not None and page_no <= 28:
            return "Substantial paragraph of report content " * 20
        return ""
    doc.export_to_markdown.side_effect = export

    converter = MagicMock()
    converter.convert.return_value = MagicMock(document=doc)
    with patch("nlp_esg.ingest_docling.DocumentConverter", return_value=converter):
        report = parse_with_docling(pdf)
    assert report is None
