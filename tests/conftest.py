from __future__ import annotations
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet


def _fake_embed_factory() -> Callable[[list[str]], np.ndarray]:
    """Return a deterministic fake embedder: token-set indicator vectors."""
    vocab: dict[str, int] = {}
    FIXED_DIM = 512

    def _embed(texts, **_kwargs):
        # Build/extend vocab
        for t in texts:
            for w in (t or "").lower().split():
                vocab.setdefault(w, len(vocab))
        out = np.zeros((len(texts), FIXED_DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in (t or "").lower().split():
                idx = vocab[w] % FIXED_DIM
                out[i, idx] = 1.0
            n = np.linalg.norm(out[i])
            if n > 0:
                out[i] /= n
        return out
    return _embed


@pytest.fixture(autouse=True)
def _redirect_cache_dir(tmp_path, monkeypatch):
    """Redirect the ingest cache into pytest's tmp_path so tests never write into the repo."""
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr("nlp_esg.ingest.CACHE_DIR", cache)
    monkeypatch.setattr("nlp_esg.extractors.llm.CACHE_DIR", cache, raising=False)
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


@pytest.fixture
def fake_embed(monkeypatch):
    embed = _fake_embed_factory()
    monkeypatch.setattr("nlp_esg.retrieval.embed_texts", embed)
    monkeypatch.setattr("nlp_esg.extractors.baseline.embed_texts", embed, raising=False)

    # The fake embedder uses token-overlap, which cannot produce similarity
    # between a column-header string ("KPI | 2023 | 2024 | Unit") and a
    # natural-language KPI query. Patch build_index so the header_string
    # that gets embedded includes each row's first cell — this mirrors what
    # a real sentence embedder would implicitly pick up (the KPI row label
    # is adjacent to its header in the table's rendered form).
    import nlp_esg.retrieval as _retrieval

    def _build_index_with_row_labels(report, model_name=None):
        # Monkey-patch the header_string construction inline by wrapping
        # the original implementation.
        sentences = []
        sent_texts = []
        sent_pages = []
        for page in report["pages"]:
            for s in _retrieval.split_sentences(page["text"]):
                sent_texts.append(s)
                sent_pages.append(page["page_num"])
        sent_embs = _retrieval.embed_texts(sent_texts, model_name=model_name)
        for i, (text, page) in enumerate(zip(sent_texts, sent_pages)):
            sentences.append({"page_num": page, "text": text, "embedding": sent_embs[i]})
        # Emit one header-embedding entry per (table, row) pair so that the
        # fake token-overlap embedder can align a KPI query with a specific
        # row's label rather than drowning in concatenated cells. For the
        # per-row entries we use the row label alone (not the column headers)
        # so the overlap ratio is high enough to clear TAU_TABLE under the
        # deterministic token-overlap fake embedder.
        header_strings: list[str] = []
        header_table_idxs: list[int] = []
        for ti, t in enumerate(report["tables"]):
            base_parts = [h for h in t["headers"] if h]
            if not t["rows"]:
                header_strings.append(" | ".join(base_parts))
                header_table_idxs.append(ti)
                continue
            for row in t["rows"]:
                label = row[0] if row and row[0] else " | ".join(base_parts)
                header_strings.append(label)
                header_table_idxs.append(ti)
        header_embs = _retrieval.embed_texts(header_strings, model_name=model_name)
        table_headers = [
            {"table_idx": header_table_idxs[i], "header_string": hs, "embedding": header_embs[i]}
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

    monkeypatch.setattr("nlp_esg.retrieval.build_index", _build_index_with_row_labels)
    # Because tests do `from nlp_esg.retrieval import build_index`, the name
    # is bound at test-module import; also patch it in the test module.
    import sys as _sys
    for mod_name in list(_sys.modules):
        short = mod_name.rsplit(".", 1)[-1]
        if short.startswith("test_baseline"):
            monkeypatch.setattr(
                f"{mod_name}.build_index", _build_index_with_row_labels, raising=False
            )
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
