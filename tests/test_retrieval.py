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


def _ix(*, pages, sentences, table_headers, tables):
    return {
        "company": "x", "report_year": 2024, "parser": "pdfplumber",
        "pages": pages, "sentences": sentences,
        "table_headers": table_headers, "tables": tables,
    }


def test_rank_pages_cosine_returns_pages_sorted_by_max_sim():
    from nlp_esg.retrieval import rank_pages_cosine
    q = np.array([1.0, 0.0], dtype=np.float32)
    e_high = np.array([1.0, 0.0], dtype=np.float32)
    e_low = np.array([0.0, 1.0], dtype=np.float32)
    indexed = _ix(
        pages=[{"page_num": 1, "text": "a"}, {"page_num": 2, "text": "b"}],
        sentences=[
            {"page_num": 1, "text": "low", "embedding": e_low},
            {"page_num": 2, "text": "high", "embedding": e_high},
        ],
        table_headers=[],
        tables=[],
    )
    ranked = rank_pages_cosine(indexed, q)
    assert ranked[0][0] == 2
    assert ranked[1][0] == 1


def test_rrf_combines_ranks():
    """A page that's rank-2 in two queries should beat a page that's rank-1 in one."""
    from nlp_esg.retrieval import _rrf_combine
    rankings = [
        [(1, 0.9), (2, 0.8), (3, 0.5)],
        [(2, 0.95), (1, 0.7), (3, 0.4)],
    ]
    fused = _rrf_combine(rankings, k=60)
    by_page = dict(fused)
    assert by_page[2] > by_page[3]
    assert abs(by_page[1] - by_page[2]) < 1e-9


def test_rank_pages_rrf_uses_multiple_queries(monkeypatch):
    """rank_pages_rrf calls rank_pages_cosine once per query and fuses."""
    from nlp_esg.retrieval import rank_pages_rrf
    indexed = _ix(
        pages=[{"page_num": i, "text": ""} for i in (1, 2, 3)],
        sentences=[], table_headers=[], tables=[],
    )
    fake_embed = lambda texts, model_name=None: np.array([[1.0, 0.0]] * len(texts), dtype=np.float32)
    monkeypatch.setattr("nlp_esg.retrieval.embed_texts", fake_embed)

    rankings = iter([
        [(1, 0.9), (2, 0.5), (3, 0.1)],
        [(2, 0.9), (1, 0.5), (3, 0.1)],
    ])
    monkeypatch.setattr(
        "nlp_esg.retrieval.rank_pages_cosine",
        lambda *a, **kw: next(rankings),
    )
    out = rank_pages_rrf(indexed, ["q1", "q2"])
    assert out[0][0] in (1, 2)
    assert out[-1][0] == 3


def test_build_index_table_header_includes_row_labels(monkeypatch):
    """Embedded header_string should include row[0] of first 5 rows."""
    from nlp_esg.retrieval import build_index

    captured: list[list[str]] = []

    def fake_embed(texts, model_name=None):
        captured.append(list(texts))
        return np.zeros((len(texts), 4), dtype=np.float32)

    monkeypatch.setattr("nlp_esg.retrieval.embed_texts", fake_embed)

    parsed = {
        "company": "eni", "report_year": 2024, "parser": "pdfplumber",
        "pages": [{"page_num": 1, "text": "x"}],
        "tables": [{
            "page_num": 1,
            "headers": ["", "2024", "2023"],
            "rows": [
                ["Total gross Scope 1 GHG emissions (MtCO2eq)", "18.95", "20.20"],
                ["Methane emissions", "0.5", "0.6"],
            ],
        }],
    }
    # use_cache=False: don't read OR write the on-disk indexed cache; otherwise
    # this synthetic 1-page test would either short-circuit on the real cache
    # or overwrite it with garbage data, depending on timing.
    build_index(parsed, use_cache=False)
    # The second embed_texts call is for table headers.
    header_strings = captured[1]
    assert any("Total gross Scope 1 GHG emissions" in hs for hs in header_strings)


def test_bm25_picks_page_with_rare_token(monkeypatch):
    """Hybrid ranking should surface the page containing the rare query token."""
    from nlp_esg.retrieval import rank_pages_hybrid
    indexed = _ix(
        pages=[
            {"page_num": 1, "text": "long narrative paragraph about climate strategy"},
            {"page_num": 2, "text": "Total Scope 1 GHG emissions MtCO2e 33.7 in 2024"},
            {"page_num": 3, "text": "another narrative page about governance"},
        ],
        sentences=[
            {"page_num": p, "text": "x",
             "embedding": np.array([0.5, 0.5], dtype=np.float32)}
            for p in (1, 2, 3)
        ],
        table_headers=[], tables=[],
    )
    fake_embed = lambda texts, model_name=None: np.array([[0.5, 0.5]] * len(texts), dtype=np.float32)
    monkeypatch.setattr("nlp_esg.retrieval.embed_texts", fake_embed)
    out = rank_pages_hybrid(indexed, ["Scope 1 emissions MtCO2e"])
    assert out[0][0] == 2


def test_rank_pages_cosine_unit_presence_boost():
    from nlp_esg.retrieval import rank_pages_cosine
    q = np.array([1.0, 0.0], dtype=np.float32)
    e_mid = np.array([0.5, 0.5], dtype=np.float32)
    indexed = _ix(
        pages=[
            {"page_num": 1, "text": "narrative no unit"},
            {"page_num": 2, "text": "data with MtCO2e marker"},
        ],
        sentences=[
            {"page_num": 1, "text": "x", "embedding": e_mid},
            {"page_num": 2, "text": "y", "embedding": e_mid},
        ],
        table_headers=[], tables=[],
    )
    ranked = rank_pages_cosine(indexed, q, unit_tokens=["mtco2e"])
    assert ranked[0][0] == 2  # boosted page wins despite same cosine
