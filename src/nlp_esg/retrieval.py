from __future__ import annotations
import numpy as np

import logging
import re
from collections import defaultdict
from functools import lru_cache
from typing import TypedDict

from sentence_transformers import SentenceTransformer, models

from nlp_esg.config import EMBEDDING_MODEL_NAME
from nlp_esg.ingest import ParsedReport, Page, TableEntry
from nlp_esg.normalize import normalize_co2

log = logging.getLogger(__name__)


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
    pages: list[Page]
    tables: list[TableEntry]
    sentences: list[Sentence]
    table_headers: list[TableHeaderEmb]


@lru_cache(maxsize=2)
def _load_model(name: str) -> SentenceTransformer:
    if name == "climatebert":
        word = models.Transformer("climatebert/distilroberta-base-climate-f")
        pooling = models.Pooling(
            word.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
        )
        log.info("loaded embedding model %r", name)
        return SentenceTransformer(modules=[word, pooling])
    if name == "minilm":
        log.info("loaded embedding model %r", name)
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
        for s in split_sentences(normalize_co2(page["text"])):
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


def rank_pages_cosine(
    report: IndexedReport,
    query_emb: np.ndarray,
    unit_tokens: list[str] | None = None,
) -> list[tuple[int, float]]:
    """Score each page by max sentence/table-header cosine sim to query.

    Optional small unit-presence bonus (+0.1) for pages whose normalised text
    contains a KPI unit token. Returns [(page_num, score)] sorted descending.
    """
    page_max: dict[int, float] = {}
    for s in report["sentences"]:
        sim = cosine_sim(query_emb, s["embedding"])
        pn = s["page_num"]
        if sim > page_max.get(pn, -1.0):
            page_max[pn] = sim
    for th in report["table_headers"]:
        sim = cosine_sim(query_emb, th["embedding"])
        pn = report["tables"][th["table_idx"]]["page_num"]
        if sim > page_max.get(pn, -1.0):
            page_max[pn] = sim

    tokens = [u.lower() for u in (unit_tokens or [])]
    scored: list[tuple[int, float]] = []
    for p in report["pages"]:
        base = page_max.get(p["page_num"], 0.0)
        bonus = 0.0
        if tokens:
            text_l = normalize_co2(p["text"]).lower()
            if any(u in text_l for u in tokens):
                bonus = 0.1
        scored.append((p["page_num"], base + bonus))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def _rrf_combine(
    rankings: list[list[tuple[int, float]]], k: int = 60
) -> list[tuple[int, float]]:
    """Reciprocal-rank fusion. Each ranking is [(page_num, score)] sorted desc."""
    fused: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, (page_num, _) in enumerate(ranking):
            fused[page_num] += 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: -x[1])


def rank_pages_rrf(
    report: IndexedReport,
    queries: list[str],
    unit_tokens: list[str] | None = None,
    k: int = 60,
) -> list[tuple[int, float]]:
    """Multi-query page ranking via reciprocal-rank fusion."""
    if not queries:
        return []
    rankings = []
    for q in queries:
        q_emb = embed_texts([q])[0]
        rankings.append(rank_pages_cosine(report, q_emb, unit_tokens=unit_tokens))
    return _rrf_combine(rankings, k=k)


_BM25_TOK_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize_for_bm25(text: str) -> list[str]:
    return _BM25_TOK_RE.findall(text.lower())


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def rank_pages_hybrid(
    report: IndexedReport,
    queries: list[str],
    alpha: float = 0.5,
    rrf_k: int = 60,
) -> list[tuple[int, float]]:
    """alpha * cosine_rrf + (1 - alpha) * bm25, both min-max normalised in [0,1]."""
    from rank_bm25 import BM25Okapi

    if not queries:
        return []

    rrf_ranking = rank_pages_rrf(report, queries, k=rrf_k)
    rrf_by_page = dict(rrf_ranking)

    page_nums = [p["page_num"] for p in report["pages"]]
    corpus = [_tokenize_for_bm25(normalize_co2(p["text"])) for p in report["pages"]]
    if not any(corpus):
        return rrf_ranking
    bm25 = BM25Okapi(corpus)
    query_tokens = _tokenize_for_bm25(" ".join(queries))
    bm25_scores = list(bm25.get_scores(query_tokens))

    rrf_vals = [rrf_by_page.get(pn, 0.0) for pn in page_nums]
    rrf_norm = _minmax(rrf_vals)
    bm25_norm = _minmax(bm25_scores)

    fused = [
        (pn, alpha * r + (1 - alpha) * b)
        for pn, r, b in zip(page_nums, rrf_norm, bm25_norm)
    ]
    fused.sort(key=lambda x: x[1], reverse=True)
    return fused
