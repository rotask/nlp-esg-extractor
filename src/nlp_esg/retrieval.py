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
