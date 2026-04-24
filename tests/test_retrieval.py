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
