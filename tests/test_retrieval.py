import numpy as np
import pytest
from scipy import sparse

from ledgar.retrieval import (durations, label_in_question, mrr, precision_at_k, rank_bank, relevance, top_k,
                              twin_index)

rng = np.random.default_rng(0)
BANK = rng.normal(size=(200, 16)).astype(np.float32)
BANK /= np.linalg.norm(BANK, axis=1, keepdims=True)


def test_top_k_matches_full_sort():
    scores = rng.normal(size=(5, 200))
    assert np.array_equal(top_k(scores, 7), np.argsort(-scores, axis=1)[:, :7])


def test_a_bank_clause_finds_itself_first():
    ranked = rank_bank(BANK[:10], BANK, 3)
    assert np.array_equal(ranked[:, 0], np.arange(10))


def test_sparse_and_dense_give_the_same_ranking():
    assert np.array_equal(rank_bank(sparse.csr_matrix(BANK[:10]), sparse.csr_matrix(BANK), 5), rank_bank(BANK[:10], BANK, 5))


def test_twins_are_removed():
    texts = [f"clause {i}" for i in range(200)]
    texts[50] = "Clause  0"                       # same text as clause 0 after normalisation
    ranked = rank_bank(BANK[:1], BANK, 5, query_texts=["clause 0"], twins=twin_index(texts))
    assert 0 not in ranked[0] and 50 not in ranked[0]


def test_filters_and_padding():
    bank_labels = np.arange(200) % 4
    ranked = rank_bank(BANK[:3], BANK, 60, query_labels=np.array([1, 2, 3]), bank_labels=bank_labels)
    for row, label in zip(ranked, [1, 2, 3]):
        assert (bank_labels[row[row >= 0]] == label).all()
        assert (row >= 0).sum() == 50 and (row[50:] == -1).all()   # only 50 clauses of each type
    allowed = np.zeros(200, dtype=bool)
    allowed[[3, 7]] = True
    assert set(rank_bank(BANK[:1], BANK, 4, allowed=allowed)[0]) == {3, 7, -1}


def test_precision_and_mrr():
    ranked = np.array([[0, 1, 2], [3, 4, -1]])
    rel = relevance(ranked, query_labels=[1, 0], bank_labels=[0, 1, 1, 1, 0])
    assert rel.tolist() == [[False, True, True], [False, True, False]]
    assert precision_at_k(rel, 1) == 0.0
    assert precision_at_k(rel, 3) == pytest.approx(3 / 6)
    assert mrr(rel) == pytest.approx(0.5)


def test_durations_digits_words_and_hyphens():
    assert durations("upon thirty (30) days' prior written notice") == {(30, "day")}
    assert durations("show me termination clauses with a 30-day notice period") == {(30, "day")}
    assert durations("for a period of twelve months, and ninety days") == {(12, "month"), (90, "day")}
    assert durations("This Agreement may be executed in counterparts.") == set()


def test_label_in_question():
    names = ["Governing Laws", "Terminations", "Termination", "Notices", "Confidentiality"]
    assert names[label_in_question("show me termination clauses with a 30-day notice", names)] in ("Terminations",
                                                                                                  "Termination")
    assert label_in_question("which law governs the contract?", names) is None
    assert names[label_in_question("Governing law of Delaware", names)] == "Governing Laws"
