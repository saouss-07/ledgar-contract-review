import numpy as np
from scipy import sparse

from ledgar.encoders import ALPHAS, rank_all
from ledgar.retrieval import rank_bank, twin_index

rng = np.random.default_rng(0)


def _normed(n, d):
    x = rng.normal(size=(n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_rank_all_matches_rank_bank_for_every_alpha():
    bank_a, bank_t = _normed(300, 16), sparse.csr_matrix(_normed(300, 32))
    q_a, q_t = bank_a[:20] + 0.1 * _normed(20, 16), sparse.csr_matrix(bank_t[:20].toarray() + 0.1 * _normed(20, 32))
    texts = [f"clause {i}" for i in range(300)]
    twins = twin_index(texts)
    ranked = rank_all({"enc": q_a}, {"enc": bank_a}, q_t, bank_t, texts[:20], twins, 5, "cpu", batch_size=7)
    assert np.array_equal(ranked[("tfidf", 0.0)], rank_bank(q_t, bank_t, 5, query_texts=texts[:20], twins=twins))
    for a in ALPHAS:
        expected = rank_bank([q_a, q_t], [bank_a, bank_t], 5, query_texts=texts[:20], twins=twins, weights=[a, 1 - a])
        assert np.array_equal(ranked[("enc", a)], expected)
    assert not any(i in ranked[("enc", 1.0)][i] for i in range(20))   # each query's twin (itself) is removed
