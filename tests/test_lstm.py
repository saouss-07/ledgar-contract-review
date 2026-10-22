import numpy as np
import torch
from torch import nn

from ledgar.lstm import (PAD, UNK, LSTMClassifier, batches, build_vocab, encode, predict_proba, tokenize,
                         train_one_epoch)


def test_tokenize_like_glove():
    assert tokenize("The Company's 30 days, NOT later.") == ["the", "company", "'s", "30", "days", ",", "not",
                                                             "later", "."]


def test_vocab_and_encoding():
    vocab = build_vocab([["a", "b", "a"], ["a", "c", "b"]], min_freq=2)
    assert vocab == {"<pad>": PAD, "<unk>": UNK, "a": 2, "b": 3}   # "c" seen once: unknown
    assert encode(["a", "c", "b", "a"], vocab, max_len=3) == [2, UNK, 3]
    assert encode([], vocab) == [UNK]                              # never an empty sequence


def test_batches_keep_every_clause_once():
    seqs = [[2] * n for n in range(1, 101)]
    seen = []
    for ids, lengths, y, g in batches(seqs, np.arange(100), 16, shuffle=True, rng=np.random.default_rng(0)):
        assert ids.shape == (len(g), int(lengths.max()))
        assert (y.numpy() == g).all()
        seen += list(g)
    assert sorted(seen) == list(range(100))


def test_padding_does_not_change_the_prediction():
    torch.manual_seed(0)
    model = LSTMClassifier(np.random.default_rng(0).normal(size=(10, 8)).astype(np.float32), n_classes=3, hidden=4)
    model.eval()
    short = torch.tensor([[2, 3, 4]])
    padded = torch.tensor([[2, 3, 4, PAD, PAD, PAD]])
    with torch.no_grad():
        assert torch.allclose(model(short, torch.tensor([3])), model(padded, torch.tensor([3])), atol=1e-6)


def test_the_loop_learns_a_toy_task():
    # class = first token: an LSTM must reach 100 % on training data
    rng = np.random.default_rng(0)
    seqs = [[int(c) + 2] + list(rng.integers(5, 20, 6)) for c in rng.integers(0, 3, 300)]
    labels = np.array([s[0] - 2 for s in seqs])
    torch.manual_seed(0)
    model = LSTMClassifier(rng.normal(0, 0.1, size=(20, 16)).astype(np.float32), n_classes=3, hidden=16, dropout=0)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-2)
    cfg = {"batch_size": 32, "clip": 1.0}
    losses = [train_one_epoch(model, seqs, labels, optimiser, nn.CrossEntropyLoss(), "cpu", rng, cfg)
              for _ in range(15)]
    assert losses[-1] < losses[0] / 5
    assert (predict_proba(model, seqs, "cpu").argmax(axis=1) == labels).mean() > 0.95
