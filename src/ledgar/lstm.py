"""Bidirectional LSTM with GloVe embeddings, trained with a hand-written PyTorch loop (course S04, TP04). Milestone m3.

Pipeline: lower-cased word tokens -> vocabulary learnt on train -> GloVe 6B 300d vectors (words GloVe does not know
start at random) -> BiLSTM -> max-pooling over the words -> linear layer with 100 outputs.
The loop is written by hand (no Trainer): forward, cross-entropy, backward, gradient clipping, AdamW step. The
epoch kept is the one with the best macro-F1 on validation (early stopping); the test is predicted once, with it.

    python -m ledgar.lstm             # about 3 min per epoch on a T4, much slower on a laptop CPU
    FAST_RUN=1 python -m ledgar.lstm  # quick check on the subsample

Outputs: results/tables/lstm_history.csv, lstm_results.csv, lstm_f1_per_class.csv, lstm_top_confusions.csv,
results/predictions/lstm_{validation,test}.csv. The weights go to models/lstm.pt (ignored by Git).
GloVe comes from the Hugging Face Hub (sentence-transformers/average_word_embeddings_glove.6B.300d), stored as
safetensors + a JSON vocabulary: no pickle, no extra dependency.
"""
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from ledgar.data import FAST_RUN, SEED, load_ledgar, set_seed
from ledgar.metrics import accuracy, f1_table, macro_f1, save_predictions, top_confusions

ROOT = Path(__file__).resolve().parents[2]
TAB_DIR = ROOT / "results" / "tables"
MODEL_DIR = ROOT / "models"
GLOVE_REPO = "sentence-transformers/average_word_embeddings_glove.6B.300d"

CONFIG = dict(max_len=256, min_freq=2, emb_dim=300, hidden=128, dropout=0.3, batch_size=64, lr=1e-3,
              weight_decay=1e-4, epochs=10, patience=2, clip=1.0)
PAD, UNK = 0, 1
_TOKEN_RE = re.compile(r"[a-z]+|'[a-z]+|\d+|[^\sa-z\d]")


def tokenize(text):
    """Lower-cased words, numbers and punctuation signs, as GloVe 6B was built."""
    return _TOKEN_RE.findall(text.lower())


def build_vocab(token_lists, min_freq=CONFIG["min_freq"]):
    """word -> index, learnt on train only. 0 = padding, 1 = unknown word."""
    counts = Counter(t for tokens in token_lists for t in tokens)
    words = sorted((w for w, c in counts.items() if c >= min_freq), key=lambda w: (-counts[w], w))
    return {"<pad>": PAD, "<unk>": UNK, **{w: i + 2 for i, w in enumerate(words)}}


def encode(tokens, vocab, max_len=CONFIG["max_len"]):
    ids = [vocab.get(t, UNK) for t in tokens[:max_len]]
    return ids or [UNK]


def load_glove(vocab, dim=CONFIG["emb_dim"], seed=SEED):
    """Embedding matrix (len(vocab), dim): GloVe rows where the word is known, small random rows otherwise.

    Returns (matrix, share of the vocabulary found in GloVe).
    """
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file

    words = json.load(open(hf_hub_download(GLOVE_REPO, "0_WordEmbeddings/whitespacetokenizer_config.json"),
                           encoding="utf-8"))["vocab"]
    weights = next(iter(load_file(hf_hub_download(GLOVE_REPO, "0_WordEmbeddings/model.safetensors")).values()))
    glove_index = {w: i for i, w in enumerate(words)}
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0, 0.1, size=(len(vocab), dim)).astype(np.float32)
    matrix[PAD] = 0
    found = 0
    for word, i in vocab.items():
        j = glove_index.get(word)
        if j is not None and i > UNK:
            matrix[i] = weights[j]
            found += 1
    return matrix, found / (len(vocab) - 2)


class LSTMClassifier(nn.Module):
    def __init__(self, embeddings, n_classes, hidden=CONFIG["hidden"], dropout=CONFIG["dropout"]):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.as_tensor(embeddings), freeze=False, padding_idx=PAD)
        self.lstm = nn.LSTM(embeddings.shape[1], hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(2 * hidden, n_classes)

    def forward(self, ids, lengths):
        x = self.dropout(self.embedding(ids))
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = self.lstm(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=ids.shape[1])
        h = h.masked_fill((ids == PAD).unsqueeze(-1), float("-inf"))   # padding never wins the max-pooling
        return self.out(self.dropout(h.max(dim=1).values))


def batches(sequences, labels, batch_size, shuffle, rng=None):
    """Padded (ids, lengths, labels) tensors. When shuffling, clauses of similar length share a batch (less padding)."""
    order = np.arange(len(sequences))
    if shuffle:
        order = rng.permutation(order)
        chunk = batch_size * 50
        order = np.concatenate([sorted(order[i:i + chunk], key=lambda j: len(sequences[j]))
                                for i in range(0, len(order), chunk)])
    groups = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    if shuffle:
        groups = [groups[i] for i in rng.permutation(len(groups))]
    for g in groups:
        lengths = torch.tensor([len(sequences[j]) for j in g])
        ids = torch.full((len(g), int(lengths.max())), PAD, dtype=torch.long)
        for row, j in enumerate(g):
            ids[row, :lengths[row]] = torch.tensor(sequences[j])
        yield ids, lengths, torch.as_tensor(np.asarray(labels)[g]), g


def train_one_epoch(model, sequences, labels, optimiser, loss_fn, device, rng, cfg=CONFIG):
    """The training loop: forward, loss, backward, clipping, optimiser step. Returns the mean loss."""
    model.train()
    total, n = 0.0, 0
    for ids, lengths, y, _ in batches(sequences, labels, cfg["batch_size"], shuffle=True, rng=rng):
        ids, y = ids.to(device), y.to(device)
        optimiser.zero_grad()
        loss = loss_fn(model(ids, lengths), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
        optimiser.step()
        total, n = total + loss.item() * len(y), n + len(y)
    return total / n


@torch.no_grad()
def predict_proba(model, sequences, device, batch_size=256):
    """Softmax probabilities (n, n_classes), in the order of `sequences`."""
    model.eval()
    probs = np.zeros((len(sequences), model.out.out_features), dtype=np.float32)
    for ids, lengths, _, g in batches(sequences, np.zeros(len(sequences), dtype=np.int64), batch_size, shuffle=False):
        probs[g] = torch.softmax(model(ids.to(device), lengths), dim=-1).cpu().numpy()
    return probs


def main(cfg=CONFIG):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds, names = load_ledgar()
    train, val, test = ds["train"], ds["validation"], ds["test"]
    n_classes, tag = len(names), "_fast" if FAST_RUN else ""
    epochs = 2 if FAST_RUN else cfg["epochs"]

    tokens = {split: [tokenize(t) for t in data["text"]] for split, data in
              (("train", train), ("validation", val), ("test", test))}
    vocab = build_vocab(tokens["train"])
    seqs = {split: [encode(t, vocab, cfg["max_len"]) for t in toks] for split, toks in tokens.items()}
    embeddings, coverage = load_glove(vocab, cfg["emb_dim"])
    print(f"vocabulary {len(vocab):,} words, {coverage:.1%} in GloVe | device {device} | FAST_RUN={FAST_RUN}",
          flush=True)

    torch.manual_seed(SEED)
    model = LSTMClassifier(embeddings, n_classes, cfg["hidden"], cfg["dropout"]).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.default_rng(SEED)
    y_train, y_val = np.asarray(train["label"]), np.asarray(val["label"])

    history, best_f1, best_state, waited = [], -1.0, None, 0
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        loss = train_one_epoch(model, seqs["train"], y_train, optimiser, loss_fn, device, rng, cfg)
        val_pred = predict_proba(model, seqs["validation"], device).argmax(axis=1)
        row = {"epoch": epoch, "train_loss": loss, "val_accuracy": accuracy(y_val, val_pred),
               "val_macro_f1": macro_f1(y_val, val_pred, n_classes), "seconds": time.perf_counter() - t0}
        history.append(row)
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
        if row["val_macro_f1"] > best_f1:
            best_f1, waited = row["val_macro_f1"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= cfg["patience"]:
                print(f"early stopping: no gain on validation for {waited} epochs")
                break
    pd.DataFrame(history).to_csv(TAB_DIR / f"lstm_history{tag}.csv", index=False)
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "vocab": vocab, "config": cfg}, MODEL_DIR / f"lstm{tag}.pt")
    best_epoch = max(history, key=lambda r: r["val_macro_f1"])["epoch"]

    results = []
    for split, data in (("validation", val), ("test", test)):
        probs = predict_proba(model, seqs[split], device)
        pred, y = probs.argmax(axis=1), np.asarray(data["label"])
        save_predictions(f"lstm{tag}", split, data["idx"], y, pred, probs.max(axis=1))
        results.append({"model": "bilstm_glove", "best_epoch": best_epoch, "split": split,
                        "accuracy": accuracy(y, pred), "macro_f1": macro_f1(y, pred, n_classes)})
        if split == "test":
            f1_table(y, pred, names).to_csv(TAB_DIR / f"lstm_f1_per_class{tag}.csv", index=False)
            top_confusions(y, pred, names, k=10).to_csv(TAB_DIR / f"lstm_top_confusions{tag}.csv", index=False)
            # Same check as the baseline: test without the clauses already seen in train
            norm = lambda s: re.sub(r"\s+", " ", s.lower()).strip()
            seen = set(map(norm, train["text"]))
            keep = np.array([norm(t) not in seen for t in data["text"]])
            results.append({"model": "bilstm_glove", "best_epoch": best_epoch, "split": "test_sans_doublons",
                            "accuracy": accuracy(y[keep], pred[keep]),
                            "macro_f1": macro_f1(y[keep], pred[keep], n_classes)})
    table = pd.DataFrame(results)
    table.to_csv(TAB_DIR / f"lstm_results{tag}.csv", index=False)
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
