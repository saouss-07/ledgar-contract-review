"""Which sentence encoder for the clause bank? Comparison of 7 pretrained models (volet B, after milestone m2).

ledgar.retrieval uses all-MiniLM-L6-v2, the encoder of the course milestone, and finds that TF-IDF beats it on these
very standardised texts. Here every encoder is evaluated the same way, without any training:
- alone: cosine between L2-normalised embeddings;
- fused with TF-IDF: alpha * cosine(encoder) + (1 - alpha) * cosine(TF-IDF), alpha chosen on validation (MRR@10).
Same protocol as ledgar.retrieval: bank = train, queries = validation (choices) and test (final numbers), a retrieved
clause is relevant when it has the type of the query, exact twins of the query removed from the ranking.

Every encoder reads at most 256 word pieces (the limit of MiniLM), so that the comparison is about the model, not the
context length. e5 expects the prefix "query: " on both sides for a symmetric task (clause -> similar clauses).
Legal-BERT is not a sentence encoder: mean pooling of its tokens tests whether legal pretraining alone is enough.

    python -m ledgar.encoders                       # GPU strongly advised: about 45 min on a T4 for the 7 models
    python -m ledgar.encoders --models minilm bge-small
    FAST_RUN=1 python -m ledgar.encoders            # quick check on the subsample

Outputs: results/tables/retrieval_encoders.csv, results/figures/retrieval_encoders.png.
Embeddings are cached in data/embeddings/ (ignored by Git).
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch

from ledgar.baseline import make_vectorizer
from ledgar.data import FAST_RUN, load_ledgar, set_seed
from ledgar.retrieval import EMB_DIR, KS, ROOT, TAB_DIR, normalise, relevance, retrieval_scores, twin_index

FIG_DIR = ROOT / "results" / "figures"
MAX_SEQ_LENGTH = 256
ALPHAS = (1.0, 0.75, 0.5, 0.25)   # alpha = 1: the encoder alone; alpha = 0 (TF-IDF alone) is computed once

ENCODERS = {   # short name -> (Hugging Face model, prefix added to every text)
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", ""),
    "mpnet": ("sentence-transformers/all-mpnet-base-v2", ""),
    "bge-small": ("BAAI/bge-small-en-v1.5", ""),
    "bge-base": ("BAAI/bge-base-en-v1.5", ""),
    "gte-base": ("thenlper/gte-base", ""),
    "e5-base": ("intfloat/e5-base-v2", "query: "),
    "legal-bert": ("nlpaueb/legal-bert-base-uncased", ""),
}


def encode_splits(key, splits, device, tag=""):
    """{split: float32 (n, dim) normalised embeddings}, cached in data/embeddings/<key>_<split>.npy.

    Returns (embeddings, info) where info has the number of parameters and the encoding time of the bank.
    """
    from sentence_transformers import SentenceTransformer

    name, prefix = ENCODERS[key]
    paths = {split: EMB_DIR / f"enc_{key}_{split}{tag}.npy" for split in splits}
    model, info = None, {"model": key, "hf_name": name}
    out = {}
    for split, texts in splits.items():
        if paths[split].exists() and len(np.load(paths[split], mmap_mode="r")) == len(texts):
            out[split] = np.load(paths[split])
            continue
        if model is None:
            model = SentenceTransformer(name, device=device)
            model.max_seq_length = MAX_SEQ_LENGTH
            if device == "cuda":
                model.half()                      # fp16 on GPU: about twice as fast, same ranking
        t0 = time.perf_counter()
        emb = model.encode([prefix + t for t in texts], batch_size=128, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=len(texts) > 1000).astype(np.float32)
        if split == "train":
            info["encode_train_s"] = round(time.perf_counter() - t0, 1)
        EMB_DIR.mkdir(parents=True, exist_ok=True)
        np.save(paths[split], emb)
        out[split] = emb
    if model is not None:
        info["params_M"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 1)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    info["dim"] = out["train"].shape[1]
    return out, info


def rank_all(query_emb, bank_emb, query_tfidf, bank_tfidf, query_texts, twins, k, device, batch_size=1000):
    """Top-k of every (encoder, alpha), plus TF-IDF alone, in one pass over the queries.

    The TF-IDF scores of a batch are computed once and reused by every encoder and every alpha.
    Returns {(encoder, alpha): (n, k) indices}, with ("tfidf", 0.0) for TF-IDF alone.
    """
    n = query_tfidf.shape[0]
    banks = {key: torch.as_tensor(e, device=device) for key, e in bank_emb.items()}
    out = {("tfidf", 0.0): np.zeros((n, k), dtype=np.int64)}
    out.update({(key, a): np.zeros((n, k), dtype=np.int64) for key in bank_emb for a in ALPHAS})
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        lexical = torch.as_tensor((query_tfidf[start:stop] @ bank_tfidf.T).toarray(), dtype=torch.float32,
                                  device=device)
        rows, cols = [], []
        for row, text in enumerate(query_texts[start:stop]):
            for col in twins.get(normalise(text), []):
                rows.append(row)
                cols.append(col)
        rows = torch.as_tensor(rows, dtype=torch.long, device=device)
        cols = torch.as_tensor(cols, dtype=torch.long, device=device)

        def top(scores):
            scores[rows, cols] = -torch.inf
            return scores.topk(k, dim=1).indices.cpu().numpy()

        out[("tfidf", 0.0)][start:stop] = top(lexical.clone())
        for key, bank in banks.items():
            semantic = torch.as_tensor(query_emb[key][start:stop], device=device) @ bank.T
            for a in ALPHAS:
                out[(key, a)][start:stop] = top(a * semantic + (1 - a) * lexical)
    return out


def plot_encoders(table, path=FIG_DIR / "retrieval_encoders.png"):
    """Test P@5 of each encoder alone and fused with TF-IDF (dot plot), TF-IDF alone as a reference line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blue, orange, ink, muted, grid, surface = "#2a78d6", "#eb6834", "#0b0b0b", "#6b6a64", "#e6e5e0", "#fcfcfb"
    test = table[table.split == "test"]
    tfidf = test.loc[test.model == "tfidf", "P@5"].iloc[0]
    alone = test[(test.model != "tfidf") & (test.alpha == 1.0)].set_index("model")["P@5"]
    fused = test[(test.model != "tfidf") & (test.alpha < 1.0)].set_index("model")
    order = fused["P@5"].sort_values().index
    y = np.arange(len(order))
    plt.rcParams.update({"font.size": 11, "axes.edgecolor": muted, "xtick.color": muted, "ytick.color": ink})
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(order) + 1.9), facecolor=surface)
    ax.set_facecolor(surface)
    ax.grid(axis="x", color=grid, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.hlines(y, alone[order], fused.loc[order, "P@5"], color=grid, lw=2, zorder=1)
    ax.scatter(alone[order], y, s=64, color=blue, edgecolor=surface, lw=1.5, zorder=3, label="encodeur seul")
    ax.scatter(fused.loc[order, "P@5"], y, s=64, color=orange, edgecolor=surface, lw=1.5, zorder=3,
               label="fusion avec TF-IDF")
    ax.axvline(tfidf, color=muted, lw=1.2, ls="--", label=f"TF-IDF seul ({tfidf:.3f})".replace(".", ","))
    ax.set_yticks(y, [f"{m} (alpha {fused.loc[m, 'alpha']:g})".replace(".", ",") for m in order])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ",")))
    ax.set_xlabel("P@5 sur le test (part des 5 résultats du bon type), alpha choisi en validation", color=ink,
                  fontsize=9.5)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=9, handletextpad=0.3,
              columnspacing=1.2, borderaxespad=0.2)
    ax.set_title("Encodeurs de phrases pour la banque de clauses", loc="left", color=ink, fontsize=12.5, pad=28)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=surface)
    plt.close(fig)
    return path


def main(models=tuple(ENCODERS)):
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds, _ = load_ledgar()
    train, val, test = ds["train"], ds["validation"], ds["test"]
    tag = "_fast" if FAST_RUN else ""
    texts = {"train": list(train["text"]), "validation": list(val["text"]), "test": list(test["text"])}
    labels = {split: np.asarray(d["label"]) for split, d in (("train", train), ("validation", val), ("test", test))}
    k = max(KS)

    emb, infos = {}, []
    for key in models:
        print(f"[{key}] encoding on {device} ...", flush=True)
        emb[key], info = encode_splits(key, texts, device, tag)
        infos.append(info)
        print(f"[{key}] {info}", flush=True)

    vec = make_vectorizer()                        # same TF-IDF as the baseline, fitted on train only
    bank_tfidf = vec.fit_transform(texts["train"])
    twins = twin_index(texts["train"])

    ranked = {}
    for split in ("validation", "test"):
        t0 = time.perf_counter()
        ranked[split] = rank_all({key: emb[key][split] for key in models}, {key: emb[key]["train"] for key in models},
                                 vec.transform(texts[split]), bank_tfidf, texts[split], twins, k, device)
        print(f"{split}: ranked in {time.perf_counter() - t0:.0f} s", flush=True)

    rows = []
    for split in ("validation", "test"):
        for (key, a), r in ranked[split].items():
            rows.append({"model": key, "alpha": a, "split": split,
                         **retrieval_scores(relevance(r, labels[split], labels["train"]))})
    scores = pd.DataFrame(rows)
    val_scores = scores[(scores.split == "validation") & (scores.alpha < 1.0) & (scores.model != "tfidf")]
    best_alpha = val_scores.loc[val_scores.groupby("model")[f"MRR@{k}"].idxmax()].set_index("model")["alpha"]

    keep = [(scores.model == "tfidf")]
    for key in models:
        keep.append((scores.model == key) & ((scores.alpha == 1.0) | (scores.alpha == best_alpha[key])))
    table = scores[np.logical_or.reduce(keep)].copy()
    table["method"] = np.where(table.model == "tfidf", "TF-IDF seul",
                               np.where(table.alpha == 1.0, "encodeur seul", "fusion avec TF-IDF"))
    info = pd.DataFrame(infos).reindex(columns=["model", "hf_name", "params_M", "dim", "encode_train_s"])
    table = table.merge(info, on="model", how="left")
    table = table.sort_values(["split", "method", "P@5"], ascending=[False, True, False])
    table.to_csv(TAB_DIR / f"retrieval_encoders{tag}.csv", index=False)
    if not FAST_RUN:
        plot_encoders(table)
    cols = ["model", "method", "alpha", "P@1", "P@5", "P@10", f"MRR@{k}"]
    print(table[table.split == "test"][cols].round(4).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", choices=list(ENCODERS), default=list(ENCODERS))
    main(tuple(parser.parse_args().models))
