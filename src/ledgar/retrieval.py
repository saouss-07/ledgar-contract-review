"""Clause retrieval with sentence embeddings (course S03, volet B). Milestone m2.

The clause bank is the train split (60 000 clauses with their type). Each clause is encoded once by the
pretrained sentence-transformer all-MiniLM-L6-v2 (no training), the vectors are L2-normalised, so the cosine
similarity is a dot product, and the top-k is a NumPy argpartition (FAISS is not needed at this size).

Two kinds of queries, as in the brief:
- a pasted clause: "find me clauses like this one";
- a lawyer's question: "show me termination clauses with a 30-day notice period". The hybrid search combines the
  semantic score with two filters: the clause type (named in the question, or predicted by the TF-IDF baseline
  until BERT from volet A is ready) and the durations found by the regex of ledgar.metadata.

Evaluation: a retrieved clause is relevant when it has the same type as the query clause. Queries are the
validation clauses (for choices) and the test clauses (final numbers). Exact twins of the query in the bank
(2.8 % of the test, see the EDA) are removed from the ranking, otherwise they would inflate Precision@1.

    python -m ledgar.retrieval            # encodes 80 000 clauses: about 3 min on a T4, 20-40 min on a laptop CPU
    FAST_RUN=1 python -m ledgar.retrieval # quick check on the subsample

Outputs: results/tables/retrieval_results.csv, retrieval_p5_per_class.csv, retrieval_examples.csv.
Embeddings and the baseline classifier are cached in data/ and models/ (ignored by Git).
"""
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ledgar.data import FAST_RUN, load_ledgar, set_seed
from ledgar.metadata import find_durations

ROOT = Path(__file__).resolve().parents[2]
TAB_DIR = ROOT / "results" / "tables"
EMB_DIR = ROOT / "data" / "embeddings"
MODEL_DIR = ROOT / "models"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"   # 384 dimensions, truncates at 256 word pieces
KS = (1, 5, 10)
BASELINE_SETTING = dict(C=30.0, class_weight="balanced")  # chosen on validation in ledgar.baseline

_WORD_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
                 "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40,
                 "forty-five": 45, "fifty": 50, "sixty": 60, "ninety": 90, "one hundred twenty": 120,
                 "one hundred eighty": 180}


def normalise(text):
    """Same normalisation as the EDA to detect exact twins."""
    return re.sub(r"\s+", " ", text.lower()).strip()


# ---------------------------------------------------------------------------------------------------------------
# Encoding and top-k
# ---------------------------------------------------------------------------------------------------------------

def load_encoder(model_name=MODEL_NAME):
    from sentence_transformers import SentenceTransformer   # imported here: the tests do not need the model
    return SentenceTransformer(model_name)


def encode(encoder, texts, batch_size=64):
    """float32 matrix (n, 384), each row of norm 1."""
    return encoder.encode(list(texts), batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True,
                          show_progress_bar=len(texts) > 1000).astype(np.float32)


def cached_embeddings(encoder, texts, name):
    """Encode once, then reload data/embeddings/<name>.npy."""
    path = EMB_DIR / f"{name}.npy"
    if path.exists():
        emb = np.load(path)
        if len(emb) == len(texts):
            return emb
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    emb = encode(encoder, texts)
    np.save(path, emb)
    return emb


def top_k(scores, k):
    """Indices of the k highest scores of each row, best first. scores: (n_queries, n_bank)."""
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1)
    return np.take_along_axis(part, order, axis=1)


def twin_index(bank_texts):
    """normalised text -> indices of the bank clauses with that text."""
    index = {}
    for i, t in enumerate(bank_texts):
        index.setdefault(normalise(t), []).append(i)
    return index


def rank_bank(query_vecs, bank_vecs, k, query_texts=None, twins=None, allowed=None, query_labels=None,
              bank_labels=None, batch_size=1000):
    """Top-k bank indices for each query (cosine = dot product of normalised vectors).

    query_vecs / bank_vecs: dense arrays or scipy sparse matrices with L2-normalised rows.
    twins + query_texts: remove from the ranking the bank clauses identical to the query.
    allowed: optional boolean vector (n_bank,), False = filtered out for every query.
    query_labels + bank_labels: keep, for each query, only the bank clauses of its own (e.g. predicted) type.
    Rows with fewer than k remaining clauses are padded with -1.
    """
    out = np.full((query_vecs.shape[0], k), -1, dtype=np.int64)
    for start in range(0, query_vecs.shape[0], batch_size):
        stop = min(start + batch_size, query_vecs.shape[0])
        scores = query_vecs[start:stop] @ bank_vecs.T
        scores = scores.toarray() if hasattr(scores, "toarray") else np.asarray(scores)
        scores = scores.astype(np.float32)
        if twins is not None:
            for row, text in enumerate(query_texts[start:stop]):
                scores[row, twins.get(normalise(text), [])] = -np.inf
        if allowed is not None:
            scores[:, ~allowed] = -np.inf
        if query_labels is not None:
            scores[np.asarray(bank_labels)[None, :] != np.asarray(query_labels)[start:stop, None]] = -np.inf
        idx = top_k(scores, k)
        valid = np.isfinite(np.take_along_axis(scores, idx, axis=1))
        out[start:stop] = np.where(valid, idx, -1)
    return out


# ---------------------------------------------------------------------------------------------------------------
# Metrics: a retrieved clause is relevant when it has the type of the query clause
# ---------------------------------------------------------------------------------------------------------------

def relevance(ranked, query_labels, bank_labels):
    """Boolean matrix (n_queries, k): True where the retrieved clause has the query's type (-1 = nothing)."""
    bank_labels = np.asarray(bank_labels)
    rel = bank_labels[np.clip(ranked, 0, None)] == np.asarray(query_labels)[:, None]
    return rel & (ranked >= 0)


def precision_at_k(rel, k):
    """Mean share of relevant clauses in the first k results."""
    return float(rel[:, :k].mean())


def mrr(rel):
    """Mean reciprocal rank of the first relevant result (0 when none in the k results)."""
    first = rel.argmax(axis=1)
    return float(np.where(rel.any(axis=1), 1.0 / (first + 1), 0.0).mean())


def retrieval_scores(rel, ks=KS):
    row = {f"P@{k}": precision_at_k(rel, k) for k in ks if k <= rel.shape[1]}
    row[f"MRR@{rel.shape[1]}"] = mrr(rel)
    return row


# ---------------------------------------------------------------------------------------------------------------
# Hybrid search for a lawyer's question: semantic score + clause type + durations
# ---------------------------------------------------------------------------------------------------------------

def _duration_key(number, unit):
    """("thirty", "days") and ("30", "day") both become (30, "day")."""
    value = int(number) if number.isdigit() else _WORD_NUMBERS.get(number.lower())
    return value, unit.lower().rstrip("s")


def durations(text):
    """Set of (value, unit) in a text. Hyphens are spaces first, so that "30-day notice" is found."""
    text = re.sub(r"(\d)\s*-\s*(?=[a-z])", r"\1 ", text, flags=re.IGNORECASE)
    return {_duration_key(n, u) for n, u in find_durations(text)}


def label_in_question(question, label_names):
    """Clause type named in the question ("termination" -> Terminations), the longest name wins, or None."""
    q = question.lower()
    hits = []
    for i, name in enumerate(label_names):
        stem = re.sub(r"s$", "", name.lower())
        if re.search(rf"\b{re.escape(stem)}", q):
            hits.append((len(stem), i))
    return max(hits)[1] if hits else None


class ClauseBank:
    """The searchable bank: texts, types, normalised embeddings, durations, and a classifier for the queries."""

    def __init__(self, texts, labels, label_names, embeddings, encoder, classifier=None):
        self.texts, self.labels, self.label_names = list(texts), np.asarray(labels), label_names
        self.embeddings, self.encoder, self.classifier = embeddings, encoder, classifier
        self.durations = [durations(t) for t in self.texts]
        self.twins = twin_index(self.texts)

    def predict_label(self, text):
        if self.classifier is None:
            return None
        return int(self.classifier.predict([text])[0])

    def search(self, query, k=5, label=None, duration=None, exclude_twins=True):
        """DataFrame of the k most similar bank clauses, after the optional filters.

        label: a class number; "predicted": predicted by the classifier (for a pasted clause, whose words such as
            "Agreement" would match a type name); "auto": named in the question, else predicted (for a question).
        duration: a (value, unit) pair such as (30, "day"), or "auto": found in the query by regex.
        """
        if label == "auto":
            label = label_in_question(query, self.label_names)
            if label is None:
                label = self.predict_label(query)
        elif label == "predicted":
            label = self.predict_label(query)
        if duration == "auto":
            found = durations(query)
            duration = next(iter(found)) if len(found) == 1 else None
        allowed = np.ones(len(self.texts), dtype=bool)
        if label is not None:
            allowed &= self.labels == label
        if duration is not None:
            allowed &= np.array([duration in d for d in self.durations])
        q = encode(self.encoder, [query])
        ranked = rank_bank(q, self.embeddings, k, query_texts=[query], twins=self.twins if exclude_twins else None,
                           allowed=allowed)[0]
        ranked = ranked[ranked >= 0]
        scores = self.embeddings[ranked] @ q[0]
        return pd.DataFrame({"rank": np.arange(1, len(ranked) + 1), "score": scores.round(4), "bank_idx": ranked,
                             "label_name": [self.label_names[self.labels[i]] for i in ranked],
                             "text": [self.texts[i] for i in ranked]}).assign(
            filter_label=None if label is None else self.label_names[label],
            filter_duration=None if duration is None else f"{duration[0]} {duration[1]}")


def baseline_classifier(train, tag=""):
    """TF-IDF + logistic regression of ledgar.baseline, refit once on train and cached in models/."""
    from sklearn.pipeline import make_pipeline

    from ledgar.baseline import fit_model, make_vectorizer

    path = MODEL_DIR / f"baseline_tfidf_logreg{tag}.joblib"
    if path.exists():
        return joblib.load(path)
    vec = make_vectorizer()
    X = vec.fit_transform(train["text"])
    clf = make_pipeline(vec, fit_model(X, train["label"], **BASELINE_SETTING))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, path)
    return clf


# Hand-checked queries (milestone m2): 3 pasted clauses from the test split and 5 lawyer's questions
QUESTIONS = [
    "show me termination clauses with a 30-day notice period",
    "governing law of the State of Delaware",
    "confidentiality obligations that survive termination of the agreement",
    "indemnification of the company's directors and officers",
    "the employee may not compete with the company for twelve months after leaving",
]
N_PASTED = 3


def main():
    set_seed()
    ds, names = load_ledgar()
    train, val, test = ds["train"], ds["validation"], ds["test"]
    tag = "_fast" if FAST_RUN else ""
    k_max = max(KS)

    t0 = time.perf_counter()
    encoder = load_encoder()
    emb = {split: cached_embeddings(encoder, data["text"], f"minilm_{split}{tag}")
           for split, data in (("train", train), ("validation", val), ("test", test))}
    print(f"embeddings {emb['train'].shape} in {time.perf_counter() - t0:.0f} s | FAST_RUN={FAST_RUN}", flush=True)

    clf = baseline_classifier(train, tag)
    vec = clf.steps[0][1]   # the TF-IDF of the classifier, fitted on train: the lexical retrieval baseline
    tfidf = {"train": vec.transform(train["text"])}
    twins = twin_index(train["text"])
    bank_labels = np.asarray(train["label"])

    rows, per_class = [], None
    for split, data in (("validation", val), ("test", test)):
        texts, labels = list(data["text"]), np.asarray(data["label"])
        pred = clf.predict(texts)                                # hybrid: bank clauses of the predicted type
        runs = {
            "tfidf_cosinus": (vec.transform(texts), tfidf["train"], None),
            "minilm": (emb[split], emb["train"], None),
            "minilm + classe predite": (emb[split], emb["train"], pred),
        }
        for method, (q, bank, filter_labels) in runs.items():
            for exclude in (False, True):
                ranked = rank_bank(q, bank, k_max, query_texts=texts, twins=twins if exclude else None,
                                   query_labels=filter_labels, bank_labels=bank_labels)
                rel = relevance(ranked, labels, bank_labels)
                rows.append({"method": method, "split": split, "twins_removed": exclude, **retrieval_scores(rel)})
                print({k: round(v, 4) if isinstance(v, float) else v for k, v in rows[-1].items()}, flush=True)
                if split == "test" and method == "minilm" and exclude:
                    per_class = pd.DataFrame({"label": labels, "relevant_at_5": rel[:, :5].mean(axis=1)})
    pd.DataFrame(rows).to_csv(TAB_DIR / f"retrieval_results{tag}.csv", index=False)

    table = per_class.groupby("label")["relevant_at_5"].agg(["count", "mean"]).reset_index()
    table.columns = ["label", "support", "P@5"]
    table.insert(1, "label_name", [names[i] for i in table["label"]])
    table.sort_values("P@5").to_csv(TAB_DIR / f"retrieval_p5_per_class{tag}.csv", index=False)

    bank = ClauseBank(train["text"], train["label"], names, emb["train"], encoder, clf)
    examples = []
    rng = np.random.default_rng(0)
    for i in rng.choice(len(test), N_PASTED, replace=False):
        res = bank.search(test[int(i)]["text"], k=5, label="predicted")
        examples.append(res.assign(query_kind="clause collee", query=test[int(i)]["text"],
                                   query_label=names[test[int(i)]["label"]]))
    for question in QUESTIONS:
        for label, duration, kind in ((None, None, "question, semantique seule"),
                                      ("auto", "auto", "question, hybride")):
            res = bank.search(question, k=5, label=label, duration=duration)
            examples.append(res.assign(query_kind=kind, query=question, query_label=None))
    ex = pd.concat(examples, ignore_index=True)
    ex["text"] = ex["text"].str.slice(0, 300)
    ex["query"] = ex["query"].str.slice(0, 300)
    cols = ["query_kind", "query", "query_label", "filter_label", "filter_duration", "rank", "score", "label_name",
            "bank_idx", "text"]
    ex[cols].to_csv(TAB_DIR / f"retrieval_examples{tag}.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
