"""Clause retrieval with sentence embeddings (course S03, volet B). Milestone m2.

The clause bank is the train split (60 000 clauses with their type). Each clause is encoded once by the
pretrained sentence-transformer all-MiniLM-L6-v2 (no training), the vectors are L2-normalised, so the cosine
similarity is a dot product, and the top-k is a NumPy argpartition (FAISS is not needed at this size).

Two kinds of queries, as in the brief:
- a pasted clause: "find me clauses like this one";
- a lawyer's question: "show me termination clauses with a 30-day notice period". The hybrid search combines the
  semantic score with two filters: the clause type (named in the question, or predicted by the TF-IDF baseline
  until BERT from volet A is ready) and the durations found by the regex of ledgar.metadata.

Two refinements, both chosen on validation:
- score fusion: alpha * cosine(MiniLM) + (1 - alpha) * cosine(TF-IDF), because the lexical baseline beats MiniLM
  alone on these very standardised texts;
- confidence fallback: the predicted type filters the bank only when the classifier is confident enough, so that a
  question outside the 100 types (e.g. non-compete) is searched in the whole bank instead of a wrong type.

Evaluation: a retrieved clause is relevant when it has the same type as the query clause. Queries are the
validation clauses (for choices) and the test clauses (final numbers). Exact twins of the query in the bank
(2.8 % of the test, see the EDA) are removed from the ranking, otherwise they would inflate Precision@1.

    python -m ledgar.retrieval            # encodes 80 000 clauses: about 3 min on a T4, 20-40 min on a laptop CPU
    FAST_RUN=1 python -m ledgar.retrieval # quick check on the subsample

Outputs: results/tables/retrieval_results.csv, retrieval_choices.csv (validation grid), retrieval_p5_per_class.csv,
retrieval_examples.csv.
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


def similarity(query_vecs, bank_vecs, start, stop, weights=None):
    """Cosine scores (stop - start, n_bank) of the queries start:stop.

    With lists of representations (e.g. [MiniLM, TF-IDF]) and their weights, the weighted sum of the cosines:
    score fusion, alpha * semantic + (1 - alpha) * lexical.
    """
    if not isinstance(query_vecs, (list, tuple)):
        query_vecs, bank_vecs, weights = [query_vecs], [bank_vecs], [1.0]
    total = 0.0
    for q, b, w in zip(query_vecs, bank_vecs, weights):
        s = q[start:stop] @ b.T
        s = s.toarray() if hasattr(s, "toarray") else np.asarray(s)
        total = total + np.float32(w) * s.astype(np.float32)
    return total


def rank_bank(query_vecs, bank_vecs, k, query_texts=None, twins=None, allowed=None, query_labels=None,
              bank_labels=None, weights=None, batch_size=1000):
    """Top-k bank indices for each query (cosine = dot product of normalised vectors).

    query_vecs / bank_vecs: dense arrays or scipy sparse matrices with L2-normalised rows, or lists of them with
    `weights` (score fusion, see `similarity`).
    twins + query_texts: remove from the ranking the bank clauses identical to the query.
    allowed: optional boolean vector (n_bank,), False = filtered out for every query.
    query_labels + bank_labels: keep, for each query, only the bank clauses of its own (e.g. predicted) type.
    Rows with fewer than k remaining clauses are padded with -1.
    """
    n = (query_vecs[0] if isinstance(query_vecs, (list, tuple)) else query_vecs).shape[0]
    out = np.full((n, k), -1, dtype=np.int64)
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        scores = similarity(query_vecs, bank_vecs, start, stop, weights)
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
    """The searchable bank: texts, types, normalised embeddings, durations, and a classifier for the queries.

    tfidf (optional): a fitted TF-IDF vectorizer; the score is then alpha * MiniLM + (1 - alpha) * TF-IDF.
    min_confidence: the predicted type is used as a filter only when the classifier's probability reaches it,
        otherwise the search runs on the whole bank (a question outside the 100 types is not forced into one).
    alpha and min_confidence are chosen on validation by main().
    """

    def __init__(self, texts, labels, label_names, embeddings, encoder, classifier=None, tfidf=None, alpha=1.0,
                 min_confidence=0.0):
        self.texts, self.labels, self.label_names = list(texts), np.asarray(labels), label_names
        self.embeddings, self.encoder, self.classifier = embeddings, encoder, classifier
        self.tfidf, self.alpha, self.min_confidence = tfidf, alpha, min_confidence
        self.bank_tfidf = tfidf.transform(self.texts) if tfidf is not None else None
        self.durations = [durations(t) for t in self.texts]
        self.twins = twin_index(self.texts)

    def predict_label(self, text):
        """(predicted type, its probability), or (None, 0.0) without a classifier."""
        if self.classifier is None:
            return None, 0.0
        proba = self.classifier.predict_proba([text])[0]
        best = int(np.argmax(proba))
        return int(self.classifier.classes_[best]), float(proba[best])

    def search(self, query, k=5, label=None, duration=None, exclude_twins=True, alpha=None):
        """DataFrame of the k most similar bank clauses, after the optional filters.

        label: a class number; "predicted": predicted by the classifier (for a pasted clause, whose words such as
            "Agreement" would match a type name); "auto": named in the question, else predicted (for a question).
            A predicted type below min_confidence is not used as a filter.
        duration: a (value, unit) pair such as (30, "day"), or "auto": found in the query by regex.
        alpha: weight of MiniLM in the score (1.0 = semantic only); default: the bank's alpha.
        """
        confidence = None
        if label == "auto":
            label = label_in_question(query, self.label_names)
            if label is None:
                label, confidence = self.predict_label(query)
        elif label == "predicted":
            label, confidence = self.predict_label(query)
        if confidence is not None and confidence < self.min_confidence:
            label = None
        if duration == "auto":
            found = durations(query)
            duration = next(iter(found)) if len(found) == 1 else None
        allowed = np.ones(len(self.texts), dtype=bool)
        if label is not None:
            allowed &= self.labels == label
        if duration is not None:
            allowed &= np.array([duration in d for d in self.durations])
        alpha = self.alpha if alpha is None else alpha
        q = encode(self.encoder, [query])
        if self.bank_tfidf is not None and alpha < 1:
            qs, banks, weights = [q, self.tfidf.transform([query])], [self.embeddings, self.bank_tfidf], [alpha, 1 - alpha]
        else:
            qs, banks, weights = q, self.embeddings, None
        twins = self.twins if exclude_twins else None
        ranked = rank_bank(qs, banks, k, query_texts=[query], twins=twins, allowed=allowed, weights=weights)[0]
        ranked = ranked[ranked >= 0]
        scores = similarity(qs, banks, 0, 1, weights)[0, ranked]
        return pd.DataFrame({"rank": np.arange(1, len(ranked) + 1), "score": scores.round(4), "bank_idx": ranked,
                             "label_name": [self.label_names[self.labels[i]] for i in ranked],
                             "text": [self.texts[i] for i in ranked]}).assign(
            filter_label=None if label is None else self.label_names[label],
            filter_duration=None if duration is None else f"{duration[0]} {duration[1]}",
            confidence=None if confidence is None else round(confidence, 3))


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
    "which clauses limit the liability of the supplier?",
]
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)          # weight of MiniLM in the fused score, chosen on validation
THRESHOLDS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)   # minimal classifier confidence to filter by type
MAX_P5_COST = 0.005   # the threshold kept is the highest one costing at most 0.5 point of P@5 on validation
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

    queries = {}
    for split, data in (("validation", val), ("test", test)):
        texts = list(data["text"])
        proba = clf.predict_proba(texts)
        queries[split] = dict(texts=texts, labels=np.asarray(data["label"]), tfidf=vec.transform(texts),
                              pred=clf.classes_[proba.argmax(axis=1)], confidence=proba.max(axis=1))

    def run(split, alpha, filtered, exclude=True):
        """Top-k for every query of the split: alpha * MiniLM + (1 - alpha) * TF-IDF, optionally predicted type only."""
        q = queries[split]
        if alpha == 1:
            qs, banks, weights = emb[split], emb["train"], None
        elif alpha == 0:
            qs, banks, weights = q["tfidf"], tfidf["train"], None
        else:
            qs, banks, weights = [emb[split], q["tfidf"]], [emb["train"], tfidf["train"]], [alpha, 1 - alpha]
        return rank_bank(qs, banks, k_max, query_texts=q["texts"], twins=twins if exclude else None,
                         query_labels=q["pred"] if filtered else None, bank_labels=bank_labels, weights=weights)

    def with_fallback(split, ranked_filtered, ranked_all, threshold):
        """Predicted-type filter only for the queries whose classifier confidence reaches the threshold."""
        return np.where((queries[split]["confidence"] >= threshold)[:, None], ranked_filtered, ranked_all)

    # Choices on validation only (twins removed): the fusion weight alpha (by MRR), then the confidence threshold
    # of the type filter (P@5: what the lawyer sees)
    choices, val_labels = [], queries["validation"]["labels"]
    ranked_val = {alpha: run("validation", alpha, filtered=False) for alpha in ALPHAS}
    for alpha, ranked in ranked_val.items():
        choices.append({"choice": "alpha", "value": alpha,
                        **retrieval_scores(relevance(ranked, val_labels, bank_labels))})
    alpha = max((c for c in choices if c["choice"] == "alpha"), key=lambda c: c[f"MRR@{k_max}"])["value"]
    filtered_val = run("validation", alpha, filtered=True)
    for threshold in THRESHOLDS:
        ranked = with_fallback("validation", filtered_val, ranked_val[alpha], threshold)
        choices.append({"choice": "seuil_confiance", "value": threshold,
                        **retrieval_scores(relevance(ranked, val_labels, bank_labels))})
    # The validation clauses all belong to the 100 types, where filtering by the predicted type always helps: the best
    # threshold there is 0, which gives no protection to questions outside the types. We keep the highest threshold
    # whose cost on validation is at most MAX_P5_COST (a guard-rail with a measured price).
    grid = [c for c in choices if c["choice"] == "seuil_confiance"]
    best_p5 = max(c["P@5"] for c in grid)
    threshold = max(c["value"] for c in grid if c["P@5"] >= best_p5 - MAX_P5_COST)
    pd.DataFrame(choices).to_csv(TAB_DIR / f"retrieval_choices{tag}.csv", index=False)
    print(f"chosen on validation: alpha = {alpha}, confidence threshold = {threshold}", flush=True)

    rows, per_class = [], None
    for split in ("validation", "test"):
        labels = queries[split]["labels"]
        for exclude in (False, True):
            ranked = {name: run(split, a, filtered=f, exclude=exclude) for name, a, f in (
                ("tfidf_cosinus", 0, False), ("minilm", 1, False), (f"fusion (alpha = {alpha})", alpha, False),
                ("minilm + classe predite", 1, True), ("fusion + classe predite", alpha, True))}
            ranked[f"fusion + classe predite si confiance >= {threshold}"] = with_fallback(
                split, ranked["fusion + classe predite"], ranked[f"fusion (alpha = {alpha})"], threshold)
            for method, r in ranked.items():
                rel = relevance(r, labels, bank_labels)
                rows.append({"method": method, "split": split, "twins_removed": exclude, **retrieval_scores(rel)})
                print({k: round(v, 4) if isinstance(v, float) else v for k, v in rows[-1].items()}, flush=True)
                if split == "test" and method == "minilm" and exclude:
                    per_class = pd.DataFrame({"label": labels, "relevant_at_5": rel[:, :5].mean(axis=1)})
    pd.DataFrame(rows).to_csv(TAB_DIR / f"retrieval_results{tag}.csv", index=False)

    table = per_class.groupby("label")["relevant_at_5"].agg(["count", "mean"]).reset_index()
    table.columns = ["label", "support", "P@5"]
    table.insert(1, "label_name", [names[i] for i in table["label"]])
    table.sort_values("P@5").to_csv(TAB_DIR / f"retrieval_p5_per_class{tag}.csv", index=False)

    bank = ClauseBank(train["text"], train["label"], names, emb["train"], encoder, clf, tfidf=vec, alpha=alpha,
                      min_confidence=threshold)
    examples = []
    rng = np.random.default_rng(0)
    for i in rng.choice(len(test), N_PASTED, replace=False):
        res = bank.search(test[int(i)]["text"], k=5, label="predicted")
        examples.append(res.assign(query_kind="clause collee, hybride", query=test[int(i)]["text"],
                                   query_label=names[test[int(i)]["label"]]))
    for question in QUESTIONS:
        for label, duration, a, kind in ((None, None, 1.0, "question, semantique seule"),
                                         ("auto", "auto", None, "question, hybride")):
            res = bank.search(question, k=5, label=label, duration=duration, alpha=a)
            examples.append(res.assign(query_kind=kind, query=question, query_label=None))
    ex = pd.concat(examples, ignore_index=True)
    ex["text"] = ex["text"].str.slice(0, 300)
    ex["query"] = ex["query"].str.slice(0, 300)
    cols = ["query_kind", "query", "query_label", "filter_label", "confidence", "filter_duration", "rank", "score",
            "label_name", "bank_idx", "text"]
    ex[cols].to_csv(TAB_DIR / f"retrieval_examples{tag}.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
