"""Baseline: TF-IDF on words and bigrams + logistic regression (course S01, "bag of words"). Milestone m1.

Every choice is made on the validation split: the regularisation C and class_weight (None or "balanced",
because LEDGAR is imbalanced, see notebooks/01_eda.ipynb). The test split is only predicted with the chosen
model, for the final comparisons (BERT versus baseline at m4, the audit).

    python -m ledgar.baseline            # full run, about 25 minutes on a laptop CPU
    FAST_RUN=1 python -m ledgar.baseline # quick check on the subsample

Outputs: results/tables/baseline_grid.csv, baseline_results.csv, baseline_f1_per_class.csv,
baseline_top_confusions.csv, and results/predictions/baseline_{validation,test}.csv.
"""
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from ledgar.data import FAST_RUN, load_ledgar, set_seed
from ledgar.metrics import accuracy, f1_table, macro_f1, save_predictions, top_confusions

TAB_DIR = Path(__file__).resolve().parents[2] / "results" / "tables"
TFIDF = dict(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)
GRID_C = [2.0, 10.0, 30.0]
GRID_WEIGHT = [None, "balanced"]


def make_vectorizer():
    return TfidfVectorizer(**TFIDF)


def fit_model(X, y, C, class_weight, seed=0):
    return LogisticRegression(C=C, class_weight=class_weight, max_iter=1000, random_state=seed).fit(X, y)


def main():
    set_seed()
    ds, names = load_ledgar()
    train, val, test = ds["train"], ds["validation"], ds["test"]
    n_classes = len(names)
    tag = "_fast" if FAST_RUN else ""

    vec = make_vectorizer()
    X_train = vec.fit_transform(train["text"])        # vocabulary learnt on train only
    X_val, X_test = vec.transform(val["text"]), vec.transform(test["text"])
    print(f"TF-IDF: {X_train.shape[1]:,} features | FAST_RUN={FAST_RUN}")

    grid, best, best_model = [], None, None
    for C in GRID_C:
        for weight in GRID_WEIGHT:
            t0 = time.perf_counter()
            model = fit_model(X_train, train["label"], C, weight)
            pred = model.predict(X_val)
            row = {"C": C, "class_weight": weight or "none", "val_accuracy": accuracy(val["label"], pred),
                   "val_macro_f1": macro_f1(val["label"], pred, n_classes), "fit_seconds": time.perf_counter() - t0}
            grid.append(row)
            print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
            if best is None or row["val_macro_f1"] > best["val_macro_f1"]:
                best, best_model = row, model
    pd.DataFrame(grid).to_csv(TAB_DIR / f"baseline_grid{tag}.csv", index=False)
    print("chosen on validation:", best)

    results = []
    for split, X, data in (("validation", X_val, val), ("test", X_test, test)):
        proba = best_model.predict_proba(X)
        pred = best_model.classes_[proba.argmax(axis=1)]
        save_predictions(f"baseline{tag}", split, data["idx"], data["label"], pred, proba.max(axis=1))
        results.append({"model": "tfidf_logreg", "C": best["C"], "class_weight": best["class_weight"], "split": split,
                        "accuracy": accuracy(data["label"], pred), "macro_f1": macro_f1(data["label"], pred, n_classes)})
        if split == "test":
            f1_table(data["label"], pred, names).to_csv(TAB_DIR / f"baseline_f1_per_class{tag}.csv", index=False)
            top_confusions(data["label"], pred, names, k=10).to_csv(TAB_DIR / f"baseline_top_confusions{tag}.csv", index=False)
            # Test without the clauses already seen in train (2.8 %, see the EDA): checks the score is not inflated
            norm = lambda s: re.sub(r"\s+", " ", s.lower()).strip()
            seen = set(map(norm, train["text"]))
            keep = np.array([norm(t) not in seen for t in data["text"]])
            y, p = np.asarray(data["label"])[keep], pred[keep]
            results.append({"model": "tfidf_logreg", "C": best["C"], "class_weight": best["class_weight"],
                            "split": "test_sans_doublons", "accuracy": accuracy(y, p), "macro_f1": macro_f1(y, p, n_classes)})
    table = pd.DataFrame(results)
    table.to_csv(TAB_DIR / f"baseline_results{tag}.csv", index=False)
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
