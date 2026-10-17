"""Metrics shared by every model, and the common predictions format.

- accuracy, macro-F1, per-class F1, confusion matrix: written in NumPy (course S01, TP01 part B) and checked
  against scikit-learn in tests/test_metrics.py.
- compute_metrics: the function the Hugging Face Trainer calls (course S06, TP06 exercise 9).
- contingency, mcnemar, paired_bootstrap: paired comparison of two models on the same test set (S06, TP06 part D).
- save_predictions / load_predictions: ONE file format for every model, so that the audit (volet C) reads the
  baseline, the LSTM and BERT the same way. Rows are keyed by `idx` (see ledgar.data), no text is stored.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd

PREDICTION_COLUMNS = ["idx", "label", "pred", "confidence"]
PRED_DIR = Path(__file__).resolve().parents[2] / "results" / "predictions"


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def confusion_matrix(y_true, y_pred, n_classes):
    """C[i, j] = number of examples of true class i predicted as j."""
    C = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(C, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return C


def per_class_f1(y_true, y_pred, n_classes):
    """F1_k = 2 TP_k / (2 TP_k + FP_k + FN_k), 0 when the class is neither present nor predicted."""
    C = confusion_matrix(y_true, y_pred, n_classes)
    tp = np.diag(C)
    fp = C.sum(axis=0) - tp
    fn = C.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    return np.divide(2 * tp, denom, out=np.zeros(n_classes), where=denom > 0), denom > 0


def macro_f1(y_true, y_pred, n_classes=None):
    """Plain mean of the per-class F1 over the classes that appear in y_true or y_pred (scikit-learn's rule)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if n_classes is None:
        n_classes = int(max(y_true.max(), y_pred.max())) + 1
    f1, seen = per_class_f1(y_true, y_pred, n_classes)
    return float(f1[seen].mean())


def compute_metrics(eval_pred):
    """For the Hugging Face Trainer: eval_pred = (logits, labels)."""
    logits, labels = eval_pred
    preds = np.asarray(logits).argmax(axis=-1)
    return {"accuracy": accuracy(labels, preds), "macro_f1": macro_f1(labels, preds, n_classes=logits.shape[-1])}


def f1_table(y_true, y_pred, label_names):
    """One row per class: support, precision, recall, F1, sorted by F1 (worst first)."""
    n = len(label_names)
    C = confusion_matrix(y_true, y_pred, n)
    tp = np.diag(C)
    support, predicted = C.sum(axis=1), C.sum(axis=0)
    f1, _ = per_class_f1(y_true, y_pred, n)
    return pd.DataFrame({
        "label": range(n), "label_name": label_names, "support": support,
        "precision": np.divide(tp, predicted, out=np.zeros(n), where=predicted > 0),
        "recall": np.divide(tp, support, out=np.zeros(n), where=support > 0),
        "f1": f1,
    }).sort_values("f1").reset_index(drop=True)


def top_confusions(y_true, y_pred, label_names, k=10):
    """The k most frequent (true, predicted) pairs of distinct classes."""
    C = confusion_matrix(y_true, y_pred, len(label_names))
    np.fill_diagonal(C, 0)
    flat = np.argsort(C, axis=None)[::-1][:k]
    rows = [(label_names[i], label_names[j], int(C[i, j])) for i, j in zip(*np.unravel_index(flat, C.shape)) if C[i, j] > 0]
    return pd.DataFrame(rows, columns=["vraie_classe", "classe_predite", "n"])


def contingency(y_true, pred_a, pred_b):
    """(a, b, c, d): both right, only A right, only B right, both wrong."""
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    ok_a, ok_b = pred_a == y_true, pred_b == y_true
    return (int(np.sum(ok_a & ok_b)), int(np.sum(ok_a & ~ok_b)), int(np.sum(~ok_a & ok_b)), int(np.sum(~ok_a & ~ok_b)))


def mcnemar(b, c):
    """McNemar's test with continuity correction: (chi2, p_value). Only the disagreements b and c count."""
    if b + c == 0:
        return 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return chi2, math.erfc(math.sqrt(chi2 / 2))


def paired_bootstrap(y_true, pred_a, pred_b, n_classes, n_boot=1000, seed=0):
    """macro-F1(A) - macro-F1(B) on the test set, and its 95 % bootstrap interval: (observed, low, high)."""
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    rng = np.random.default_rng(seed)
    n = len(y_true)
    observed = macro_f1(y_true, pred_a, n_classes) - macro_f1(y_true, pred_b, n_classes)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        s = rng.integers(0, n, n)
        diffs[i] = macro_f1(y_true[s], pred_a[s], n_classes) - macro_f1(y_true[s], pred_b[s], n_classes)
    low, high = np.percentile(diffs, [2.5, 97.5])
    return float(observed), float(low), float(high)


def save_predictions(model_name, split, idx, y_true, y_pred, confidence, pred_dir=PRED_DIR):
    """Write results/predictions/<model>_<split>.csv with the columns idx, label, pred, confidence.

    confidence = probability of the predicted class (max softmax), used for calibration and abstention.
    """
    df = pd.DataFrame({"idx": np.asarray(idx), "label": np.asarray(y_true), "pred": np.asarray(y_pred),
                       "confidence": np.round(np.asarray(confidence, dtype=float), 6)})
    if not df["confidence"].between(0, 1).all():
        raise ValueError("confidence must be a probability in [0, 1]")
    path = Path(pred_dir) / f"{model_name}_{split}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_predictions(model_name, split, pred_dir=PRED_DIR):
    df = pd.read_csv(Path(pred_dir) / f"{model_name}_{split}.csv")
    missing = set(PREDICTION_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns {missing}")
    return df
