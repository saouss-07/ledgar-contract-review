import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix as sk_confusion

from ledgar.metrics import (accuracy, compute_metrics, confusion_matrix, contingency, f1_table, load_predictions,
                            macro_f1, mcnemar, paired_bootstrap, save_predictions, top_confusions)

rng = np.random.default_rng(0)
Y = rng.integers(0, 100, 5000)
P = np.where(rng.random(5000) < 0.7, Y, rng.integers(0, 100, 5000))   # 70 % right, the rest random


def test_matches_scikit_learn():
    assert accuracy(Y, P) == pytest.approx(accuracy_score(Y, P))
    assert macro_f1(Y, P, 100) == pytest.approx(f1_score(Y, P, average="macro"))
    assert np.array_equal(confusion_matrix(Y, P, 100), sk_confusion(Y, P, labels=range(100)))


def test_matches_scikit_learn_with_missing_classes():
    y, p = np.array([0, 0, 1, 5]), np.array([0, 1, 1, 1])     # classes 2-4 never seen, 5 never predicted
    assert macro_f1(y, p, 10) == pytest.approx(f1_score(y, p, average="macro", zero_division=0))


def test_course_toy_example():
    # TP06 exercise 9: per-class F1 = 2/3, 2/3, 1 -> macro-F1 = 7/9
    m = compute_metrics((np.array([[2.0, 0, 0], [0, 1, 0], [0, 3, 0], [0, 0, 1]]), np.array([0, 0, 1, 2])))
    assert m["accuracy"] == pytest.approx(0.75) and m["macro_f1"] == pytest.approx(7 / 9)


def test_mcnemar_course_values():
    # S06 exercise 5 / TP06 exercise 20
    chi2, p = mcnemar(60, 40)
    assert chi2 == pytest.approx(3.61) and p == pytest.approx(0.0574, abs=1e-4)
    chi2, p = mcnemar(25, 5)
    assert chi2 == pytest.approx(361 / 30) and p == pytest.approx(0.00052, abs=1e-5)
    assert mcnemar(0, 0) == (0.0, 1.0)
    assert contingency([0, 1, 2, 3, 0], [0, 1, 0, 0, 0], [0, 2, 2, 0, 1]) == (1, 2, 1, 1)


def test_paired_bootstrap():
    obs, low, high = paired_bootstrap(Y, P, Y, n_classes=100, n_boot=200)
    assert obs < 0 and high < 0, "a perfect model B must be significantly better"
    assert paired_bootstrap(Y, P, P, n_classes=100, n_boot=20)[1:] == (0.0, 0.0)


def test_tables():
    names = [f"c{k}" for k in range(100)]
    t = f1_table(Y, P, names)
    assert len(t) == 100 and t["f1"].is_monotonic_increasing and t["support"].sum() == len(Y)
    conf = top_confusions(np.array([0, 0, 0, 1]), np.array([1, 1, 2, 1]), names[:3], k=5)
    assert conf.iloc[0].tolist() == ["c0", "c1", 2]


def test_predictions_roundtrip(tmp_path):
    path = save_predictions("baseline", "test", [3, 7], [1, 2], [1, 0], [0.9, 0.4], pred_dir=tmp_path)
    assert path.name == "baseline_test.csv"
    df = load_predictions("baseline", "test", pred_dir=tmp_path)
    assert df.columns.tolist() == ["idx", "label", "pred", "confidence"] and df["idx"].tolist() == [3, 7]
    with pytest.raises(ValueError):
        save_predictions("bad", "test", [0], [0], [0], [1.5], pred_dir=tmp_path)
