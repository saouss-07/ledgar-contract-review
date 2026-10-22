import numpy as np

from ledgar.audit import (change_figures, choose_threshold, expected_calibration_error, swap_company_form, swap_gender,
                          swap_jurisdiction, wilson)


def test_swap_gender_both_ways_and_case():
    assert swap_gender("He shall devote his time") == "She shall devote her time"
    assert swap_gender("she resigns and her rights") == "he resigns and his rights"
    assert swap_gender("hereunder and therein") == "hereunder and therein"


def test_swap_jurisdiction():
    assert swap_jurisdiction("the laws of the State of New York") == "the laws of the State of England and Wales"
    assert swap_jurisdiction("THE LAWS OF THE STATE OF DELAWARE", "France") == "THE LAWS OF THE STATE OF FRANCE"
    assert swap_jurisdiction("may be executed in counterparts") == "may be executed in counterparts"


def test_company_form_and_figures():
    assert swap_company_form("Acme Inc. and Beta LLC") == "Acme SAS and Beta GmbH"
    assert change_figures("thirty (30) days notice and $5,000") == "thirty (45) days notice and €5,000"
    assert change_figures("within 10 business days") == "within 25 business days"
    assert change_figures("Section 30 of the Agreement") == "Section 30 of the Agreement"


def test_threshold_reaches_target_on_the_data_it_is_chosen_on():
    rng = np.random.default_rng(0)
    conf = rng.random(2000)
    ok = rng.random(2000) < conf            # higher confidence -> more often right
    thr = choose_threshold(conf, ok, target=0.9)
    assert ok[conf >= thr].mean() >= 0.9
    assert choose_threshold(np.array([0.9, 0.8]), np.array([False, False]), target=0.9) == 1.0


def test_calibration_and_wilson():
    conf = np.full(1000, 0.8)
    assert expected_calibration_error(conf, np.arange(1000) < 800) < 1e-9   # 80 % right at 80 % confidence
    assert expected_calibration_error(conf, np.ones(1000, bool)) > 0.19
    lo, hi = wilson(80, 100)
    assert lo < 0.8 < hi and 0.70 < lo < 0.72
