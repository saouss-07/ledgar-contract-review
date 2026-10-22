import numpy as np
import pandas as pd

from ledgar.audit import (BERT_SAME_COVERAGE, COMBINED, change_figures, choose_threshold, combine_rules, combined_ood,
                          expected_calibration_error, subgroup_table, swap_company_form, swap_gender, swap_jurisdiction,
                          wilson)


def test_combine_rules_and_equal_coverage():
    b_pred, b_conf = [0, 1, 2, 3, 4], [0.9, 0.9, 0.5, 0.9, 0.9]
    l_pred, l_conf = [0, 1, 2, 9, 4], [0.999, 0.80, 0.999, 0.999, 0.70]
    rules = combine_rules(b_pred, b_conf, l_pred, l_conf, {"baseline": 0.74, "legalbert_full_es": 0.99})
    # combined: agree (0, 1, 2, 4) AND baseline sure (not 2) -> clauses 0, 1, 4
    assert rules[COMBINED][0].tolist() == [True, True, False, False, True]
    assert rules["Legal-BERT seul"][0].tolist() == [True, False, True, True, False]
    # same coverage = Legal-BERT's 3 most confident clauses (0, 2, 3), whatever the baseline says
    assert rules[BERT_SAME_COVERAGE][0].sum() == rules[COMBINED][0].sum() == 3
    assert rules[BERT_SAME_COVERAGE][0].tolist() == [True, False, True, True, False]
    assert (rules[COMBINED][1] == np.array(l_pred)).all(), "the combined rule keeps Legal-BERT's labels"


def test_subgroup_macro_f1_ignores_types_absent_from_the_group():
    df = pd.DataFrame({"g": ["a", "a", "a", "b"], "label": [0, 0, 1, 2], "m_pred": [0, 5, 1, 2]})
    df["m_ok"] = df["m_pred"] == df["label"]
    t = subgroup_table(df, models={"m": "m"}, columns=["g"]).set_index("groupe")
    # group a: type 0 has F1 = 2/3, type 1 has F1 = 1; the wrongly predicted type 5 must not count as a 0
    assert abs(t.loc["a", "m_macro_f1"] - (2 / 3 + 1) / 2) < 1e-9
    assert t.loc["a", "m_accuracy"] == 2 / 3 and t.loc["b", "m_macro_f1"] == 1.0


def test_combined_rule_catches_a_confident_bert_error():
    keys = {"type_de_sonde": ["x", "y", "z"], "clause": ["a", "b", "c"], "type_attendu": ["A", "B", None]}
    base = pd.DataFrame({**keys, "type_predit": ["A", "B", "F"], "confiance": [0.9, 0.9, 0.3],
                         "correct": [True, True, False], "envoyee_avocat": [False, False, True]})
    bert = pd.DataFrame({**keys, "type_predit": ["A", "C", "F"], "confiance": [0.999, 0.999, 0.999],
                         "correct": [True, False, False], "envoyee_avocat": [False, False, False]})
    m = combined_ood(base, bert, threshold_baseline=0.74)
    assert m["envoyee_avocat_combine"].tolist() == [False, True, True]   # agree + sure / disagree / baseline unsure
    assert m["erreur_non_relue_bert_seul"].sum() == 2 and m["erreur_non_relue_combine"].sum() == 0


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
