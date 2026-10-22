"""Audit (volet C): where can the clause classifier be trusted, and where not?

Part 1 works on the saved predictions of every model (results/predictions/, common format), joined by `idx`
with the subgroup attributes of ledgar.metadata:
    - accuracy and macro-F1 per subgroup (class frequency, length, truncation at 256 tokens, jurisdiction,
      party, contract type, gendered pronouns, boilerplate vs novel wording);
    - calibration (is the confidence honest?) and abstention: a confidence threshold, chosen on VALIDATION, under
      which a clause is sent to a lawyer; its coverage and accuracy are then reported on test.
Part 2 needs a model that predicts new texts. It runs on the baseline (refitted in ~4 min with the setting chosen
in ledgar.baseline); for Legal-BERT, call counterfactual_report / ood_report on Colab with its predict function.
    - counterfactual tests: change something that must NOT change the clause type (gender, jurisdiction, company
      form, figures) and measure the flip rate;
    - out-of-distribution probes: European-style clauses in English, French clauses, types outside the 100.

    python -m ledgar.audit              # ~10 min on a laptop CPU
Outputs: results/tables/audit_*.csv and results/figures/audit_*.png
"""
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ledgar.baseline import fit_model, make_vectorizer
from ledgar.data import load_ledgar, set_seed
from ledgar.metadata import add_metadata, find_jurisdiction
from ledgar.metrics import load_predictions, per_class_f1

ROOT = Path(__file__).resolve().parents[2]
TAB_DIR, FIG_DIR = ROOT / "results" / "tables", ROOT / "results" / "figures"
MODELS = {"baseline": "TF-IDF (baseline)", "legalbert_full_es": "Legal-BERT", "lstm": "BiLSTM"}
TARGET_ACCURACY = 0.95      # accuracy wanted on the clauses the system labels alone
MIN_GROUP = 50              # smaller groups are reported but flagged as too small to conclude
BERT_TOKENIZER = "nlpaueb/legal-bert-base-uncased"


# ---------------------------------------------------------------- part 1: the test frame and its subgroups

def similarity_to_train(train_texts, texts, chunk=500):
    """Highest TF-IDF cosine of each text with any training clause (1.0 = a copy of a training clause)."""
    vec = make_vectorizer()
    X_tr = vec.fit_transform(train_texts).astype(np.float32)
    X = vec.transform(texts).astype(np.float32)
    best = np.empty(X.shape[0], dtype=np.float32)
    for s in range(0, X.shape[0], chunk):
        best[s:s + chunk] = (X[s:s + chunk] @ X_tr.T).max(axis=1).toarray().ravel()
    return best


def wording_group(sim):
    return pd.cut(sim, bins=[-1, 0.80, 0.95, 2], labels=["nouvelle (< 0,80)", "proche (0,80-0,95)", "quasi-copie (≥ 0,95)"],
                  right=False)


def build_frame(split, ds, names, models=MODELS):
    """One row per clause of `split`: text attributes + label + pred / confidence / ok of every model."""
    from transformers import AutoTokenizer

    df = add_metadata(ds[split].to_pandas())
    counts = pd.read_csv(TAB_DIR / "class_counts.csv")
    df["freq_group"] = df["label"].map(dict(zip(counts["label"], counts["freq_group"])))
    tok = AutoTokenizer.from_pretrained(BERT_TOKENIZER)
    df["n_tokens"] = [len(ids) for ids in tok(df["text"].tolist())["input_ids"]]
    df["truncated_256"] = np.where(df["n_tokens"] > 256, "tronquée (> 256)", "entière (≤ 256)")
    df["max_sim_train"] = similarity_to_train(ds["train"]["text"], df["text"].tolist())
    df["wording"] = wording_group(df["max_sim_train"])
    for m in models:
        p = load_predictions(m, split)
        assert (p["idx"].to_numpy() == df["idx"].to_numpy()).all(), f"{m}: predictions not aligned on idx"
        df[f"{m}_pred"], df[f"{m}_conf"] = p["pred"].to_numpy(), p["confidence"].to_numpy()
        df[f"{m}_ok"] = df[f"{m}_pred"] == df["label"]
    return df


def wilson(k, n, z=1.96):
    """95 % Wilson interval of a proportion k / n."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    centre, half = (p + z * z / (2 * n)) / (1 + z * z / n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


SUBGROUPS = ["freq_group", "length_group", "truncated_256", "wording", "jurisdiction_group", "party_type",
             "contract_type", "gender", "all_caps", "has_amount", "has_duration"]


def subgroup_table(df, models=MODELS, columns=SUBGROUPS):
    rows = []
    for col in columns:
        for value, g in df.groupby(col, observed=True):
            row = {"attribut": col, "groupe": str(value), "n": len(g), "part": len(g) / len(df), "assez_grand": len(g) >= MIN_GROUP}
            for m in models:
                k = int(g[f"{m}_ok"].sum())
                lo, hi = wilson(k, len(g))
                # macro-F1 over the types present in the group: a type the model wrongly predicts there does not
                # add a class with F1 = 0 (that would mix the group's quality with how many types it attracts)
                f1, _ = per_class_f1(g["label"], g[f"{m}_pred"], 100)
                row.update({f"{m}_accuracy": k / len(g), f"{m}_acc_ic_bas": lo, f"{m}_acc_ic_haut": hi,
                            f"{m}_macro_f1": float(f1[np.unique(g["label"])].mean())})
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- part 1: calibration and abstention

def expected_calibration_error(conf, ok, n_bins=15):
    """Mean |accuracy - confidence| over confidence bins, weighted by their size (0 = perfectly honest)."""
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(ok[mask].mean() - conf[mask].mean())
    return float(ece)


def choose_threshold(conf, ok, target=TARGET_ACCURACY):
    """Lowest confidence threshold such that the clauses at or above it reach `target` accuracy (on validation)."""
    order = np.argsort(-conf)
    acc_at_k = np.cumsum(ok[order]) / np.arange(1, len(ok) + 1)
    good = np.where(acc_at_k >= target)[0]
    return float(conf[order][good.max()]) if len(good) else 1.0


def calibration_tables(val, test, models=MODELS):
    calib, abst = [], []
    for m in models:
        cv, ov = val[f"{m}_conf"].to_numpy(), val[f"{m}_ok"].to_numpy()
        ct, ot = test[f"{m}_conf"].to_numpy(), test[f"{m}_ok"].to_numpy()
        calib.append({"modele": m, "ece_test": expected_calibration_error(ct, ot), "confiance_moyenne_test": ct.mean(),
                      "accuracy_test": ot.mean(), "confiance_moyenne_erreurs": ct[~ot].mean(),
                      "erreurs_avec_confiance_sup_0_9": float(np.mean(ct[~ot] >= 0.9))})
        thr = choose_threshold(cv, ov)
        auto = ct >= thr
        abst.append({"modele": m, "cible_accuracy": TARGET_ACCURACY, "seuil_choisi_sur_validation": thr,
                     "test_part_automatique": auto.mean(), "test_accuracy_automatique": ot[auto].mean(),
                     "test_part_envoyee_avocat": 1 - auto.mean(), "test_accuracy_clauses_envoyees": ot[~auto].mean()})
    return pd.DataFrame(calib), pd.DataFrame(abst)


# ---------------------------------------------------------------- part 2: counterfactual tests and OOD probes

def _case_like(src, word):
    return word.upper() if src.isupper() else word.capitalize() if src[0].isupper() else word


def swap_gender(text):
    """he/him/his/himself <-> she/her/hers/herself. 'her' becomes 'his' (possessive reading, the usual one in contracts)."""
    table = {"he": "she", "him": "her", "his": "her", "himself": "herself",
             "she": "he", "her": "his", "hers": "his", "herself": "himself"}
    return re.sub(r"\b(he|him|his|himself|she|her|hers|herself)\b",
                  lambda m: _case_like(m.group(0), table[m.group(0).lower()]), text, flags=re.IGNORECASE)


def swap_jurisdiction(text, target="England and Wales"):
    place = find_jurisdiction(text)
    if place is None:
        return text
    return re.sub(re.escape(place), lambda m: _case_like(m.group(0), target) if m.group(0).isupper() else target, text,
                  flags=re.IGNORECASE)


def swap_company_form(text):
    table = [(r"\bInc\.", "SAS"), (r"\bL\.L\.C\.|\bLLC\b", "GmbH"), (r"\bCorporation\b", "S.A."), (r"\bCorp\.", "S.A."),
             (r"\bLtd\.?", "SARL")]
    for pattern, repl in table:
        text = re.sub(pattern, repl, text)
    return text


def change_figures(text):
    """Every number in a duration or amount + 15 ('30 days' -> '45 days'), and $ -> €."""
    text = re.sub(r"\b(\d{1,3})(\s*\)?\s*(?:calendar\s+|business\s+)?(?:days?|months?|years?|weeks?))\b",
                  lambda m: f"{int(m.group(1)) + 15}{m.group(2)}", text, flags=re.IGNORECASE)
    return text.replace("$", "€")


PERTURBATIONS = {
    "genre (he ↔ she)": (swap_gender, lambda d: d["gender"] != "aucun"),
    "juridiction → England and Wales": (swap_jurisdiction, lambda d: d["jurisdiction_group"] != "aucune"),
    "juridiction → France": (lambda t: swap_jurisdiction(t, "France"), lambda d: d["jurisdiction_group"] != "aucune"),
    "forme de société (Inc. → SAS…)": (swap_company_form, lambda d: d["text"].str.contains(r"\bInc\.|\bLLC\b|\bCorporation\b|\bLtd")),
    "chiffres (+15, $ → €)": (change_figures, lambda d: d["has_duration"] | d["has_amount"]),
}


def counterfactual_report(predict_proba, df, model_name):
    """predict_proba(list_of_texts) -> (n, 100) array. Flip rate = share of clauses whose prediction changes."""
    rows = []
    for name, (perturb, applies) in PERTURBATIONS.items():
        sub = df[applies(df)]
        new_texts = [perturb(t) for t in sub["text"]]
        changed = np.array([a != b for a, b in zip(sub["text"], new_texts)])
        sub, new_texts = sub[changed], [t for t, c in zip(new_texts, changed) if c]
        if len(sub) == 0:
            continue
        before = predict_proba(sub["text"].tolist()).argmax(1)
        after = predict_proba(new_texts).argmax(1)
        y = sub["label"].to_numpy()
        rows.append({"modele": model_name, "perturbation": name, "n": len(sub), "taux_flip": float(np.mean(before != after)),
                     "accuracy_avant": float(np.mean(before == y)), "accuracy_apres": float(np.mean(after == y))})
    return pd.DataFrame(rows)


# Hand-written probes (illustrative, not a benchmark): expected type, or None when no LEDGAR type fits
OOD_PROBES = [
    ("anglais, style européen", "This Agreement shall be governed by and construed in accordance with the laws of England and Wales.", "Governing Laws"),
    ("anglais, style européen", "Any dispute arising out of or in connection with this Agreement shall be submitted to the exclusive jurisdiction of the courts of Paris, France.", "Submission To Jurisdiction"),
    ("anglais, style européen", "Each party shall comply with Regulation (EU) 2016/679 (the General Data Protection Regulation) when processing personal data under this Agreement.", "Compliance With Laws"),
    ("anglais, style européen", "The Recipient shall keep all Confidential Information strictly confidential and shall not disclose it to any third party without the prior written consent of the Discloser.", "Confidentiality"),
    ("français", "Le présent contrat est régi par le droit français et sera interprété conformément à celui-ci.", "Governing Laws"),
    ("français", "Chacune des parties peut résilier le présent contrat moyennant un préavis écrit de trente (30) jours.", "Terminations"),
    ("français", "Le destinataire s'engage à garder strictement confidentielles les informations transmises et à ne pas les divulguer à des tiers.", "Confidentiality"),
    ("type absent des 100", "For a period of two years after the termination of his employment, the Employee shall not engage in any business that competes with the Company.", None),
    ("type absent des 100", "The Supplier's total liability under this Agreement shall not exceed the fees paid by the Customer in the twelve months preceding the claim.", None),
]


def ood_report(predict_proba, names, threshold, model_name):
    proba = predict_proba([t for _, t, _ in OOD_PROBES])
    rows = []
    for (kind, text, expected), p in zip(OOD_PROBES, proba):
        pred, conf = names[int(p.argmax())], float(p.max())
        rows.append({"modele": model_name, "type_de_sonde": kind, "clause": text[:90] + ("…" if len(text) > 90 else ""),
                     "type_attendu": expected or "aucun des 100", "type_predit": pred, "confiance": round(conf, 3),
                     "correct": pred == expected, "envoyee_avocat": conf < threshold})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
STYLE = {"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
         "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
         "grid.linewidth": 0.6, "axes.axisbelow": True, "font.size": 10, "axes.titlesize": 12,
         "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left"}


def plot_subgroups(table, path, models=("baseline", "legalbert_full_es")):
    keep = [("wording", None), ("freq_group", None), ("truncated_256", None), ("jurisdiction_group", None)]
    rows = pd.concat([table[(table.attribut == a) & table.assez_grand] for a, _ in keep])
    labels = {"wording": "Formulation", "freq_group": "Fréquence du type", "truncated_256": "Longueur", "jurisdiction_group": "Juridiction"}
    rows = rows.iloc[::-1].reset_index(drop=True)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 0.42 * len(rows) + 1.6))
        y = np.arange(len(rows))
        for m, color, dy in zip(models, (BLUE, ORANGE), (-0.12, 0.12)):
            acc = 100 * rows[f"{m}_accuracy"]
            err = 100 * np.vstack([rows[f"{m}_accuracy"] - rows[f"{m}_acc_ic_bas"], rows[f"{m}_acc_ic_haut"] - rows[f"{m}_accuracy"]])
            ax.errorbar(acc, y + dy, xerr=err, fmt="o", ms=6, color=color, elinewidth=1, capsize=0, label=MODELS[m])
        ax.set_yticks(y, [f"{labels[a]} · {g} (n={n})" for a, g, n in zip(rows.attribut, rows.groupe, rows.n)])
        ax.set_xlabel("accuracy au test (%), intervalle à 95 %")
        nov = table[(table.attribut == "wording") & (table.groupe.str.startswith("nouvelle"))].iloc[0]
        cop = table[(table.attribut == "wording") & (table.groupe.str.startswith("quasi"))].iloc[0]
        fig.suptitle(f"Formulations nouvelles : −{100 * (cop.baseline_accuracy - nov.baseline_accuracy):.0f} points d'accuracy "
                     f"face aux formules types", x=0.02, ha="left", fontsize=12, fontweight="bold", color=INK)
        ax.legend(frameon=False, loc="lower left")
        ax.grid(axis="y", visible=False)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)


def plot_abstention(val, test, abst, path, models=("baseline", "legalbert_full_es")):
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 4))
        for m, color in zip(models, (BLUE, ORANGE)):
            conf, ok = test[f"{m}_conf"].to_numpy(), test[f"{m}_ok"].to_numpy()
            order = np.argsort(-conf)
            coverage = np.arange(1, len(ok) + 1) / len(ok)
            ax.plot(100 * coverage, 100 * np.cumsum(ok[order]) / np.arange(1, len(ok) + 1), color=color, lw=2, label=MODELS[m])
            a = abst[abst.modele == m].iloc[0]
            ax.plot(100 * a.test_part_automatique, 100 * a.test_accuracy_automatique, "o", ms=8, color=color)
        ax.axhline(100 * TARGET_ACCURACY, color=INK_2, lw=0.8, ls="--")
        ax.set_xlabel("part des clauses étiquetées sans relecture (%)")
        ax.set_ylabel("accuracy sur ces clauses (%)")
        ax.set_ylim(80, 100.5)
        b = abst[abst.modele == "legalbert_full_es"].iloc[0]
        ax.set_title(f"À {100 * TARGET_ACCURACY:.0f} % d'accuracy, Legal-BERT étiquette seul {100 * b.test_part_automatique:.0f} % des clauses")
        ax.legend(frameon=False, loc="lower left")
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)


# ---------------------------------------------------------------- main

def main():
    set_seed()
    ds, names = load_ledgar(fast_run=False)
    print("building validation and test frames…", flush=True)
    val, test = build_frame("validation", ds, names), build_frame("test", ds, names)

    sub = subgroup_table(test)
    sub.to_csv(TAB_DIR / "audit_subgroups.csv", index=False)
    calib, abst = calibration_tables(val, test)
    calib.to_csv(TAB_DIR / "audit_calibration.csv", index=False)
    abst.to_csv(TAB_DIR / "audit_abstention.csv", index=False)
    plot_subgroups(sub, FIG_DIR / "audit_subgroups.png")
    plot_abstention(val, test, abst, FIG_DIR / "audit_abstention.png")
    print(calib.round(3).to_string(index=False), "\n", abst.round(3).to_string(index=False), flush=True)

    print("refitting the baseline for the counterfactual and OOD tests (~4 min)…", flush=True)
    vec = make_vectorizer()
    X = vec.fit_transform(ds["train"]["text"])
    model = fit_model(X, ds["train"]["label"], C=30.0, class_weight="balanced")
    predict_proba = lambda texts: model.predict_proba(vec.transform(texts))
    cf = counterfactual_report(predict_proba, test, "baseline")
    cf.to_csv(TAB_DIR / "audit_counterfactuals.csv", index=False)
    thr = float(abst.loc[abst.modele == "baseline", "seuil_choisi_sur_validation"].iloc[0])
    ood = ood_report(predict_proba, names, thr, "baseline")
    ood.to_csv(TAB_DIR / "audit_ood.csv", index=False)
    print(cf.round(3).to_string(index=False), "\n", ood.to_string(index=False))


if __name__ == "__main__":
    main()
