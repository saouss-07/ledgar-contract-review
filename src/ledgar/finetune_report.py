"""Volet A (milestone m4): results tables of the fine-tuned Legal-BERT runs against the common baseline.

The runs themselves are trained on a Colab GPU by notebooks/partie_A_notebook.ipynb, which writes
results/predictions/<model>_{validation,test}.csv (common format, see ledgar.metrics) and
results/tables/finetune_metrics_<model>.json (parameters, training time, GPU memory, scores).
This module only reads those files, so it runs on any CPU in a few seconds:

    python -m ledgar.finetune_report

Outputs in results/tables/:
- finetune_tradeoff.csv       one row per model: trainable parameters, time, GPU memory, scores
- finetune_paired_tests.csv   McNemar (accuracy) + paired bootstrap (macro-F1) on the same test set
- finetune_f1_by_freq_group.csv  mean per-class F1 by frequency group (groups of class_counts.csv)
- finetune_f1_per_class.csv   per-class F1 of every model, with the training support
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ledgar.metrics import accuracy, contingency, load_predictions, macro_f1, mcnemar, paired_bootstrap, per_class_f1

TAB_DIR = Path(__file__).resolve().parents[2] / "results" / "tables"
N_CLASSES = 100

# (file name in results/predictions, label shown in the tables)
MODELS = [
    ("baseline", "TF-IDF + régression logistique (baseline commune, m1)"),
    ("legalbert_full_3ep", "Legal-BERT, fine-tuning complet, 3 epochs (pré-enregistré)"),
    ("legalbert_lora_r4", "Legal-BERT + LoRA r=4, 3 epochs"),
    ("legalbert_lora_r8", "Legal-BERT + LoRA r=8, 3 epochs"),
    ("legalbert_lora_r16", "Legal-BERT + LoRA r=16, 3 epochs"),
    ("legalbert_full_es", "Legal-BERT, fine-tuning complet, arrêt anticipé (exploratoire)"),
]

PAIRS = [
    ("legalbert_full_3ep", "baseline"),
    ("legalbert_full_es", "baseline"),
    ("legalbert_full_es", "legalbert_full_3ep"),
    ("legalbert_full_3ep", "legalbert_lora_r8"),
    ("legalbert_lora_r8", "baseline"),
    ("legalbert_lora_r16", "legalbert_lora_r4"),
]


def tradeoff_table(preds, y_test):
    base = pd.read_csv(TAB_DIR / "baseline_results.csv").set_index("split")
    rows = []
    for name, label in MODELS:
        p = preds[name]
        row = {"modele": name, "description": label,
               "test_accuracy": accuracy(y_test, p), "test_macro_f1": macro_f1(y_test, p, N_CLASSES)}
        if name == "baseline":
            row.update({"val_macro_f1": base.loc["validation", "macro_f1"],
                        "test_macro_f1_sans_doublons": base.loc["test_sans_doublons", "macro_f1"]})
        else:
            m = json.load(open(TAB_DIR / f"finetune_metrics_{name}.json"))
            row.update({"params_entraines": m["trainable_params"], "pct_entraines": m["trainable_pct"],
                        "epochs_max": m["epochs"], "temps_min": m["train_time_min"],
                        "memoire_gpu_go": m["peak_gpu_mem_gb"], "val_macro_f1": m["validation"]["macro_f1"],
                        "test_macro_f1_sans_doublons": m["test_dedup"]["macro_f1"]})
        rows.append(row)
    cols = ["modele", "description", "params_entraines", "pct_entraines", "epochs_max", "temps_min",
            "memoire_gpu_go", "val_macro_f1", "test_accuracy", "test_macro_f1", "test_macro_f1_sans_doublons"]
    return pd.DataFrame(rows)[cols]


def paired_tests(preds, y_test):
    rows = []
    for a, b in PAIRS:
        _, only_a, only_b, _ = contingency(y_test, preds[a], preds[b])
        chi2, p = mcnemar(only_a, only_b)
        diff, low, high = paired_bootstrap(y_test, preds[a], preds[b], N_CLASSES)
        rows.append({"modele_A": a, "modele_B": b, "seul_A_juste": only_a, "seul_B_juste": only_b,
                     "mcnemar_chi2": chi2, "mcnemar_p": p,
                     "delta_accuracy": accuracy(y_test, preds[a]) - accuracy(y_test, preds[b]),
                     "delta_macro_f1": diff, "ic95_bas": low, "ic95_haut": high,
                     "macro_f1_significatif": bool(low > 0 or high < 0)})
    return pd.DataFrame(rows)


def per_class(preds, y_test):
    counts = pd.read_csv(TAB_DIR / "class_counts.csv").sort_values("label").reset_index(drop=True)
    table = counts[["label", "label_name", "train", "test", "freq_group"]].copy()
    for name, _ in MODELS:
        table[f"f1_{name}"] = per_class_f1(y_test, preds[name], N_CLASSES)[0]
    return table


def main():
    preds, y_test = {}, None
    for name, _ in MODELS:
        df = load_predictions(name, "test").sort_values("idx")
        if y_test is None:
            y_test, idx = df["label"].to_numpy(), df["idx"].to_numpy()
        if not (np.array_equal(df["idx"].to_numpy(), idx) and np.array_equal(df["label"].to_numpy(), y_test)):
            raise ValueError(f"{name}: the test predictions are not aligned with the baseline")
        preds[name] = df["pred"].to_numpy()

    outputs = {
        "finetune_tradeoff.csv": tradeoff_table(preds, y_test),
        "finetune_paired_tests.csv": paired_tests(preds, y_test),
    }
    table = per_class(preds, y_test)
    f1_cols = [c for c in table.columns if c.startswith("f1_")]
    outputs["finetune_f1_per_class.csv"] = table
    outputs["finetune_f1_by_freq_group.csv"] = (
        table.groupby("freq_group")[f1_cols].mean().reindex(["fréquente", "moyenne", "rare"]).reset_index())

    for file, df in outputs.items():
        df.to_csv(TAB_DIR / file, index=False)
        print(f"\n{file}\n{df.round(4).to_string(index=False)}")


if __name__ == "__main__":
    main()
