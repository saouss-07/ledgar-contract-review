"""Loading LEDGAR and making every run reproducible.

Every script and notebook of the project gets its data from here, so that the three of us
train and evaluate on exactly the same splits.
"""
import os
import random

import numpy as np
import torch
from datasets import DatasetDict, load_dataset

SEED = 0
FAST_RUN = os.environ.get("FAST_RUN", "0") == "1"   # True: small subsample, to check that a script runs

# FAST_RUN sizes per split (the official splits are 60 000 / 10 000 / 10 000)
FAST_SIZES = {"train": 2000, "validation": 500, "test": 500}


def set_seed(seed=SEED):
    """Fix every source of randomness we use: Python, NumPy, PyTorch (CPU and GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_ledgar(fast_run=FAST_RUN, seed=SEED):
    """Return (dataset, label_names).

    dataset is a DatasetDict with the official LexGLUE splits "train", "validation" and "test".
    Each row is {"idx": int, "text": str, "label": int}. "idx" is the position of the clause in its
    official split: it stays the same in FAST_RUN, so predictions and metadata can always be joined on it.
    The test split is for the final evaluation only: every choice is made on "validation".
    """
    raw = load_dataset("coastalcph/lex_glue", "ledgar")
    label_names = raw["train"].features["label"].names
    ds = DatasetDict({
        split: raw[split].map(lambda row, i: {"idx": i}, with_indices=True)
        for split in ("train", "validation", "test")
    })
    if fast_run:
        ds = DatasetDict({
            split: ds[split].shuffle(seed=seed).select(range(n))
            for split, n in FAST_SIZES.items()
        })
    return ds, label_names
