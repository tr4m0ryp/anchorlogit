"""Bank Marketing (OpenML 1461): the expected null.

The reference is gradient boosting on every feature, fitted once on a disjoint
half; the residual features hold no information the reference lacks, so the
method must not hurt (kappa near 1, small deviation). Eight disjoint chunks of
the other half give eight repeats with nested training sets and 1,200 test rows.

Run: OMP_NUM_THREADS=2 python -m bench.bank [--repeats 8]
"""

import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from bench.common.arms import Task
from bench.common.reference import hgb, local_reference, ref_logit
from bench.common.runner import run
from bench.datasets import fetch

N_TRAIN = (250, 1000)
N_TEST = 1200


def load():
    f = fetch(1461).frame
    y = (f["Class"].astype(str) == "2").astype(int).to_numpy()
    feats = f.drop(columns="Class")
    cat = [c for c in feats if str(feats[c].dtype) == "category"]
    ordinal = feats.copy()
    for c in cat:
        ordinal[c] = ordinal[c].cat.codes
    onehot = pd.get_dummies(feats, columns=cat, dtype=float).astype(float)
    mask = np.array([c in cat for c in feats])
    return onehot.to_numpy(), ordinal.to_numpy(float), mask, y


def draws(repeats, n_train=N_TRAIN, ref_rows=None):
    """``ref_rows(Xord) -> bool mask`` restricts the reference's training half."""
    Xoh, Xord, mask, y = load()
    idx = np.arange(len(y))
    ref_idx, pool = train_test_split(idx, test_size=0.5, stratify=y, random_state=0)
    if ref_rows is not None:
        ref_idx = ref_idx[ref_rows(Xord[ref_idx])]
    ref = hgb(0, categorical_features=mask).fit(Xord[ref_idx], y[ref_idx])
    r_all = ref_logit(ref, Xord)
    pool = np.random.default_rng(0).permutation(pool)
    chunk = max(n_train) + N_TEST
    for rep in range(repeats):
        part = pool[rep * chunk:(rep + 1) * chunk]
        te = part[max(n_train):]
        for n in n_train:
            tr = part[:n]
            sc = StandardScaler().fit(Xoh[tr])
            Xtr, Xte = sc.transform(Xoh[tr]), sc.transform(Xoh[te])
            lr = (lambda s, a=Xord[tr], b=y[tr], c=Xord[te]:
                  local_reference(a, b, c, s, categorical_features=mask))
            yield n, rep, Task(Xtr, y[tr], r_all[tr], Xte, y[te], r_all[te], seed=rep,
                               local_reference=lr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--name", default="bank")
    a = ap.parse_args()
    meta = {"source": "https://www.openml.org/d/1461", "n_train": list(N_TRAIN),
            "n_test_per_repeat": N_TEST, "reference": "HistGradientBoosting on all features, "
            "disjoint stratified half", "repeats": a.repeats,
            "se_method": "between-repeat SD/sqrt(R) over disjoint chunks; se_within = iid row "
            "bootstrap, 2000 resamples"}
    run(a.name, draws(a.repeats), meta, block=1)


if __name__ == "__main__":
    main()
