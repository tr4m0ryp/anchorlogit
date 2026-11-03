"""Electricity / Elec2 (OpenML 151): a drifting, time-ordered stream.

For each of 12 disjoint origins ``o`` (from row 19,000, after the
Victoria columns start to vary): the reference is gradient boosting fitted
on every row before ``o - 1000 - 48`` (one day of embargo before the longest
training window), so it ages; the residual model trains on the ``n`` rows just
before ``o`` and is tested on the next 336 rows (one week). Features: day of
week (one-hot), period, prices, demands, transfer.

Run: OMP_NUM_THREADS=2 python -m bench.elec2 [--repeats 12]
"""

import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from bench.common.arms import Task
from bench.common.reference import hgb, local_reference, ref_logit
from bench.common.runner import run
from bench.datasets import fetch

N_TRAIN = (250, 1000)
N_TEST = 336
EMBARGO = 48
FIRST, STEP = 19000, 1500  # vic* and transfer are constant before row 17,424


def load():
    f = fetch(151).frame
    y = (f["class"] == "UP").astype(int).to_numpy()
    num = f[["period", "nswprice", "nswdemand", "vicprice", "vicdemand", "transfer"]]
    day = pd.get_dummies(f["day"].astype(int), prefix="day", dtype=float)
    X = pd.concat([num.astype(float), day], axis=1).to_numpy()
    return X, y


def draws(repeats, n_train=N_TRAIN):
    X, y = load()
    nmax = max(n_train)
    for rep in range(repeats):
        o = FIRST + STEP * rep
        cut = o - nmax - EMBARGO
        ref = hgb(rep).fit(X[:cut], y[:cut])
        te = np.arange(o, o + N_TEST)
        for n in n_train:
            tr = np.arange(o - n, o)
            sc = StandardScaler().fit(X[tr])
            Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
            lr = (lambda s, a=X[tr], b=y[tr], c=X[te]: local_reference(a, b, c, s))
            yield n, rep, Task(Xtr, y[tr], ref_logit(ref, X[tr]), Xte, y[te],
                               ref_logit(ref, X[te]), seed=rep, local_reference=lr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=12)
    ap.add_argument("--name", default="elec2")
    a = ap.parse_args()
    meta = {"source": "https://www.openml.org/d/151", "n_train": list(N_TRAIN),
            "n_test_per_repeat": N_TEST, "origins": [FIRST + STEP * k for k in range(a.repeats)],
            "reference": "HistGradientBoosting on all rows before origin - 1048",
            "repeats": a.repeats,
            "se_method": "between-window SD/sqrt(R) over disjoint (train, test) windows; "
            "se_within = moving-block bootstrap of the test week, block 48 rows, 2000 resamples. "
            "Windows share an expanding reference history, so they are not strictly independent."}
    run(a.name, draws(a.repeats), meta, block=EMBARGO)


if __name__ == "__main__":
    main()
