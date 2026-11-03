"""HIGGS (OpenML 23512): strong reference, residual features that re-express it.

Reference: gradient boosting on the 21 low-level features, fitted once on a
disjoint 60,000 rows. Residual model: ridge logistic on all 28 features (the 7
high-level features are functions of the low-level ones). Ten disjoint chunks
of the remaining rows give ten independent repeats; in each, the training sets
are nested prefixes and the test set is 1,500 rows.

Run: OMP_NUM_THREADS=2 python -m bench.higgs [--repeats 10]
"""

import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
from sklearn.preprocessing import StandardScaler

from bench.common.arms import Task
from bench.common.reference import hgb, local_reference, ref_logit
from bench.common.runner import run
from bench.datasets import fetch

N_TRAIN = (250, 500, 1000, 2000)
N_TEST = 1500
N_REF = 60000


def load():
    frame = fetch(23512).frame.dropna()
    y = frame["class"].astype(int).to_numpy()
    X = frame.drop(columns="class").astype(float).to_numpy()
    low = np.arange(21)
    return X, y, low


def draws(repeats, n_train=N_TRAIN, ref_all=False):
    """``ref_all`` builds the reference from all 28 features (a stronger one)."""
    X, y, low = load()
    if ref_all:
        low = np.arange(X.shape[1])
    order = np.random.default_rng(0).permutation(len(y))
    ref_idx, pool = order[:N_REF], order[N_REF:]
    ref = hgb(0).fit(X[ref_idx][:, low], y[ref_idx])
    r_all = ref_logit(ref, X[:, low])
    chunk = max(n_train) + N_TEST
    high = np.setdiff1d(np.arange(X.shape[1]), low)
    for rep in range(repeats):
        idx = pool[rep * chunk:(rep + 1) * chunk]
        te = idx[max(n_train):]
        for n in n_train:
            tr = idx[:n]
            sc = StandardScaler().fit(X[tr])
            Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
            lr = (lambda s, a=X[tr][:, low], b=y[tr], c=X[te][:, low]:
                  local_reference(a, b, c, s))
            yield n, rep, Task(Xtr, y[tr], r_all[tr], Xte, y[te], r_all[te], seed=rep,
                               keep_cols=high if high.size else None, local_reference=lr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--name", default="higgs")
    a = ap.parse_args()
    meta = {"source": "https://www.openml.org/d/23512", "n_train": list(N_TRAIN),
            "n_test_per_repeat": N_TEST, "reference": "HistGradientBoosting on 21 low-level "
            "features, 60,000 disjoint rows", "repeats": a.repeats,
            "se_method": "between-repeat SD/sqrt(R) over disjoint chunks (independent "
            "given the fixed reference); se_within = iid row bootstrap, 2000 resamples"}
    run(a.name, draws(a.repeats), meta, block=1)


if __name__ == "__main__":
    main()
