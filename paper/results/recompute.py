"""Paired differences recomputed from the saved per-row test log-odds.

    python paper/results/recompute.py [--raw DIR] [--out results/recompute.json]

--raw   the benchmark's per-row test log-odds (<name>_test_logodds.npz; keys n<N>_r<R>_<arm> and
        n<N>_r<R>_y), default $ANCHORLOGIT_BENCH_RAW or <repo>/../_data/anchorlogit_bench

For each (result file, n_train, ours - arm) this writes:
  mean        mean over repeats of the per-row log-loss difference (stored log-odds are float32,
              so it can differ from the JSON's float64 mean in the 5th decimal)
  se_rep      between-repeat SD / sqrt(R), recomputed (must match the JSON's "se")
  se_boot2    two-stage bootstrap: resample repeats, then rows inside each drawn repeat
              (Elec2: moving blocks of 48 rows = one day), 2,000 draws, seed 0
  se_rows     row bootstrap of the pooled rows, repeats ignored (test-row noise only)
There are no clock-hour blocks in these public data: the time-ordered set (Elec2) uses day blocks.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
B = 2000
BLOCK = {"elec2": 48}
MAIN = ["a1_plain", "a2_ref", "a2_ref_platt", "a3_ref_feature", "a4_ref_feature_x100",
        "a5_offset_k1", "a7_anchored_cv"]
PAIRS = {
    "higgs": [("a6_offset_kcv", a) for a in MAIN] + [("a7_anchored_cv", "a4_ref_feature_x100"),
                                                     ("a6_offset_kcv", "a8_kcv_no_source_cols"),
                                                     ("a6_offset_kcv", "a9_local_insample"),
                                                     ("a6_offset_kcv", "a10_lgbm_init")],
    "bank": [("a6_offset_kcv", a) for a in MAIN] + [("a7_anchored_cv", "a4_ref_feature_x100")],
    "elec2": [("a6_offset_kcv", a) for a in MAIN] + [("a7_anchored_cv", "a4_ref_feature_x100")],
    "higgs_smalln_1se": [("a6_offset_kcv", "a4_ref_feature_x100"),
                         ("a6_offset_kcv", "a5_offset_k1")],
    "higgs_smalln_1se_controls": [("v_kappa_1se", a) for a in
                                  ("a1_plain", "a2_ref", "a4_ref_feature_x100", "v_free_1se",
                                   "v_k1_1se")] + [("v_free_1se", "a4_ref_feature_x100")],
    "higgs_strongref": [("a6_offset_kcv", a) for a in ("a2_ref", "a4_ref_feature_x100",
                                                       "a5_offset_k1")]
    + [("v_kappa_1se", a) for a in ("a2_ref", "a4_ref_feature_x100", "v_free_1se")],
    "bank_shiftref": [("a6_offset_kcv", a) for a in ("a2_ref", "a2_ref_platt",
                                                     "a4_ref_feature_x100", "a5_offset_k1",
                                                     "v_free_1se")]
    + [("v_kappa_1se", "v_free_1se")],
}


def rowloss(y, raw):
    raw = raw.astype(np.float64)
    return np.logaddexp(0.0, raw) - y * raw


def boot_means(d, rng, block):
    n = d.size
    if block <= 1:
        return d[rng.integers(0, n, size=(B, n))].mean(axis=1)
    k = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(B, k))
    idx = (starts[:, :, None] + np.arange(block)).reshape(B, -1)[:, :n]
    return d[idx].mean(axis=1)


def load(path):
    z = np.load(path)
    reps = {}
    for k in z.files:
        m = re.match(r"n(\d+)_r(\d+)_(.+)$", k)
        reps.setdefault(m.group(1), {}).setdefault(int(m.group(2)), {})[m.group(3)] = z[k]
    return reps


def pair_stats(reps, ours, arm, block):
    rng = np.random.default_rng(0)
    ds = []
    for _, r in sorted(reps.items()):
        y = r["y"].astype(np.float64)
        ds.append(rowloss(y, r[ours]) - rowloss(y, r[arm]))
    R = len(ds)
    means = np.array([d.mean() for d in ds])
    inner = np.stack([boot_means(d, rng, block) for d in ds])          # (R, B)
    pick = rng.integers(0, R, size=(B, R))
    two = inner[pick, np.arange(B)[:, None]].mean(axis=1)
    pooled = np.concatenate(ds)
    rows = boot_means(pooled, rng, block)
    return {"mean": float(means.mean()), "se_rep": float(means.std(ddof=1) / np.sqrt(R)),
            "se_boot2": float(two.std(ddof=1)), "se_rows": float(rows.std(ddof=1)),
            "repeats": R, "n_rows": int(pooled.size), "wins": int((means < 0).sum())}


def main():
    ap = argparse.ArgumentParser()
    default = os.environ.get("ANCHORLOGIT_BENCH_RAW", str(REPO.parent / "_data" / "anchorlogit_bench"))
    ap.add_argument("--raw", default=default)
    ap.add_argument("--out", default=str(HERE / "recompute.json"))
    a = ap.parse_args()
    out = {}
    for name, pairs in PAIRS.items():
        reps_by_n = load(Path(a.raw) / f"{name}_test_logodds.npz")
        out[name] = {}
        for n, reps in reps_by_n.items():
            arms = set.intersection(*(set(r) for r in reps.values()))
            out[name][n] = {f"{o} - {b}": pair_stats(reps, o, b, BLOCK.get(name, 1))
                            for o, b in pairs if o in arms and b in arms}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
