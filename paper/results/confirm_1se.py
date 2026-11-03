"""Confirmation run of the kept one-SE rule on HIGGS draws the variant runs never used.

    OMP_NUM_THREADS=2 python paper/results/confirm_1se.py [--out FILE]

The variant files higgs_smalln_1se*.json use repeats 0..9 of bench.higgs.draws with
n_train = (100, 150, 250, 1000), i.e. pool chunks of 2,500 rows. The pool holds 15 such
chunks, so repeats 10..14 are disjoint from every row the one-SE rule was selected on.
Same arms as higgs_smalln_1se_controls (cheap arms + the three one-SE arms), n = 100 and
150 only. Writes paper/results/confirm_1se.json; touches nothing under bench/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from bench import higgs  # noqa: E402
from bench.common.arms import ARMS, run_arms  # noqa: E402
from bench.common.runner import _aggregate, _choices  # noqa: E402
from bench.variants.selection import ONE_SE  # noqa: E402

LAYOUT = (100, 150, 250, 1000)  # keeps the variant runs' chunk size (2,500 rows)
USED, TOTAL, KEEP_N = 10, 15, (100, 150)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "confirm_1se.json"))
    a = ap.parse_args()
    t0 = time.time()
    names = dict(ARMS, **{k: v[0] for k, v in ONE_SE.items()})
    by_n: dict[int, list] = {}
    for n, rep, task in higgs.draws(TOTAL, LAYOUT):
        if rep < USED or n not in KEEP_N:
            continue
        assert len(task.yte) == higgs.N_TEST, "pool too small for a fresh chunk"
        raw, info = run_arms(task, skip=("grid", "a8", "a9", "a10"))
        for arm, (_, fn) in ONE_SE.items():
            raw[arm], info[arm] = fn(task, raw, info)
        by_n.setdefault(n, []).append({"y": task.yte, "r": task.rte, "raw": raw,
                                       "info": info, "seed": rep})
        print(f"n={n} rep={rep} {time.time() - t0:.0f}s", flush=True)
    out = {"variant": "confirm_1se", "repeats": list(range(USED, TOTAL)),
           "note": "fresh HIGGS draws (repeats 10..14), disjoint from the rows the one-SE "
           "rule was selected on", "by_n_train": {}}
    for n, recs in sorted(by_n.items()):
        agg = _aggregate(recs, 1, names)
        agg["choices"] = _choices(recs)
        out["by_n_train"][str(n)] = agg
    out["runtime_s"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
