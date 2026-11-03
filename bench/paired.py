"""Add paired differences for one more 'ours' arm to a finished result file.

Reads the saved per-example test log-odds (outside the repository) and writes
``paired["<ours> - <arm>"]`` for every other arm into bench/results/<name>.json,
with the same SE method as the run (block length from ``--block``).

Run: python -m bench.paired <name> <ours_arm> [--block 1]
"""

import argparse
import json
import re

import numpy as np

from bench.common.metrics import boot_se, paired, row_losses
from bench.common.runner import RAW, RESULTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("ours")
    ap.add_argument("--block", type=int, default=1)
    a = ap.parse_args()
    path = RESULTS / f"{a.name}.json"
    res = json.loads(path.read_text())
    z = np.load(RAW / f"{a.name}_test_logodds.npz")
    keys = {}
    for k in z.files:
        m = re.match(r"n(\d+)_r(\d+)_(.+)$", k)
        keys.setdefault(m.group(1), {}).setdefault(int(m.group(2)), {})[m.group(3)] = z[k]
    for n, reps in keys.items():
        block = res["by_n_train"][n]
        arms = [x for x in res["arms"] if all(x in r for r in reps.values())]
        for arm in arms:
            if arm == a.ours:
                continue
            out = {}
            for mi, metric in enumerate(("logloss", "brier")):
                ds, ses = [], []
                for rep, r in sorted(reps.items()):
                    y = r["y"].astype(float)
                    d = (row_losses(y, r[a.ours].astype(float))[mi]
                         - row_losses(y, r[arm].astype(float))[mi])
                    ds.append(d.mean())
                    ses.append(boot_se(d, block=a.block, seed=rep))
                out[metric] = paired(ds, ses, sum(len(r["y"]) for r in reps.values()))
            block["paired"][f"{a.ours} - {arm}"] = out
    path.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
