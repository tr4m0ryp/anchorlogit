"""Run all arms over (n_train, repeat) draws and write bench/results/<name>.json."""

import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from .arms import ARMS, run_arms
from .metrics import boot_se, paired, row_losses, summary

RESULTS = Path(__file__).resolve().parents[1] / "results"
RAW = Path(os.environ.get("ANCHORLOGIT_BENCH_RAW",
                          "../_data/anchorlogit_bench"))
OURS = ("a6_offset_kcv", "a7_anchored_cv", "v_kappa_1se", "v_k1_1se", "v_free_1se")


def _aggregate(records, block, names):
    arms = [a for a in names if all(a in rec["raw"] for rec in records)]
    out = {"arms": {}, "paired": {}}
    for a in arms:
        ms = [summary(rec["y"], rec["raw"][a], rec["r"]) for rec in records]
        agg = {}
        for key in ms[0]:
            vals = [m[key] for m in ms if m[key] is not None]
            agg[key] = float(np.mean(vals)) if vals else None
        agg["tail_n"] = int(sum(m["tail_n"] for m in ms))
        out["arms"][a] = agg
    n_rows = sum(len(rec["y"]) for rec in records)
    for ours in [o for o in OURS if o in arms]:
        for a in arms:
            if a == ours:
                continue
            res = {}
            for mi, metric in enumerate(("logloss", "brier")):
                ds, ses = [], []
                for rec in records:
                    d = (row_losses(rec["y"], rec["raw"][ours])[mi]
                         - row_losses(rec["y"], rec["raw"][a])[mi])
                    ds.append(d.mean())
                    ses.append(boot_se(d, block=block, seed=rec["seed"]))
                res[metric] = paired(ds, ses, n_rows)
            out["paired"][f"{ours} - {a}"] = res
    return out


def _choices(records):
    keep = {}
    for a in ("a6_offset_kcv", "a7_anchored_cv", "a8_kcv_no_source_cols",
              "a9_local_crossfit", "a9_local_insample", "a1_plain", "a3_ref_feature",
              *[a for a in records[0]["info"] if a.startswith("v")]):
        vals = [rec["info"].get(a) for rec in records if a in rec["info"]]
        if vals:
            keep[a] = vals
    return keep


def run(name, draws, meta, block=1, extra=None, skip=()):
    """``draws`` yields (n_train, repeat, Task). Writes JSON, returns the dict.

    ``extra`` maps added arm names to ``(description, fn)``; ``fn(task, raw, info)``
    returns ``(test_log_odds, info_dict)``.
    """
    t0 = time.time()
    extra = extra or {}
    names = dict(ARMS, **{k: v[0] for k, v in extra.items()})
    by_n = {}
    for n, rep, task in draws:
        raw, info = run_arms(task, skip=skip)
        for arm, (_, fn) in extra.items():
            raw[arm], info[arm] = fn(task, raw, info)
        by_n.setdefault(n, []).append({"y": task.yte, "r": task.rte, "raw": raw,
                                       "info": info, "seed": rep})
        print(f"[{name}] n={n} rep={rep} {time.time() - t0:.0f}s "
              f"a1={summary(task.yte, raw['a1_plain'], task.rte)['logloss']:.4f} "
              f"a3={summary(task.yte, raw['a3_ref_feature'], task.rte)['logloss']:.4f} "
              f"a6={summary(task.yte, raw.get('a6_offset_kcv', task.rte), task.rte)['logloss']:.4f} "
              f"k={info.get('a6_offset_kcv', {}).get('kappa')}", flush=True)
    result = {"dataset": name, "meta": meta, "arms": names,
              "se_method": meta.get("se_method"), "by_n_train": {}}
    RAW.mkdir(parents=True, exist_ok=True)
    dump = {}
    for n, recs in sorted(by_n.items()):
        result["by_n_train"][str(n)] = _aggregate(recs, block, names)
        result["by_n_train"][str(n)]["choices"] = _choices(recs)
        for rec in recs:
            for a, v in rec["raw"].items():
                dump[f"n{n}_r{rec['seed']}_{a}"] = v.astype(np.float32)
            dump[f"n{n}_r{rec['seed']}_y"] = rec["y"].astype(np.int8)
    np.savez_compressed(RAW / f"{name}_test_logodds.npz", **dump)
    result["runtime_s"] = round(time.time() - t0, 1)
    result["env"] = {"python": platform.python_version(), "threads": 2}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(result, indent=1, default=float))
    return result
