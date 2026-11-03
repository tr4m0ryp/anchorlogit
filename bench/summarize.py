"""Print the key comparisons of every bench/results/*.json as a text table.

Run: python -m bench.summarize [name ...]
"""

import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
SHOW = ("a1_plain", "a2_ref", "a3_ref_feature", "a4_ref_feature_x100", "a5_offset_k1",
        "a6_offset_kcv", "a7_anchored_cv", "a8_kcv_no_source_cols", "a9_local_crossfit",
        "a9_local_insample", "a10_lgbm_init", "v_kappa_1se")


def show(path):
    d = json.loads(path.read_text())
    print(f"== {d['dataset']}  ({d.get('runtime_s')} s)")
    for n, v in d["by_n_train"].items():
        ll = {a: v["arms"][a]["logloss"] for a in SHOW if a in v["arms"]}
        print(f" n={n}: " + " ".join(f"{a.split('_')[0]}={x:.4f}" for a, x in ll.items()))
        for ours in ("a6_offset_kcv", "a7_anchored_cv"):
            parts = []
            for a in ("a1_plain", "a2_ref", "a3_ref_feature", "a4_ref_feature_x100",
                      "a5_offset_k1"):
                p = v["paired"].get(f"{ours} - {a}")
                if p:
                    q = p["logloss"]
                    parts.append(f"{a.split('_')[0]} {q['mean']:+.4f}({q['se']:.4f})")
            print(f"   {ours.split('_')[0]} - : " + "  ".join(parts))
        ks = [c["kappa"] for c in v["choices"].get("a6_offset_kcv", [])]
        print(f"   kappa_cv picks: {ks}")


if __name__ == "__main__":
    names = sys.argv[1:]
    paths = [RESULTS / f"{n}.json" for n in names] or sorted(RESULTS.glob("*.json"))
    for p in paths:
        show(p)
