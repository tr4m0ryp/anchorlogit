"""Cross-cell counts, recomputed SEs, equivalence checks, tests and patch size for numbers.tex."""
from __future__ import annotations

import re
import sys
from pathlib import Path

from numfmt import Out, f4, intc

MAIN_CELLS = {"higgs": ("250", "500", "1000", "2000"), "bank": ("250", "1000"),
              "elec2": ("250", "1000")}


def _cells(res, key):
    for name, ns in MAIN_CELLS.items():
        for n in ns:
            yield name, n, res[name]["by_n_train"][n]["paired"][key]["logloss"]


def counts(o: Out, res, results_dir):
    """How many of the 8 main cells each comparison wins / loses beyond 2 SE."""
    o.c("higgs.json + bank.json + elec2.json: by_n_train.*.paired[key].logloss over the 8 main"
        " (dataset, n) cells; Beat = mean < -2 se, Lose = mean > +2 se")
    o.m("nMainCells", str(sum(len(v) for v in MAIN_CELLS.values())))
    for key, T in (("a6_offset_kcv - a4_ref_feature_x100", "KcvFree"),
                   ("a7_anchored_cv - a4_ref_feature_x100", "AnchFree"),
                   ("a6_offset_kcv - a3_ref_feature", "KcvFeat"),
                   ("a6_offset_kcv - a5_offset_k1", "KcvKOne"),
                   ("a6_offset_kcv - a1_plain", "KcvPlain"),
                   ("a6_offset_kcv - a2_ref", "KcvRef")):
        cs = list(_cells(res, key))
        o.m(f"nBeat{T}", str(sum(p["mean"] < -2 * p["se"] for *_, p in cs)))
        o.m(f"nLose{T}", str(sum(p["mean"] > 2 * p["se"] for *_, p in cs)))
    n_pairs = sum(len(c["paired"]) for d in res.values() for c in d["by_n_train"].values())
    n_cells = sum(len(d["by_n_train"]) for d in res.values())
    o.c("all 7 result files: number of (file, n) cells and of paired comparisons stored")
    o.m("nAllCells", str(n_cells))
    o.m("nAllPairs", intc(n_pairs))
    o.c("bench/results/*.json: arms in the main runs (higgs.json arms) and one-SE variant arms")
    o.m("nMainArms", str(len(res["higgs"]["arms"])))
    o.m("nVariantArms", str(len(res["higgs_smalln_1se_controls"]["arms"])
                            - len(res["higgs"]["arms"])))
    o.m("nResultFiles", str(len(res)))
    repo = Path(results_dir).resolve().parents[1]
    sys.path.insert(0, str(repo))
    from bench.common.grid import C_GRID, KAPPA_GRID, STRENGTH_GRID
    o.c("bench/common/grid.py: C_GRID, KAPPA_GRID, STRENGTH_GRID (5-fold CV, bench/common/arms.py)")
    o.m("nCGridSize", str(len(C_GRID)))
    o.m("nCGridLo", f"$10^{{{int(round(__import__('math').log10(C_GRID[0])))}}}$")
    o.m("nCGridHi", f"{C_GRID[-1]:g}")
    o.m("nKappaGridSize", str(len(KAPPA_GRID)))
    o.m("nKappaGrid", ", ".join(f"{k:g}" for k in KAPPA_GRID))
    o.m("nStrengthGrid", ", ".join("$\\infty$" if s == float("inf") else f"{s:g}"
                                   for s in STRENGTH_GRID))
    o.m("nFolds", "5")


def recomputed(o: Out, rc, res, DATA, SIZE, tag_pair):
    worst = 0.0
    beat_boot = {"a6_offset_kcv - a4_ref_feature_x100": 0, "a7_anchored_cv - a4_ref_feature_x100": 0}
    for name, by_n in rc.items():
        D = DATA[name]
        for n, pairs in by_n.items():
            for key, p in pairs.items():
                T = tag_pair(key)
                o.c(f"recompute.json: {name}.{n}['{key}'] .se_boot2 / .se_rows (per-row recomputed)")
                o.m(f"n{D}{SIZE[n]}D{T}SeBoot", f4(p["se_boot2"]))
                o.m(f"n{D}{SIZE[n]}D{T}SeRows", f4(p["se_rows"]))
                js = res[name]["by_n_train"][n]["paired"].get(key)
                if js:
                    worst = max(worst, abs(js["logloss"]["se"] - p["se_rep"]),
                                abs(js["logloss"]["mean"] - p["mean"]))
                if key in beat_boot and n in MAIN_CELLS.get(name, ()):
                    beat_boot[key] += p["mean"] < -2 * p["se_boot2"]
    o.c("recompute.json vs bench JSON: max |difference| in mean and between-repeat SE (float32 rows)")
    o.m("nRecomputeMaxDiff", f"{worst:.1e}")
    o.c("recompute.json: main cells where ours - a4 < -2 SeBoot")
    o.m("nBeatKcvFreeBoot", str(beat_boot["a6_offset_kcv - a4_ref_feature_x100"]))
    o.m("nBeatAnchFreeBoot", str(beat_boot["a7_anchored_cv - a4_ref_feature_x100"]))


def _sci(x):
    m, e = f"{x:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def twins(o: Out, t):
    for k, T in (("statsmodels", "Sm"), ("glum", "Glum"), ("upstream_zero_offset", "Up"),
                 ("lemma2", "LemTwo")):
        o.c(f"twins.json: {k}_max (max abs difference over cases) and {k}_cases")
        o.m(f"nTwin{T}Cases", str(t[f"{k}_cases"]))
        for m, q in (("coef", "Coef"), ("eta", "Eta"), ("anchor_coef_gap", "Coef")):
            if m in t[f"{k}_max"]:
                o.m(f"nTwin{T}{q}", _sci(t[f"{k}_max"][m]))
    env = t["env"]
    o.c("twins.json: env (versions used for the equivalence check)")
    for k, T in (("sklearn", "Sklearn"), ("statsmodels", "Statsmodels"), ("glum", "Glum"),
                 ("python", "Python"), ("anchorlogit", "Anchorlogit")):
        o.m(f"nVer{T}", env[k])


def tests_and_patch(o: Out, patch: Path, logs: Path):
    lib_add = lib_del = test_add = 0
    cur = ""
    for line in patch.read_text().splitlines():
        if line.startswith("diff --git"):
            cur = line.split()[2]
        elif line.startswith("+") and not line.startswith("+++"):
            if "/tests/" in cur:
                test_add += 1
            else:
                lib_add += 1
        elif line.startswith("-") and not line.startswith("---"):
            lib_del += 0 if "/tests/" in cur else 1
    o.c(f"{patch.name}: added / removed lines outside tests, added test lines")
    o.m("nPatchLibAdd", str(lib_add))
    o.m("nPatchLibDel", str(lib_del))
    o.m("nPatchTestAdd", str(test_add))
    log = (logs / "check_patch.log").read_text()
    up = re.search(r"(\d+) passed, (\d+) deselected", log)
    o.c("paper/.harvest/check_patch.log: upstream offset tests passed on the patched scikit-learn")
    o.m("nUpstreamOffsetTests", up.group(1))
    rest = log[up.end():]
    marks = "".join(re.sub(r"\[.*", "", ln).strip() for ln in rest.splitlines()
                    if re.search(r"\[\s*\d+%\]", ln))
    o.c("paper/.harvest/check_patch.log: package tests on the patched scikit-learn (dots, s, F)")
    o.m("nPkgTestsPatchedPass", str(marks.count(".")))
    o.m("nPkgTestsPatchedSkip", str(marks.count("s")))
    o.m("nPkgTestsPatchedFail", str(marks.count("F")))
    st = re.search(r"(\d+) passed, (\d+) skipped", (logs / "pytest_stock.log").read_text())
    o.c("paper/.harvest/pytest_stock.log: package tests on stock scikit-learn (passed, skipped)")
    o.m("nPkgTestsStockPass", st.group(1))
    o.m("nPkgTestsStockSkip", st.group(2))
