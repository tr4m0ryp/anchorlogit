"""Write results/numbers.tex: one \\newcommand per number the paper may cite.

    python paper/results/make_numbers.py [--results DIR] [--recompute FILE] [--twins FILE]
                                         [--patch FILE] [--logs DIR] [--out FILE]

--results    the benchmark result files (default <repo>/bench/results: the public evidence)
--recompute  recompute.json from recompute.py (per-row paired SEs, default this directory)
--twins      twins.json from twins.py (output equivalence, default this directory)
--patch      the upstream scikit-learn patch (default <repo>/patches/scikit-learn-offset.patch)
--logs       pytest logs of the patched and stock test runs (default paper/.harvest)

Every macro is preceded by a % comment naming its file and JSON path. Three SEs appear:
  ...Se      the benchmark's own reported SE: between-repeat SD / sqrt(R) over disjoint
             (train, test) draws; the benchmark calls a difference a win when |d| > 2 Se.
  ...SeW     the benchmark's se_within: pooled within-test-set bootstrap (test-row noise only).
  ...SeBoot  recomputed here from per-row outputs: two-stage bootstrap (repeats, then rows;
             Elec2 rows in day blocks of 48), 2,000 draws. ...SeRows = pooled row bootstrap.
Macro names: n<Data><Size><Quantity><Arm>. Data: Hg higgs, Bk bank, El elec2, Hs higgs_smalln_1se,
Hc higgs_smalln_1se_controls, Sr higgs_strongref, Bs bank_shiftref. Size: Hund 100, HundFifty 150,
TwoFifty 250, FiveHund 500, OneK 1000, TwoK 2000. Paired: n<Data><Size>D<Ours><Arm> = ours - arm.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from numfmt import Out, f4, sgn4, intc  # noqa: E402
from numbers_extra import counts, recomputed, tests_and_patch, twins  # noqa: E402

DATA = {"higgs": "Hg", "bank": "Bk", "elec2": "El", "higgs_smalln_1se": "Hs",
        "higgs_smalln_1se_controls": "Hc", "higgs_strongref": "Sr", "bank_shiftref": "Bs"}
SIZE = {"100": "Hund", "150": "HundFifty", "250": "TwoFifty", "500": "FiveHund", "1000": "OneK",
        "2000": "TwoK"}
ARM = {"a1_plain": "Plain", "a2_ref": "Ref", "a2_ref_platt": "Platt", "a3_ref_feature": "Feat",
       "a4_ref_feature_x100": "Free", "a5_offset_k1": "KOne", "a6_offset_kcv": "Kcv",
       "a7_anchored_cv": "Anch", "a8_kcv_no_source_cols": "NoSrc", "a9_local_crossfit": "LocCf",
       "a9_local_insample": "LocIn", "a10_lgbm_init": "Gbm", "v_kappa_1se": "Rule",
       "v_k1_1se": "KOneRule", "v_free_1se": "FreeRule"}
METRIC = {"logloss": "LL", "brier": "Brier", "ece15": "ECE", "cal_slope": "Slope",
          "tail90_logloss": "TailLL"}


def tag_pair(key):
    ours, arm = key.split(" - ")
    return ARM[ours] + ARM[arm]


def one_file(o: Out, path: Path, D: str):
    d = json.loads(path.read_text())
    fn = path.name
    meta = d["meta"]
    o.c(f"{fn}: meta.repeats, meta.n_test_per_repeat (if present), runtime_s")
    o.m(f"n{D}Repeats", str(meta["repeats"]))
    if "n_test_per_repeat" in meta:
        o.m(f"n{D}TestPerRep", intc(meta["n_test_per_repeat"]))
    if d.get("runtime_s"):
        o.m(f"n{D}RuntimeMin", str(round(d["runtime_s"] / 60)))
    for n, cell in d["by_n_train"].items():
        S = SIZE[n]
        for arm, ms in cell["arms"].items():
            for k, q in METRIC.items():
                if ms.get(k) is not None:
                    o.c(f"{fn}: by_n_train.{n}.arms.{arm}.{k}")
                    o.m(f"n{D}{S}{ARM[arm]}{q}", f4(ms[k]))
        for key, p in cell["paired"].items():
            T, ll, br = tag_pair(key), p["logloss"], p["brier"]
            o.c(f"{fn}: by_n_train.{n}.paired['{key}'].logloss .mean/.se/.se_within/.p_two_sided/"
                ".wins/.repeats; .brier.mean/.se")
            o.m(f"n{D}{S}D{T}", sgn4(ll["mean"]))
            o.m(f"n{D}{S}D{T}Se", f4(ll["se"]))
            o.m(f"n{D}{S}D{T}SeW", f4(ll["se_within"]))
            pv = ll["p_two_sided"]
            o.m(f"n{D}{S}D{T}P", "n/a" if pv != pv else f"{pv:.3f}")
            o.m(f"n{D}{S}D{T}Wins", f"{ll['wins']}/{ll['repeats']}")
            o.m(f"n{D}{S}D{T}Br", sgn4(br["mean"]))
            o.m(f"n{D}{S}D{T}BrSe", f4(br["se"]))
        if n in ("250", "1000") or D in ("Hg",):
            first = next(iter(cell["paired"].values()))["logloss"]
            o.c(f"{fn}: by_n_train.{n}.paired[*].logloss.n_rows (test rows pooled over repeats)")
            o.m(f"n{D}{S}TestRows", intc(first["n_rows"]))
        for arm, picks in cell["choices"].items():
            ks = [c["kappa"] for c in picks if "kappa" in c]
            if ks and arm in ("a6_offset_kcv", "v_kappa_1se", "a7_anchored_cv"):
                o.c(f"{fn}: by_n_train.{n}.choices.{arm}[*].kappa: median, count == 1, count")
                o.m(f"n{D}{S}{ARM[arm]}KappaMed", f"{statistics.median(ks):g}")
                o.m(f"n{D}{S}{ARM[arm]}KappaOnes", f"{sum(k == 1.0 for k in ks)}/{len(ks)}")
            if arm == "a7_anchored_cv":
                ls = [c["strength"] for c in picks]
                o.c(f"{fn}: by_n_train.{n}.choices.a7_anchored_cv[*].strength: count == inf")
                o.m(f"n{D}{S}AnchFixedCount",
                    f"{sum(x == float('inf') for x in ls)}/{len(ls)}")
            if arm == "v_free_1se":
                cs = [c["anchor_coef"] for c in picks]
                o.c(f"{fn}: by_n_train.{n}.choices.v_free_1se[*].anchor_coef: median")
                o.m(f"n{D}{S}FreeRuleCoefMed", f"{statistics.median(cs):.2f}")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(REPO / "bench" / "results"))
    ap.add_argument("--recompute", default=str(HERE / "recompute.json"))
    ap.add_argument("--twins", default=str(HERE / "twins.json"))
    ap.add_argument("--patch", default=str(REPO / "patches" / "scikit-learn-offset.patch"))
    ap.add_argument("--logs", default=str(HERE.parent / ".harvest"))
    ap.add_argument("--out", default=str(HERE / "numbers.tex"))
    a = ap.parse_args()
    o = Out()
    o.c("Generated by results/make_numbers.py; do not edit by hand.")
    o.c("Se = between-repeat SE (benchmark's own), SeW = within-test bootstrap (benchmark's own),")
    o.c("SeBoot / SeRows = recomputed here from per-row test log-odds (see recompute.py).")
    res = {}
    for name, D in DATA.items():
        res[name] = one_file(o, Path(a.results) / f"{name}.json", D)
    counts(o, res, a.results)
    recomputed(o, json.loads(Path(a.recompute).read_text()), res, DATA, SIZE, tag_pair)
    twins(o, json.loads(Path(a.twins).read_text()))
    tests_and_patch(o, Path(a.patch), Path(a.logs))
    Path(a.out).write_text(o.text())
    print(f"wrote {a.out}: {o.count} macros")


if __name__ == "__main__":
    main()
