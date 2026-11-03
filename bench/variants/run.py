"""Variants that test what would have to be true for the offset to win.

- higgs_smalln_1se: smaller training sets (100, 150, 250, 1000) and the
  one-SE kappa rule (a stronger prior toward the reference).
- higgs_strongref: the reference sees all 28 features, so the residual
  features only re-express it (strong reference, noisy re-expressions).
- bank_shiftref: the reference is trained only on clients under 40 and is
  wrong for the older ones, a subgroup the features (age) can detect.

Run: OMP_NUM_THREADS=2 python -m bench.variants.run <variant> [--repeats R]
"""

import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

from bench import bank, higgs
from bench.common.runner import run
from bench.variants.selection import ONE_SE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=["higgs_smalln_1se", "higgs_smalln_1se_controls",
                                        "higgs_strongref", "bank_shiftref"])
    ap.add_argument("--repeats", type=int, default=10)
    a = ap.parse_args()
    meta = {"variant": a.variant, "repeats": a.repeats,
            "se_method": "between-repeat SD/sqrt(R) over disjoint chunks; se_within = iid "
            "row bootstrap, 2000 resamples"}
    if a.variant == "higgs_smalln_1se":
        n = (100, 150, 250, 1000)
        meta.update(source="https://www.openml.org/d/23512", n_train=list(n))
        run(a.variant, higgs.draws(a.repeats, n), meta, extra=ONE_SE, skip=("a9", "a10"))
    elif a.variant == "higgs_smalln_1se_controls":
        n = (100, 150, 250, 1000)
        meta.update(source="https://www.openml.org/d/23512", n_train=list(n),
                    note="same draws as higgs_smalln_1se; only the cheap arms and the "
                    "one-SE arms")
        run(a.variant, higgs.draws(a.repeats, n), meta, extra=ONE_SE,
            skip=("grid", "a8", "a9", "a10"))
    elif a.variant == "higgs_strongref":
        n = (100, 250, 1000)
        meta.update(source="https://www.openml.org/d/23512", n_train=list(n),
                    reference="HistGradientBoosting on all 28 features, 60,000 disjoint rows")
        run(a.variant, higgs.draws(a.repeats, n, ref_all=True), meta, extra=ONE_SE,
            skip=("a9",))
    else:
        n = (250, 1000)
        meta.update(source="https://www.openml.org/d/1461", n_train=list(n),
                    reference="HistGradientBoosting on clients under 40 of the disjoint half")
        rows = lambda Xord: Xord[:, 0] < 40  # column V1 = age
        run(a.variant, bank.draws(a.repeats, n, ref_rows=rows), meta, extra=ONE_SE,
            skip=("a9",))


if __name__ == "__main__":
    main()
