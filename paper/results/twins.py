"""Output-equivalence check: anchorlogit's offset fit against reference implementations.

    python paper/results/twins.py [--out results/twins.json]

Arms compared on synthetic binary data with a known offset (anchorlogit's tests/conftest.make_data
generator, re-implemented here so the paper does not import the test suite):
  statsmodels GLM(Binomial, offset=)      unpenalised, both anchorlogit solvers, weighted and not
  glum GeneralizedLinearRegressor(offset=) ridge at three C values (alpha = 1 / (C * sum w))
  pristine scikit-learn LogisticRegression offset = 0 (anchorlogit must equal upstream)
  Lemma 2 identity: anchored penalty 0.5*lam*(b_z - kappa)^2 == offset kappa*z + column z/sqrt(lam)
Writes the max absolute difference in coefficients and in the raw score x@w+b+offset per check.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
from scipy.special import expit

HERE = Path(__file__).resolve().parent


def make_data(n, p, seed, offset_scale=1.0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    w = np.linspace(1.0, -1.0, p)
    offset = offset_scale * rng.normal(size=n)
    y = (rng.random(n) < expit(X @ w + 0.3 + offset)).astype(int)
    return X, y, offset


def vs_statsmodels(fit, sizes):
    import statsmodels.api as sm
    out = []
    for n, p, seed in sizes:
        X, y, off = make_data(n, p, seed, 1.5)
        for weighted in (False, True):
            sw = np.random.default_rng(seed + 1).integers(1, 4, size=n) if weighted else None
            kw = {} if sw is None else {"freq_weights": sw}
            res = sm.GLM(y, sm.add_constant(X), family=sm.families.Binomial(), offset=off,
                         **kw).fit(tol=1e-12)
            for solver in ("lbfgs", "newton"):
                w, b, _ = fit(X, y, off, sample_weight=sw, C=np.inf, solver=solver, tol=1e-11)
                eta = X @ w + b + off
                eta_sm = res.predict(sm.add_constant(X), offset=off, which="linear")
                out.append({"n": n, "p": p, "weighted": weighted, "solver": solver,
                            "coef": float(np.max(np.abs(np.r_[b, w] - res.params))),
                            "eta": float(np.max(np.abs(eta - eta_sm)))})
    return out


def vs_glum(fit, sizes):
    import glum
    out = []
    for n, p, seed in sizes:
        X, y, off = make_data(n, p, seed, 1.5)
        for C in (0.02, 0.5, 5.0):
            g = glum.GeneralizedLinearRegressor(family="binomial", alpha=1.0 / (C * n),
                                                l1_ratio=0.0, gradient_tol=1e-12, max_iter=1000)
            g.fit(X, y, offset=off)
            pg = g.predict(X, offset=off)
            for solver in ("lbfgs", "newton"):
                w, b, _ = fit(X, y, off, C=C, solver=solver, tol=1e-11)
                out.append({"n": n, "p": p, "C": C, "solver": solver,
                            "coef": float(np.max(np.abs(np.r_[b - g.intercept_, w - g.coef_]))),
                            "eta": float(np.max(np.abs(X @ w + b + off - np.log(pg / (1 - pg)))))})
    return out


def vs_upstream(fit, sizes):
    from sklearn.linear_model import LogisticRegression
    out = []
    for n, p, seed in sizes:
        X, y, _ = make_data(n, p, seed)
        for C in (0.02, 0.5, 5.0):
            m = LogisticRegression(C=C, solver="lbfgs", tol=1e-12, max_iter=10000).fit(X, y)
            w, b, _ = fit(X, y, np.zeros(n), C=C, solver="newton", tol=1e-11)
            out.append({"n": n, "p": p, "C": C,
                        "coef": float(np.max(np.abs(np.r_[b - m.intercept_[0], w - m.coef_[0]])))})
    return out


def lemma2(fit, sizes):
    out = []
    for n, p, seed in sizes:
        X, y, z = make_data(n, p, seed, 1.5)
        for kappa in (0.25, 1.0):
            for lam in (0.1, 1.0, 10.0):
                D = np.column_stack([X, z])
                pf = np.append(np.ones(p), lam)
                w1, b1, _ = fit(D, y, kappa * z, C=0.5, penalty_factor=pf, solver="newton",
                                tol=1e-11)
                w2, b2, _ = fit(np.column_stack([X, z / np.sqrt(lam)]), y, kappa * z, C=0.5,
                                solver="newton", tol=1e-11)
                eta1 = D @ w1 + b1 + kappa * z
                eta2 = X @ w2[:-1] + w2[-1] * z / np.sqrt(lam) + b2 + kappa * z
                out.append({"n": n, "kappa": kappa, "lam": lam,
                            "anchor_coef_gap": float(abs((kappa + w1[-1])
                                                         - (kappa + w2[-1] / np.sqrt(lam)))),
                            "eta": float(np.max(np.abs(eta1 - eta2)))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "twins.json"))
    a = ap.parse_args()
    import anchorlogit
    import sklearn
    import statsmodels
    import glum
    from anchorlogit import fit_offset_logistic as fit
    sizes = [(500, 4, 21), (2000, 10, 7), (5000, 20, 3)]
    t0 = time.time()
    res = {"statsmodels": vs_statsmodels(fit, sizes), "glum": vs_glum(fit, sizes),
           "upstream_zero_offset": vs_upstream(fit, sizes), "lemma2": lemma2(fit, sizes)}
    for k in list(res):
        res[k + "_max"] = {m: max(r[m] for r in res[k]) for m in res[k][0]
                           if m in ("coef", "eta", "anchor_coef_gap")}
        res[k + "_cases"] = len(res[k])
    res["env"] = {"python": platform.python_version(), "anchorlogit": anchorlogit.__version__,
                  "sklearn": sklearn.__version__, "statsmodels": statsmodels.__version__,
                  "glum": glum.__version__, "numpy": np.__version__, "seconds": time.time() - t0}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k.endswith(("_max", "_cases", "env"))},
                     indent=1))


if __name__ == "__main__":
    main()
