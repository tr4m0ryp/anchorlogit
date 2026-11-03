"""One cross-validated grid over (kappa, anchor_strength, C) for the offset arms.

Every offset arm is a restriction of the same grid, so the arms share folds and
their CV losses are computed once:

- no reference:      kappa = 0, anchor_strength = inf
- classical offset:  kappa = 1, anchor_strength = inf
- fractional offset: kappa in KAPPA_GRID, anchor_strength = inf
- anchored:          every (kappa, anchor_strength) pair
"""

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from anchorlogit import fit_offset_logistic

C_GRID = 10.0 ** np.arange(-4.0, 1.01, 0.5)
KAPPA_GRID = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
STRENGTH_GRID = (np.inf, 10.0, 1.0, 0.1, 0.0)
TOL = 1e-7


def configs(kappas=KAPPA_GRID, strengths=STRENGTH_GRID):
    """(kappa, strength) pairs; strength 0 leaves kappa without effect, so one entry."""
    out = []
    for lam in strengths:
        for k in ((0.0,) if lam == 0 else kappas):
            out.append((float(k), float(lam)))
    return out


def _design(X, r, lam):
    if np.isinf(lam):
        return X, None
    return np.column_stack([X, r]), np.append(np.ones(X.shape[1]), lam)


def fit_one(X, y, r, kappa, lam, C, init=None):
    """Fit one configuration; returns (coef, intercept)."""
    D, pf = _design(X, r, lam)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        w, b, _ = fit_offset_logistic(D, y, offset=kappa * r, C=C, penalty_factor=pf,
                                      solver="newton", tol=TOL, max_iter=200,
                                      coef_init=init)
    return w, b


def raw_of(X, r, kappa, lam, w, b):
    D, _ = _design(X, r, lam)
    return D @ w + b + kappa * r


def half_logloss(y, raw):
    return np.logaddexp(0.0, raw) - y * raw


def cv_folds(X, y, r, folds, cfgs, c_grid=C_GRID):
    """Held-out log-loss per fold, shape (n_folds, len(cfgs), len(c_grid))."""
    out = np.zeros((len(folds), len(cfgs), len(c_grid)))
    for f, (tr, te) in enumerate(folds):
        for i, (k, lam) in enumerate(cfgs):
            init = None
            for j, C in enumerate(c_grid):
                w, b = fit_one(X[tr], y[tr], r[tr], k, lam, C, init)
                init = np.append(w, b)
                out[f, i, j] = half_logloss(y[te], raw_of(X[te], r[te], k, lam, w, b)).mean()
    return out


def cv_table(X, y, r, folds, cfgs, c_grid=C_GRID):
    """Mean held-out log-loss, shape (len(cfgs), len(c_grid)) (equal-size folds)."""
    return cv_folds(X, y, r, folds, cfgs, c_grid).mean(axis=0)


def select(table, cfgs, allowed, c_grid=C_GRID):
    """Best (kappa, strength, C) among the configurations ``allowed`` accepts.

    Ties within 1e-12 go to the larger C, then to the later config.
    """
    best, pick = np.inf, None
    for i, cfg in enumerate(cfgs):
        if not allowed(*cfg):
            continue
        for j, C in enumerate(c_grid):
            if table[i, j] <= best + 1e-12:
                best, pick = table[i, j], (cfg[0], cfg[1], float(C))
    return pick, float(best)


def fit_predict(Xtr, ytr, rtr, Xte, rte, pick):
    k, lam, C = pick
    w, b = fit_one(Xtr, ytr, rtr, k, lam, C)
    out = {"kappa": k, "strength": lam, "C": C}
    if not np.isinf(lam):
        out["anchor_coef"] = float(k + w[-1])
    return raw_of(Xte, rte, k, lam, w, b), out
