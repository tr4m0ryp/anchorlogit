"""Every arm of the evidence plan on one (train, test) draw.

``Task`` carries standardised features, the reference log-odds of train and
test rows (out of sample for the training rows), the labels, the columns that
do not feed the reference (ablation 8) and an optional ``local_reference``
callable that refits the reference on the training rows (ablation 9).
Each arm returns test log-odds; the caller scores them.
"""

import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold

from anchorlogit import fit_offset_logistic

from .grid import C_GRID, KAPPA_GRID, configs, cv_table, fit_predict, select

ARMS = {
    "a1_plain": "pristine LogisticRegressionCV, no reference",
    "a2_ref": "reference alone",
    "a2_ref_platt": "reference recalibrated (Platt on its log-odds)",
    "a3_ref_feature": "pristine LogisticRegressionCV, reference log-odds as a standardised feature",
    "a4_ref_feature_x100": "same, reference column x100 (nearly free coefficient)",
    "a5_offset_k1": "offset, kappa = 1 (classical GLM offset), C by CV",
    "a6_offset_kcv": "offset, (kappa, C) by CV (ours)",
    "a7_anchored_cv": "anchored, (kappa, anchor_strength, C) by CV (ours, general)",
    "a8_kcv_no_source_cols": "ablation: a6 without the columns the reference is built from",
    "a9_local_crossfit": "ablation: reference refit on the training rows, cross-fitted, a6 grid",
    "a9_local_insample": "ablation: same reference, in-sample on the training rows (leaks)",
    "a10_lgbm_init": "LightGBM started at kappa_cv x reference (init_score)",
}


@dataclass
class Task:
    Xtr: np.ndarray
    ytr: np.ndarray
    rtr: np.ndarray
    Xte: np.ndarray
    yte: np.ndarray
    rte: np.ndarray
    seed: int
    keep_cols: Optional[np.ndarray] = None
    local_reference: Optional[Callable] = None
    info: dict = field(default_factory=dict)


def _folds(y, seed):
    return list(StratifiedKFold(5, shuffle=True, random_state=seed).split(y, y))


def _pristine(X, y, Xte, folds):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", FutureWarning)
        m = LogisticRegressionCV(Cs=C_GRID, cv=folds, scoring="neg_log_loss",
                                 max_iter=5000, tol=1e-6).fit(X, y)
    return m.decision_function(Xte), {"C": float(m.C_[0])}


def _offset_arms(t, folds, out, info):
    cfgs = configs()
    table = cv_table(t.Xtr, t.ytr, t.rtr, folds, cfgs)
    rules = {
        "a5_offset_k1": lambda k, lam: np.isinf(lam) and k == 1.0,
        "a6_offset_kcv": lambda k, lam: np.isinf(lam),
        "a7_anchored_cv": lambda k, lam: True,
    }
    for arm, rule in rules.items():
        pick, cvl = select(table, cfgs, rule)
        out[arm], info[arm] = fit_predict(t.Xtr, t.ytr, t.rtr, t.Xte, t.rte, pick)
        info[arm]["cv_logloss"] = cvl


def _kappa_cv(Xtr, ytr, rtr, Xte, rte, folds):
    cfgs = configs(KAPPA_GRID, (np.inf,))
    pick, cvl = select(cv_table(Xtr, ytr, rtr, folds, cfgs), cfgs, lambda k, lam: True)
    raw, inf = fit_predict(Xtr, ytr, rtr, Xte, rte, pick)
    inf["cv_logloss"] = cvl
    return raw, inf


def run_arms(t: Task, skip=()):
    folds = _folds(t.ytr, t.seed)
    out, info = {}, {}
    out["a1_plain"], info["a1_plain"] = _pristine(t.Xtr, t.ytr, t.Xte, folds)
    out["a2_ref"] = t.rte.copy()
    w, b, _ = fit_offset_logistic(t.rtr[:, None], t.ytr, C=np.inf, solver="newton")
    out["a2_ref_platt"], info["a2_ref_platt"] = t.rte * w[0] + b, {"a": float(w[0]), "b": b}
    mu, sd = t.rtr.mean(), t.rtr.std() or 1.0
    ztr, zte = (t.rtr - mu) / sd, (t.rte - mu) / sd
    for arm, scale in (("a3_ref_feature", 1.0), ("a4_ref_feature_x100", 100.0)):
        out[arm], info[arm] = _pristine(np.column_stack([t.Xtr, scale * ztr]), t.ytr,
                                        np.column_stack([t.Xte, scale * zte]), folds)
    if "grid" not in skip:
        _offset_arms(t, folds, out, info)
    if t.keep_cols is not None and "a8" not in skip:
        k = t.keep_cols
        out["a8_kcv_no_source_cols"], info["a8_kcv_no_source_cols"] = _kappa_cv(
            t.Xtr[:, k], t.ytr, t.rtr, t.Xte[:, k], t.rte, folds)
    if t.local_reference is not None and "a9" not in skip:
        r_cf, r_in, r_te = t.local_reference(t.seed)
        out["a9_local_crossfit"], info["a9_local_crossfit"] = _kappa_cv(
            t.Xtr, t.ytr, r_cf, t.Xte, r_te, folds)
        out["a9_local_insample"], info["a9_local_insample"] = _kappa_cv(
            t.Xtr, t.ytr, r_in, t.Xte, r_te, folds)
    if "a10" not in skip and "a6_offset_kcv" in info:
        kap = info["a6_offset_kcv"]["kappa"]
        gbm = LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=7,
                             min_child_samples=20, reg_lambda=1.0, n_jobs=2,
                             random_state=t.seed, verbose=-1)
        gbm.fit(t.Xtr, t.ytr, init_score=kap * t.rtr)
        out["a10_lgbm_init"] = gbm.predict(t.Xte, raw_score=True) + kap * t.rte
        info["a10_lgbm_init"] = {"kappa": kap}
    return out, info
