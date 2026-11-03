"""A stronger prior: the one-standard-error rule toward the reference.

Among the configurations whose mean CV log-loss is within one fold-SE of the
best, take the largest kappa, then the smallest C (strongest ridge). At small
n the CV minimum is noisy; this rule trusts the reference unless the data
clearly say otherwise. The same C rule applied to the free column and to the
classical offset (kappa = 1) are the controls that separate the effect of the
offset from the effect of a stronger ridge.
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold

from ..common.grid import C_GRID, KAPPA_GRID, configs, cv_folds, fit_predict


def one_se(task, cfgs):
    folds = list(StratifiedKFold(5, shuffle=True, random_state=task.seed)
                 .split(task.ytr, task.ytr))
    per = cv_folds(task.Xtr, task.ytr, task.rtr, folds, cfgs)
    mean = per.mean(axis=0)
    se = per.std(axis=0, ddof=1) / np.sqrt(per.shape[0])
    i, j = np.unravel_index(np.argmin(mean), mean.shape)
    ok = mean <= mean[i, j] + se[i, j]
    best = None
    for a, (k, lam) in enumerate(cfgs):
        for b, C in enumerate(C_GRID):
            if ok[a, b] and (best is None or (k, -C) > best[0]):
                best = ((k, -C), (k, lam, float(C)), float(mean[a, b]))
    out, inf = fit_predict(task.Xtr, task.ytr, task.rtr, task.Xte, task.rte, best[1])
    inf["cv_logloss"] = best[2]
    return out, inf


def one_se_kappa(task, raw=None, info=None):
    return one_se(task, configs(KAPPA_GRID, (np.inf,)))


def one_se_k1(task, raw=None, info=None):
    return one_se(task, [(1.0, np.inf)])


def one_se_free(task, raw=None, info=None):
    return one_se(task, [(0.0, 0.0)])


ONE_SE = {
    "v_kappa_1se": ("offset, kappa by the one-SE rule toward kappa = 1, C toward the "
                    "strongest ridge", one_se_kappa),
    "v_k1_1se": ("control: offset kappa = 1, C by the same one-SE rule", one_se_k1),
    "v_free_1se": ("control: reference as an unpenalised column, C by the same one-SE "
                   "rule", one_se_free),
}
