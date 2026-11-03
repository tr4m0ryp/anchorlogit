"""Reference models: a histogram gradient-boosting classifier and its log-odds."""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

from anchorlogit import logit_of

CLIP = 1e-4


def hgb(seed, **kw):
    params = dict(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                  early_stopping=False, random_state=seed)
    params.update(kw)
    return HistGradientBoostingClassifier(**params)


def ref_logit(model, X):
    return logit_of(model.predict_proba(X)[:, 1], clip=CLIP)


def local_reference(Xtr, ytr, Xte, seed, folds=5, **kw):
    """(cross-fitted train, in-sample train, test) log-odds of a reference refit on Xtr."""
    small = dict(max_iter=100, min_samples_leaf=10)
    small.update(kw)
    full = hgb(seed, **small).fit(Xtr, ytr)
    r_in, r_te = ref_logit(full, Xtr), ref_logit(full, Xte)
    r_cf = np.empty(len(ytr))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(Xtr, ytr):
        r_cf[te] = ref_logit(hgb(seed, **small).fit(Xtr[tr], ytr[tr]), Xtr[te])
    return r_cf, r_in, r_te
