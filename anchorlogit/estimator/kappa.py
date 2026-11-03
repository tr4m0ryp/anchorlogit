"""Choosing the trusted fraction ``kappa`` of the reference log-odds by CV."""

import numpy as np
from sklearn.model_selection import StratifiedKFold, check_cv

from ..core.loss import pointwise_loss
from ..core.solver import fit_offset_logistic

DEFAULT_KAPPA_GRID = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)


def _splitter(cv, random_state):
    if cv is None:
        cv = 5
    if isinstance(cv, (int, np.integer)):
        return StratifiedKFold(int(cv), shuffle=True, random_state=random_state)
    return check_cv(cv, classifier=True)


def search_kappa(design, y, r, extra, sample_weight, fit_kwargs, grid, cv,
                 random_state, groups=None):
    """Mean held-out log-loss for each ``kappa`` in ``grid``.

    Every fold refits the same penalised model with offset
    ``kappa * r + extra`` on the training part and scores the held-out part
    with the same offset. Returns ``(best_kappa, losses)`` where ``losses`` has
    one mean log-loss per grid value; ties go to the larger kappa (more trust
    in the reference when the data cannot tell).
    """
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError("kappa_grid must be a non-empty 1-d sequence")
    sw = np.ones(y.shape[0]) if sample_weight is None else np.asarray(sample_weight, float)
    folds = list(_splitter(cv, random_state).split(design, y, groups))
    losses = np.zeros(grid.size)
    for k, kappa in enumerate(grid):
        offset = kappa * r + extra
        total, weight = 0.0, 0.0
        for train, test in folds:
            coef, b, _ = fit_offset_logistic(
                design[train], y[train], offset=offset[train],
                sample_weight=sw[train], **fit_kwargs)
            raw = design[test] @ coef + b + offset[test]
            total += float(sw[test] @ pointwise_loss(y[test], raw))
            weight += float(sw[test].sum())
        losses[k] = total / weight
    best = np.flatnonzero(losses <= losses.min() + 1e-12)[-1]
    return float(grid[best]), losses
