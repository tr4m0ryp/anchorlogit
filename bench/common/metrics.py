"""Per-row losses, summary metrics and paired differences with standard errors."""

import numpy as np
from scipy import stats
from scipy.special import expit

from anchorlogit import fit_offset_logistic

TAIL = 3.0


def row_losses(y, raw):
    """Natural-log loss and Brier score per row."""
    ll = np.logaddexp(0.0, raw) - y * raw
    return ll, (expit(raw) - y) ** 2


def ece(y, raw, bins=15):
    """Expected calibration error, equal-mass bins."""
    p = expit(raw)
    order = np.argsort(p, kind="stable")
    total = 0.0
    for idx in np.array_split(order, bins):
        if idx.size:
            total += idx.size * abs(p[idx].mean() - y[idx].mean())
    return total / y.size


def calibration_slope(y, raw):
    w, _, _ = fit_offset_logistic(raw[:, None], y, C=np.inf, solver="newton")
    return float(w[0])


def summary(y, raw, r):
    ll, br = row_losses(y, raw)
    tail = np.abs(r) >= TAIL
    q90 = np.abs(r) >= np.quantile(np.abs(r), 0.9)
    return {
        "logloss": float(ll.mean()), "brier": float(br.mean()),
        "ece15": float(ece(y, raw)), "cal_slope": calibration_slope(y, raw),
        "tail_logloss": float(ll[tail].mean()) if tail.any() else None,
        "tail_n": int(tail.sum()),
        "tail90_logloss": float(ll[q90].mean()),
    }


def boot_se(d, n_boot=2000, block=1, seed=0):
    """SE of mean(d): iid row bootstrap (block=1) or moving-block bootstrap."""
    rng = np.random.default_rng(seed)
    n = d.size
    if block <= 1:
        idx = rng.integers(0, n, size=(n_boot, n))
        return float(d[idx].mean(axis=1).std(ddof=1))
    starts = np.arange(n - block + 1)
    k = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        s = rng.choice(starts, size=k)
        means[b] = d[(s[:, None] + np.arange(block)).ravel()[:n]].mean()
    return float(means.std(ddof=1))


def paired(per_repeat_d, per_repeat_se, n_rows):
    """Aggregate per-repeat mean differences (independent repeats).

    se: between-repeat SD / sqrt(R) (includes training-draw variance; primary).
    se_within: pooled within-test-set bootstrap SE (test-row noise only).
    """
    d = np.asarray(per_repeat_d, float)
    R = d.size
    se = float(d.std(ddof=1) / np.sqrt(R)) if R > 1 else float("nan")
    se_w = float(np.sqrt(np.sum(np.square(per_repeat_se))) / R)
    t = float(d.mean() / se) if se > 0 else float("nan")
    p = float(2 * stats.t.sf(abs(t), R - 1)) if se > 0 else float("nan")
    h = float(stats.t.ppf(0.95, R - 1) * se) if R > 1 else float("nan")
    return {"mean": float(d.mean()), "se": se, "se_within": se_w, "t": t, "p_two_sided": p,
            "ci90": [float(d.mean() - h), float(d.mean() + h)], "repeats": R,
            "n_rows": int(n_rows), "wins": int((d < 0).sum())}
