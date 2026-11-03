"""The method's lemmas as exact numerical checks.

Lemma 1: a fixed offset kappa * z equals a coefficient on z fixed at kappa.
Lemma 2: the anchored penalty 0.5 * lam * (beta_z - kappa)^2 equals the offset
         kappa * z plus an ordinary ridge column z / sqrt(lam).
Lemma 3: the objective is convex, the solution unique, every start converges
         to it; the Newton line search does not count the offset twice.
"""

import numpy as np
import pytest
from scipy import optimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from anchorlogit import AnchoredLogisticRegression, OffsetLogisticRegression, fit_offset_logistic
from anchorlogit.core.loss import pointwise_loss


def anchored_data(n=400, p=4, seed=0):
    """Features, a reference log-odds column z (last), and labels."""
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(n, p))
    z = 1.5 * F[:, 0] + rng.normal(scale=1.0, size=n)
    y = (rng.random(n) < expit(0.6 * z + F @ np.linspace(0.5, -0.5, p) + 0.2)).astype(int)
    return F, z, y


def anchored_objective(theta, F, z, y, C, lam, kappa):
    w, beta, b = theta[:-2], theta[-2], theta[-1]
    raw = F @ w + beta * z + b
    pen = 0.5 * w @ w + 0.5 * lam * (beta - kappa) ** 2
    return C * float(pointwise_loss(y, raw).sum()) + pen


def fit_anchored(F, z, y, **kw):
    X = np.column_stack([F, z])
    params = dict(anchor=-1, anchor_scale="logit", clip=0.0, solver="newton", tol=1e-12)
    params.update(kw)
    return AnchoredLogisticRegression(**params).fit(X, y), X


@pytest.mark.parametrize("kappa", [0.0, 0.25, 1.0])
def test_lemma1_fixed_offset_is_a_fixed_coefficient(kappa):
    F, z, y = anchored_data()
    m, X = fit_anchored(F, z, y, kappa=kappa, anchor_strength=np.inf, C=0.3)
    assert m.anchor_coef_ == kappa
    off = OffsetLogisticRegression(C=0.3, solver="newton", tol=1e-12).fit(F, y, offset=kappa * z)
    np.testing.assert_allclose(m.coef_, off.coef_, atol=1e-12)
    np.testing.assert_allclose(m.decision_function(X), off.decision_function(F, offset=kappa * z),
                               atol=1e-12)
    # the same model written with z as a column whose coefficient is pinned at kappa
    def obj(t):
        return anchored_objective(np.r_[t[:-1], kappa, t[-1]], F, z, y, 0.3, 0.0, kappa)
    direct = optimize.minimize(obj, np.zeros(F.shape[1] + 1), method="BFGS",
                               options={"gtol": 1e-9}).x
    np.testing.assert_allclose(np.r_[m.coef_.ravel(), m.intercept_], direct, atol=1e-6)


@pytest.mark.parametrize("lam", [0.01, 1.0, 100.0])
@pytest.mark.parametrize("kappa", [0.0, 0.5, 1.0])
def test_lemma2_anchored_penalty_is_offset_plus_rescaled_column(lam, kappa):
    F, z, y = anchored_data(seed=1)
    m, X = fit_anchored(F, z, y, kappa=kappa, anchor_strength=lam, C=0.2)
    # (a) offset kappa * z plus the ordinary ridge column z / sqrt(lam)
    G = np.column_stack([F, z / np.sqrt(lam)])
    off = OffsetLogisticRegression(C=0.2, solver="newton", tol=1e-12).fit(G, y, offset=kappa * z)
    u = off.coef_.ravel()[-1]
    np.testing.assert_allclose(m.coef_.ravel(), off.coef_.ravel()[:-1], atol=1e-8)
    assert m.anchor_coef_ == pytest.approx(kappa + u / np.sqrt(lam), abs=1e-8)
    assert m.intercept_[0] == pytest.approx(off.intercept_[0], abs=1e-8)
    # (b) direct minimisation of the anchored objective
    direct = optimize.minimize(anchored_objective, np.zeros(F.shape[1] + 2),
                               args=(F, z, y, 0.2, lam, kappa), method="BFGS",
                               options={"gtol": 1e-9}).x
    np.testing.assert_allclose(np.r_[m.coef_.ravel(), m.anchor_coef_, m.intercept_],
                               direct, atol=1e-5)


def test_lemma2_limits_free_column_ridge_and_fixed_offset():
    F, z, y = anchored_data(seed=2)
    X = np.column_stack([F, z])
    # lam = 0: z is an ordinary unpenalised feature
    free, _ = fit_anchored(F, z, y, kappa=0.7, anchor_strength=0.0, C=0.1)
    pf = OffsetLogisticRegression(C=0.1, solver="newton", tol=1e-12,
                                  penalty_factor=np.r_[np.ones(F.shape[1]), 0.0]).fit(X, y)
    np.testing.assert_allclose(free.decision_function(X), pf.decision_function(X), atol=1e-8)
    near_free, _ = fit_anchored(F, z, y, kappa=0.7, anchor_strength=1e-9, C=0.1)
    assert near_free.anchor_coef_ == pytest.approx(free.anchor_coef_, abs=1e-6)
    # lam = 1, kappa = 0: ordinary ridge with z as a feature (scikit-learn)
    ridge, _ = fit_anchored(F, z, y, kappa=0.0, anchor_strength=1.0, C=0.1)
    sk = LogisticRegression(C=0.1, tol=1e-12, max_iter=10_000).fit(X, y)
    np.testing.assert_allclose(np.r_[ridge.coef_.ravel(), ridge.anchor_coef_], sk.coef_.ravel(),
                               atol=1e-6)
    # lam -> inf: the fixed offset
    fixed, _ = fit_anchored(F, z, y, kappa=0.4, anchor_strength=np.inf, C=0.1)
    stiff, _ = fit_anchored(F, z, y, kappa=0.4, anchor_strength=1e9, C=0.1)
    np.testing.assert_allclose(stiff.decision_function(X), fixed.decision_function(X), atol=1e-5)


@pytest.mark.parametrize("solver", ["lbfgs", "newton"])
def test_lemma3_unique_solution_from_any_start(solver):
    F, z, y = anchored_data(seed=3)
    sols = []
    for seed in range(4):
        start = np.random.default_rng(seed).normal(scale=3.0, size=F.shape[1] + 1)
        coef, b, _ = fit_offset_logistic(F, y, 0.5 * z, C=0.5, solver=solver, tol=1e-11,
                                         coef_init=start)
        sols.append(np.r_[coef, b])
    np.testing.assert_allclose(np.ptp(np.array(sols), axis=0), 0.0, atol=1e-6)


def test_lemma3_newton_line_search_with_a_large_offset():
    """A direction that carried the offset would overshoot and stall; ours converges fast."""
    F, z, y = anchored_data(seed=4)
    offset = 8.0 * z
    coef, b, n_iter = fit_offset_logistic(F, y, offset, C=1.0, solver="newton", tol=1e-11)
    ref, rb, _ = fit_offset_logistic(F, y, offset, C=1.0, solver="lbfgs", tol=1e-11)
    assert n_iter <= 30
    np.testing.assert_allclose(coef, ref, atol=1e-6)
    assert b == pytest.approx(rb, abs=1e-6)


def test_starts_at_the_prior():
    """kappa = 1, no intercept, C -> 0: the prediction is the reference probability."""
    F, z, y = anchored_data(seed=5)
    q = expit(z)
    X = np.column_stack([F, q])
    m = AnchoredLogisticRegression(anchor=-1, kappa=1.0, C=1e-10, fit_intercept=False,
                                   clip=1e-9).fit(X, y)
    np.testing.assert_allclose(m.predict_proba(X)[:, 1], q, atol=1e-6)
