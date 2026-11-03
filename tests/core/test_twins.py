"""Cross-implementation equality: statsmodels and glum fits with an offset."""

import numpy as np
import pytest

from anchorlogit import OffsetLogisticRegression, fit_offset_logistic
from tests.conftest import make_data

SOLVERS = ["lbfgs", "newton"]


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("weighted", [False, True])
def test_unpenalised_equals_statsmodels_glm_offset(solver, weighted):
    sm = pytest.importorskip("statsmodels.api")
    X, y, offset = make_data(n=500, p=4, seed=21, offset_scale=1.5)
    sw = np.random.default_rng(3).integers(1, 4, size=y.shape[0]) if weighted else None
    kw = {} if sw is None else {"freq_weights": sw}
    res = sm.GLM(y, sm.add_constant(X), family=sm.families.Binomial(), offset=offset,
                 **kw).fit(tol=1e-12)
    coef, b, _ = fit_offset_logistic(X, y, offset, sample_weight=sw, C=np.inf,
                                     solver=solver, tol=1e-11)
    np.testing.assert_allclose(np.r_[b, coef], res.params, atol=1e-6)
    np.testing.assert_allclose(
        X @ coef + b + offset, res.predict(sm.add_constant(X), offset=offset, which="linear"),
        atol=1e-6)


@pytest.mark.parametrize("solver", SOLVERS)
def test_logit_with_offset_equals_statsmodels_logit(solver):
    sm = pytest.importorskip("statsmodels.api")
    X, y, offset = make_data(n=400, p=3, seed=8)
    res = sm.Logit(y, sm.add_constant(X), offset=offset).fit(disp=0, method="newton",
                                                              tol=1e-12, maxiter=100)
    m = OffsetLogisticRegression(C=np.inf, solver=solver, tol=1e-11).fit(X, y, offset=offset)
    np.testing.assert_allclose(np.r_[m.intercept_, m.coef_.ravel()], res.params, atol=1e-6)


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("C", [0.02, 0.5, 5.0])
@pytest.mark.parametrize("weighted", [False, True])
def test_ridge_equals_glum_offset(solver, C, weighted):
    glum = pytest.importorskip("glum")
    X, y, offset = make_data(n=500, p=5, seed=13, offset_scale=1.5)
    sw = np.random.default_rng(4).uniform(0.5, 2.0, size=y.shape[0]) if weighted else None
    total = y.shape[0] if sw is None else sw.sum()
    # glum minimises sum_i s_i/S * logloss_i + alpha/2 ||w||^2: alpha = 1 / (C S).
    g = glum.GeneralizedLinearRegressor(
        family="binomial", alpha=1.0 / (C * total), l1_ratio=0.0, fit_intercept=True,
        gradient_tol=1e-12, max_iter=1000)
    g.fit(X, y, sample_weight=sw, offset=offset)
    coef, b, _ = fit_offset_logistic(X, y, offset, sample_weight=sw, C=C, solver=solver,
                                     tol=1e-11)
    np.testing.assert_allclose(coef, g.coef_, atol=1e-6)
    assert b == pytest.approx(g.intercept_, abs=1e-6)
    np.testing.assert_allclose(
        X @ coef + b + offset, np.log(g.predict(X, offset=offset) / (1 - g.predict(X, offset=offset))),
        atol=1e-6)
