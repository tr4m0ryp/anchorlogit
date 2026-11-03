"""Solver properties: equivalence, convexity, shrinkage to the offset, validation."""

import numpy as np
import pytest
from scipy import optimize, sparse
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from anchorlogit import fit_offset_logistic, offset_logistic_regression
from anchorlogit.core.loss import pointwise_loss
from tests.conftest import make_data

SOLVERS = ["lbfgs", "newton"]


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("offset_kind", ["none", "zeros"])
def test_no_offset_equals_sklearn_logistic_regression(solver, fit_intercept, offset_kind):
    X, y, _ = make_data()
    offset = None if offset_kind == "none" else np.zeros(y.shape[0])
    ref = LogisticRegression(C=0.7, fit_intercept=fit_intercept, tol=1e-12,
                             max_iter=10_000).fit(X, y)
    coef, b, _ = fit_offset_logistic(X, y, offset, C=0.7, fit_intercept=fit_intercept,
                                     solver=solver, tol=1e-10)
    np.testing.assert_allclose(coef, ref.coef_.ravel(), atol=1e-6)
    assert b == pytest.approx(ref.intercept_[0], abs=1e-6)


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("C", [0.05, 1.0, np.inf])
def test_matches_direct_minimisation_of_the_stated_objective(solver, C):
    X, y, offset = make_data(n=200, p=3, seed=5)
    sw = np.random.default_rng(2).uniform(0.5, 1.5, size=y.shape[0])

    def objective(theta):
        raw = X @ theta[:-1] + theta[-1] + offset
        pen = 0.0 if np.isinf(C) else 0.5 * theta[:-1] @ theta[:-1] / C
        return float(sw @ pointwise_loss(y, raw)) + pen

    direct = optimize.minimize(objective, np.zeros(4), method="BFGS",
                               options={"gtol": 1e-10}).x
    coef, b, _ = fit_offset_logistic(X, y, offset, sample_weight=sw, C=C, solver=solver,
                                     tol=1e-11)
    np.testing.assert_allclose(np.append(coef, b), direct, atol=1e-5)


def test_lbfgs_and_newton_agree_tightly():
    X, y, offset = make_data(n=500, p=6, seed=7, offset_scale=2.0)
    a = fit_offset_logistic(X, y, offset, C=0.3, solver="lbfgs", tol=1e-12)
    b = fit_offset_logistic(X, y, offset, C=0.3, solver="newton", tol=1e-12)
    np.testing.assert_allclose(a[0], b[0], atol=1e-7)
    assert a[1] == pytest.approx(b[1], abs=1e-7)


@pytest.mark.parametrize("solver", SOLVERS)
def test_constant_offset_moves_only_the_intercept(solver):
    X, y, _ = make_data()
    c = 1.7
    w0, b0, _ = fit_offset_logistic(X, y, None, solver=solver, tol=1e-11)
    w1, b1, _ = fit_offset_logistic(X, y, np.full(y.shape[0], c), solver=solver, tol=1e-11)
    np.testing.assert_allclose(w1, w0, atol=1e-7)
    assert b1 == pytest.approx(b0 - c, abs=1e-7)


@pytest.mark.parametrize("solver", SOLVERS)
def test_heavy_ridge_starts_and_stays_at_the_offset(solver):
    """Without an intercept and with C -> 0 the prediction is the offset's own."""
    X, y, offset = make_data(offset_scale=2.0)
    coef, b, _ = fit_offset_logistic(X, y, offset, C=1e-9, fit_intercept=False,
                                     solver=solver)
    assert b == 0.0
    np.testing.assert_allclose(expit(X @ coef + offset), expit(offset), atol=1e-6)


def test_zero_iterations_return_the_offset_prediction():
    X, y, offset = make_data()
    with pytest.warns(ConvergenceWarning):
        coef, b, _ = fit_offset_logistic(X, y, offset, solver="newton", max_iter=0)
    assert np.all(coef == 0) and b == 0.0


def test_offset_that_is_the_signal_improves_the_fit():
    X, y, offset = make_data(offset_scale=3.0)
    w0, b0, _ = fit_offset_logistic(X, y, None)
    w1, b1, _ = fit_offset_logistic(X, y, offset)
    without = pointwise_loss(y, X @ w0 + b0).mean()
    with_offset = pointwise_loss(y, X @ w1 + b1 + offset).mean()
    assert with_offset < without - 0.05


@pytest.mark.parametrize("solver", SOLVERS)
def test_integer_sample_weights_equal_row_repetition(solver):
    X, y, offset = make_data(n=150, p=3, seed=11)
    sw = np.random.default_rng(0).integers(1, 4, size=y.shape[0])
    rep = np.repeat(np.arange(y.shape[0]), sw)
    a = fit_offset_logistic(X, y, offset, sample_weight=sw, C=0.5, solver=solver,
                            tol=1e-11)
    b = fit_offset_logistic(X[rep], y[rep], offset[rep], C=0.5, solver=solver, tol=1e-11)
    np.testing.assert_allclose(a[0], b[0], atol=1e-7)
    assert a[1] == pytest.approx(b[1], abs=1e-7)


def test_penalty_factor_zero_is_unpenalised():
    X, y, offset = make_data(n=300, p=3, seed=4)
    free = fit_offset_logistic(X, y, offset, C=np.inf, solver="newton", tol=1e-11)
    pf = fit_offset_logistic(X, y, offset, C=0.01, penalty_factor=np.zeros(3),
                             solver="newton", tol=1e-11)
    np.testing.assert_allclose(pf[0], free[0], atol=1e-8)


def test_sparse_equals_dense_and_float32_accepted():
    X, y, offset = make_data()
    dense = fit_offset_logistic(X, y, offset, tol=1e-11)
    sp = fit_offset_logistic(sparse.csr_matrix(X), y, offset, tol=1e-11)
    f32 = fit_offset_logistic(X.astype(np.float32), y, offset.astype(np.float32), tol=1e-11)
    np.testing.assert_allclose(sp[0], dense[0], atol=1e-7)
    np.testing.assert_allclose(f32[0], dense[0], atol=1e-5)


@pytest.mark.parametrize("bad", ["short", "2d", "nan", "inf"])
def test_bad_offset_raises(bad):
    X, y, _ = make_data(n=50)
    offset = {"short": np.zeros(49), "2d": np.zeros((50, 2)),
              "nan": np.r_[np.nan, np.zeros(49)], "inf": np.r_[np.inf, np.zeros(49)]}[bad]
    with pytest.raises(ValueError, match="offset"):
        fit_offset_logistic(X, y, offset)


def test_other_bad_inputs_raise():
    X, y, _ = make_data(n=50)
    with pytest.raises(ValueError, match="solver"):
        fit_offset_logistic(X, y, solver="saga")
    with pytest.raises(ValueError, match="0 and 1"):
        fit_offset_logistic(X, y + 1)
    with pytest.raises(ValueError, match="C must be positive"):
        fit_offset_logistic(X, y, C=0.0)
    with pytest.raises(ValueError, match="penalty_factor"):
        fit_offset_logistic(X, y, penalty_factor=-np.ones(X.shape[1]))
    with pytest.raises(ValueError, match="dense"):
        fit_offset_logistic(sparse.csr_matrix(X), y, solver="newton")


def test_functional_form_and_backends():
    X, y, offset = make_data()
    coef, b = offset_logistic_regression(X, y, offset, C=0.5)
    ref = fit_offset_logistic(X, y, offset, C=0.5)
    np.testing.assert_allclose(coef, ref[0])
    auto = offset_logistic_regression(X, y, offset, C=0.5, backend="auto", tol=1e-10)
    np.testing.assert_allclose(auto[0], coef, atol=1e-5)
    with pytest.raises(ValueError, match="backend"):
        offset_logistic_regression(X, y, offset, backend="other")
