"""The loss, gradient, Hessian and Hessian-vector product against finite differences."""

import numpy as np
import pytest
from scipy import sparse

from anchorlogit import OffsetBinomialLoss
from tests.conftest import make_data


def _loss(fit_intercept, weighted, sparse_x=False, alpha_zero=False):
    X, y, offset = make_data(n=120, p=4, seed=3)
    rng = np.random.default_rng(1)
    sw = rng.uniform(0.2, 2.0, size=y.shape[0]) if weighted else np.ones(y.shape[0])
    alpha = np.zeros(4) if alpha_zero else np.array([0.1, 0.0, 0.5, 2.0])
    Xm = sparse.csr_matrix(X) if sparse_x else X
    return OffsetBinomialLoss(Xm, y.astype(float), offset, sw, alpha, fit_intercept)


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize("sparse_x", [True, False])
def test_gradient_hessian_hessp_match_finite_differences(fit_intercept, weighted, sparse_x):
    loss = _loss(fit_intercept, weighted, sparse_x)
    rng = np.random.default_rng(0)
    params = rng.normal(scale=0.5, size=4 + int(fit_intercept))
    value, grad = loss.value_grad(params)
    assert value == pytest.approx(loss.value(params), abs=1e-14)
    eps = 1e-6
    num_grad = np.array([
        (loss.value(params + eps * e) - loss.value(params - eps * e)) / (2 * eps)
        for e in np.eye(params.size)])
    np.testing.assert_allclose(grad, num_grad, rtol=1e-6, atol=1e-8)
    H = loss.hessian(params)
    num_H = np.array([
        (loss.value_grad(params + eps * e)[1] - loss.value_grad(params - eps * e)[1])
        / (2 * eps) for e in np.eye(params.size)])
    np.testing.assert_allclose(H, num_H, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(H, H.T, atol=1e-14)
    v = rng.normal(size=params.size)
    np.testing.assert_allclose(loss.hessp(params, v), H @ v, rtol=1e-10, atol=1e-12)


def test_value_is_mean_logloss_plus_penalty():
    loss = _loss(True, False, alpha_zero=False)
    params = np.array([0.3, -0.2, 0.1, 0.4, -0.5])
    raw = loss.X @ params[:-1] + params[-1] + loss.offset
    p = 1 / (1 + np.exp(-raw))
    y = loss.y
    expected = np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p)))
    expected += 0.5 * np.sum(loss.alpha * params[:-1] ** 2)
    assert loss.value(params) == pytest.approx(expected, rel=1e-12)


def test_linear_part_excludes_the_offset():
    loss = _loss(True, False)
    params = np.array([0.3, -0.2, 0.1, 0.4, -0.5])
    np.testing.assert_allclose(loss.raw(params) - loss.linear(params), loss.offset)


def test_loss_is_stable_for_large_log_odds():
    X = np.array([[1.0], [-1.0]])
    loss = OffsetBinomialLoss(X, np.array([1.0, 0.0]), np.array([800.0, -800.0]),
                              np.ones(2), np.zeros(1), True)
    value, grad = loss.value_grad(np.zeros(2))
    assert np.isfinite(value) and value < 1e-300 + 1e-12
    assert np.all(np.isfinite(grad))
