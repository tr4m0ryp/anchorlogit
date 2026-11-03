"""Solvers for ridge logistic regression with a fixed offset.

Two solvers share :class:`OffsetBinomialLoss`:

- ``"lbfgs"``: scipy's L-BFGS-B on the objective and its gradient. Works with
  dense and sparse ``X``.
- ``"newton"``: damped Newton with an Armijo backtracking line search on a dense
  Cholesky system. Converges to machine precision in a few iterations when the
  number of features is moderate, which is what the exactness tests use.

In the Newton line search the trial raw prediction is
``raw(params + t d) = raw(params) + t * (X d_w + d_b)``. The direction term is
linear in ``d`` and is computed without the offset; adding the offset to it
would count the offset twice per step.
"""

import warnings

import numpy as np
from scipy import linalg, optimize
from sklearn.exceptions import ConvergenceWarning

from .loss import OffsetBinomialLoss
from .validation import check_offset_inputs

SOLVERS = ("lbfgs", "newton")


def _alpha(C, penalty_factor, n_features, total_weight):
    factor = np.ones(n_features) if penalty_factor is None else penalty_factor
    if np.isinf(C):
        return np.zeros(n_features)
    return factor / (C * total_weight)


def _lbfgs(loss, x0, max_iter, tol):
    res = optimize.minimize(
        loss.value_grad,
        x0,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_iter, "gtol": tol, "ftol": 64 * np.finfo(float).eps,
                 "maxcor": 30, "maxls": 50},
    )
    if not res.success and res.nit >= max_iter:
        warnings.warn(f"L-BFGS did not converge in {max_iter} iterations; "
                      "increase max_iter.", ConvergenceWarning, stacklevel=3)
    return res.x, int(res.nit)


def _newton(loss, x0, max_iter, tol):
    params = x0.copy()
    raw = loss.raw(params)
    value = loss.value_from_raw(params, raw)
    for it in range(1, max_iter + 1):
        _, grad = loss.value_grad(params)
        if np.max(np.abs(grad)) <= tol:
            return params, it - 1
        H = loss.hessian(params, raw)
        try:
            step = -linalg.solve(H, grad, assume_a="pos")
        except (linalg.LinAlgError, ValueError):
            step = -linalg.lstsq(H, grad)[0]
        direction_linear = loss.linear(step)  # no offset: linear in the step
        slope = float(grad @ step)
        t = 1.0
        for _ in range(60):
            trial = params + t * step
            trial_raw = raw + t * direction_linear
            trial_value = loss.value_from_raw(trial, trial_raw)
            if trial_value <= value + 1e-4 * t * slope + 16 * np.finfo(float).eps * abs(value):
                break
            t *= 0.5
        else:
            break
        params, raw, value = trial, trial_raw, trial_value
    _, grad = loss.value_grad(params)
    if np.max(np.abs(grad)) > tol:
        warnings.warn(f"Newton solver stopped with gradient {np.max(np.abs(grad)):.2e} "
                      f"> tol={tol}.", ConvergenceWarning, stacklevel=3)
    return params, max_iter


def fit_offset_logistic(X, y, offset=None, sample_weight=None, C=1.0,
                        penalty_factor=None, fit_intercept=True, solver="lbfgs",
                        max_iter=1000, tol=1e-8, coef_init=None):
    """Fit ``P(y=1|x) = sigmoid(x @ w + b + offset)`` under a ridge penalty.

    Minimises ``C * sum_i s_i l(y_i, eta_i) + 0.5 * sum_j a_j w_j**2`` where
    ``a`` is ``penalty_factor`` (default all ones). ``C=np.inf`` drops the
    penalty.

    Parameters
    ----------
    X : array-like or sparse matrix of shape (n_samples, n_features)
    y : array-like of shape (n_samples,), values in {0, 1}
    offset : array-like of shape (n_samples,), default=None
        Known log-odds added to ``X @ w + b``. ``None`` means zeros.
    sample_weight : array-like of shape (n_samples,), default=None
    C : float, default=1.0
        Inverse regularisation strength, as in scikit-learn.
    penalty_factor : array-like of shape (n_features,), default=None
        Non-negative multiplier of each coefficient's L2 penalty.
    fit_intercept : bool, default=True
    solver : {"lbfgs", "newton"}, default="lbfgs"
    max_iter : int, default=1000
    tol : float, default=1e-8
        Stop when the largest absolute gradient entry of the scaled objective
        is below ``tol``.
    coef_init : array-like, default=None
        Starting point ``[w, b]`` (or ``w`` without intercept). Default zeros,
        so the fit starts at the offset's own prediction.

    Returns
    -------
    coef : ndarray of shape (n_features,)
    intercept : float
    n_iter : int
    """
    if solver not in SOLVERS:
        raise ValueError(f"solver must be one of {SOLVERS}, got {solver!r}")
    if not (C > 0):
        raise ValueError(f"C must be positive, got {C!r}")
    X, y, offset, sample_weight, penalty_factor = check_offset_inputs(
        X, y, offset, sample_weight, penalty_factor)
    n_features = X.shape[1]
    alpha = _alpha(C, penalty_factor, n_features, sample_weight.sum())
    loss = OffsetBinomialLoss(X, y, offset, sample_weight, alpha, fit_intercept)
    n_params = n_features + int(fit_intercept)
    if coef_init is None:
        x0 = np.zeros(n_params)
    else:
        x0 = np.asarray(coef_init, dtype=float).ravel()
        if x0.shape[0] != n_params:
            raise ValueError(f"coef_init must have {n_params} entries, got {x0.shape[0]}")
    if solver == "newton" and hasattr(X, "toarray"):
        raise ValueError("solver='newton' needs a dense X; use solver='lbfgs' for sparse X")
    run = _lbfgs if solver == "lbfgs" else _newton
    params, n_iter = run(loss, x0, max_iter, tol)
    w, b = loss.split(params)
    return np.asarray(w, dtype=float), float(b), n_iter
