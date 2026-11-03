"""Functional entry point and the scikit-learn backend switch."""

import inspect

import numpy as np
from sklearn.linear_model import LogisticRegression

from .solver import fit_offset_logistic

BACKENDS = ("auto", "native", "sklearn")


def sklearn_supports_offset():
    """True when the installed ``LogisticRegression.fit`` accepts ``offset=``.

    That is the case once the upstream change for scikit-learn issue #24614
    is merged (or the patch in ``patches/`` is applied).
    """
    return "offset" in inspect.signature(LogisticRegression.fit).parameters


def offset_logistic_regression(X, y, offset, C=1.0, sample_weight=None,
                               fit_intercept=True, solver="lbfgs", max_iter=1000,
                               tol=1e-8, backend="native"):
    """Ridge logistic regression with a fixed offset, functional form.

    Fits ``P(y=1|x) = sigmoid(x @ coef + intercept + offset)`` by minimising
    ``C * sum_i s_i * logloss_i + 0.5 * ||coef||^2``, scikit-learn's objective.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,), values in {0, 1}
    offset : array-like of shape (n_samples,) or None
        Known log-odds per sample.
    C : float, default=1.0
    sample_weight : array-like of shape (n_samples,), default=None
    fit_intercept : bool, default=True
    solver : {"lbfgs", "newton"}, default="lbfgs"
        Solver of the native backend. The scikit-learn backend maps "newton"
        to "newton-cholesky".
    max_iter : int, default=1000
    tol : float, default=1e-8
    backend : {"native", "sklearn", "auto"}, default="native"
        ``"sklearn"`` calls ``LogisticRegression.fit(offset=...)`` and needs a
        scikit-learn with offset support; ``"auto"`` uses it when present and
        the native solver otherwise.

    Returns
    -------
    coef : ndarray of shape (n_features,)
    intercept : float
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    use_sklearn = backend == "sklearn" or (backend == "auto" and sklearn_supports_offset())
    if use_sklearn:
        if not sklearn_supports_offset():
            raise ImportError("the installed scikit-learn has no offset= in "
                              "LogisticRegression.fit; use backend='native'")
        sk_solver = "newton-cholesky" if solver == "newton" else solver
        model = LogisticRegression(C=C, fit_intercept=fit_intercept, solver=sk_solver,
                                   max_iter=max_iter, tol=tol)
        model.fit(X, np.asarray(y), sample_weight=sample_weight, offset=offset)
        return model.coef_.ravel().copy(), float(model.intercept_[0])
    coef, intercept, _ = fit_offset_logistic(
        X, y, offset=offset, sample_weight=sample_weight, C=C,
        fit_intercept=fit_intercept, solver=solver, max_iter=max_iter, tol=tol)
    return coef, intercept
