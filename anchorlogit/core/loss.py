"""Binary logistic loss with a fixed per-sample offset added to the linear score.

The model is ``eta_i = x_i @ w + b + o_i`` with a known offset ``o_i``. The
objective minimised by the solvers is, up to a positive constant,

    C * sum_i s_i * l(y_i, eta_i) + 0.5 * sum_j a_j * w_j**2

with ``l(y, eta) = log(1 + exp(eta)) - y * eta``, sample weights ``s_i`` and a
per-feature penalty factor ``a_j`` (1 for ordinary ridge, 0 for an unpenalised
coefficient). The intercept is never penalised. For numerical scaling the
functions below work with the objective divided by ``C * sum(s)``, which has the
same minimiser.
"""

import numpy as np
from scipy.special import expit


def pointwise_loss(y, raw):
    """Half-binomial loss ``log(1 + exp(raw)) - y * raw``, computed stably."""
    return np.logaddexp(0.0, raw) - y * raw


class OffsetBinomialLoss:
    """Objective, gradient and Hessian of the offset logistic model.

    Parameters
    ----------
    X : ndarray or sparse matrix of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,), values in {0, 1}
    offset : ndarray of shape (n_samples,)
        Added to ``X @ w + b`` for every sample. It is data, not a parameter:
        it never enters a step direction.
    sample_weight : ndarray of shape (n_samples,)
    alpha : ndarray of shape (n_features,)
        Per-feature L2 strength, already divided by ``C * sum(sample_weight)``.
    fit_intercept : bool
    """

    def __init__(self, X, y, offset, sample_weight, alpha, fit_intercept):
        self.X = X
        self.y = y
        self.offset = offset
        self.sw = sample_weight / sample_weight.sum()
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.n_features = X.shape[1]

    def split(self, params):
        """Return ``(w, b)`` from the flat parameter vector."""
        if self.fit_intercept:
            return params[:-1], params[-1]
        return params, 0.0

    def linear(self, params):
        """``X @ w + b`` without the offset: used for step directions."""
        w, b = self.split(params)
        return np.asarray(self.X @ w).ravel() + b

    def raw(self, params):
        """Raw prediction ``X @ w + b + offset``."""
        return self.linear(params) + self.offset

    def value_from_raw(self, params, raw):
        w, _ = self.split(params)
        data = float(self.sw @ pointwise_loss(self.y, raw))
        return data + 0.5 * float(self.alpha @ (w * w))

    def value(self, params):
        return self.value_from_raw(params, self.raw(params))

    def value_grad(self, params):
        """Objective and its gradient (the L-BFGS callback)."""
        raw = self.raw(params)
        w, _ = self.split(params)
        resid = self.sw * (expit(raw) - self.y)
        grad_w = np.asarray(self.X.T @ resid).ravel() + self.alpha * w
        value = self.value_from_raw(params, raw)
        if self.fit_intercept:
            return value, np.append(grad_w, resid.sum())
        return value, grad_w

    def hessian(self, params, raw=None):
        """Dense Hessian of the objective (the Newton solver)."""
        if raw is None:
            raw = self.raw(params)
        p = expit(raw)
        h = self.sw * p * (1.0 - p)
        X = self.X.toarray() if hasattr(self.X, "toarray") else self.X
        if self.fit_intercept:
            X = np.hstack([X, np.ones((X.shape[0], 1), dtype=X.dtype)])
        H = (X.T * h) @ X
        diag = np.zeros(H.shape[0])
        diag[: self.n_features] = self.alpha
        H[np.diag_indices_from(H)] += diag
        return H

    def hessp(self, params, vec):
        """Hessian-vector product."""
        p = expit(self.raw(params))
        h = self.sw * p * (1.0 - p)
        Xv = self.linear(vec)
        out_w = np.asarray(self.X.T @ (h * Xv)).ravel()
        w_part = vec[: self.n_features]
        out_w = out_w + self.alpha * w_part
        if self.fit_intercept:
            return np.append(out_w, float(h @ Xv))
        return out_w
