"""``OffsetLogisticRegression``: the ``offset=`` API on stock scikit-learn.

The offset is data: it is passed to ``fit`` and to every prediction method,
like ``sample_weight``, and never stored. ``offset=None`` gives ordinary ridge
logistic regression.
"""

import numpy as np
from scipy.special import expit, log_expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted

from .._compat import validate_data
from .solver import SOLVERS, fit_offset_logistic
from .validation import check_binary_labels, check_offset


class OffsetLogisticRegression(ClassifierMixin, BaseEstimator):
    """Binary ridge logistic regression with a per-sample offset.

    ``P(y = classes_[1] | x, o) = sigmoid(x @ coef_ + intercept_ + o)``.

    Parameters
    ----------
    C : float, default=1.0
        Inverse regularisation strength; ``np.inf`` fits without penalty.
    fit_intercept : bool, default=True
    solver : {"lbfgs", "newton"}, default="lbfgs"
    max_iter : int, default=1000
    tol : float, default=1e-8
    penalty_factor : array-like of shape (n_features,), default=None
        Multiplier of each coefficient's L2 penalty (0 leaves it free).

    Attributes
    ----------
    coef_ : ndarray of shape (1, n_features)
    intercept_ : ndarray of shape (1,)
    classes_ : ndarray of shape (2,)
    n_iter_ : ndarray of shape (1,)
    n_features_in_ : int
    feature_names_in_ : ndarray of str, when fitted on a DataFrame
    """

    def __init__(self, C=1.0, fit_intercept=True, solver="lbfgs", max_iter=1000,
                 tol=1e-8, penalty_factor=None):
        self.C = C
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.penalty_factor = penalty_factor

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        tags.input_tags.sparse = True
        return tags

    def fit(self, X, y, sample_weight=None, offset=None):
        """Fit the model.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
        y : array-like of shape (n_samples,), two classes
        sample_weight : array-like of shape (n_samples,), default=None
        offset : array-like of shape (n_samples,), default=None
            Known log-odds of ``classes_[1]`` per training sample.
        """
        if self.solver not in SOLVERS:
            raise ValueError(f"solver must be one of {SOLVERS}, got {self.solver!r}")
        X, y = validate_data(self, X, y, accept_sparse=["csr", "csc"],
                             dtype=[np.float64, np.float32])
        check_binary_labels(y)
        enc = LabelEncoder().fit(y)
        self.classes_ = enc.classes_
        coef, intercept, n_iter = fit_offset_logistic(
            X, enc.transform(y), offset=offset, sample_weight=sample_weight, C=self.C,
            penalty_factor=self.penalty_factor, fit_intercept=self.fit_intercept,
            solver=self.solver, max_iter=self.max_iter, tol=self.tol)
        self.coef_ = coef.reshape(1, -1)
        self.intercept_ = np.array([intercept])
        self.n_iter_ = np.array([n_iter])
        return self

    def decision_function(self, X, offset=None):
        """Log-odds of ``classes_[1]``: ``X @ coef_ + intercept_ + offset``."""
        check_is_fitted(self)
        X = validate_data(self, X, accept_sparse=["csr", "csc"], reset=False,
                          dtype=[np.float64, np.float32])
        scores = np.asarray(X @ self.coef_.ravel()).ravel() + self.intercept_[0]
        return scores + check_offset(offset, X.shape[0])

    def predict_proba(self, X, offset=None):
        """Class probabilities, shape (n_samples, 2)."""
        p = expit(self.decision_function(X, offset=offset))
        return np.column_stack([1.0 - p, p])

    def predict_log_proba(self, X, offset=None):
        """Log of :meth:`predict_proba`, computed stably."""
        s = self.decision_function(X, offset=offset)
        return np.column_stack([log_expit(-s), log_expit(s)])

    def predict(self, X, offset=None):
        """Class labels."""
        check_is_fitted(self, "coef_")
        return self.classes_[(self.decision_function(X, offset=offset) > 0).astype(int)]

    def score(self, X, y, sample_weight=None, offset=None):
        """Mean accuracy with the given offset."""
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X, offset=offset), sample_weight=sample_weight)
