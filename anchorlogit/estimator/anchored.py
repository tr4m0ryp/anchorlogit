"""``AnchoredLogisticRegression``: ridge logistic regression that starts from a reference."""

import numbers

import numpy as np
from scipy.special import expit, log_expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_array, column_or_1d
from sklearn.utils.validation import check_is_fitted

from .._compat import validate_data
from ..core.solver import SOLVERS, fit_offset_logistic
from ..core.validation import check_binary_labels, check_offset
from .anchor import AnchorResolver
from .kappa import DEFAULT_KAPPA_GRID, search_kappa


class AnchoredLogisticRegression(ClassifierMixin, BaseEstimator):
    """Ridge logistic regression that starts from a reference log-odds.

    With reference log-odds ``r`` (from ``anchor``) the model is

        logit P(y=1 | x) = x @ coef_ + intercept_ + anchor_coef_ * r

    fitted by minimising ``C * sum_i s_i * logloss_i + 0.5 * ||coef||^2
    + 0.5 * anchor_strength * (anchor_coef - kappa)^2``. One knob spans three
    models: ``anchor_strength=np.inf`` fixes ``anchor_coef_ = kappa`` (a fixed
    offset ``kappa * r``); ``0`` leaves it free and unpenalised (the reference
    as an ordinary feature); anything in between pulls it toward ``kappa``.
    ``kappa=1, anchor_strength=np.inf`` is the classical GLM offset;
    ``0 < kappa < 1`` trusts only part of the reference. Internally the
    anchored coefficient is an offset ``kappa * r`` plus a column ``r`` with
    penalty factor ``anchor_strength`` (see the tests for the exact identity).

    Parameters
    ----------
    anchor : None, str, int, classifier, CrossFitReference or callable
        Where the reference comes from: a column of ``X`` (name or index), a
        fitted classifier (its ``predict_proba``), an unfitted classifier
        (cross-fitted on the training rows), a ``CrossFitReference``, or a
        callable ``X -> values``. ``None`` gives plain ridge logistic regression.
    anchor_scale : {"prob", "logit"}, default="prob"
        How a column or callable expresses the reference.
    kappa : float >= 0 or "cv", default=1.0
        Fraction of the reference log-odds used as offset; ``"cv"`` picks it
        from ``kappa_grid`` by ``cv``-fold log-loss.
    anchor_strength : float in [0, inf], default=np.inf
    keep_anchor_column : bool, default=False
        Keep an anchor column of ``X`` among the ordinary features too.
    clip : float, default=1e-3
        Probability clipping before the log-odds.
    C : float, default=1.0
        Inverse ridge strength on the ordinary features.
    fit_intercept : bool, default=True
    standardize : bool, default=False
        Standardise the ordinary features inside the fit (coefficients are
        reported on the original scale). The reference is never rescaled.
    solver : {"lbfgs", "newton"}, default="lbfgs"
    max_iter : int, default=1000
    tol : float, default=1e-8
    cv : int or splitter, default=None
        Folds for ``kappa="cv"`` (``None`` means 5).
    kappa_grid : sequence of float, default=None
        Candidates for ``kappa="cv"``; ``None`` means (0, .1, .25, .5, .75, 1).
    crossfit_cv : int or splitter, default=5
        Folds when an unfitted classifier is the anchor.
    random_state : int or None, default=0

    Attributes
    ----------
    coef_ : ndarray of shape (1, n_used_features)
    coef_names_ : list of str, the ordinary features in ``coef_`` order
    intercept_ : ndarray of shape (1,)
    kappa_ : float, the kappa used
    anchor_coef_ : float, the fitted coefficient on the reference log-odds
    reference_ : the fitted anchor resolver
    cv_losses_ : ndarray, mean CV log-loss per ``kappa_grid`` value (``kappa="cv"``)
    classes_, n_features_in_, feature_names_in_, n_iter_
    """

    def __init__(self, anchor=None, anchor_scale="prob", kappa=1.0,
                 anchor_strength=np.inf, keep_anchor_column=False, clip=1e-3, C=1.0,
                 fit_intercept=True, standardize=False, solver="lbfgs", max_iter=1000,
                 tol=1e-8, cv=None, kappa_grid=None, crossfit_cv=5, random_state=0):
        self.anchor = anchor
        self.anchor_scale = anchor_scale
        self.kappa = kappa
        self.anchor_strength = anchor_strength
        self.keep_anchor_column = keep_anchor_column
        self.clip = clip
        self.C = C
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.cv = cv
        self.kappa_grid = kappa_grid
        self.crossfit_cv = crossfit_cv
        self.random_state = random_state

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        return tags

    def _check_params(self):
        if self.solver not in SOLVERS:
            raise ValueError(f"solver must be one of {SOLVERS}, got {self.solver!r}")
        if not (self.anchor_strength >= 0):
            raise ValueError("anchor_strength must be in [0, inf]")
        if not (self.kappa == "cv" or (isinstance(self.kappa, numbers.Real)
                                       and np.isfinite(self.kappa) and self.kappa >= 0)):
            raise ValueError(f"kappa must be a finite number >= 0 or 'cv', got {self.kappa!r}")

    def _uses_column(self):
        return self.reference_.kind_ != "none" and np.isfinite(self.anchor_strength)

    def _design(self, F, r):
        if self.standardize:
            F = (F - self.x_mean_) / self.x_scale_
        return np.column_stack([F, r]) if self._uses_column() else F

    def fit(self, X, y, sample_weight=None, offset=None, groups=None):
        """Fit the model.

        ``offset`` (log-odds, shape (n_samples,)) is added on top of the
        anchor's offset; pass the same values to the prediction methods.
        ``groups`` reaches group-aware splitters in ``cv`` and the cross-fit.
        """
        self._check_params()
        if not hasattr(X, "shape"):
            X = np.asarray(X)
        check_array(X, accept_sparse=False, dtype=None)
        validate_data(self, X, skip_check_array=True)
        y = column_or_1d(y, warn=True)
        check_binary_labels(y)
        enc = LabelEncoder().fit(y)
        self.classes_ = enc.classes_
        y01 = enc.transform(y).astype(float)
        names = getattr(self, "feature_names_in_", None)
        self.reference_ = AnchorResolver(self.anchor, self.anchor_scale, self.clip,
                                         self.crossfit_cv, self.random_state).setup(X, names)
        if self.reference_.kind_ == "crossfit":
            r = self.reference_.crossfit_.fit_transform(X, y01, groups)
        else:
            r = self.reference_.logit(X, y01, training=True)
        F = self.reference_.features(X, self.keep_anchor_column)
        self.coef_names_ = self.reference_.feature_names(names, X.shape[1],
                                                         self.keep_anchor_column)
        self.x_mean_ = F.mean(axis=0) if self.standardize else None
        if self.standardize:
            sd = F.std(axis=0)
            self.x_scale_ = np.where(sd > 0, sd, 1.0)
        design = self._design(F, r)
        extra = check_offset(offset, X.shape[0])
        n_used = F.shape[1]
        pf = None
        if self._uses_column():
            pf = np.append(np.ones(n_used), float(self.anchor_strength))
        fit_kwargs = dict(C=self.C, penalty_factor=pf, fit_intercept=self.fit_intercept,
                          solver=self.solver, max_iter=self.max_iter, tol=self.tol)
        has_anchor = self.reference_.kind_ != "none"
        if self.kappa == "cv" and has_anchor:
            grid = DEFAULT_KAPPA_GRID if self.kappa_grid is None else self.kappa_grid
            self.kappa_, self.cv_losses_ = search_kappa(
                design, y01, r, extra, sample_weight, fit_kwargs, grid, self.cv,
                self.random_state, groups)
        else:
            self.kappa_ = 0.0 if self.kappa == "cv" or not has_anchor else float(self.kappa)
        coef, b, n_iter = fit_offset_logistic(
            design, y01, offset=self.kappa_ * r + extra, sample_weight=sample_weight,
            **fit_kwargs)
        deviation = coef[n_used] if self._uses_column() else 0.0
        w = coef[:n_used]
        if self.standardize:
            w = w / self.x_scale_
            b = b - float(w @ self.x_mean_)
        self.coef_ = w.reshape(1, -1)
        self.intercept_ = np.array([b])
        self.anchor_coef_ = float(self.kappa_ + deviation) if has_anchor else 0.0
        self.n_iter_ = np.array([n_iter])
        return self

    def reference_logit(self, X):
        """Reference log-odds of new rows (full-fit when cross-fitted)."""
        check_is_fitted(self, "reference_")
        if not hasattr(X, "shape"):
            X = np.asarray(X)
        return self.reference_.logit(X, training=False)

    def decision_function(self, X, offset=None):
        """Log-odds of ``classes_[1]``."""
        check_is_fitted(self, "coef_")
        if not hasattr(X, "shape"):
            X = np.asarray(X)
        check_array(X, accept_sparse=False, dtype=None)
        validate_data(self, X, skip_check_array=True, reset=False)
        r = self.reference_logit(X)
        F = self.reference_.features(X, self.keep_anchor_column)
        scores = F @ self.coef_.ravel() + self.intercept_[0] + self.anchor_coef_ * r
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
