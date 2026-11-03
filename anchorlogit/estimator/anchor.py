"""Resolving the ``anchor`` argument into reference log-odds and a design matrix."""

import numbers

import numpy as np
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from ..reference.crossfit import CrossFitReference, _positive_proba
from ..reference.transforms import logit_of, to_logit


def _is_fitted(est):
    try:
        check_is_fitted(est)
        return True
    except (NotFittedError, TypeError):
        return False


def _column_index(anchor, X, feature_names):
    """Position of the anchor column in X, or None if the anchor is not a column."""
    if isinstance(anchor, str):
        if feature_names is None:
            raise ValueError(f"anchor={anchor!r} is a column name but X has no column "
                             "names; pass a DataFrame or an integer index")
        names = list(feature_names)
        if anchor not in names:
            raise ValueError(f"anchor column {anchor!r} not found in X")
        return names.index(anchor)
    if isinstance(anchor, numbers.Integral) and not isinstance(anchor, bool):
        n = X.shape[1]
        if not -n <= anchor < n:
            raise ValueError(f"anchor index {anchor} out of range for {n} columns")
        return int(anchor) % n
    return None


class AnchorResolver:
    """Computes the reference log-odds of rows of ``X``.

    Kinds of anchor:

    - ``None``: no reference (log-odds 0).
    - a column name or integer index of ``X``: that column, read as a
      probability (``scale="prob"``) or log-odds (``scale="logit"``).
    - a fitted classifier: its ``predict_proba`` on ``X`` (a model trained on
      other data).
    - an unfitted classifier: cross-fitted with :class:`CrossFitReference`
      (``crossfit_cv`` folds).
    - a :class:`CrossFitReference`: used as given (cloned).
    - any other callable ``X -> values``: called on ``X``; values on ``scale``.
    """

    def __init__(self, anchor, scale, clip, crossfit_cv, random_state):
        self.anchor = anchor
        self.scale = scale
        self.clip = clip
        self.crossfit_cv = crossfit_cv
        self.random_state = random_state

    def setup(self, X, feature_names):
        a = self.anchor
        self.column_ = _column_index(a, X, feature_names)
        self.crossfit_ = None
        self.kind_ = "none" if a is None else "column" if self.column_ is not None else None
        if self.kind_ is not None:
            return self
        if isinstance(a, CrossFitReference):
            self.crossfit_, self.kind_ = clone(a), "crossfit"
        elif hasattr(a, "predict_proba"):
            if _is_fitted(a):
                self.kind_ = "fitted"
            else:
                self.crossfit_ = CrossFitReference(a, cv=self.crossfit_cv, clip=self.clip,
                                                   random_state=self.random_state)
                self.kind_ = "crossfit"
        elif callable(a):
            self.kind_ = "callable"
        else:
            raise TypeError(f"unsupported anchor {a!r}: use None, a column name or index, "
                            "a classifier, a CrossFitReference or a callable")
        return self

    def _column(self, X):
        col = X.iloc[:, self.column_] if hasattr(X, "iloc") else X[:, self.column_]
        return np.asarray(col, dtype=float)

    def logit(self, X, y=None, training=False):
        """Reference log-odds of each row; cross-fitted when ``training``."""
        n = X.shape[0]
        if self.kind_ == "none":
            r = np.zeros(n)
        elif self.kind_ == "column":
            r = to_logit(self._column(X), self.scale, self.clip)
        elif self.kind_ == "fitted":
            r = logit_of(_positive_proba(self.anchor, X), clip=self.clip)
        elif self.kind_ == "crossfit":
            r = self.crossfit_.fit_transform(X, y) if training else self.crossfit_.transform(X)
        else:
            r = to_logit(np.asarray(self.anchor(X), dtype=float).ravel(), self.scale, self.clip)
        r = np.asarray(r, dtype=float).ravel()
        if r.shape[0] != n or not np.all(np.isfinite(r)):
            raise ValueError("the anchor must give one finite reference value per row "
                             "(use FallbackReference for missing values, or clip > 0)")
        return r

    def features(self, X, keep_anchor_column):
        """``X`` as a float array, without the anchor column unless kept."""
        arr = np.asarray(X, dtype=float)
        if self.kind_ == "column" and not keep_anchor_column:
            arr = np.delete(arr, self.column_, axis=1)
        return arr

    def feature_names(self, names, n_features, keep_anchor_column):
        names = list(names) if names is not None else [f"x{i}" for i in range(n_features)]
        if self.kind_ == "column" and not keep_anchor_column:
            names.pop(self.column_)
        return names

