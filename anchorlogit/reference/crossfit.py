"""Cross-fitted reference log-odds: no training row sees its own label."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import StratifiedKFold, check_cv
from sklearn.utils.validation import check_is_fitted

from .transforms import logit_of


def _positive_proba(estimator, X):
    proba = estimator.predict_proba(X)
    classes = list(getattr(estimator, "classes_", [0, 1]))
    if len(classes) != 2:
        raise ValueError("the reference estimator must be a binary classifier")
    return np.asarray(proba)[:, 1]


class CrossFitReference(TransformerMixin, BaseEstimator):
    """Out-of-fold log-odds of a reference classifier.

    When a reference model and the model built on top of it are fitted on the
    same rows, the in-sample reference is optimistic and the second model
    learns to over-trust it. Cross-fitting avoids that: each training row gets
    the prediction of a copy of ``estimator`` fitted without the row's fold;
    new rows get the prediction of a copy fitted on every training row.

    ``fit_transform(X, y)`` returns the out-of-fold log-odds of the training
    rows; ``transform(X)`` returns the full-fit log-odds. They differ on the
    training rows on purpose (scikit-learn's ``TargetEncoder`` follows the same
    convention).

    Parameters
    ----------
    estimator : classifier with ``predict_proba``
        Cloned, never modified.
    cv : int, cross-validation generator or iterable, default=5
        An int means ``StratifiedKFold(cv, shuffle=True, random_state=random_state)``.
        Pass ``TimeSeriesSplit`` for time-ordered rows (then a row's reference
        uses only earlier rows) or a group splitter with ``groups``.
    clip : float, default=1e-3
        Probability clipping before the log-odds.
    fill_value : float, default=0.0
        Log-odds given to training rows that are in no test fold (the first
        block of a time-ordered splitter). 0.0 means "no reference".
    random_state : int or None, default=0

    Attributes
    ----------
    estimator_ : the copy fitted on all rows
    oof_assigned_ : bool ndarray, which training rows got an out-of-fold value
    """

    def __init__(self, estimator, cv=5, clip=1e-3, fill_value=0.0, random_state=0):
        self.estimator = estimator
        self.cv = cv
        self.clip = clip
        self.fill_value = fill_value
        self.random_state = random_state

    def _splitter(self):
        if isinstance(self.cv, (int, np.integer)):
            return StratifiedKFold(int(self.cv), shuffle=True, random_state=self.random_state)
        return check_cv(self.cv, classifier=True)

    def fit(self, X, y, groups=None):
        """Fit the full-data copy only (use ``fit_transform`` for training rows)."""
        self.estimator_ = clone(self.estimator).fit(X, y)
        return self

    def fit_transform(self, X, y, groups=None):
        """Fit, and return the out-of-fold log-odds of the training rows."""
        y = np.asarray(y)
        n = y.shape[0]
        oof = np.full(n, np.nan)
        for train, test in self._splitter().split(X, y, groups):
            model = clone(self.estimator).fit(_take(X, train), y[train])
            oof[test] = logit_of(_positive_proba(model, _take(X, test)), clip=self.clip)
        self.oof_assigned_ = np.isfinite(oof)
        oof[~self.oof_assigned_] = self.fill_value
        self.fit(X, y)
        return oof

    def transform(self, X):
        """Log-odds of new rows from the copy fitted on every training row."""
        check_is_fitted(self, "estimator_")
        return logit_of(_positive_proba(self.estimator_, X), clip=self.clip)


def _take(X, idx):
    if hasattr(X, "iloc"):
        return X.iloc[idx]
    return X[idx]
