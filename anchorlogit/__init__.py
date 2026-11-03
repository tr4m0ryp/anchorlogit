"""anchorlogit: logistic regression that starts from a reference log-odds.

When a probability for each example already exists (an earlier model, an
external forecast, a known prior), fit a ridge logistic model that starts from
(a fraction of) that probability's log-odds and learns only the correction the
features add.

Public API
----------
AnchoredLogisticRegression
    The estimator: the reference comes from a column, a classifier, a
    cross-fitted reference or a callable; ``kappa`` and ``anchor_strength``
    set how much it is trusted.
OffsetLogisticRegression
    Ridge logistic regression with ``offset=`` in ``fit`` and every
    prediction method (the API proposed for scikit-learn in issue #24614).
offset_logistic_regression, fit_offset_logistic
    Functional forms.
CrossFitReference
    Out-of-fold reference log-odds for training rows.
FallbackReference, logit_of
    Helpers to build reference log-odds.
"""

from .core import (
    OffsetBinomialLoss,
    OffsetLogisticRegression,
    fit_offset_logistic,
    offset_logistic_regression,
    sklearn_supports_offset,
)
from .estimator import AnchoredLogisticRegression
from .reference import CrossFitReference, FallbackReference, logit_of, to_logit

__version__ = "0.1.0"

__all__ = [
    "AnchoredLogisticRegression",
    "CrossFitReference",
    "FallbackReference",
    "OffsetBinomialLoss",
    "OffsetLogisticRegression",
    "fit_offset_logistic",
    "logit_of",
    "offset_logistic_regression",
    "sklearn_supports_offset",
    "to_logit",
]
