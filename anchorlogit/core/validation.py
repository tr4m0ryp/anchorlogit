"""Input checks shared by the offset solvers and estimators."""

import numpy as np
from sklearn.utils import check_array, check_consistent_length
from sklearn.utils.multiclass import check_classification_targets, type_of_target


def check_offset(offset, n_samples, name="offset"):
    """Return ``offset`` as a finite float64 vector of length ``n_samples``.

    ``None`` becomes zeros.
    """
    if offset is None:
        return np.zeros(n_samples)
    arr = np.asarray(offset, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.ravel()
    if arr.ndim != 1 or arr.shape[0] != n_samples:
        raise ValueError(f"{name} must have shape ({n_samples},), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite (no NaN or inf)")
    return arr


def check_binary_target(y):
    """Return ``y`` as float 0 / 1. Raises on anything that is not binary 0 / 1."""
    y = np.asarray(y)
    if y.ndim != 1:
        y = y.ravel()
    vals = np.unique(y)
    if not np.all(np.isin(vals, [0, 1])):
        raise ValueError(f"y must contain only 0 and 1, got values {vals[:5]}")
    return y.astype(float)


def check_binary_labels(y):
    """Raise unless ``y`` holds exactly two classes (any labels)."""
    check_classification_targets(y)
    y_type = type_of_target(y, input_name="y")
    if y_type != "binary":
        raise ValueError("Only binary classification is supported. The type of the "
                         f"target is {y_type}.")


def check_offset_inputs(X, y, offset, sample_weight, penalty_factor):
    """Validate the arrays of :func:`fit_offset_logistic`."""
    X = check_array(X, accept_sparse=["csr", "csc"], dtype=[np.float64, np.float32])
    if X.dtype != np.float64:
        X = X.astype(np.float64)
    n_samples, n_features = X.shape
    y = check_binary_target(y)
    check_consistent_length(X, y)
    offset = check_offset(offset, n_samples)
    if sample_weight is None:
        sample_weight = np.ones(n_samples)
    else:
        sample_weight = check_offset(sample_weight, n_samples, name="sample_weight")
        if np.any(sample_weight < 0) or sample_weight.sum() <= 0:
            raise ValueError("sample_weight must be non-negative and not all zero")
    if penalty_factor is not None:
        penalty_factor = np.asarray(penalty_factor, dtype=float).ravel()
        if penalty_factor.shape[0] != n_features:
            raise ValueError(f"penalty_factor must have {n_features} entries, "
                             f"got {penalty_factor.shape[0]}")
        if not np.all(np.isfinite(penalty_factor)) or np.any(penalty_factor < 0):
            raise ValueError("penalty_factor must be finite and non-negative")
    return X, y, offset, sample_weight, penalty_factor
