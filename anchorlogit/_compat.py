"""Small shims across supported scikit-learn versions."""

try:  # scikit-learn >= 1.6
    from sklearn.utils.validation import validate_data
except ImportError:  # pragma: no cover - older scikit-learn
    def validate_data(estimator, X="no_validation", y="no_validation", **kwargs):
        return estimator._validate_data(X, y, **kwargs)

__all__ = ["validate_data"]
