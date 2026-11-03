"""Agreement with scikit-learn's own offset= once it exists (patches/scikit-learn-offset.patch).

Skipped on a stock scikit-learn; ``bash patches/check_patch.sh`` runs them
against a patched copy.
"""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from anchorlogit import OffsetLogisticRegression, offset_logistic_regression, sklearn_supports_offset
from tests.conftest import make_data

needs_offset = pytest.mark.skipif(not sklearn_supports_offset(),
                                  reason="scikit-learn without offset= in LogisticRegression.fit")


@needs_offset
@pytest.mark.parametrize("solver", ["lbfgs", "newton-cg", "newton-cholesky"])
@pytest.mark.parametrize("C", [0.1, 1.0])
def test_patched_logistic_regression_equals_native(solver, C):
    X, y, offset = make_data(offset_scale=2.0)
    sk = LogisticRegression(C=C, solver=solver, tol=1e-10, max_iter=1000).fit(X, y, offset=offset)
    ours = OffsetLogisticRegression(C=C, solver="newton", tol=1e-11).fit(X, y, offset=offset)
    np.testing.assert_allclose(sk.coef_, ours.coef_, atol=1e-6)
    np.testing.assert_allclose(sk.predict_proba(X, offset=offset),
                               ours.predict_proba(X, offset=offset), atol=1e-7)


@needs_offset
def test_sklearn_backend_of_the_functional_form():
    X, y, offset = make_data()
    a = offset_logistic_regression(X, y, offset, C=0.5, backend="sklearn", solver="newton")
    b = offset_logistic_regression(X, y, offset, C=0.5, backend="native", solver="newton")
    np.testing.assert_allclose(a[0], b[0], atol=1e-6)


def test_sklearn_backend_unavailable_raises_or_runs():
    X, y, offset = make_data()
    if sklearn_supports_offset():
        offset_logistic_regression(X, y, offset, backend="sklearn")
    else:
        with pytest.raises(ImportError, match="offset"):
            offset_logistic_regression(X, y, offset, backend="sklearn")
