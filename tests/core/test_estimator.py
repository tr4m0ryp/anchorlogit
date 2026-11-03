"""OffsetLogisticRegression: the offset= API on stock scikit-learn."""

import pickle

import numpy as np
import pytest
from scipy.special import expit
from sklearn import config_context
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import parametrize_with_checks

from anchorlogit import OffsetLogisticRegression, fit_offset_logistic
from tests.conftest import make_data


@parametrize_with_checks([OffsetLogisticRegression()])
def test_sklearn_estimator_checks(estimator, check):
    check(estimator)


def test_offset_none_matches_sklearn_predictions():
    X, y, _ = make_data()
    ours = OffsetLogisticRegression(C=0.5, tol=1e-11).fit(X, y)
    ref = LogisticRegression(C=0.5, tol=1e-12, max_iter=10_000).fit(X, y)
    np.testing.assert_allclose(ours.predict_proba(X), ref.predict_proba(X), atol=1e-7)
    np.testing.assert_array_equal(ours.predict(X), ref.predict(X))


def test_prediction_adds_the_offset():
    X, y, offset = make_data()
    m = OffsetLogisticRegression().fit(X, y, offset=offset)
    base = X @ m.coef_.ravel() + m.intercept_[0]
    np.testing.assert_allclose(m.decision_function(X, offset=offset), base + offset)
    np.testing.assert_allclose(m.decision_function(X), base)
    np.testing.assert_allclose(m.predict_proba(X, offset=offset)[:, 1], expit(base + offset))
    np.testing.assert_allclose(np.exp(m.predict_log_proba(X, offset=offset)),
                               m.predict_proba(X, offset=offset), rtol=1e-12)
    np.testing.assert_array_equal(m.predict(X, offset=offset), (base + offset > 0).astype(int))


def test_string_labels_and_classes_order():
    X, y, offset = make_data()
    labels = np.where(y == 1, "yes", "no")
    m = OffsetLogisticRegression().fit(X, labels, offset=offset)
    np.testing.assert_array_equal(m.classes_, ["no", "yes"])
    num = OffsetLogisticRegression().fit(X, y, offset=offset)
    np.testing.assert_allclose(m.coef_, num.coef_)


def test_multiclass_raises():
    X, y, _ = make_data()
    with pytest.raises(ValueError, match="Only binary"):
        OffsetLogisticRegression().fit(X, y + (X[:, 0] > 1))


def test_nothing_stored_clone_and_pickle():
    X, y, offset = make_data()
    m = OffsetLogisticRegression(C=0.3).fit(X, y, offset=offset)
    assert not any("offset" in k for k in vars(m))
    c = clone(m)
    assert not hasattr(c, "coef_") and c.get_params() == m.get_params()
    m2 = pickle.loads(pickle.dumps(m))
    np.testing.assert_array_equal(m2.decision_function(X, offset=offset),
                                  m.decision_function(X, offset=offset))


def test_estimator_equals_functional_form():
    X, y, offset = make_data()
    m = OffsetLogisticRegression(C=0.2, solver="newton").fit(X, y, offset=offset)
    coef, b, _ = fit_offset_logistic(X, y, offset, C=0.2, solver="newton")
    np.testing.assert_allclose(m.coef_.ravel(), coef)
    assert m.intercept_[0] == b


def test_pipeline_routes_the_offset_with_metadata_routing():
    X, y, offset = make_data()
    with config_context(enable_metadata_routing=True):
        est = (OffsetLogisticRegression(tol=1e-11)
               .set_fit_request(offset=True)
               .set_predict_proba_request(offset=True))
        pipe = make_pipeline(StandardScaler(), est).fit(X, y, offset=offset)
        p = pipe.predict_proba(X, offset=offset)
    Xs = StandardScaler().fit_transform(X)
    direct = OffsetLogisticRegression(tol=1e-11).fit(Xs, y, offset=offset)
    np.testing.assert_allclose(p, direct.predict_proba(Xs, offset=offset), atol=1e-8)
