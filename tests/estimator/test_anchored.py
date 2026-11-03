"""AnchoredLogisticRegression: anchor kinds, kappa by CV, scikit-learn API."""

import pickle

import numpy as np
import pytest
from scipy.special import expit
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import parametrize_with_checks

from anchorlogit import (AnchoredLogisticRegression, CrossFitReference,
                         OffsetLogisticRegression, logit_of)


@parametrize_with_checks([AnchoredLogisticRegression(),
                          AnchoredLogisticRegression(anchor=0, anchor_scale="logit",
                                                     kappa=0.5, anchor_strength=1.0)])
def test_sklearn_estimator_checks(estimator, check):
    check(estimator)


def ref_data(n=600, seed=0, informative=True):
    """X = 3 features + a reference probability column ``q`` (last)."""
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(n, 3))
    r = 2.0 * rng.normal(size=n)
    q = expit(r)
    signal = r + 0.8 * F[:, 0] if informative else np.zeros(n)
    y = (rng.random(n) < expit(signal)).astype(int)
    return F, q, y


def test_anchor_none_is_plain_ridge_logistic():
    F, _, y = ref_data()
    a = AnchoredLogisticRegression(C=0.5).fit(F, y)
    b = OffsetLogisticRegression(C=0.5).fit(F, y)
    np.testing.assert_allclose(a.coef_, b.coef_)
    assert a.anchor_coef_ == 0.0


def test_column_by_name_index_and_callable_agree():
    pd = pytest.importorskip("pandas")
    F, q, y = ref_data()
    X = np.column_stack([F, q])
    df = pd.DataFrame(X, columns=["a", "b", "c", "ref_p"])
    by_name = AnchoredLogisticRegression(anchor="ref_p", kappa=0.5).fit(df, y)
    by_index = AnchoredLogisticRegression(anchor=3, kappa=0.5).fit(X, y)
    np.testing.assert_allclose(by_name.coef_, by_index.coef_)
    assert by_name.coef_names_ == ["a", "b", "c"]
    off = OffsetLogisticRegression().fit(F, y, offset=0.5 * logit_of(q))
    np.testing.assert_allclose(by_index.coef_, off.coef_)
    # a callable reads any columns; X stays whole, like keep_anchor_column=True
    fn = AnchoredLogisticRegression(anchor=lambda X: X[:, 3], kappa=0.5).fit(X, y)
    kept = AnchoredLogisticRegression(anchor=3, kappa=0.5, keep_anchor_column=True).fit(X, y)
    np.testing.assert_allclose(fn.predict_proba(X), kept.predict_proba(X))


def test_fitted_classifier_anchor_uses_its_predictions():
    F, _, y = ref_data()
    pre = LogisticRegression().fit(F[:300], y[:300])
    m = AnchoredLogisticRegression(anchor=pre, kappa=1.0).fit(F[300:], y[300:])
    r = logit_of(pre.predict_proba(F[300:])[:, 1])
    off = OffsetLogisticRegression().fit(F[300:], y[300:], offset=r)
    np.testing.assert_allclose(m.coef_, off.coef_)
    np.testing.assert_allclose(m.reference_logit(F[300:]), r)


def test_unfitted_classifier_is_cross_fitted():
    F, _, y = ref_data()
    clf = RandomForestClassifier(n_estimators=20, random_state=0, n_jobs=1)
    m = AnchoredLogisticRegression(anchor=clf, kappa=1.0, crossfit_cv=4).fit(F, y)
    oof = CrossFitReference(clf, cv=4).fit_transform(F, y)
    off = OffsetLogisticRegression().fit(F, y, offset=oof)
    np.testing.assert_allclose(m.coef_, off.coef_, atol=1e-10)
    # prediction uses the full fit, which differs from the out-of-fold values
    assert np.max(np.abs(m.reference_logit(F) - oof)) > 0.1
    explicit = AnchoredLogisticRegression(anchor=CrossFitReference(clf, cv=4)).fit(F, y)
    np.testing.assert_allclose(explicit.coef_, m.coef_)


def test_in_sample_reference_leaks_and_cross_fitting_does_not():
    """An overfitting reference looks perfect in sample; cross-fitting exposes it."""
    rng = np.random.default_rng(0)
    F = rng.normal(size=(400, 5))
    y = (rng.random(400) < 0.5).astype(int)  # labels unrelated to F
    clf = RandomForestClassifier(n_estimators=30, random_state=0, n_jobs=1)
    in_sample = logit_of(clone(clf).fit(F, y).predict_proba(F)[:, 1])
    leaky = AnchoredLogisticRegression(anchor=lambda X: in_sample, anchor_scale="logit",
                                       kappa="cv").fit(F, y)
    honest = AnchoredLogisticRegression(anchor=clf, kappa="cv").fit(F, y)
    assert leaky.kappa_ >= 0.5
    assert honest.kappa_ <= 0.1


def test_kappa_cv_trusts_a_good_reference_and_ignores_noise():
    F, q, y = ref_data(seed=1)
    X = np.column_stack([F, q])
    good = AnchoredLogisticRegression(anchor=3, kappa="cv").fit(X, y)
    assert good.kappa_ >= 0.75 and good.cv_losses_.shape == (6,)
    F2, q2, y2 = ref_data(seed=2, informative=False)
    noise = AnchoredLogisticRegression(anchor=3, kappa="cv").fit(np.column_stack([F2, q2]), y2)
    assert noise.kappa_ <= 0.1


def test_permuted_labels_leave_no_learned_deviation():
    """With labels permuted, the anchored coefficient's deviation and the features vanish."""
    F, q, y = ref_data(seed=3)
    X = np.column_stack([F, q])
    yp = np.random.default_rng(0).permutation(y)
    m = AnchoredLogisticRegression(anchor=3, kappa=0.0, anchor_strength=1.0, C=0.05).fit(X, yp)
    real = AnchoredLogisticRegression(anchor=3, kappa=0.0, anchor_strength=1.0, C=0.05).fit(X, y)
    assert abs(m.anchor_coef_) < 0.1 < real.anchor_coef_
    assert np.max(np.abs(m.coef_)) < 0.25 < real.coef_[0, 0]


def test_extra_offset_keep_column_and_standardize():
    F, q, y = ref_data()
    X = np.column_stack([F, q])
    extra = np.random.default_rng(1).normal(size=y.shape[0])
    m = AnchoredLogisticRegression(anchor=3, kappa=0.5).fit(X, y, offset=extra)
    off = OffsetLogisticRegression().fit(F, y, offset=0.5 * logit_of(q) + extra)
    np.testing.assert_allclose(m.coef_, off.coef_)
    np.testing.assert_allclose(m.decision_function(X, offset=extra),
                               off.decision_function(F, offset=0.5 * logit_of(q) + extra))
    kept = AnchoredLogisticRegression(anchor=3, keep_anchor_column=True).fit(X, y)
    assert kept.coef_.shape == (1, 4)
    scale = 10.0 ** np.arange(3)
    s = AnchoredLogisticRegression(anchor=3, standardize=True, tol=1e-11).fit(
        np.column_stack([F * scale, q]), y)
    Fs = StandardScaler().fit_transform(F)
    manual = OffsetLogisticRegression(tol=1e-11).fit(Fs, y, offset=logit_of(q))
    np.testing.assert_allclose(s.predict_proba(np.column_stack([F * scale, q])),
                               manual.predict_proba(Fs, offset=logit_of(q)), atol=1e-8)


def test_clone_pickle_and_grid_search_over_the_knob():
    F, q, y = ref_data()
    X = np.column_stack([F, q])
    m = AnchoredLogisticRegression(anchor=3, kappa=0.5, anchor_strength=1.0).fit(X, y)
    m2 = pickle.loads(pickle.dumps(m))
    np.testing.assert_array_equal(m2.predict_proba(X), m.predict_proba(X))
    assert clone(m).get_params() == m.get_params()
    grid = GridSearchCV(AnchoredLogisticRegression(anchor=3),
                        {"kappa": [0.25, 1.0], "anchor_strength": [np.inf, 1.0, 0.0]},
                        scoring="neg_log_loss", cv=3, n_jobs=1).fit(X, y)
    assert set(grid.best_params_) == {"kappa", "anchor_strength"}


def test_bad_parameters_raise():
    F, q, y = ref_data(n=60)
    X = np.column_stack([F, q])
    for kw, match in [({"kappa": -0.1}, "kappa"), ({"anchor_strength": -1.0}, "anchor_strength"),
                      ({"solver": "saga"}, "solver"), ({"anchor": "q"}, "column names"),
                      ({"anchor": 9}, "out of range"), ({"anchor": 1.5}, "unsupported")]:
        with pytest.raises((ValueError, TypeError), match=match):
            AnchoredLogisticRegression(**kw).fit(X, y)
