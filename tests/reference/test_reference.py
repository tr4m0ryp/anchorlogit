"""logit_of, FallbackReference and CrossFitReference (no label leakage, no lookahead)."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, TimeSeriesSplit

from anchorlogit import CrossFitReference, FallbackReference, logit_of, to_logit
from tests.conftest import make_data


def test_logit_of_values_clip_and_nan():
    p = np.array([0.5, 0.75, 0.0, 1.0, np.nan])
    out = logit_of(p, clip=1e-3)
    assert out[0] == 0.0
    assert out[1] == pytest.approx(np.log(3.0))
    assert out[2] == pytest.approx(-np.log(999.0)) and out[3] == pytest.approx(np.log(999.0))
    assert np.isnan(out[4])
    assert np.isinf(logit_of([0.0], clip=0.0)[0])
    with pytest.raises(ValueError):
        logit_of([1.2])
    with pytest.raises(ValueError):
        logit_of([0.3], clip=0.6)
    np.testing.assert_array_equal(to_logit([1.5, -2.0], scale="logit"), [1.5, -2.0])
    with pytest.raises(ValueError):
        to_logit([0.5], scale="odds")


def test_fallback_reference_first_finite_value_per_row():
    X = np.array([[0.9, 0.2], [np.nan, 0.3], [np.nan, np.nan]])
    ref = FallbackReference([lambda X: X[:, 0], lambda X: X[:, 1]], default=0.0)
    out = ref(X)
    np.testing.assert_allclose(out, [logit_of(0.9), logit_of(0.3), 0.0])
    np.testing.assert_array_equal(ref.source_index(X), [0, 1, -1])
    with pytest.raises(ValueError):
        FallbackReference([])


def _crossfit(cv, X, y, groups=None):
    cf = CrossFitReference(LogisticRegression(), cv=cv)
    return cf, cf.fit_transform(X, y, groups)


def test_out_of_fold_value_does_not_depend_on_own_label():
    """Flipping one row's label leaves that row's cross-fitted log-odds unchanged."""
    X, y, _ = make_data(n=200, p=3)
    cv = list(GroupKFold(5).split(X, y, np.arange(200) % 5))
    _, oof = _crossfit(cv, X, y)
    for i in (0, 17, 133):
        y2 = y.copy()
        y2[i] = 1 - y2[i]
        _, oof2 = _crossfit(cv, X, y2)
        assert oof2[i] == oof[i]
        assert not np.allclose(oof2, oof)  # other rows did see the flipped label


def test_out_of_fold_differs_from_in_sample_and_transform_uses_full_fit():
    X, y, _ = make_data(n=300, p=3)
    cf, oof = _crossfit(5, X, y)
    full = LogisticRegression().fit(X, y)
    np.testing.assert_allclose(cf.transform(X), logit_of(full.predict_proba(X)[:, 1]))
    assert np.max(np.abs(oof - cf.transform(X))) > 1e-3
    assert cf.oof_assigned_.all()


def test_time_ordered_splitter_has_no_lookahead():
    """With TimeSeriesSplit a row's reference depends only on earlier rows."""
    X, y, _ = make_data(n=300, p=3)
    cf, oof = _crossfit(TimeSeriesSplit(4), X, y)
    first_block = ~cf.oof_assigned_
    assert first_block[:60].all() and np.all(oof[first_block] == 0.0)
    cut = 180
    X2, y2 = X.copy(), y.copy()
    X2[cut:] = np.random.default_rng(9).normal(size=X2[cut:].shape)
    y2[cut:] = 1 - y2[cut:]
    _, oof2 = _crossfit(TimeSeriesSplit(4), X2, y2)
    # a row before `cut` is predicted by a model fitted on rows before it only
    np.testing.assert_array_equal(oof2[:cut], oof[:cut])
    assert not np.allclose(oof2[cut:], oof[cut:])


def test_group_splitter_receives_groups():
    X, y, _ = make_data(n=200, p=3)
    groups = np.repeat(np.arange(20), 10)
    cf, oof = _crossfit(GroupKFold(4), X, y, groups)
    assert cf.oof_assigned_.all() and np.all(np.isfinite(oof))
