"""Quickstart: correct an existing model's probabilities with a few labelled rows.

A reference model was trained elsewhere on part of the features. On a small new
sample we fit a ridge logistic model that starts from (a fraction of) the
reference's log-odds and learns only the correction the other features add.

Run:  python examples/quickstart.py
"""

import numpy as np
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV

from anchorlogit import AnchoredLogisticRegression, logit_of


def make_population(n, rng):
    """Six features; the outcome depends on all six and an interaction."""
    X = rng.normal(size=(n, 6))
    eta = 1.2 * X[:, 0] - X[:, 1] + 0.8 * X[:, 0] * X[:, 2] + 0.7 * X[:, 4] - 0.6 * X[:, 5]
    y = (rng.random(n) < expit(eta)).astype(int)
    return X, y


def main(n_train=400, seed=0):
    rng = np.random.default_rng(seed)
    # 1. a reference model fitted on other data that only saw features 0-3
    X_old, y_old = make_population(20_000, rng)
    ref = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(X_old[:, :4], y_old)

    # 2. a small new labelled sample and a large test sample, reference as a column
    def with_reference(X):
        return np.column_stack([X, ref.predict_proba(X[:, :4])[:, 1]])

    X_tr, y_tr = make_population(n_train, rng)
    X_te, y_te = make_population(20_000, rng)
    A_tr, A_te = with_reference(X_tr), with_reference(X_te)
    q_te = A_te[:, -1]

    arms = {}
    arms["reference alone"] = q_te
    plain = LogisticRegression(C=1.0).fit(X_tr, y_tr)
    arms["logistic, no reference"] = plain.predict_proba(X_te)[:, 1]
    as_feature = np.column_stack([X_tr, logit_of(A_tr[:, -1])])
    feat = LogisticRegression(C=1.0).fit(as_feature, y_tr)
    arms["reference log-odds as a feature"] = feat.predict_proba(
        np.column_stack([X_te, logit_of(q_te)]))[:, 1]
    for name, kw in [("offset, kappa = 1", dict(kappa=1.0)),
                     ("offset, kappa by CV", dict(kappa="cv"))]:
        m = AnchoredLogisticRegression(anchor=-1, C=1.0, **kw).fit(A_tr, y_tr)
        arms[name] = m.predict_proba(A_te)[:, 1]
        if kw["kappa"] == "cv":
            print(f"kappa chosen by CV: {m.kappa_}")
    grid = GridSearchCV(AnchoredLogisticRegression(anchor=-1, C=1.0),
                        {"kappa": [0.0, 0.5, 1.0], "anchor_strength": [np.inf, 10.0, 1.0, 0.0]},
                        scoring="neg_log_loss", cv=5).fit(A_tr, y_tr)
    arms["anchored, (kappa, strength) by CV"] = grid.predict_proba(A_te)[:, 1]
    print(f"grid choice: {grid.best_params_}, anchor coefficient "
          f"{grid.best_estimator_.anchor_coef_:.3f}")

    print(f"\ntest log-loss, {n_train} training rows, {len(y_te)} test rows")
    for name, p in arms.items():
        print(f"  {name:<36s} {log_loss(y_te, p):.4f}")


if __name__ == "__main__":
    main()
