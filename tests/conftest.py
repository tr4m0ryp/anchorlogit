import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from scipy.special import expit  # noqa: E402


def make_data(n=400, p=5, seed=0, offset_scale=1.0):
    """Synthetic binary data with a known offset that carries part of the signal."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    w = np.linspace(1.0, -1.0, p)
    offset = offset_scale * rng.normal(size=n)
    y = (rng.random(n) < expit(X @ w + 0.3 + offset)).astype(int)
    return X, y, offset


@pytest.fixture
def data():
    return make_data()
