"""Public datasets for the benchmark, cached outside the repository."""

import os
from pathlib import Path

DATA_HOME = os.environ.get("ANCHORLOGIT_DATA",
                           str(Path(__file__).resolve().parents[3] / "_data" / "openml"))


def fetch(data_id):
    from sklearn.datasets import fetch_openml
    return fetch_openml(data_id=data_id, data_home=DATA_HOME, as_frame=True, parser="auto")
