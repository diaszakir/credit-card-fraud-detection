"""Shared fixtures: small SYNTHETIC frames in the raw creditcard.csv format.

The numbers in these frames are random test data, not results; they only exercise the code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import RAW_COLUMNS, V_COLUMNS


def make_raw_frame(n: int = 600, n_fraud: int = 12, seed: int = 0) -> pd.DataFrame:
    """Random frame with the raw column names / dtypes: sorted integer Time with ties."""
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {"Time": np.sort(rng.integers(0, n // 2, size=n)).astype("float64")}
    for column in V_COLUMNS:
        data[column] = rng.normal(size=n)
    amount = np.round(rng.gamma(2.0, 30.0, size=n), 2)
    amount[::50] = 0.0
    data["Amount"] = amount
    target = np.zeros(n, dtype="int64")
    target[rng.choice(n, size=n_fraud, replace=False)] = 1
    data["Class"] = target
    return pd.DataFrame(data)[RAW_COLUMNS]


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    return make_raw_frame()


@pytest.fixture
def raw_csv(tmp_path, raw_frame):
    path = tmp_path / "creditcard.csv"
    raw_frame.to_csv(path, index=False)
    return path


@pytest.fixture
def processed_frame() -> pd.DataFrame:
    """Synthetic processed frame (ids + split) spanning ~5 hours."""
    from src.config import TIME_COLUMN
    from src.data.memory import optimize_dtypes
    from src.data.splitting import build_processed_frame

    raw = make_raw_frame(n=3000, n_fraud=90, seed=3)
    rng = np.random.default_rng(3)
    raw[TIME_COLUMN] = np.sort(rng.integers(0, 5 * 3600, size=len(raw))).astype("float64")
    return build_processed_frame(optimize_dtypes(raw))
