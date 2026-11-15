"""Integration checks on the REAL processed dataset (skipped when it has not been built)."""

import numpy as np
import pytest

from src.config import (
    EXPECTED_FRAUD,
    EXPECTED_ROWS,
    FEATURE_COLUMNS,
    ID_COLUMN,
    RAW_COLUMNS,
    RAW_CSV_PATH,
    SPLIT_COLUMN,
    SPLIT_MANIFEST_PATH,
    SPLIT_NAMES,
    TRANSACTIONS_PARQUET_PATH,
)
from src.data.api import load_split_manifest, load_transactions, load_xy
from src.data.loading import read_raw_csv
from src.data.splitting import validate_split

pytestmark = pytest.mark.skipif(
    not TRANSACTIONS_PARQUET_PATH.is_file() or not RAW_CSV_PATH.is_file(),
    reason="processed dataset not built; run `python -m src.data.cli run-all`",
)


@pytest.fixture(scope="module")
def real():
    return load_transactions()


def test_real_dataset_matches_unique_raw_rows(real):
    raw = read_raw_csv()
    assert len(raw) == EXPECTED_ROWS
    assert int(raw["Class"].sum()) == EXPECTED_FRAUD
    unique = raw.drop_duplicates(subset=RAW_COLUMNS, keep="first")
    assert len(real) == len(unique)
    assert int(real["Class"].sum()) == int(unique["Class"].sum())
    assert real.isna().sum().sum() == 0


def test_real_split_is_valid_and_complete(real):
    summary = validate_split(real)
    assert sum(info["n_rows"] for info in summary.values()) == len(real)
    assert sum(info["n_fraud"] for info in summary.values()) == int(real["Class"].sum())


def test_real_ids_are_stable_positions(real):
    raw = read_raw_csv()
    expected_ids = raw.index[~raw.duplicated(subset=RAW_COLUMNS, keep="first")]
    assert np.array_equal(real[ID_COLUMN].to_numpy(), expected_ids.to_numpy())


def test_committed_manifest_agrees_with_processed_data(real):
    if not SPLIT_MANIFEST_PATH.is_file():
        pytest.skip("manifest missing")
    manifest = load_split_manifest()
    assert np.array_equal(manifest[ID_COLUMN].to_numpy(), real[ID_COLUMN].to_numpy())
    assert (manifest[SPLIT_COLUMN].astype(str).to_numpy() == real[SPLIT_COLUMN].astype(str).to_numpy()).all()


def test_xy_views_cover_every_row_exactly_once():
    seen = 0
    for name in SPLIT_NAMES:
        X, y = load_xy(name)
        assert list(X.columns) == FEATURE_COLUMNS and len(X) == len(y)
        seen += len(X)
    assert seen == len(read_raw_csv().drop_duplicates(subset=RAW_COLUMNS))
