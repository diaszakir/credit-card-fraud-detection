"""Integration checks on the REAL processed dataset (skipped when it has not been built)."""

import numpy as np
import pytest

from src.config import (
    EXPECTED_FRAUD,
    EXPECTED_ROWS,
    FEATURE_COLUMNS,
    ID_COLUMN,
    SPLIT_COLUMN,
    SPLIT_MANIFEST_PATH,
    SPLIT_NAMES,
    TRANSACTIONS_PARQUET_PATH,
)
from src.data.api import load_split_manifest, load_transactions, load_xy
from src.data.splitting import validate_split

pytestmark = pytest.mark.skipif(
    not TRANSACTIONS_PARQUET_PATH.is_file(),
    reason="processed dataset not built; run `python -m src.data.cli run-all`",
)


@pytest.fixture(scope="module")
def real():
    return load_transactions()


def test_real_dataset_matches_the_known_totals(real):
    assert len(real) == EXPECTED_ROWS
    assert int(real["Class"].sum()) == EXPECTED_FRAUD
    assert real.isna().sum().sum() == 0


def test_real_split_is_valid_and_complete(real):
    summary = validate_split(real)
    assert sum(info["n_rows"] for info in summary.values()) == EXPECTED_ROWS
    assert sum(info["n_fraud"] for info in summary.values()) == EXPECTED_FRAUD


def test_real_ids_are_stable_positions(real):
    assert np.array_equal(real[ID_COLUMN].to_numpy(), np.arange(EXPECTED_ROWS))


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
    assert seen == EXPECTED_ROWS
