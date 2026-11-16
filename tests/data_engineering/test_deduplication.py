"""Synthetic checks for exact raw duplicates and stable ids after removal."""

import numpy as np
import pandas as pd
import pytest

from src.config import ID_COLUMN, RAW_COLUMNS, TIME_COLUMN
from src.data.api import load_split_manifest, load_xy
from src.data.io_utils import read_json
from src.data.memory import optimize_dtypes
from src.data.pipeline import step_optimize, step_split
from src.data.splitting import build_processed_frame, remove_duplicate_transactions, validate_split
from src.data.storage import save_parquet
from src.db.builder import build_database
from src.db.crosscheck import crosscheck_with_pandas
from src.db.engine import get_engine
from src.db.queries import run_all
from tests.data_engineering.conftest import make_raw_frame


def make_raw_with_duplicates():
    raw = make_raw_frame(n=20, n_fraud=10)
    raw[TIME_COLUMN] = np.arange(20, dtype="float64")
    raw["Class"] = (np.arange(20) % 3 != 2).astype(int)
    # Three extra copies change the uncleaned cut positions; all keep the original Time.
    return pd.concat([raw.iloc[:5], raw.iloc[[4, 4, 4]], raw.iloc[5:]], ignore_index=True)


def test_removes_exact_copies_and_preserves_first_raw_ids():
    raw = make_raw_with_duplicates()
    before = raw.copy()
    unique = remove_duplicate_transactions(raw)
    assert len(unique) == 20
    assert unique[ID_COLUMN].tolist() == list(range(5)) + list(range(8, 23))
    assert not unique.duplicated(subset=RAW_COLUMNS).any()
    pd.testing.assert_frame_equal(raw, before)


def test_duplicate_comparison_uses_all_raw_columns():
    raw = make_raw_frame(n=4, n_fraud=2)
    raw.iloc[1:] = raw.iloc[0].to_numpy()
    raw.loc[1, "Class"] = 1 - raw.loc[0, "Class"]
    raw.loc[3, "Time"] += 1
    # Row 2 repeats row 0 despite the distinct-label row between them.
    unique = remove_duplicate_transactions(raw)
    assert unique[ID_COLUMN].tolist() == [0, 1, 3]


def test_near_equal_raw_values_are_not_removed_after_rounding(tmp_path):
    raw = make_raw_frame(n=2, n_fraud=0)
    raw.iloc[1] = raw.iloc[0]
    raw.loc[0, "V1"] = 1.0
    raw.loc[1, "V1"] = 1.0 + 1e-9
    csv_path = tmp_path / "raw.csv"
    raw.to_csv(csv_path, index=False)
    optimized = step_optimize(csv_path, tmp_path / "interim.parquet")
    assert len(optimized) == 2
    assert optimized[ID_COLUMN].tolist() == [0, 1]
    assert optimized.loc[0, "V1"] == optimized.loc[1, "V1"]


def test_clean_pipeline_splits_unique_rows_and_preserves_ids_in_sqlite(tmp_path):
    raw = make_raw_with_duplicates()
    csv_path = tmp_path / "raw.csv"
    interim = tmp_path / "interim.parquet"
    transactions = tmp_path / "transactions.parquet"
    manifest_path = tmp_path / "manifest.parquet"
    summary_path = tmp_path / "summary.json"
    raw.to_csv(csv_path, index=False)

    step_optimize(csv_path, interim)
    processed = step_split(interim, transactions, manifest_path, summary_path)
    summary = read_json(summary_path)
    assert [summary[name]["n_rows"] for name in ("train", "validation", "test")] == [14, 3, 3]
    assert summary == validate_split(processed)
    assert processed.loc[processed["split"] == "train", "Time"].max() == 13
    assert processed.loc[processed["split"] == "validation", "Time"].min() == 14
    assert processed.loc[processed["split"] == "test", "Time"].min() == 17

    manifest = load_split_manifest(manifest_path)
    assert manifest[ID_COLUMN].tolist() == processed[ID_COLUMN].tolist()
    ids = []
    for name in ("train", "validation", "test"):
        X, y = load_xy(name, path=transactions)
        expected = processed[processed["split"] == name]
        assert X.index.tolist() == expected[ID_COLUMN].tolist()
        assert y.tolist() == expected["Class"].tolist()
        ids.extend(X.index)
    assert len(ids) == len(set(ids)) == len(processed)

    db_path = tmp_path / "fraud.db"
    build_database(transactions, db_path)
    engine = get_engine(db_path)
    try:
        results = run_all(engine)
        assert results["q12_duplicate_transactions"].empty
        assert all(crosscheck_with_pandas(engine, processed).values())
        stored_ids = pd.read_sql_query("SELECT transaction_id FROM transactions ORDER BY transaction_id", engine)
        assert stored_ids[ID_COLUMN].tolist() == processed[ID_COLUMN].tolist()
    finally:
        engine.dispose()


@pytest.mark.parametrize("invalid", ["duplicate", "negative", "fractional", "missing", "out_of_order"])
def test_validate_split_rejects_invalid_original_ids(invalid):
    unique = remove_duplicate_transactions(make_raw_with_duplicates())
    processed = build_processed_frame(optimize_dtypes(unique))
    if invalid == "duplicate":
        processed.loc[1, ID_COLUMN] = processed.loc[0, ID_COLUMN]
    elif invalid == "negative":
        processed.loc[0, ID_COLUMN] = -1
    elif invalid == "fractional":
        processed[ID_COLUMN] = processed[ID_COLUMN].astype("float64") + 0.5
    elif invalid == "missing":
        processed[ID_COLUMN] = processed[ID_COLUMN].astype("Int32")
        processed.loc[0, ID_COLUMN] = pd.NA
    else:
        processed.loc[[0, 1], ID_COLUMN] = processed.loc[[1, 0], ID_COLUMN].to_numpy()
    with pytest.raises(ValueError, match="transaction_id"):
        validate_split(processed)


def test_split_rejects_old_interim_without_original_ids(tmp_path):
    old_path = save_parquet(optimize_dtypes(make_raw_with_duplicates()), tmp_path / "old.parquet")
    with pytest.raises(ValueError, match="optimize"):
        step_split(old_path)
