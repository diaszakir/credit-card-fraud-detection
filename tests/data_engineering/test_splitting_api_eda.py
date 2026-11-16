import numpy as np
import pandas as pd
import pytest

from src.config import (
    FEATURE_COLUMNS,
    ID_COLUMN,
    PROCESSED_COLUMNS,
    SPLIT_COLUMN,
    SPLIT_NAMES,
    TIME_COLUMN,
)
from src.data.api import load_split_manifest, load_transactions, load_xy
from src.data.memory import optimize_dtypes
from src.data.splitting import (
    add_split_column,
    assign_transaction_ids,
    build_processed_frame,
    chronological_split,
    find_cut_positions,
    split_summary,
    validate_split,
)
from src.data.storage import save_parquet
from tests.data_engineering.conftest import make_raw_frame


@pytest.fixture
def processed():
    """Synthetic processed frame spanning ~5 hours so the hourly EDA has several buckets."""
    raw = make_raw_frame(n=3000, n_fraud=90, seed=3)
    rng = np.random.default_rng(3)
    raw[TIME_COLUMN] = np.sort(rng.integers(0, 5 * 3600, size=len(raw))).astype("float64")
    return build_processed_frame(optimize_dtypes(raw))


# ----------------------------------------------------------------------------- cut positions
def test_cut_is_moved_to_the_next_distinct_time():
    # 20 rows; nominal 70 % cut = 14, rows 13/14/15 share Time 7
    time = np.array([0, 0, 1, 1, 2, 3, 4, 5, 6, 6, 6, 6, 6, 7, 7, 7, 8, 9, 10, 11])
    first, second = find_cut_positions(time, 0.70, 0.15)
    assert first == 16 and time[first - 1] < time[first]  # moved past the tie at Time 7
    assert time[second - 1] < time[second] and first < second < len(time)


def test_cut_equals_nominal_position_when_no_tie():
    time = np.arange(100)
    assert find_cut_positions(time, 0.70, 0.15) == (70, 85)


@pytest.mark.parametrize("train,val", [(0.0, 0.1), (0.9, 0.2), (1.0, 0.1), (0.5, 0.0)])
def test_invalid_fractions_raise(train, val):
    with pytest.raises(ValueError):
        find_cut_positions(np.arange(100), train, val)


def test_unsorted_time_and_degenerate_data_raise():
    with pytest.raises(ValueError, match="sorted"):
        find_cut_positions(np.array([3, 1, 2, 4, 5, 6, 7, 8, 9, 10]), 0.7, 0.15)
    with pytest.raises(ValueError, match="non-empty"):
        find_cut_positions(np.zeros(50), 0.7, 0.15)  # one timestamp -> one split only


# ----------------------------------------------------------------------------- ids and labels
def test_ids_are_positions_and_unsorted_input_is_refused(raw_frame):
    with_ids = assign_transaction_ids(raw_frame)
    assert with_ids.columns[0] == ID_COLUMN
    assert str(with_ids[ID_COLUMN].dtype) == "int32"
    assert np.array_equal(with_ids[ID_COLUMN], np.arange(len(raw_frame)))
    with pytest.raises(ValueError, match="already present"):
        assign_transaction_ids(with_ids)
    with pytest.raises(ValueError, match="not ordered"):
        assign_transaction_ids(raw_frame.iloc[::-1].reset_index(drop=True))


def test_processed_frame_follows_the_contract(processed):
    assert list(processed.columns) == PROCESSED_COLUMNS
    assert list(processed[SPLIT_COLUMN].cat.categories) == list(SPLIT_NAMES)
    summary = validate_split(processed)
    assert sum(s["n_rows"] for s in summary.values()) == len(processed)
    # rows are in chronological order and each split is a contiguous block
    labels = processed[SPLIT_COLUMN].astype(str).tolist()
    assert labels == sorted(labels, key=SPLIT_NAMES.index)


def test_split_is_deterministic(processed):
    again = chronological_split(processed.drop(columns=[SPLIT_COLUMN]))
    pd.testing.assert_series_equal(again, processed[SPLIT_COLUMN])


def test_no_timestamp_is_shared_between_splits(processed):
    ranges = {n: processed.loc[processed[SPLIT_COLUMN] == n, TIME_COLUMN] for n in SPLIT_NAMES}
    assert ranges["train"].max() < ranges["validation"].min()
    assert ranges["validation"].max() < ranges["test"].min()


def test_validate_split_detects_violations(processed):
    shuffled = processed.sample(frac=1.0, random_state=0).reset_index(drop=True)
    with pytest.raises(ValueError, match="transaction_id"):
        validate_split(shuffled)

    no_fraud_in_test = processed.copy()
    test_rows = no_fraud_in_test[SPLIT_COLUMN] == "test"
    no_fraud_in_test.loc[test_rows, "Class"] = 0
    with pytest.raises(ValueError, match="both classes"):
        validate_split(no_fraud_in_test)

    overlapping = processed.copy()
    overlapping.loc[overlapping.index[-1], SPLIT_COLUMN] = "train"  # last row (latest Time) -> train
    with pytest.raises(ValueError, match="overlap"):
        validate_split(overlapping)


def test_add_split_column_position(raw_frame):
    framed = add_split_column(assign_transaction_ids(raw_frame))
    assert list(framed.columns[:2]) == [ID_COLUMN, SPLIT_COLUMN]


# ----------------------------------------------------------------------------- read API
def test_api_returns_consistent_views(tmp_path, processed):
    path = save_parquet(processed, tmp_path / "transactions.parquet", row_group_size=500)
    manifest_path = save_parquet(processed[[ID_COLUMN, SPLIT_COLUMN]], tmp_path / "manifest.parquet")

    X, y = load_xy("validation", path=path)
    expected = processed[processed[SPLIT_COLUMN] == "validation"]
    assert X.index.name == ID_COLUMN and list(X.columns) == FEATURE_COLUMNS
    assert np.array_equal(X.index, expected[ID_COLUMN]) and np.array_equal(y.to_numpy(), expected["Class"])

    everything = load_transactions(path=path)
    assert len(everything) == len(processed)
    only_test = load_transactions(split="test", columns=[ID_COLUMN, "Class"], path=path)
    assert list(only_test.columns) == [ID_COLUMN, "Class"]

    manifest = load_split_manifest(manifest_path)
    assert manifest[SPLIT_COLUMN].value_counts().to_dict() == processed[SPLIT_COLUMN].value_counts().to_dict()
    # the three splits never share an id
    ids = [set(load_xy(s, path=path)[0].index) for s in SPLIT_NAMES]
    assert not (ids[0] & ids[1]) and not (ids[1] & ids[2]) and not (ids[0] & ids[2])


def test_api_errors(tmp_path):
    with pytest.raises(ValueError, match="unknown split"):
        load_transactions(split="holdout", path=tmp_path / "x.parquet")
    with pytest.raises(FileNotFoundError, match="run-all"):
        load_xy("train", path=tmp_path / "missing.parquet")


# ----------------------------------------------------------------------------- EDA smoke test
def test_eda_smoke_creates_all_outputs(tmp_path, processed):
    from src.data.eda import hourly_profile, run_eda

    profile = hourly_profile(processed)
    assert profile["n_transactions"].sum() == len(processed)
    assert profile["n_fraud"].sum() == processed["Class"].sum()

    result = run_eda(processed, figures_dir=tmp_path / "fig", tables_dir=tmp_path / "tab")
    assert len(list((tmp_path / "fig").glob("*.png"))) == 7
    assert len(list((tmp_path / "fig").glob("*.html"))) == 2
    assert (tmp_path / "tab" / "eda_summary.json").is_file()
    assert result["split_summary"] == split_summary(processed)
