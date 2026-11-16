import numpy as np
import pandas as pd
import pytest

from src.config import AMOUNT_COLUMN, RAW_COLUMNS, TIME_COLUMN, V_COLUMNS
from src.data.loading import DataLoadError, load_csv_chunked, read_raw_csv
from src.data.memory import (
    OPTIMIZED_DTYPES,
    cast_report,
    compare_memory,
    max_relative_error,
    memory_mib,
    optimize_dtypes,
)


def test_optimize_dtypes_applies_policy_and_saves_memory(raw_frame):
    optimized = optimize_dtypes(raw_frame)
    for column, dtype in OPTIMIZED_DTYPES.items():
        assert str(optimized[column].dtype) == dtype
    assert memory_mib(optimized) < memory_mib(raw_frame)
    # the input frame is not modified
    assert str(raw_frame[TIME_COLUMN].dtype) == "float64"


def test_optimize_dtypes_is_lossless_for_ints_and_amount(raw_frame):
    optimized = optimize_dtypes(raw_frame)
    assert np.array_equal(optimized[TIME_COLUMN].to_numpy(), raw_frame[TIME_COLUMN].to_numpy())
    assert np.array_equal(optimized[AMOUNT_COLUMN].to_numpy(), raw_frame[AMOUNT_COLUMN].to_numpy())
    assert np.array_equal(optimized["Class"].to_numpy(), raw_frame["Class"].to_numpy())
    for column in V_COLUMNS:
        assert max_relative_error(raw_frame[column], optimized[column]) < 1e-6


def test_optimize_dtypes_refuses_unsafe_casts(raw_frame):
    fractional_time = raw_frame.copy()
    fractional_time.loc[3, TIME_COLUMN] = 1.5
    with pytest.raises(ValueError, match="non-integer"):
        optimize_dtypes(fractional_time)

    huge_time = raw_frame.copy()
    huge_time.loc[3, TIME_COLUMN] = 3e9
    with pytest.raises(ValueError, match="does not fit"):
        optimize_dtypes(huge_time)

    tiny_difference = raw_frame.copy()
    tiny_difference["V1"] = 1.0 + 1e-9  # float32 cannot represent this -> relative error 1e-9 is fine
    optimize_dtypes(tiny_difference)  # ... so this must pass
    with pytest.raises(ValueError, match="loses precision"):
        optimize_dtypes(raw_frame, float_rtol=1e-12)


def test_cast_report_and_memory_summary(raw_frame):
    optimized = optimize_dtypes(raw_frame)
    report = {row["column"]: row for row in cast_report(raw_frame, optimized)}
    assert report[TIME_COLUMN]["dtype_before"] == "float64"
    assert report[TIME_COLUMN]["dtype_after"] == "int32"
    assert report[TIME_COLUMN]["values_identical"]
    assert report["V1"]["max_relative_error"] < 1e-6
    summary = compare_memory(raw_frame, optimized)
    assert summary["mib_saved"] > 0 and 0 < summary["saved_percent"] < 100


@pytest.mark.parametrize("chunksize", [1, 7, 100, 10_000])
def test_chunked_load_equals_optimized_full_load(raw_csv, chunksize):
    chunked = load_csv_chunked(raw_csv, chunksize=chunksize)
    full = optimize_dtypes(read_raw_csv(raw_csv))
    pd.testing.assert_frame_equal(chunked, full)


def test_scientific_notation_time_token_is_handled(tmp_path, raw_frame):
    path = tmp_path / "creditcard.csv"
    text = raw_frame.assign(**{TIME_COLUMN: raw_frame[TIME_COLUMN].astype(int)}).to_csv(index=False)
    lines = text.splitlines()
    first, rest = lines[1].split(",", 1)
    lines[1] = "1e+05," + rest  # same quirk as the real file
    path.write_text("\n".join(lines) + "\n")
    df = load_csv_chunked(path, chunksize=100)
    assert df.loc[0, TIME_COLUMN] == 100_000
    assert str(df[TIME_COLUMN].dtype) == "int32"


def test_loaders_raise_clear_errors(tmp_path):
    with pytest.raises(DataLoadError, match="not found"):
        read_raw_csv(tmp_path / "missing.csv")
    with pytest.raises(DataLoadError, match="not found"):
        load_csv_chunked(tmp_path / "missing.csv")

    wrong = tmp_path / "wrong.csv"
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(wrong, index=False)
    with pytest.raises(DataLoadError, match="missing columns"):
        read_raw_csv(wrong)

    empty = tmp_path / "empty.csv"
    empty.write_text("")
    with pytest.raises(DataLoadError):
        read_raw_csv(empty)


def test_raw_columns_order_is_enforced(raw_csv):
    df = read_raw_csv(raw_csv)
    assert list(df.columns) == RAW_COLUMNS
