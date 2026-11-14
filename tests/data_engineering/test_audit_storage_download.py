import numpy as np
import pandas as pd
import pytest

from src.config import RAW_COLUMNS, V_COLUMNS
from src.data.audit import audit_dataframe, iqr_outlier_table, scan_time_tokens
from src.data.download import DownloadError, download_dataset, sha256_of, verify_dataset
from src.data.storage import load_parquet, save_parquet


def _tiny_frame() -> pd.DataFrame:
    """4 rows with a known duplicate, a zero amount and one timestamp shared by 2 rows."""
    base = {c: [0.1, 0.1, 0.5, 0.9] for c in V_COLUMNS}
    return pd.DataFrame(
        {
            "Time": [0.0, 0.0, 1.0, 2.0],
            **base,
            "Amount": [10.0, 10.0, 0.0, 5.5],
            "Class": [0, 0, 1, 0],
        }
    )[RAW_COLUMNS]


def test_audit_counts_known_quantities():
    audit = audit_dataframe(_tiny_frame())
    assert audit["overview"]["n_rows"] == 4
    assert audit["missing_values"]["total_missing"] == 0
    assert audit["duplicates"]["n_repeated_rows"] == 1
    assert audit["duplicates"]["n_rows_in_duplicate_groups"] == 2
    assert audit["duplicates"]["repeated_rows_among_fraud"] == 0
    assert audit["class_balance"]["n_fraud"] == 1
    assert audit["class_balance"]["fraud_rate_percent"] == pytest.approx(25.0)
    assert audit["amount"]["n_zero"] == 1 and audit["amount"]["n_negative"] == 0
    assert audit["time"]["is_sorted_non_decreasing"] is True
    assert audit["time"]["max_rows_sharing_one_timestamp"] == 2
    assert any("share one timestamp" in note for note in audit["findings"])


def test_audit_detects_missing_values_and_unsorted_time():
    df = _tiny_frame()
    df.loc[1, "V3"] = np.nan
    df.loc[0, "Time"] = 5.0
    audit = audit_dataframe(df)
    assert audit["missing_values"]["columns_with_missing"] == {"V3": 1}
    assert audit["time"]["is_sorted_non_decreasing"] is False
    assert any("NOT ordered" in note for note in audit["findings"])


def test_audit_rejects_wrong_schema():
    with pytest.raises(ValueError, match="columns missing"):
        audit_dataframe(pd.DataFrame({"a": [1]}))


def test_audit_with_source_file_reports_hash_and_time_tokens(raw_csv, raw_frame):
    audit = audit_dataframe(raw_frame, source_path=raw_csv)
    assert audit["source_file"]["sha256"] == sha256_of(raw_csv)
    assert scan_time_tokens(raw_csv)["n_tokens"] == len(raw_frame)


def test_iqr_outlier_table_flags_obvious_outlier():
    df = pd.DataFrame({"x": [1.0] * 20 + [1000.0], "Class": [0] * 21})
    table = iqr_outlier_table(df)
    assert table.loc[0, "n_outliers"] == 1


def test_parquet_roundtrip_projection_and_filter(tmp_path, raw_frame):
    df = raw_frame.assign(split=np.where(np.arange(len(raw_frame)) < 400, "train", "test"))
    path = save_parquet(df, tmp_path / "x.parquet", row_group_size=100)
    pd.testing.assert_frame_equal(load_parquet(path), df, check_dtype=False)
    assert list(load_parquet(path, columns=["Time", "Class"]).columns) == ["Time", "Class"]
    only_test = load_parquet(path, filters=[("split", "==", "test")])
    assert len(only_test) == 200 and set(only_test["split"]) == {"test"}
    with pytest.raises(FileNotFoundError):
        load_parquet(tmp_path / "missing.parquet")


def test_verify_dataset_reports_problems(raw_csv, raw_frame, tmp_path):
    ok = verify_dataset(raw_csv, expected_rows=len(raw_frame), expected_fraud=int(raw_frame["Class"].sum()))
    assert ok["ok"] and ok["problems"] == []
    bad = verify_dataset(raw_csv, expected_rows=1, expected_fraud=1)
    assert not bad["ok"] and len(bad["problems"]) == 2
    missing = verify_dataset(tmp_path / "nope.csv")
    assert not missing["ok"] and "not found" in missing["problems"][0]


def test_download_uses_urls_and_verifies(raw_csv, raw_frame, tmp_path):
    dest = tmp_path / "out" / "creditcard.csv"
    expected = dict(expected_rows=len(raw_frame), expected_fraud=int(raw_frame["Class"].sum()))
    path = download_dataset(dest, urls=[raw_csv.as_uri()], **expected)
    assert path == dest and sha256_of(dest) == sha256_of(raw_csv)
    # second call is a no-op because the file is already verified
    assert download_dataset(dest, urls=["file:///does/not/exist"], **expected) == dest


def test_download_falls_back_to_next_url_then_fails_with_help(raw_csv, raw_frame, tmp_path):
    expected = dict(expected_rows=len(raw_frame), expected_fraud=int(raw_frame["Class"].sum()))
    dest = tmp_path / "creditcard.csv"
    download_dataset(dest, urls=["file:///does/not/exist.csv", raw_csv.as_uri()], **expected)
    assert dest.is_file()

    other = tmp_path / "other" / "creditcard.csv"
    with pytest.raises(DownloadError, match="kaggle"):
        download_dataset(other, urls=["file:///does/not/exist.csv"], **expected)
    assert not other.exists()

    with pytest.raises(DownloadError, match="verification failed"):
        download_dataset(tmp_path / "wrong" / "creditcard.csv", urls=[raw_csv.as_uri()], expected_rows=5)
