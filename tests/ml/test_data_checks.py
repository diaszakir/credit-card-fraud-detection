"""Synthetic data only: schema, manifests and expanding time folds."""

import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS, ID_COLUMN, PROCESSED_COLUMNS, SPLIT_COLUMN, TIME_COLUMN
from src.data.download import sha256_of
from src.data.io_utils import read_json, write_json
from src.data.memory import optimize_dtypes
from src.data.splitting import build_processed_frame, remove_duplicate_transactions, split_summary
from src.data.storage import save_parquet
from src.ml.data_checks import (
    check_ml_frame,
    main,
    make_time_folds,
    review_repeated_patterns,
    shared_feature_patterns,
    summarize_time_folds,
    validate_ml_data,
)
from tests.data_engineering.conftest import make_raw_frame


@pytest.fixture
def data_files(tmp_path):
    raw = make_raw_frame(n=1200, n_fraud=240)
    raw[TIME_COLUMN] = np.arange(len(raw)) // 2
    raw["Class"] = (np.arange(len(raw)) % 5 == 0).astype("int64")
    raw = pd.concat([raw.iloc[:16], raw.iloc[[15]], raw.iloc[16:]], ignore_index=True)
    raw_path = tmp_path / "raw.csv"
    raw.to_csv(raw_path, index=False)
    processed = build_processed_frame(optimize_dtypes(remove_duplicate_transactions(raw)))
    transactions_path = save_parquet(processed, tmp_path / "transactions.parquet")
    manifest_path = save_parquet(processed[[ID_COLUMN, SPLIT_COLUMN]], tmp_path / "manifest.parquet")
    audit_path = write_json({
        "overview": {"n_rows": len(raw)},
        "duplicates": {"n_repeated_rows": 1},
        "source_file": {"sha256": sha256_of(raw_path)},
    }, tmp_path / "audit.json")
    summary_path = write_json(split_summary(processed), tmp_path / "summary.json")
    return {
        "transactions_path": transactions_path,
        "manifest_path": manifest_path,
        "audit_path": audit_path,
        "raw_path": raw_path,
        "summary_path": summary_path,
    }


def test_data_check_keeps_test_rows_closed(data_files, monkeypatch):
    import src.ml.data_checks as checks

    original = checks.load_transactions
    requested_splits = []

    def guarded_loader(*, split, path):
        requested_splits.append(split)
        assert split in ("train", "validation")
        return original(split=split, path=path)

    monkeypatch.setattr(checks, "load_transactions", guarded_loader)
    report = validate_ml_data(**data_files)
    assert requested_splits == ["train", "validation"]
    assert report["status"] == "ok" and report["test_rows_loaded"] is False
    assert report["feature_names"] == FEATURE_COLUMNS
    assert report["cv"]["n_splits"] == 3
    assert len(report["cv"]["folds_sha256"]) == 64
    assert "n_fraud" not in report["split_metadata"]["test"]


@pytest.mark.parametrize("problem", ["missing_column", "dtype", "infinite", "target", "ids", "time"])
def test_frame_checks_reject_invalid_inputs(data_files, problem):
    frame = pd.read_parquet(data_files["transactions_path"], filters=[("split", "==", "train")])
    if problem == "missing_column":
        frame = frame.drop(columns=["V1"])
    elif problem == "dtype":
        frame["V1"] = frame["V1"].astype("float64")
    elif problem == "infinite":
        frame.loc[0, "V1"] = np.inf
    elif problem == "target":
        frame.loc[0, "Class"] = 3
    elif problem == "ids":
        frame.loc[1, ID_COLUMN] = frame.loc[0, ID_COLUMN]
    else:
        frame.loc[0, TIME_COLUMN] = frame[TIME_COLUMN].max()
    with pytest.raises(ValueError):
        check_ml_frame(frame, "train")


def test_time_folds_are_deterministic_expanding_and_do_not_split_timestamps(data_files):
    train = pd.read_parquet(data_files["transactions_path"], filters=[("split", "==", "train")])
    folds = make_time_folds(train)
    repeated = make_time_folds(train)
    previous_training = set()
    for (training, validation), (same_train, same_val) in zip(folds, repeated):
        assert np.array_equal(training, same_train) and np.array_equal(validation, same_val)
        assert not set(training) & set(validation)
        assert previous_training.issubset(set(training))
        assert train.iloc[training][TIME_COLUMN].max() < train.iloc[validation][TIME_COLUMN].min()
        assert set(train.iloc[training]["Class"]) == {0, 1}
        assert set(train.iloc[validation]["Class"]) == {0, 1}
        assert int(train.iloc[training]["Class"].sum()) >= 6
        previous_training = set(training)
    summaries = summarize_time_folds(train, folds)
    assert summaries == summarize_time_folds(train, repeated)
    changed_ids = train.copy()
    changed_ids[ID_COLUMN] += 10000
    changed = summarize_time_folds(changed_ids, folds)
    assert summaries[0]["ids_sha256"] != changed[0]["ids_sha256"]


def test_folds_reject_validation_data_and_insufficient_smote_examples(data_files):
    train = pd.read_parquet(data_files["transactions_path"], filters=[("split", "==", "train")])
    validation = pd.read_parquet(data_files["transactions_path"], filters=[("split", "==", "validation")])
    with pytest.raises(ValueError, match="only the train"):
        make_time_folds(validation)
    scarce = train.copy()
    scarce["Class"] = 0
    scarce.loc[scarce.index[:5], "Class"] = 1
    with pytest.raises(ValueError, match="SMOTE"):
        make_time_folds(scarce)
    train["Class"] = 0
    with pytest.raises(ValueError, match="both classes"):
        make_time_folds(train)


@pytest.mark.parametrize("n_splits", [0, 1, True, 2.5])
def test_invalid_fold_count_is_rejected(data_files, n_splits):
    frame = pd.read_parquet(data_files["transactions_path"], filters=[("split", "==", "train")])
    with pytest.raises(ValueError, match="n_splits"):
        make_time_folds(frame, n_splits=n_splits)


def test_shared_patterns_are_reported_without_deleting_rows(data_files):
    processed = pd.read_parquet(data_files["transactions_path"])
    train = processed[processed["split"] == "train"].copy()
    validation = processed[processed["split"] == "validation"].copy()
    validation.loc[validation.index[0], FEATURE_COLUMNS] = train.iloc[0][FEATURE_COLUMNS].to_numpy()
    size_before = len(train) + len(validation)
    result = shared_feature_patterns(train, validation)
    assert result["shared_patterns"] == 1
    assert result["train_rows"] == result["validation_rows"] == 1
    assert result["status"] == "review_required"
    assert len(train) + len(validation) == size_before


def test_manifest_repeated_ids_and_old_raw_hash_are_rejected(data_files):
    manifest = pd.read_parquet(data_files["manifest_path"])
    manifest.loc[1, ID_COLUMN] = manifest.loc[0, ID_COLUMN]
    save_parquet(manifest, data_files["manifest_path"])
    with pytest.raises(ValueError, match="manifest transaction_id"):
        validate_ml_data(**data_files)
    data_files["raw_path"].write_text("changed source\n")
    with pytest.raises(ValueError, match="SHA-256"):
        validate_ml_data(**data_files)


def test_metadata_time_overlap_is_rejected_without_opening_test(data_files):
    summary = read_json(data_files["summary_path"])
    summary["test"]["time_min"] = summary["validation"]["time_max"]
    write_json(summary, data_files["summary_path"])
    with pytest.raises(ValueError, match="Time ranges overlap"):
        validate_ml_data(**data_files)


def test_cli_writes_reproducible_json_report(data_files, tmp_path):
    output = tmp_path / "checks.json"
    args = ["--output", str(output)]
    for flag, name in (("transactions", "transactions_path"), ("manifest", "manifest_path"),
                       ("audit", "audit_path"), ("raw", "raw_path"), ("summary", "summary_path")):
        args.extend(["--" + flag, str(data_files[name])])
    assert main(args) == 0
    report = read_json(output)
    assert report["status"] == "ok" and report["test_rows_loaded"] is False
    assert report["raw_sha256"] == sha256_of(data_files["raw_path"])
    assert main(args) == 0
    assert read_json(output) == report


def add_recurring_features(data_files, problem=None):
    """Create one recurring synthetic operation at a later Time."""
    raw = pd.read_csv(data_files["raw_path"])
    processed = pd.read_parquet(data_files["transactions_path"])
    train_row = processed[(processed["split"] == "train") & (processed["Class"] == 0)].iloc[0]
    validation_row = processed[(processed["split"] == "validation") & (processed["Class"] == 0)].iloc[0]
    if problem == "same_time":
        train_row = processed[(processed["split"] == "train") & (processed["Class"] == 0)].iloc[-1]
    train_id, validation_id = int(train_row[ID_COLUMN]), int(validation_row[ID_COLUMN])
    raw.loc[validation_id, FEATURE_COLUMNS] = raw.loc[train_id, FEATURE_COLUMNS].to_numpy()
    if problem == "rounding":
        raw.loc[train_id, "V1"] = 1.0
        raw.loc[validation_id, "V1"] = 1.0 + 1e-9
    elif problem == "labels":
        raw.loc[validation_id, "Class"] = 1
    elif problem == "same_time":
        raw.loc[train_id, TIME_COLUMN] = raw.loc[validation_id, TIME_COLUMN]
    for source_id in (train_id, validation_id):
        mask = processed[ID_COLUMN] == source_id
        for column in FEATURE_COLUMNS:
            processed.loc[mask, column] = np.asarray(raw.loc[source_id, column], dtype=processed[column].dtype)
        processed.loc[mask, "Class"] = int(raw.loc[source_id, "Class"])
        processed.loc[mask, TIME_COLUMN] = int(raw.loc[source_id, TIME_COLUMN])
    raw.to_csv(data_files["raw_path"], index=False)
    save_parquet(processed, data_files["transactions_path"])
    audit = read_json(data_files["audit_path"])
    audit["source_file"]["sha256"] = sha256_of(data_files["raw_path"])
    write_json(audit, data_files["audit_path"])
    return processed[processed["split"] == "train"], processed[processed["split"] == "validation"]


def test_recurring_review_reads_only_raw_prefix_and_accepts_distinct_times(data_files, monkeypatch):
    train, validation = add_recurring_features(data_files)
    original = pd.read_csv
    expected_rows = int(validation[ID_COLUMN].max()) + 1
    requested_rows = []

    def guarded_read_csv(path, **kwargs):
        assert kwargs["nrows"] == expected_rows
        requested_rows.append(kwargs["nrows"])
        return original(path, **kwargs)

    monkeypatch.setattr("src.ml.data_checks.pd.read_csv", guarded_read_csv)
    review = review_repeated_patterns(train, validation, data_files["raw_path"])
    assert requested_rows == [expected_rows]
    assert review["decision"] == "accepted_recurring_features"
    assert review["raw_prefix_rows_read"] == expected_rows
    assert review["raw_prefix_last_transaction_id"] == expected_rows - 1
    assert review["test_rows_loaded"] is False
    assert review["pair_count"] == review["raw_exact_feature_pairs"] == 1
    assert review["class_pair_counts"]["0->0"] == 1
    assert review["rounding_only_pairs"] == review["conflicting_label_pairs"] == 0
    assert review["time_gap_min_seconds"] > 0
    assert review["novel_validation_rows"] == len(validation) - 1


def test_retaining_recurring_features_requires_explicit_review_flag(data_files):
    add_recurring_features(data_files)
    manifest_before = sha256_of(data_files["manifest_path"])
    assert validate_ml_data(**data_files)["status"] == "needs_duplicate_review"
    report = validate_ml_data(**data_files, review_recurring=True)
    assert report["status"] == "ok_with_recurring_features"
    assert report["recurring_feature_policy"] == "chronological_recurring_features_retained"
    assert report["selection_population"] == "full_validation"
    assert report["novel_validation_sensitivity_required"] is True
    assert sha256_of(data_files["manifest_path"]) == manifest_before


@pytest.mark.parametrize("problem, field", [("rounding", "rounding_only_pairs"), ("labels", "conflicting_label_pairs")])
def test_rounding_only_or_conflicting_labels_remain_for_review(data_files, problem, field):
    add_recurring_features(data_files, problem)
    report = validate_ml_data(**data_files, review_recurring=True)
    assert report["status"] == "needs_duplicate_review"
    assert report["recurring_feature_policy"] is None
    assert report["recurring_feature_review"]["decision"] == "needs_review"
    assert report["recurring_feature_review"][field] == 1


def test_same_time_pairs_cannot_be_accepted_as_recurring_operations(data_files):
    train, validation = add_recurring_features(data_files, "same_time")
    review = review_repeated_patterns(train, validation, data_files["raw_path"])
    assert review["decision"] == "needs_review"
    assert review["nonpositive_time_gap_pairs"] == 1


def test_recurring_review_detects_processed_source_label_mismatch(data_files):
    train, validation = add_recurring_features(data_files)
    validation = validation.copy()
    repeated_index = validation.loc[validation["Class"] == 0].index[0]
    validation.loc[repeated_index, "Class"] = 1
    with pytest.raises(ValueError, match="raw Time or Class"):
        review_repeated_patterns(train, validation, data_files["raw_path"])


def test_cli_exposes_explicit_recurring_review_flag(data_files, tmp_path):
    add_recurring_features(data_files)
    output = tmp_path / "recurring_checks.json"
    args = ["--output", str(output), "--review-recurring"]
    for flag, name in (("transactions", "transactions_path"), ("manifest", "manifest_path"),
                       ("audit", "audit_path"), ("raw", "raw_path"), ("summary", "summary_path")):
        args.extend(["--" + flag, str(data_files[name])])
    assert main(args) == 0
    report = read_json(output)
    assert report["status"] == "ok_with_recurring_features"
    assert report["test_rows_loaded"] is False
