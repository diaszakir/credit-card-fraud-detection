"""Check ML inputs and build time folds without loading test features or labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import (
    AMOUNT_COLUMN,
    AUDIT_JSON_PATH,
    EXPERIMENTS_DIR,
    FEATURE_COLUMNS,
    ID_COLUMN,
    PROCESSED_COLUMNS,
    RAW_COLUMNS,
    RAW_CSV_PATH,
    SPLIT_COLUMN,
    SPLIT_MANIFEST_PATH,
    SPLIT_NAMES,
    SPLIT_SUMMARY_JSON_PATH,
    TARGET_COLUMN,
    TIME_COLUMN,
    TRANSACTIONS_PARQUET_PATH,
    V_COLUMNS,
)
from src.data.api import load_split_manifest, load_transactions
from src.data.download import sha256_of
from src.data.io_utils import read_json, write_json


def check_ml_frame(frame: pd.DataFrame, split: str) -> None:
    """Validate one loaded train/validation frame before any model sees it."""
    if list(frame.columns) != PROCESSED_COLUMNS:
        raise ValueError(f"{split}: columns must follow PROCESSED_COLUMNS")
    if frame.empty:
        raise ValueError(f"{split}: empty split")

    dtypes = {ID_COLUMN: "int32", TIME_COLUMN: "int32", TARGET_COLUMN: "int8"}
    dtypes.update({name: "float32" for name in V_COLUMNS})
    dtypes[AMOUNT_COLUMN] = "float64"
    for name, dtype in dtypes.items():
        if str(frame[name].dtype) != dtype:
            raise ValueError(f"{split}: {name} must have dtype {dtype}")

    ids = frame[ID_COLUMN]
    if ids.isna().any() or (ids < 0).any() or ids.duplicated().any() or not ids.is_monotonic_increasing:
        raise ValueError(f"{split}: transaction_id must be unique, increasing and non-negative")
    if not frame[TIME_COLUMN].is_monotonic_increasing or (frame[TIME_COLUMN] < 0).any():
        raise ValueError(f"{split}: Time must be increasing and non-negative")
    if not frame[SPLIT_COLUMN].eq(split).all():
        raise ValueError(f"{split}: unexpected split labels")
    if set(frame[TARGET_COLUMN].unique()) != {0, 1}:
        raise ValueError(f"{split}: Class must contain both labels 0 and 1")
    if not np.isfinite(frame[FEATURE_COLUMNS].to_numpy(dtype="float64")).all():
        raise ValueError(f"{split}: features contain missing or infinite values")
    if (frame[AMOUNT_COLUMN] < 0).any():
        raise ValueError(f"{split}: Amount cannot be negative")


def make_time_folds(
    train: pd.DataFrame,
    n_splits: int = 3,
    k_neighbors: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Use expanding training windows and equal time intervals inside train only.

    For three folds the time range is divided into four intervals: 1 -> 2,
    1+2 -> 3, 1+2+3 -> 4. Positions refer to the supplied row order.
    """
    if not isinstance(n_splits, int) or isinstance(n_splits, bool) or n_splits < 2:
        raise ValueError("n_splits must be an integer of at least 2")
    if not isinstance(k_neighbors, int) or isinstance(k_neighbors, bool) or k_neighbors < 1:
        raise ValueError("k_neighbors must be a positive integer")
    if train.empty or not {TIME_COLUMN, TARGET_COLUMN, ID_COLUMN}.issubset(train.columns):
        raise ValueError("train must contain Time, Class and transaction_id")
    if SPLIT_COLUMN in train.columns and not train[SPLIT_COLUMN].eq("train").all():
        raise ValueError("CV may use only the train split")

    time = train[TIME_COLUMN].to_numpy(dtype="float64")
    labels = train[TARGET_COLUMN].to_numpy()
    if not np.isfinite(time).all() or np.any(np.diff(time) < 0):
        raise ValueError("train Time must be finite and sorted")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("train Class must contain only labels 0 and 1")

    edges = np.linspace(time[0], time[-1] + 1, n_splits + 2)
    folds = []
    for number in range(n_splits):
        train_positions = np.flatnonzero(time < edges[number + 1])
        validation_positions = np.flatnonzero(
            (time >= edges[number + 1]) & (time < edges[number + 2])
        )
        if not len(train_positions) or not len(validation_positions):
            raise ValueError(f"fold {number + 1}: empty training or validation interval")
        for name, positions in (("train", train_positions), ("validation", validation_positions)):
            counts = np.bincount(labels[positions].astype(int), minlength=2)
            if (counts == 0).any():
                raise ValueError(f"fold {number + 1}: {name} must contain both classes")
            if name == "train" and counts.min() <= k_neighbors:
                raise ValueError(
                    f"fold {number + 1}: SMOTE needs at least {k_neighbors + 1} "
                    "examples of each class in its training interval"
                )
        folds.append((train_positions, validation_positions))
    return folds


def summarize_time_folds(train: pd.DataFrame, folds: list[tuple[np.ndarray, np.ndarray]]) -> list[dict]:
    """Describe each fold and hash its exact ordered transaction ids."""
    summaries = []
    for number, (train_positions, validation_positions) in enumerate(folds, start=1):
        parts = {"train": train.iloc[train_positions], "validation": train.iloc[validation_positions]}
        ids = {name: part[ID_COLUMN].astype("int64").tolist() for name, part in parts.items()}
        digest = hashlib.sha256(json.dumps(ids, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        summary = {"fold": number, "ids_sha256": digest}
        for name, part in parts.items():
            summary[name] = {
                "n_rows": len(part),
                "n_fraud": int(part[TARGET_COLUMN].sum()),
                "n_legitimate": int((part[TARGET_COLUMN] == 0).sum()),
                "time_min": int(part[TIME_COLUMN].min()),
                "time_max": int(part[TIME_COLUMN].max()),
                "id_min": int(part[ID_COLUMN].min()),
                "id_max": int(part[ID_COLUMN].max()),
            }
        summaries.append(summary)
    return summaries


def shared_feature_patterns(train: pd.DataFrame, validation: pd.DataFrame) -> dict:
    """Report equal model inputs across splits; do not silently remove any rows."""
    train_hash = pd.util.hash_pandas_object(train[FEATURE_COLUMNS], index=False)
    validation_hash = pd.util.hash_pandas_object(validation[FEATURE_COLUMNS], index=False)
    common = np.intersect1d(train_hash.to_numpy(), validation_hash.to_numpy())
    left = train.loc[train_hash.isin(common), FEATURE_COLUMNS]
    right = validation.loc[validation_hash.isin(common), FEATURE_COLUMNS]
    # Verify candidate hash matches using the actual values, so a hash collision is not a duplicate.
    shared = left.drop_duplicates().merge(right.drop_duplicates(), on=FEATURE_COLUMNS, how="inner")
    n_train = len(left.merge(shared, on=FEATURE_COLUMNS, how="inner", validate="many_to_one"))
    n_validation = len(right.merge(shared, on=FEATURE_COLUMNS, how="inner", validate="many_to_one"))
    return {
        "shared_patterns": len(shared),
        "train_rows": n_train,
        "validation_rows": n_validation,
        "status": "review_required" if len(shared) else "none",
        "scope": "29 model features only; Time and Class are excluded",
    }


def review_repeated_patterns(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    raw_path: Path = RAW_CSV_PATH,
) -> dict:
    """Review recurring inputs using only the raw prefix ending at validation.

    Equal vectors at different times may be distinct future operations. This
    report makes that policy explicit without changing the approved split.
    """
    check_ml_frame(train, "train")
    check_ml_frame(validation, "validation")
    columns = [ID_COLUMN, TIME_COLUMN, TARGET_COLUMN, *FEATURE_COLUMNS]
    pairs = train[columns].merge(
        validation[columns], on=FEATURE_COLUMNS, suffixes=("_train", "_validation")
    )
    result = {
        "pair_count": len(pairs),
        "class_pair_counts": {},
        "raw_exact_feature_pairs": 0,
        "rounding_only_pairs": 0,
        "conflicting_label_pairs": 0,
        "nonpositive_time_gap_pairs": 0,
        "time_gap_min_seconds": None,
        "time_gap_max_seconds": None,
        "raw_prefix_rows_read": 0,
        "raw_prefix_last_transaction_id": None,
        "test_rows_loaded": False,
        "novel_validation_rows": len(validation),
        "decision": "no_shared_features",
    }
    if pairs.empty:
        return result

    last_id = int(validation[ID_COLUMN].max())
    raw = pd.read_csv(raw_path, nrows=last_id + 1, usecols=RAW_COLUMNS)
    if len(raw) != last_id + 1:
        raise ValueError("raw prefix does not cover all validation transaction ids")
    result["raw_prefix_rows_read"] = len(raw)
    result["raw_prefix_last_transaction_id"] = last_id
    train_ids = pairs[f"{ID_COLUMN}_train"].to_numpy()
    validation_ids = pairs[f"{ID_COLUMN}_validation"].to_numpy()

    for name, ids, frame in (("train", train_ids, train), ("validation", validation_ids, validation)):
        source = raw.iloc[np.unique(ids)]
        stored = frame.set_index(ID_COLUMN).loc[np.unique(ids)]
        # A recurring-feature decision must refer to the actual original rows.
        if not np.array_equal(source[TIME_COLUMN].to_numpy(), stored[TIME_COLUMN].to_numpy()) or not np.array_equal(
            source[TARGET_COLUMN].to_numpy(), stored[TARGET_COLUMN].to_numpy()
        ):
            raise ValueError(f"{name}: repeated rows disagree with raw Time or Class")
        cast = source[FEATURE_COLUMNS].astype({column: frame[column].dtype for column in FEATURE_COLUMNS})
        if not np.array_equal(cast.to_numpy(), stored[FEATURE_COLUMNS].to_numpy()):
            raise ValueError(f"{name}: repeated feature values disagree with the raw source")

    source_train = raw.iloc[train_ids][FEATURE_COLUMNS].to_numpy(dtype="float64")
    source_validation = raw.iloc[validation_ids][FEATURE_COLUMNS].to_numpy(dtype="float64")
    equal = np.all(source_train == source_validation, axis=1)
    gaps = pairs[f"{TIME_COLUMN}_validation"] - pairs[f"{TIME_COLUMN}_train"]
    train_labels = pairs[f"{TARGET_COLUMN}_train"]
    validation_labels = pairs[f"{TARGET_COLUMN}_validation"]
    result["raw_exact_feature_pairs"] = int(equal.sum())
    result["rounding_only_pairs"] = int((~equal).sum())
    result["conflicting_label_pairs"] = int((train_labels != validation_labels).sum())
    result["nonpositive_time_gap_pairs"] = int((gaps <= 0).sum())
    result["time_gap_min_seconds"] = int(gaps.min())
    result["time_gap_max_seconds"] = int(gaps.max())
    result["novel_validation_rows"] = len(validation) - len(np.unique(validation_ids))
    for before in (0, 1):
        for after in (0, 1):
            result["class_pair_counts"][f"{before}->{after}"] = int(
                ((train_labels == before) & (validation_labels == after)).sum()
            )
    accepted = (
        result["rounding_only_pairs"] == 0
        and result["conflicting_label_pairs"] == 0
        and result["nonpositive_time_gap_pairs"] == 0
    )
    result["decision"] = "accepted_recurring_features" if accepted else "needs_review"
    return result


def validate_ml_data(
    transactions_path: Path = TRANSACTIONS_PARQUET_PATH,
    manifest_path: Path = SPLIT_MANIFEST_PATH,
    audit_path: Path = AUDIT_JSON_PATH,
    raw_path: Path = RAW_CSV_PATH,
    summary_path: Path = SPLIT_SUMMARY_JSON_PATH,
    n_splits: int = 3,
    k_neighbors: int = 5,
    review_recurring: bool = False,
) -> dict:
    """Check provenance, manifest and train/validation; keep test row values closed.

    Test boundaries are checked from the saved split summary, not by loading test
    features or labels. The original and processed files are hashed as opaque bytes.
    """
    audit = read_json(audit_path)
    summary = read_json(summary_path)
    raw_sha256 = sha256_of(raw_path)
    if raw_sha256 != audit.get("source_file", {}).get("sha256"):
        raise ValueError("raw CSV SHA-256 does not match the saved audit")

    parquet = pq.ParquetFile(transactions_path)
    if parquet.schema_arrow.names != PROCESSED_COLUMNS:
        raise ValueError("processed Parquet columns do not match the data contract")
    expected_types = {ID_COLUMN: pa.int32(), TIME_COLUMN: pa.int32(), TARGET_COLUMN: pa.int8()}
    expected_types.update({name: pa.float32() for name in V_COLUMNS})
    expected_types[AMOUNT_COLUMN] = pa.float64()
    for name, dtype in expected_types.items():
        if parquet.schema_arrow.field(name).type != dtype:
            raise ValueError(f"processed Parquet: {name} must have type {dtype}")

    manifest = load_split_manifest(manifest_path)
    if list(manifest.columns) != [ID_COLUMN, SPLIT_COLUMN] or manifest.empty:
        raise ValueError("manifest must contain transaction_id and split")
    ids = manifest[ID_COLUMN]
    if (
        str(ids.dtype) != "int32" or ids.isna().any() or (ids < 0).any()
        or ids.duplicated().any() or not ids.is_monotonic_increasing
    ):
        raise ValueError("manifest transaction_id must contain unique increasing non-negative int32 ids")
    if manifest[SPLIT_COLUMN].isna().any() or not manifest[SPLIT_COLUMN].isin(SPLIT_NAMES).all():
        raise ValueError("manifest contains unknown or missing split labels")
    split_order = manifest[SPLIT_COLUMN].astype(str).map({name: i for i, name in enumerate(SPLIT_NAMES)})
    if not split_order.is_monotonic_increasing:
        raise ValueError("manifest splits must be consecutive chronological blocks")
    if parquet.metadata.num_rows != len(manifest):
        raise ValueError("processed row count does not match the manifest")

    raw_rows = audit.get("overview", {}).get("n_rows")
    repeated_rows = audit.get("duplicates", {}).get("n_repeated_rows")
    if not isinstance(raw_rows, int) or not isinstance(repeated_rows, int):
        raise ValueError("raw audit must include row and duplicate counts")
    if ids.max() >= raw_rows or raw_rows - repeated_rows != len(manifest):
        raise ValueError("manifest does not match the audited raw deduplication counts or original ids")

    split_metadata = {}
    previous_time_max = None
    for name in SPLIT_NAMES:
        part = manifest[manifest[SPLIT_COLUMN] == name]
        info = summary.get(name, {})
        required = ("n_rows", "id_min", "id_max", "time_min", "time_max")
        if any(key not in info for key in required) or part.empty:
            raise ValueError(f"{name}: missing split metadata or empty manifest partition")
        if len(part) != info["n_rows"] or int(part[ID_COLUMN].min()) != info["id_min"] or int(part[ID_COLUMN].max()) != info["id_max"]:
            raise ValueError(f"{name}: manifest disagrees with saved split metadata")
        if info["time_min"] > info["time_max"] or (
            previous_time_max is not None and previous_time_max >= info["time_min"]
        ):
            raise ValueError("saved Time ranges overlap or are out of order")
        previous_time_max = info["time_max"]
        split_metadata[name] = {key: info[key] for key in required}

    frames = {}
    for name in ("train", "validation"):
        frame = load_transactions(split=name, path=transactions_path)
        check_ml_frame(frame, name)
        expected_ids = manifest.loc[manifest[SPLIT_COLUMN] == name, ID_COLUMN].to_numpy()
        if not np.array_equal(frame[ID_COLUMN].to_numpy(), expected_ids):
            raise ValueError(f"{name}: processed transaction ids disagree with the manifest")
        info = split_metadata[name]
        if int(frame[TIME_COLUMN].min()) != info["time_min"] or int(frame[TIME_COLUMN].max()) != info["time_max"]:
            raise ValueError(f"{name}: processed Time range disagrees with split metadata")
        info["n_fraud"] = int(frame[TARGET_COLUMN].sum())
        info["n_legitimate"] = int((frame[TARGET_COLUMN] == 0).sum())
        frames[name] = frame

    duplicates = shared_feature_patterns(frames["train"], frames["validation"])
    folds = make_time_folds(frames["train"], n_splits, k_neighbors)
    fold_summaries = summarize_time_folds(frames["train"], folds)
    fold_digest = hashlib.sha256("".join(item["ids_sha256"] for item in fold_summaries).encode()).hexdigest()
    report = {
        "status": "needs_duplicate_review" if duplicates["shared_patterns"] else "ok",
        "raw_sha256": raw_sha256,
        "processed_sha256": sha256_of(transactions_path),
        "split_manifest_sha256": sha256_of(manifest_path),
        "preprocessing": "raw-exact-dedup-before-float32",
        "feature_names": list(FEATURE_COLUMNS),
        "feature_dtypes": {name: str(frames["train"][name].dtype) for name in FEATURE_COLUMNS},
        "positive_label": 1,
        "split_metadata": split_metadata,
        "test_rows_loaded": False,
        "test_boundary_check": "saved split metadata only; no test features or labels loaded",
        "shared_feature_patterns": duplicates,
        "cv": {
            "method": "expanding_equal_time_intervals",
            "n_splits": n_splits,
            "smote_k_neighbors": k_neighbors,
            "folds_sha256": fold_digest,
            "folds": fold_summaries,
        },
    }
    if review_recurring:
        review = review_repeated_patterns(frames["train"], frames["validation"], raw_path)
        report["recurring_feature_review"] = review
        report["recurring_feature_review_requested"] = True
        report["selection_population"] = "full_validation"
        report["novel_validation_sensitivity_required"] = bool(duplicates["shared_patterns"])
        accepted = review["decision"] in ("accepted_recurring_features", "no_shared_features")
        report["recurring_feature_policy"] = "chronological_recurring_features_retained" if accepted else None
        if accepted and duplicates["shared_patterns"]:
            report["status"] = "ok_with_recurring_features"
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=EXPERIMENTS_DIR / "data_checks.json")
    parser.add_argument("--transactions", type=Path, default=TRANSACTIONS_PARQUET_PATH)
    parser.add_argument("--manifest", type=Path, default=SPLIT_MANIFEST_PATH)
    parser.add_argument("--audit", type=Path, default=AUDIT_JSON_PATH)
    parser.add_argument("--raw", type=Path, default=RAW_CSV_PATH)
    parser.add_argument("--summary", type=Path, default=SPLIT_SUMMARY_JSON_PATH)
    parser.add_argument("--review-recurring", action="store_true", help="Review raw repeated features and retain distinct-time operations explicitly")
    args = parser.parse_args(argv)
    try:
        report = validate_ml_data(
            args.transactions, args.manifest, args.audit, args.raw, args.summary,
            review_recurring=args.review_recurring,
        )
        write_json(report, args.output)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Data check failed: {exc}\n")
    print(f"Data checks: {report['status']}; report: {args.output}")
    return 2 if report["status"] == "needs_duplicate_review" else 0


if __name__ == "__main__":
    raise SystemExit(main())
