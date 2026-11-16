"""Freeze the validation-selected pipeline and verify the package for Adam."""

import argparse
import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.config import EXPERIMENTS_DIR, ROOT_DIR
from src.data.api import load_xy
from src.data.io_utils import read_json
from src.ml.artifacts import file_sha256, load_artifact, save_artifact
from src.ml.comparison import choose_best, load_results
from src.ml.experiment import CONFIG_PATH, write_atomic_json
from src.ml.metrics import compute_metrics
from src.ml.prediction import predict_transactions

FINAL_MODEL = ROOT_DIR / "models" / "fraud_pipeline.joblib"
FINAL_METADATA = FINAL_MODEL.with_suffix(".json")
SELECTION_LOCK = ROOT_DIR / "models" / "selection_lock.json"


def array_sha256(values, dtype):
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes()).hexdigest()


def check_lock_metadata(lock, metadata):
    fields = ("threshold", "comparison", "min_precision", "artifact_sha256",
              "dataset_sha256", "processed_sha256", "split_sha256", "source_sha256", "config_sha256")
    if lock.get("selected_run_id") != metadata.get("run_id"):
        raise ValueError("Selection lock disagrees with the model run_id")
    for name in fields:
        if name not in lock or name not in metadata or lock[name] != metadata[name]:
            raise ValueError(f"Selection lock disagrees with metadata.{name}")


def freeze_model():
    config = read_json(CONFIG_PATH)
    config["config_sha256"] = file_sha256(CONFIG_PATH)
    selected = choose_best(load_results(EXPERIMENTS_DIR / "results"), config)
    previous = read_json(EXPERIMENTS_DIR / "selected_run.json")
    if previous["run_id"] != selected["run_id"]:
        raise ValueError("Saved selection is inconsistent; compare the validation results first")
    if SELECTION_LOCK.exists():
        lock = read_json(SELECTION_LOCK)
        if lock["selected_run_id"] != selected["run_id"]:
            raise ValueError("A different model is already frozen; do not silently overwrite the selection")
        if file_sha256(FINAL_MODEL) != lock["artifact_sha256"] or file_sha256(FINAL_METADATA) != lock["metadata_sha256"]:
            raise ValueError("Frozen package changed; inspect it before proceeding")
        check_lock_metadata(lock, read_json(FINAL_METADATA))
        return lock
    model_path = ROOT_DIR / selected["model_path"]
    if file_sha256(model_path) != selected["artifact_sha256"]:
        raise ValueError("Selected training artifact changed")
    pipeline, metadata = load_artifact(model_path)
    for name in ("run_id", "threshold", "min_precision", "dataset_sha256", "processed_sha256", "split_sha256", "source_sha256", "config_sha256", "scores_sha256"):
        if name not in metadata or metadata[name] != selected[name]:
            raise ValueError(f"Selected artifact metadata disagrees with run result: {name}")
    scores_path = ROOT_DIR / selected["scores_file"]
    if file_sha256(scores_path) != selected["scores_sha256"]:
        raise ValueError("Selected validation cache changed before freezing")
    scores = pd.read_parquet(scores_path)
    metadata.update({
        "frozen": True, "frozen_at": datetime.now(timezone.utc).isoformat(),
        "source_paths": {"model": selected["model_path"], "metadata": selected["metadata_path"],
                         "validation_scores": selected["scores_file"]},
        "model_path": str(FINAL_MODEL.relative_to(ROOT_DIR)),
        "metadata_path": str(FINAL_METADATA.relative_to(ROOT_DIR)),
        "selection_rule": config["model_ties"], "evaluation_owner": "Adam",
        "test_evaluated": False, "refit_on_train_validation": False,
        "validation_scores_values_sha256": array_sha256(scores["score"], "<f8"),
        "validation_ids_sha256": array_sha256(scores["transaction_id"], "<i8"),
        "validation_labels_sha256": array_sha256(scores["y_true"], "i1"),
    })
    saved = save_artifact(pipeline, metadata, FINAL_MODEL)
    lock = {
        "selected_run_id": selected["run_id"], "model_path": saved["model_path"],
        "metadata_path": saved["metadata_path"], "artifact_sha256": saved["artifact_sha256"],
        "metadata_sha256": file_sha256(FINAL_METADATA), "threshold": saved["threshold"],
        "comparison": ">=", "min_precision": 0.85,
        "dataset_sha256": saved["dataset_sha256"], "processed_sha256": saved["processed_sha256"],
        "split_sha256": saved["split_sha256"],
        "source_sha256": saved["source_sha256"], "config_sha256": saved["config_sha256"],
        "frozen_at": saved["frozen_at"], "test_evaluated": False, "evaluation_owner": "Adam",
    }
    write_atomic_json(lock, SELECTION_LOCK)
    return lock


def make_synthetic_example(metadata):
    """Synthetic inputs exercise the API; their scores are not quality metrics."""
    rng = np.random.default_rng(42)
    names = metadata["feature_names"]
    frame = pd.DataFrame(rng.normal(size=(3, len(names))), columns=names,
                         index=["synthetic_1", "synthetic_2", "synthetic_3"])
    frame["Amount"] = np.array([0.0, 20.0, 100.0])
    return frame.astype(metadata["feature_dtypes"])


def verify_package():
    lock = read_json(SELECTION_LOCK)
    if file_sha256(FINAL_MODEL) != lock["artifact_sha256"] or file_sha256(FINAL_METADATA) != lock["metadata_sha256"]:
        raise ValueError("Frozen model or metadata does not match the selection lock")
    pipeline, metadata = load_artifact(FINAL_MODEL)
    check_lock_metadata(lock, metadata)
    cache_path = ROOT_DIR / metadata["source_paths"]["validation_scores"]
    if file_sha256(cache_path) != metadata["scores_sha256"]:
        raise ValueError("Saved validation predictions changed")
    cache = pd.read_parquet(cache_path)
    X_validation, y_validation = load_xy("validation")
    if not np.array_equal(X_validation.index, cache["transaction_id"]) or not np.array_equal(y_validation, cache["y_true"]):
        raise ValueError("Validation ids or labels differ from the original run")
    predictions = predict_transactions(X_validation, pipeline, metadata)
    scores_equal = np.array_equal(predictions["score"], cache["score"])
    classes_equal = np.array_equal(predictions["y_pred"], cache["y_pred"])
    if not scores_equal or not classes_equal:
        raise ValueError("Probabilities or classes changed after artifact reload")
    metrics = compute_metrics(y_validation, predictions["score"], metadata["threshold"], metadata["min_precision"])
    if metrics != metadata["validation_metrics"]:
        raise ValueError("Metrics changed after artifact reload")

    bad_schema = X_validation.iloc[:2].drop(columns=[metadata["feature_names"][0]])
    try:
        predict_transactions(bad_schema, pipeline, metadata)
    except ValueError as exc:
        schema_error = str(exc)
    else:
        raise ValueError("Invalid feature schema was unexpectedly accepted")
    synthetic = make_synthetic_example(metadata)
    synthetic_predictions = predict_transactions(synthetic, pipeline, metadata)
    example = {
        "synthetic": True, "purpose": "API integration only; not observed transactions or model quality",
        "input": synthetic.reset_index(names="example_id").to_dict(orient="records"),
        "predictions": synthetic_predictions.reset_index(names="example_id").to_dict(orient="records"),
    }
    write_atomic_json(example, EXPERIMENTS_DIR / "synthetic_prediction_check.json")
    report = {
        "selected_run_id": metadata["run_id"], "artifact_sha256": metadata["artifact_sha256"],
        "validation_rows_checked": len(X_validation), "scores_identical_after_reload": scores_equal,
        "classes_identical_after_reload": classes_equal, "metrics_identical_after_reload": True,
        "invalid_schema_rejected": True, "schema_error": schema_error,
        "synthetic_example_checked": True, "threshold": metadata["threshold"],
        "comparison": ">=", "test_evaluated": False, "validation_metrics": metrics,
    }
    write_atomic_json(report, EXPERIMENTS_DIR / "handoff_verification.json")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "verify"])
    args = parser.parse_args()
    report = freeze_model() if args.command == "freeze" else verify_package()
    print(report)


if __name__ == "__main__":
    main()
