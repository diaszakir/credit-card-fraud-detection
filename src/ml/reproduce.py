"""Восстановить замороженную модель из рецепта без CV и повторного выбора порога."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.config import (
    EXPERIMENTS_DIR, ID_COLUMN, RAW_CSV_PATH, ROOT_DIR,
    SPLIT_MANIFEST_PATH, TRANSACTIONS_PARQUET_PATH,
)
from src.data.api import load_xy
from src.data.io_utils import read_json
from src.ml.artifacts import (
    _validate_metadata, _validate_versions, file_sha256, load_artifact, save_artifact,
)
from src.ml.experiment import CONFIG_PATH, write_atomic_json
from src.ml.handoff import FINAL_METADATA, FINAL_MODEL, SELECTION_LOCK, check_lock_metadata
from src.ml.metrics import compute_metrics
from src.ml.pipelines import make_pipeline
from src.ml.prediction import predict_transactions, validate_features

DATA_CHECKS_PATH = EXPERIMENTS_DIR / "data_checks.json"
REPORT_PATH = EXPERIMENTS_DIR / "reproduction_check.json"
VALIDATION_DIGESTS = {
    "validation_scores_values_sha256": ("score", "<f8"),
    "validation_ids_sha256": ("transaction_id", "<i8"),
    "validation_labels_sha256": ("y_true", "i1"),
}


def source_sha256():
    """Сохранить Linux recipe: имена frozen-файлов всегда используют POSIX slash."""
    relative_names = (
        "src/ml/experiment.py", "src/ml/pipelines.py", "src/ml/metrics.py",
        "src/ml/prediction.py", "src/ml/data_checks.py", "src/ml/artifacts.py",
        "src/ml/cli.py", "src/config.py", "src/data/api.py",
    )
    digest = hashlib.sha256()
    root = Path(ROOT_DIR)
    for name in relative_names:
        path = root / name
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def array_sha256(values, dtype):
    """Хешировать значения в согласованном порядке байтов, независимо от Parquet."""
    values = np.ascontiguousarray(np.asarray(values, dtype=np.dtype(dtype)))
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _check_versions(metadata):
    _validate_versions(metadata)
    aliases = {"sklearn": "scikit-learn", "imblearn": "imbalanced-learn"}
    saved = metadata.get("library_versions", {})
    required = {"numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn", "joblib", "pyarrow"}
    missing = sorted(required - {aliases.get(name, name) for name in saved})
    if missing or not isinstance(saved.get("python", metadata.get("python_version")), str):
        raise ValueError(f"Frozen recipe lacks complete library_versions: {missing}")
    # При восстановлении важны также pandas и pyarrow из исходного запуска.
    for name, expected in saved.items():
        if name == "python":
            continue  # Общая проверка выше сравнивает major/minor Python.
        package = aliases.get(name, name)
        try:
            installed = version(package)
        except PackageNotFoundError as exc:
            raise ValueError(f"reproduction requires missing package {package}") from exc
        if installed != expected:
            raise ValueError(f"reproduction requires {package}=={expected}; installed version is {installed}")


def _read_recipe():
    metadata = read_json(FINAL_METADATA)
    lock = read_json(SELECTION_LOCK)
    if file_sha256(FINAL_METADATA) != lock.get("metadata_sha256"):
        raise ValueError("Frozen metadata does not match the selection lock")
    check_lock_metadata(lock, metadata)
    _validate_metadata(metadata, require_threshold=True)
    _check_versions(metadata)
    if metadata.get("frozen") is not True or metadata.get("test_evaluated") is not False:
        raise ValueError("Reproduction requires a frozen recipe before test evaluation")
    for field in VALIDATION_DIGESTS:
        value = metadata.get(field)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Frozen recipe requires {field}; do not recreate a threshold from validation")
    return metadata, lock


def _check_inputs(metadata):
    config = read_json(CONFIG_PATH)
    audit = read_json(DATA_CHECKS_PATH)
    if audit.get("status") not in ("ok", "ok_with_recurring_features"):
        raise ValueError("Repeat the ML data checks and resolve their findings before reproduction")
    inputs = (
        ("dataset_sha256", RAW_CSV_PATH, "raw_sha256"),
        ("processed_sha256", TRANSACTIONS_PARQUET_PATH, "processed_sha256"),
        ("split_sha256", SPLIT_MANIFEST_PATH, "split_manifest_sha256"),
    )
    for field, path, audit_field in inputs:
        expected = metadata.get(field)
        if expected is None or file_sha256(path) != expected or audit.get(audit_field) != expected:
            raise ValueError(f"Reproduction input disagrees with frozen {field}; rebuild and check the original data")
    if file_sha256(CONFIG_PATH) != metadata.get("config_sha256"):
        raise ValueError("Reproduction configuration differs from the frozen config_sha256")
    if source_sha256() != metadata.get("source_sha256"):
        raise ValueError("Reproduction source differs from the frozen source_sha256")
    for field in ("seed", "min_precision", "positive_label", "comparison", "protocol_version"):
        if config.get(field) != metadata.get(field):
            raise ValueError(f"Reproduction config.{field} disagrees with the frozen recipe")
    if config.get("resource_policy") != {"n_jobs": 1, "blas_threads": 1}:
        raise ValueError("Reproduction requires the original single-worker resource policy")
    return config


def _validation_cache_path(metadata):
    source = metadata.get("source_paths", {}).get("validation_scores")
    if not isinstance(source, str) or not source:
        raise ValueError("Frozen recipe requires source_paths.validation_scores")
    path = (Path(ROOT_DIR) / source).resolve()
    if not path.is_relative_to(Path(ROOT_DIR).resolve()):
        raise ValueError("Validation cache must stay inside the project directory")
    return path


def reproduce_selected():
    """Восстановить только отсутствующий joblib и сохранить прежнее решение модели."""
    metadata, lock = _read_recipe()
    if Path(FINAL_MODEL).exists():
        if file_sha256(FINAL_MODEL) != lock["artifact_sha256"]:
            raise ValueError("Existing frozen model differs from its lock; reproduction will not overwrite it")
        _, loaded = load_artifact(FINAL_MODEL, FINAL_METADATA)
        check_lock_metadata(lock, loaded)
        return {"status": "already_present", "run_id": metadata["run_id"], "fit_performed": False,
                "artifact_sha256": metadata["artifact_sha256"], "test_evaluated": False}

    config = _check_inputs(metadata)
    cache_path = _validation_cache_path(metadata)
    pipeline = make_pipeline(metadata["model_name"], metadata["imbalance_strategy"],
                             seed=metadata["seed"], params=metadata["params"])
    X_train, y_train = load_xy("train")
    X_train = validate_features(X_train, metadata)
    if len(X_train) != metadata.get("train_rows"):
        raise ValueError("Reproduction train row count differs from the frozen recipe")
    with threadpool_limits(limits=config["resource_policy"]["blas_threads"]):
        pipeline.fit(X_train, y_train)
        X_validation, y_validation = load_xy("validation")
        predictions = predict_transactions(X_validation, pipeline, metadata)

    values = {"score": predictions["score"], "transaction_id": X_validation.index, "y_true": y_validation}
    for field, (name, dtype) in VALIDATION_DIGESTS.items():
        if array_sha256(values[name], dtype) != metadata[field]:
            raise ValueError(f"Reproduced validation does not match {field}; frozen files were not replaced")
    metrics = compute_metrics(y_validation, predictions["score"], metadata["threshold"], metadata["min_precision"])
    if metrics != metadata["validation_metrics"]:
        raise ValueError("Reproduced validation metrics differ; frozen files were not replaced")
    if len(X_validation) != metadata.get("validation_rows"):
        raise ValueError("Reproduction validation row count differs from the frozen recipe")
    if Path(FINAL_MODEL).exists():
        raise ValueError("Frozen model appeared during reproduction; it will not be overwritten")

    # Только после всех сверок готовим новые файлы. Порог и выбранный run_id не меняются.
    cache = pd.DataFrame({
        ID_COLUMN: X_validation.index, "split": "validation",
        "y_true": np.asarray(y_validation, dtype="int8"), "score": predictions["score"].to_numpy(),
        "y_pred": predictions["y_pred"].to_numpy(),
    })
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    Path(FINAL_MODEL).parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".reproduce-", dir=Path(FINAL_MODEL).parent) as directory:
        staged_cache = Path(directory) / "validation.parquet"
        cache.to_parquet(staged_cache, index=False)
        reproduced = dict(metadata)
        reproduced.update({
            "reproduced_from_artifact_sha256": metadata["artifact_sha256"],
            "reproduced_at": datetime.now(timezone.utc).isoformat(),
            "reproduction_command": "./.venv/bin/python -m src.ml.reproduce",
            "scores_sha256": file_sha256(staged_cache),
        })
        staged_model = Path(directory) / Path(FINAL_MODEL).name
        saved = save_artifact(pipeline, reproduced, staged_model)
        saved["model_size_bytes"] = staged_model.stat().st_size
        staged_metadata = staged_model.with_suffix(".json")
        write_atomic_json(saved, staged_metadata)
        updated_lock = dict(lock, artifact_sha256=saved["artifact_sha256"],
                            metadata_sha256=file_sha256(staged_metadata))
        check_lock_metadata(updated_lock, saved)
        staged_lock = Path(directory) / "selection_lock.json"
        write_atomic_json(updated_lock, staged_lock)
        staged_cache.replace(cache_path)
        staged_model.replace(FINAL_MODEL)
        staged_metadata.replace(FINAL_METADATA)
        staged_lock.replace(SELECTION_LOCK)

    report = {
        "status": "reproduced", "run_id": metadata["run_id"], "fit_performed": True,
        "fitted_splits": ["train"], "evaluated_splits": ["validation"],
        "cv_performed": False, "threshold_retuned": False, "test_evaluated": False,
        "threshold": metadata["threshold"], "comparison": metadata["comparison"],
        "scores_values_match": True, "validation_ids_match": True,
        "validation_labels_match": True, "validation_metrics_match": True,
        "validation_rows": len(X_validation), "validation_metrics": metrics,
        "original_artifact_sha256": metadata["artifact_sha256"],
        "artifact_sha256": saved["artifact_sha256"],
        "artifact_binary_identical": metadata["artifact_sha256"] == saved["artifact_sha256"],
        "reproduced_at": saved["reproduced_at"],
    }
    write_atomic_json(report, REPORT_PATH)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        report = reproduce_selected()
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Reproduction failed: {exc}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
