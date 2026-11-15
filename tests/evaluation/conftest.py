"""Синтетический «проект» во временной папке для тестов Evaluation & Application.

Все числа здесь — тестовые данные, а не результаты экспериментов. Модель обучается на
16 искусственных строках только для проверки контракта загрузки и прогноза.
"""

from __future__ import annotations

import json
import platform
from importlib.metadata import version

import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS
from src.evaluation.sources import ROOT_ENV_VAR
from src.ml.artifacts import file_sha256, save_artifact
from src.ml.metrics import compute_metrics
from src.ml.pipelines import make_pipeline
from src.ml.prediction import predict_transactions

DTYPES = {name: "float64" if name == "Amount" else "float32" for name in FEATURE_COLUMNS}
LOCK_FIELDS = ("threshold", "comparison", "min_precision", "artifact_sha256", "dataset_sha256",
               "processed_sha256", "split_sha256", "source_sha256", "config_sha256")


def feature_frame(v1_values, index=None) -> pd.DataFrame:
    frame = pd.DataFrame(0.0, index=range(len(v1_values)), columns=FEATURE_COLUMNS)
    frame["V1"] = v1_values
    frame["Amount"] = 20.0
    if index is not None:
        frame.index = pd.Index(index, name="transaction_id")
    return frame.astype(DTYPES)


def write_lock(root, metadata):
    lock = {"selected_run_id": metadata["run_id"],
            "metadata_sha256": file_sha256(root / "models" / "fraud_pipeline.json")}
    lock.update({name: metadata[name] for name in LOCK_FIELDS})
    (root / "models" / "selection_lock.json").write_text(json.dumps(lock))


@pytest.fixture
def empty_project(tmp_path, monkeypatch):
    """Чистый клон без данных и модели."""
    monkeypatch.setenv(ROOT_ENV_VAR, str(tmp_path))
    monkeypatch.delenv("FRAUD_APP_STUB", raising=False)
    return tmp_path


@pytest.fixture
def project(tmp_path, monkeypatch):
    """Проект с синтетической замороженной моделью, сравнением, SQL-результатом и кэшем."""
    monkeypatch.setenv(ROOT_ENV_VAR, str(tmp_path))
    monkeypatch.delenv("FRAUD_APP_STUB", raising=False)

    y_train = np.tile([0, 1], 8)
    X_train = feature_frame(np.where(y_train == 1, 4.0, -4.0))
    pipeline = make_pipeline("logistic_regression")
    pipeline.fit(X_train, y_train)

    run_id = "synthetic_selected"
    metadata = {
        "schema_version": 1, "protocol_version": "synthetic-eval-test", "run_id": run_id,
        "model_name": "logistic_regression", "imbalance_strategy": "none", "tuning": "fixed",
        "split": "validation", "frozen": True, "test_evaluated": False, "provenance": "synthetic_fixture",
        "feature_names": list(FEATURE_COLUMNS), "feature_dtypes": DTYPES,
        "positive_label": 1, "threshold": 0.5, "comparison": ">=", "min_precision": 0.85,
        "dataset_sha256": "a" * 64, "processed_sha256": "b" * 64, "split_sha256": "c" * 64,
        "source_sha256": "d" * 64, "config_sha256": "e" * 64,
        "library_versions": {
            "python": platform.python_version(),
            **{name: version(name) for name in ("scikit-learn", "imbalanced-learn", "joblib", "numpy", "scipy")},
        },
        "source_paths": {"validation_scores": f"reports/experiments/cache/{run_id}/validation.parquet"},
    }
    # Кэш validation: 4 строки, по одной на каждый исход TP/TN/FN/FP.
    X_val = feature_frame([4.0, -4.0, -4.0, 4.0], index=[201, 202, 203, 204])
    y_val = np.array([1, 0, 1, 0])
    predicted = predict_transactions(X_val, pipeline, metadata)
    metadata["validation_metrics"] = compute_metrics(y_val, predicted["score"].to_numpy(), 0.5)
    model_path = tmp_path / "models" / "fraud_pipeline.joblib"
    metadata = save_artifact(pipeline, metadata, model_path)
    write_lock(tmp_path, metadata)
    cache = tmp_path / metadata["source_paths"]["validation_scores"]
    cache.parent.mkdir(parents=True)
    pd.DataFrame({"transaction_id": X_val.index, "split": "validation", "y_true": y_val,
                  "score": predicted["score"].to_numpy(), "y_pred": predicted["y_pred"].to_numpy()}).to_parquet(cache)

    experiments = tmp_path / "reports" / "experiments"
    pd.DataFrame([
        {"run_id": run_id, "model_name": "logistic_regression", "imbalance_strategy": "none", "tuning": "fixed",
         "status": "ok", "threshold": 0.5, "precision": 0.5, "recall": 0.5, "pr_auc": 0.5,
         "average_precision": 0.5, "fit_seconds": 0.1, "peak_memory_mb": 1.0, "model_size_bytes": 10,
         "tn": 1, "fp": 1, "fn": 1, "tp": 1},
        {"run_id": "synthetic_dummy", "model_name": "dummy", "imbalance_strategy": "none", "tuning": "fixed",
         "status": "no_feasible_threshold", "threshold": None, "precision": None, "recall": None, "pr_auc": 0.5,
         "average_precision": 0.5, "fit_seconds": 0.1, "peak_memory_mb": 1.0, "model_size_bytes": 10},
    ]).to_csv(experiments / "comparison.csv", index=False)

    (tmp_path / "reports" / "split_summary.json").write_text(json.dumps({
        "train": {"n_rows": 10, "n_fraud": 1}, "validation": {"n_rows": 4, "n_fraud": 2},
    }))
    sql = tmp_path / "reports" / "sql_results"
    sql.mkdir(parents=True)
    pd.DataFrame({"split_name": ["train"], "n_transactions": [10]}).to_csv(sql / "q01_split_overview.csv", index=False)

    return {"root": tmp_path, "pipeline": pipeline, "metadata": metadata, "model_path": model_path,
            "X_val": X_val, "y_val": y_val, "predicted": predicted}
