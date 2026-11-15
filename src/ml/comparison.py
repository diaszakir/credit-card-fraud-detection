"""Сравнить validation-запуски и сохранить таблицы для SQLite и отчёта."""

import json
from numbers import Real
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ROOT_DIR
from src.data.io_utils import read_json, write_json
from src.data.storage import save_parquet
from src.ml.artifacts import file_sha256


HASH_FIELDS = ("dataset_sha256", "processed_sha256", "split_sha256", "source_sha256", "config_sha256")
MODEL_ORDER = ["logistic_regression", "random_forest", "dummy"]
STRATEGY_ORDER = ["none", "class_weight", "smote", "undersampling"]
PREDICTION_COLUMNS = ["transaction_id", "split", "y_true", "score", "y_pred"]


def finite_number(value, name, *, allow_none=False):
    if value is None and allow_none:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


def check_result(result):
    """Не смешивать test, демонстрации и результаты разных протоколов."""
    if not isinstance(result, dict):
        raise ValueError("each run result must be a dictionary")
    for name in ("run_id", "model_name", "imbalance_strategy"):
        if not isinstance(result.get(name), str) or not result[name]:
            raise ValueError(f"run result requires a non-empty {name}")
    if result.get("provenance") != "real":
        raise ValueError("selection and export require provenance='real'; synthetic runs are not eligible")
    if result.get("split") != "validation":
        raise ValueError("model comparison may use only validation results")
    if result.get("status") not in ("ok", "no_feasible_threshold"):
        raise ValueError("unknown run status")
    if result.get("tuning") not in ("fixed", "cv"):
        raise ValueError("tuning must be fixed or cv")
    if "threshold" not in result:
        raise ValueError("run result requires threshold, including null for unavailable thresholds")
    threshold = finite_number(result["threshold"], "threshold", allow_none=True)
    if threshold is not None and not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    minimum = finite_number(result.get("min_precision"), "min_precision")
    if not 0 < minimum <= 1:
        raise ValueError("min_precision must be in (0, 1]")
    if not isinstance(result.get("seed"), int) or isinstance(result["seed"], bool):
        raise ValueError("seed must be an integer")
    python_version = result.get("python_version")
    if not isinstance(python_version, str) or not (python_version == "3.12" or python_version.startswith("3.12.")):
        raise ValueError("python_version must describe Python 3.12")
    versions = result.get("library_versions")
    if not isinstance(versions, dict) or not versions or any(
        not isinstance(name, str) or not name or not isinstance(value, str) or not value
        for name, value in versions.items()
    ):
        raise ValueError("library_versions must contain package names and version strings")
    for name in HASH_FIELDS:
        digest = result.get(name)
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"{name} must be a SHA-256 hex digest")

    metrics = result.get("validation_metrics")
    if not isinstance(metrics, dict):
        raise ValueError("validation_metrics must be a dictionary")
    for name in ("pr_auc", "average_precision"):
        value = finite_number(metrics.get(name), name)
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    for name, value in metrics.items():
        if value is not None and not isinstance(value, (bool, np.bool_)):
            finite_number(value, f"validation_metrics.{name}")
    for name in ("precision", "recall", "accuracy"):
        if metrics.get(name) is not None and not 0 <= metrics[name] <= 1:
            raise ValueError(f"{name} must be in [0, 1]")


def check_results(results, config=None):
    """Проверить общий источник, разбиение и конфигурацию перед сравнением."""
    if not results:
        raise ValueError("no experiment results found")
    seen = set()
    for result in results:
        check_result(result)
        if result["run_id"] in seen:
            raise ValueError(f"duplicate run_id: {result['run_id']}")
        seen.add(result["run_id"])

    first = results[0]
    for name in (*HASH_FIELDS, "seed", "min_precision", "python_version", "library_versions"):
        if any(result[name] != first[name] for result in results[1:]):
            raise ValueError(f"runs disagree on {name}; do not compare different data or protocols")
    if config is not None:
        minimum = finite_number(config.get("min_precision"), "config.min_precision")
        if minimum != first["min_precision"]:
            raise ValueError("results do not match config.min_precision")
        if "seed" in config and config["seed"] != first["seed"]:
            raise ValueError("results do not match config.seed")
        if "config_sha256" in config and config["config_sha256"] != first["config_sha256"]:
            raise ValueError("results do not match config_sha256")


def load_results(results_dir):
    """Читать только каталог run JSON; порядок возвращаемых запусков стабилен."""
    directory = Path(results_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"results directory not found: {directory}")
    results = [read_json(path) for path in sorted(directory.glob("*.json"))]
    check_results(results)
    return sorted(results, key=lambda result: result["run_id"])


def is_feasible(result, min_precision):
    metrics = result["validation_metrics"]
    return (
        result["status"] == "ok"
        and result["threshold"] is not None
        and metrics.get("precision_constraint_met") is True
        and metrics.get("precision") is not None
        and metrics.get("recall") is not None
        and metrics["precision"] >= min_precision
    )


def choose_best(results, config):
    """Recall, precision, PR-AUC, затем заранее заданный порядок и run_id."""
    check_results(results, config)
    models = config.get("model_order", MODEL_ORDER)
    strategies = config.get("strategy_order", STRATEGY_ORDER)
    if not isinstance(models, list) or not isinstance(strategies, list) or len(set(models)) != len(models) or len(set(strategies)) != len(strategies):
        raise ValueError("model_order and strategy_order must be lists without duplicates")
    for result in results:
        if result["model_name"] not in models or result["imbalance_strategy"] not in strategies:
            raise ValueError("run model or imbalance strategy is absent from the fixed tie order")
    candidates = [result for result in results if is_feasible(result, config["min_precision"])]
    if not candidates:
        raise ValueError("no run satisfies min_precision; discuss the next step before opening test")

    def order(result):
        metrics = result["validation_metrics"]
        return (
            -metrics["recall"], -metrics["precision"], -metrics["pr_auc"],
            models.index(result["model_name"]), strategies.index(result["imbalance_strategy"]), result["run_id"],
        )

    return min(candidates, key=order)


def read_predictions(result):
    """Проверить сохранённые решения перед передачей в SQLite."""
    source = result.get("scores_file")
    if not isinstance(source, (str, Path)) or not str(source):
        raise ValueError(f"{result['run_id']}: scores_file is required")
    path = Path(source)
    if not path.is_absolute():
        path = ROOT_DIR / path
    expected_sha256 = result.get("scores_sha256")
    if not isinstance(expected_sha256, str) or file_sha256(path) != expected_sha256:
        raise ValueError(f"{result['run_id']}: validation cache SHA-256 mismatch")
    frame = pd.read_parquet(path)
    missing = [name for name in PREDICTION_COLUMNS if name not in frame.columns]
    if missing or frame.empty:
        raise ValueError(f"{result['run_id']}: scores file is empty or lacks columns {missing}")
    frame = frame[PREDICTION_COLUMNS].copy()
    ids = frame["transaction_id"]
    if not pd.api.types.is_integer_dtype(ids.dtype) or ids.isna().any() or (ids < 0).any() or ids.duplicated().any():
        raise ValueError("prediction transaction_id must contain unique non-negative integers")
    if not frame["split"].eq("validation").all():
        raise ValueError("prediction files may contain only validation rows")
    for name in ("y_true", "y_pred"):
        if frame[name].isna().any() or not frame[name].isin([0, 1]).all():
            raise ValueError(f"prediction {name} must contain labels 0 and 1")
    score = frame["score"]
    if not pd.api.types.is_numeric_dtype(score.dtype) or pd.api.types.is_bool_dtype(score.dtype) or pd.api.types.is_complex_dtype(score.dtype):
        raise ValueError("prediction scores must be real numbers")
    values = score.to_numpy(dtype="float64", na_value=np.nan)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("prediction scores must be finite and in [0, 1]")
    expected = (values >= result["threshold"]).astype("int8")
    if not np.array_equal(frame["y_pred"].to_numpy(), expected):
        raise ValueError("cached y_pred does not use the saved >= threshold")
    frame.insert(0, "run_id", result["run_id"])
    return frame


def export_results(results, output_dir):
    """Записать реальные результаты; недоступные метрики не заменять нулями."""
    check_results(results)
    ordered = sorted(results, key=lambda result: result["run_id"])
    minimum = ordered[0]["min_precision"]
    runs, metrics, comparison, predictions = [], [], [], []
    first_ids = first_labels = None
    for result in ordered:
        try:
            params_json = json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("run metadata must contain finite JSON-serializable values") from exc
        runs.append({
            "run_id": result["run_id"], "model_name": result["model_name"],
            "imbalance_strategy": result["imbalance_strategy"], "params_json": params_json,
            "fit_seconds": result.get("fit_seconds"), "model_size_bytes": result.get("model_size_bytes"),
            "peak_memory_mb": result.get("peak_memory_mb"), "threshold": result["threshold"],
            "created_at": result.get("created_at"),
        })
        row = {name: result.get(name) for name in (
            "run_id", "model_name", "imbalance_strategy", "tuning", "status", "threshold",
            "min_precision", "fit_seconds", "cv_seconds", "peak_memory_mb", "model_size_bytes",
        )}
        row.update(result["validation_metrics"])
        comparison.append(row)
        for name, value in result["validation_metrics"].items():
            if value is not None:
                metrics.append({
                    "run_id": result["run_id"], "model_name": result["model_name"],
                    "imbalance_strategy": result["imbalance_strategy"], "split": "validation",
                    "metric": name, "value": float(value),
                })
        if is_feasible(result, minimum):
            frame = read_predictions(result)
            ids = frame["transaction_id"].to_numpy()
            labels = frame["y_true"].to_numpy()
            if first_ids is not None and (not np.array_equal(ids, first_ids) or not np.array_equal(labels, first_labels)):
                raise ValueError("prediction files disagree on validation ids, row order or labels")
            first_ids, first_labels = ids, labels
            predictions.append(frame)

    combined = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(columns=["run_id", *PREDICTION_COLUMNS])
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {name: directory / filename for name, filename in (
        ("runs", "runs.parquet"), ("metrics", "metrics.parquet"),
        ("predictions", "predictions.parquet"), ("comparison", "comparison.csv"), ("summary", "summary.json"),
    )}
    save_parquet(pd.DataFrame(runs), paths["runs"])
    save_parquet(pd.DataFrame(metrics), paths["metrics"])
    save_parquet(combined, paths["predictions"])
    pd.DataFrame(comparison).to_csv(paths["comparison"], index=False)
    summary = {name: ordered[0][name] for name in (
        *HASH_FIELDS, "seed", "min_precision", "split", "provenance", "python_version", "library_versions",
    )}
    summary.update({
        "n_runs": len(ordered), "run_ids": [result["run_id"] for result in ordered],
        "feasible_run_ids": [result["run_id"] for result in ordered if is_feasible(result, minimum)],
        "no_feasible_threshold_run_ids": [result["run_id"] for result in ordered if result["status"] == "no_feasible_threshold"],
        "metric_methods": {"pr_auc": "trapezoidal auc(recall, precision)", "average_precision": "average_precision_score"},
    })
    write_json(summary, paths["summary"])
    return paths
