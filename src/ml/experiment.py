"""Run one isolated train/validation experiment and save enough evidence to resume."""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from threadpoolctl import threadpool_limits

from src.config import EXPERIMENTS_DIR, FEATURE_COLUMNS, ID_COLUMN, ROOT_DIR, TARGET_COLUMN
from src.data.api import load_transactions
from src.data.io_utils import read_json, to_jsonable
from src.ml.artifacts import file_sha256, save_artifact
from src.ml.data_checks import check_ml_frame, make_time_folds
from src.ml.metrics import compute_metrics, select_threshold
from src.ml.pipelines import MODEL_NAMES, STRATEGIES, make_pipeline, parameter_grid
from src.ml.prediction import fraud_scores, predict_transactions, validate_features
from src.ml.progress import save_checkpoint

CONFIG_PATH = Path(__file__).with_name("experiment_config.json")
RESULTS_DIR = EXPERIMENTS_DIR / "results"
MODELS_DIR = ROOT_DIR / "models" / "runs"
CACHE_DIR = EXPERIMENTS_DIR / "cache"


def write_atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(to_jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_sha256():
    digest = hashlib.sha256()
    paths = [Path(__file__).with_name(name) for name in (
        "experiment.py", "pipelines.py", "metrics.py", "prediction.py", "data_checks.py", "artifacts.py", "cli.py",
    )]
    paths += [ROOT_DIR / "src" / "config.py", ROOT_DIR / "src" / "data" / "api.py"]
    for path in paths:
        digest.update(str(path.relative_to(ROOT_DIR)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def library_versions():
    packages = ("numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn", "joblib", "pyarrow")
    return {"python": platform.python_version(), **{name: version(name) for name in packages}}


def peak_memory():
    """Peak RSS in MiB, with an explicit unavailable result instead of a false zero."""
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/self/status").read_text().splitlines():
                if line.startswith("VmHWM:"):
                    value = int(line.split()[1]) / 1024
                    if value > 0:
                        return value, "/proc/self/status VmHWM (KiB / 1024)"
        except (OSError, ValueError):
            pass
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        value /= 1024**2 if sys.platform == "darwin" else 1024
        if value > 0:
            return value, "resource.getrusage(RUSAGE_SELF).ru_maxrss"
    except (ImportError, OSError, ValueError):
        pass
    return None, "Peak RSS is unavailable on this host; no memory comparison can be claimed."


def novel_validation_mask(X_train, X_validation):
    """Find exact repeated input vectors without using labels or a hash-only shortcut."""
    known = X_train.drop_duplicates()
    matched = X_validation.reset_index().merge(known, on=FEATURE_COLUMNS, how="inner")
    return ~X_validation.index.isin(matched[ID_COLUMN])


def fit_with_cv(X, y, train, model_name, strategy, config, tune):
    pipeline = make_pipeline(model_name, strategy, seed=config["seed"])
    cv_seconds = 0.0
    cv_results = []
    best_params = {}
    if tune:
        folds = make_time_folds(train, config["cv"]["n_splits"], config["smote_k_neighbors"])
        search = GridSearchCV(
            pipeline, parameter_grid(model_name), cv=folds, scoring="average_precision",
            refit=False, n_jobs=1, error_score="raise",
        )
        started = perf_counter()
        search.fit(X, y)
        cv_seconds = perf_counter() - started
        best_params = search.best_params_
        for index, params in enumerate(search.cv_results_["params"]):
            cv_results.append({
                "params": params,
                "mean_average_precision": float(search.cv_results_["mean_test_score"][index]),
                "std_average_precision": float(search.cv_results_["std_test_score"][index]),
                "fold_average_precision": [float(search.cv_results_[f"split{i}_test_score"][index]) for i in range(len(folds))],
            })
        pipeline = make_pipeline(model_name, strategy, seed=config["seed"], params=best_params)
    started = perf_counter()
    pipeline.fit(X, y)
    fit_seconds = perf_counter() - started
    return pipeline, best_params, cv_results, cv_seconds, fit_seconds


def run_experiment(model_name, strategy="none", tune=False, force=False, stage=None):
    """Read train and validation only; test predictions belong to the final evaluator."""
    config = read_json(CONFIG_PATH)
    audit = read_json(EXPERIMENTS_DIR / "data_checks.json")
    if audit["status"] not in ("ok", "ok_with_recurring_features"):
        raise ValueError("Resolve the recorded data checks before training")
    if audit["cv"]["n_splits"] != config["cv"]["n_splits"] or audit["cv"]["smote_k_neighbors"] != config["smote_k_neighbors"]:
        raise ValueError("CV configuration changed; repeat the data checks before training")
    if config["positive_label"] != 1 or config["comparison"] != ">=" or config["resource_policy"]["n_jobs"] != 1:
        raise ValueError("This protocol requires positive_label=1, comparison=>= and n_jobs=1")
    if model_name == "dummy" and tune:
        raise ValueError("Dummy baseline has no hyperparameter search")
    data_hash = audit["raw_sha256"]
    split_hash = audit["split_manifest_sha256"]
    config_hash = file_sha256(CONFIG_PATH)
    code_hash = source_sha256()
    environment_versions = library_versions()
    identity = {"model": model_name, "strategy": strategy, "tune": tune, "data": data_hash,
                "split": split_hash, "processed": audit["processed_sha256"], "config": config_hash,
                "source": code_hash, "environment": environment_versions}
    suffix = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    mode = "cv" if tune else "fixed"
    run_id = f"{model_name}_{strategy}_{mode}_{suffix}"
    result_path = RESULTS_DIR / f"{run_id}.json"
    # Check real inputs even when reusing a completed run after an interruption.
    from src.config import TRANSACTIONS_PARQUET_PATH, SPLIT_MANIFEST_PATH
    if file_sha256(TRANSACTIONS_PARQUET_PATH) != audit["processed_sha256"] or file_sha256(SPLIT_MANIFEST_PATH) != split_hash:
        raise ValueError("Data changed since data_checks; repeat the audit before experiments")
    if result_path.exists() and not force:
        result = read_json(result_path)
        model_path = ROOT_DIR / result["model_path"]
        scores_path = ROOT_DIR / result["scores_file"]
        if file_sha256(model_path) != result["artifact_sha256"] or file_sha256(scores_path) != result["scores_sha256"]:
            raise ValueError(f"Cached files are inconsistent for {run_id}; inspect them before --force")
        print(f"Cached completed run: {run_id}", flush=True)
        return result
    if stage is not None:
        save_checkpoint(stage, "in_progress", f"Running {run_id}; resume this command after interruption. Completed result JSONs are reused.")
    print(f"Starting {run_id}: train + validation only", flush=True)
    train = load_transactions(split="train")
    validation = load_transactions(split="validation")
    check_ml_frame(train, "train")
    check_ml_frame(validation, "validation")
    X_train = train.set_index(ID_COLUMN)[FEATURE_COLUMNS]
    y_train = train[TARGET_COLUMN].to_numpy()
    X_validation = validation.set_index(ID_COLUMN)[FEATURE_COLUMNS]
    y_validation = validation[TARGET_COLUMN].to_numpy()
    schema = {"feature_names": list(FEATURE_COLUMNS), "feature_dtypes": {name: str(X_train[name].dtype) for name in FEATURE_COLUMNS}}
    X_train = validate_features(X_train, schema)
    X_validation = validate_features(X_validation, schema)
    with threadpool_limits(limits=config["resource_policy"]["blas_threads"]):
        pipeline, params, cv_results, cv_seconds, fit_seconds = fit_with_cv(
            X_train, y_train, train, model_name, strategy, config, tune,
        )
        scores = fraud_scores(pipeline, X_validation, config["positive_label"])
        threshold_result = select_threshold(y_validation, scores, config["min_precision"])
        threshold = threshold_result["threshold"]
        prediction_metadata = dict(schema, positive_label=1, comparison=">=", threshold=threshold)
        predictions = None if threshold is None else predict_transactions(X_validation, pipeline, prediction_metadata)
        if predictions is not None and not np.array_equal(predictions["score"].to_numpy(), scores):
            raise ValueError("Shared prediction function disagrees with validation scores")
        metrics = compute_metrics(y_validation, scores, threshold, config["min_precision"])
        novel = novel_validation_mask(X_train, X_validation)
        if len(np.unique(y_validation[novel])) == 2:
            novel_metrics = compute_metrics(y_validation[novel], scores[novel], threshold, config["min_precision"])
        else:
            novel_metrics = {"status": "unavailable", "reason": "Novel validation must contain both classes; the full validation remains the selection population."}
    memory_mb, memory_method = peak_memory()
    command = f"./.venv/bin/python -m src.ml.cli run --model {model_name} --strategy {strategy}"
    if tune:
        command += " --tune"
    scores_path = CACHE_DIR / run_id / "validation.parquet"
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    score_frame = pd.DataFrame({
        ID_COLUMN: X_validation.index, "split": "validation", "y_true": y_validation,
        "score": scores, "y_pred": pd.array([None] * len(scores), dtype="Int8") if threshold is None else predictions["y_pred"].to_numpy(),
    })
    temporary = scores_path.with_suffix(".tmp.parquet")
    score_frame.to_parquet(temporary, index=False)
    temporary.replace(scores_path)
    model_path = MODELS_DIR / f"{run_id}.joblib"
    result = {
        "schema_version": 1, "protocol_version": config["protocol_version"],
        "run_id": run_id, "model_version": run_id, "model_name": model_name,
        "imbalance_strategy": strategy, "tuning": mode, "provenance": "real", "split": "validation",
        "status": threshold_result["status"], "threshold": threshold, "min_precision": config["min_precision"],
        "positive_label": 1, "comparison": ">=", **schema,
        "validation_metrics": metrics, "threshold_selection": threshold_result,
        "novel_validation_metrics": novel_metrics,
        "novel_validation_rows": int(novel.sum()), "novel_validation_fraud": int(y_validation[novel].sum()),
        "selection_population": "full_validation", "recurring_feature_policy": config["recurring_features"]["policy"],
        "train_rows": len(train), "validation_rows": len(validation),
        "seed": config["seed"], "params": params,
        "classifier_parameters": pipeline.named_steps["clf"].get_params(),
        "pipeline_parameters": {name: str(value) for name, value in pipeline.get_params(deep=False).items()},
        "cv_results": cv_results, "cv_plan": audit["cv"],
        "fit_seconds": fit_seconds, "cv_seconds": cv_seconds, "peak_memory_mb": memory_mb,
        "memory_measurement": memory_method,
        "memory_scope": "Fresh isolated process through CV (if requested), fit and prediction; excludes artifact serialization",
        "time_measurement": "perf_counter around final full train pipeline.fit; CV measured separately",
        "resource_policy": config["resource_policy"],
        "dataset_sha256": data_hash, "processed_sha256": audit["processed_sha256"], "split_sha256": split_hash,
        "config_sha256": config_hash, "source_sha256": code_hash,
        "python_version": platform.python_version(), "library_versions": environment_versions,
        "source_revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, capture_output=True, text=True, check=True).stdout.strip(),
        "command": command, "created_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path.relative_to(ROOT_DIR)), "metadata_path": str(model_path.with_suffix(".json").relative_to(ROOT_DIR)),
        "scores_file": str(scores_path.relative_to(ROOT_DIR)), "scores_sha256": file_sha256(scores_path),
        "test_evaluated": False,
    }
    result = save_artifact(pipeline, result, model_path)
    result["model_size_bytes"] = model_path.stat().st_size
    # Keep the human-readable run metadata and the result catalogue consistent.
    write_atomic_json(result, model_path.with_suffix(".json"))
    write_atomic_json(result, result_path)
    print(json.dumps({"run_id": run_id, "status": result["status"], "threshold": threshold,
                      "validation": metrics, "fit_seconds": fit_seconds}, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--strategy", choices=STRATEGIES, default="none")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--stage", type=int)
    args = parser.parse_args()
    run_experiment(args.model, args.strategy, args.tune, args.force, args.stage)


if __name__ == "__main__":
    main()
