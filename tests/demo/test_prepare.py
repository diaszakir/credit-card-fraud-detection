"""All files and classes here are synthetic; no real transactions are loaded."""

import json
import platform
import shutil
import sys
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import app.prepare as prepare
from src.config import FEATURE_COLUMNS, ID_COLUMN, PROCESSED_COLUMNS
from src.data.io_utils import read_json, write_json
from src.ml.artifacts import file_sha256, save_artifact


class SyntheticScores:
    """A test-only fitted object: V1 already contains the synthetic score."""

    classes_ = np.array([0, 1])
    n_features_in_ = 29
    feature_names_in_ = np.asarray(FEATURE_COLUMNS)

    def predict_proba(self, X):
        score = X["V1"].to_numpy(dtype="float64")
        return np.column_stack([1 - score, score])


@pytest.fixture
def synthetic_demo(tmp_path, monkeypatch):
    paths = {
        "ROOT_DIR": tmp_path,
        "RAW_CSV_PATH": tmp_path / "data/raw/creditcard.csv",
        "TRANSACTIONS_PARQUET_PATH": tmp_path / "data/processed/transactions.parquet",
        "SPLIT_MANIFEST_PATH": tmp_path / "data/processed/split_manifest.parquet",
        "AUDIT_JSON_PATH": tmp_path / "reports/data_audit.json",
        "SPLIT_SUMMARY_JSON_PATH": tmp_path / "reports/split_summary.json",
        "DATA_CHECKS_PATH": tmp_path / "reports/experiments/data_checks.json",
        "EXAMPLES_PATH": tmp_path / "reports/experiments/demo_examples.json",
        "PREPARATION_STATE": tmp_path / "reports/experiments/demo_preparation.json",
        "FINAL_MODEL": tmp_path / "models/fraud_pipeline.joblib",
        "FINAL_METADATA": tmp_path / "models/fraud_pipeline.json",
        "SELECTION_LOCK": tmp_path / "models/selection_lock.json",
        "CONFIG_PATH": tmp_path / "src/ml/experiment_config.json",
    }
    for name, path in paths.items():
        monkeypatch.setattr(prepare, name, path)
    source_sha256 = "a" * 64
    monkeypatch.setattr(prepare, "source_sha256", lambda: source_sha256)
    write_json({"protocol_version": "synthetic-v1", "min_precision": 0.85}, paths["CONFIG_PATH"])
    seed = tmp_path / "synthetic_seed"
    seed.mkdir()
    X = pd.DataFrame(0.0, index=pd.Index(np.arange(100, 120, dtype="int32"), name=ID_COLUMN), columns=FEATURE_COLUMNS)
    X["V1"] = [0.9] * 9 + [0.1] * 9 + [0.1, 0.9]
    X["Amount"] = np.arange(1, 21, dtype="float64")
    dtypes = {name: "float32" for name in FEATURE_COLUMNS}
    dtypes["Amount"] = "float64"
    X = X.astype(dtypes)
    y = pd.Series([1] * 9 + [0] * 9 + [1, 0], index=X.index, name="Class", dtype="int8")
    model = SyntheticScores()
    scores = model.predict_proba(X)[:, 1]
    cache = pd.DataFrame({ID_COLUMN: X.index, "split": "validation", "y_true": y.to_numpy(),
                          "score": scores, "y_pred": (scores >= 0.5).astype("int8")})
    seed_cache = seed / "validation.parquet"
    cache.to_parquet(seed_cache, index=False)
    cache_relative = "reports/experiments/cache/synthetic-run/validation.parquet"
    cache_path = tmp_path / cache_relative
    processed = X.reset_index()
    processed["split"] = "validation"
    processed["Time"] = np.arange(len(processed), dtype="int32")
    processed["Class"] = y.to_numpy()
    processed = processed[PROCESSED_COLUMNS]
    seed_processed = seed / "transactions.parquet"
    processed.to_parquet(seed_processed, index=False)
    seed_manifest = seed / "manifest.parquet"
    processed[[ID_COLUMN, "split"]].to_parquet(seed_manifest, index=False)
    seed_raw = seed / "raw.csv"
    processed.drop(columns=[ID_COLUMN, "split"]).to_csv(seed_raw, index=False)
    metadata = {
        "schema_version": 1, "protocol_version": "synthetic-v1", "run_id": "synthetic-run",
        "feature_names": list(FEATURE_COLUMNS), "feature_dtypes": dtypes,
        "positive_label": 1, "threshold": 0.5, "comparison": ">=", "min_precision": 0.85,
        "frozen": True, "test_evaluated": False, "validation_rows": len(X),
        "model_path": "models/fraud_pipeline.joblib", "metadata_path": "models/fraud_pipeline.json",
        "source_paths": {"validation_scores": cache_relative}, "scores_sha256": file_sha256(seed_cache),
        "dataset_sha256": file_sha256(seed_raw), "processed_sha256": file_sha256(seed_processed),
        "split_sha256": file_sha256(seed_manifest), "source_sha256": source_sha256,
        "config_sha256": file_sha256(paths["CONFIG_PATH"]),
        "library_versions": {"python": platform.python_version(), **{
            name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn", "joblib", "pyarrow")
        }},
    }
    seed_model = seed / "fraud_pipeline.joblib"
    metadata = save_artifact(model, metadata, seed_model)
    paths["FINAL_METADATA"].parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(seed_model.with_suffix(".json"), paths["FINAL_METADATA"])
    lock = {name: metadata[name] for name in (
        "threshold", "comparison", "min_precision", "artifact_sha256", "dataset_sha256",
        "processed_sha256", "split_sha256", "source_sha256", "config_sha256",
    )}
    lock.update({"selected_run_id": metadata["run_id"], "metadata_sha256": file_sha256(paths["FINAL_METADATA"])})
    write_json(lock, paths["SELECTION_LOCK"])
    write_json({"source_file": {"sha256": metadata["dataset_sha256"]}}, paths["AUDIT_JSON_PATH"])
    write_json({"synthetic": True}, paths["SPLIT_SUMMARY_JSON_PATH"])
    audit = {"status": "ok", "raw_sha256": metadata["dataset_sha256"],
             "processed_sha256": metadata["processed_sha256"], "split_manifest_sha256": metadata["split_sha256"]}
    write_json(audit, paths["DATA_CHECKS_PATH"])
    requested_splits, commands = [], []

    def load_validation(split, *, path):
        assert split == "validation", "the demo must never load test rows"
        assert path == paths["TRANSACTIONS_PARQUET_PATH"]
        requested_splits.append(split)
        return X.copy(), y.copy()

    def copy(source, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    def run(command, *, cwd, check):
        assert command[0] == sys.executable and command[1] == "-m"
        assert cwd == tmp_path and check is True
        action = command[2:]
        commands.append(action)
        if action == ["src.data.cli", "download"]:
            copy(seed_raw, paths["RAW_CSV_PATH"])
        elif action == ["src.data.cli", "optimize"]:
            pass
        elif action == ["src.data.cli", "split"]:
            copy(seed_processed, paths["TRANSACTIONS_PARQUET_PATH"])
            copy(seed_manifest, paths["SPLIT_MANIFEST_PATH"])
        elif action == ["src.ml.data_checks", "--review-recurring"]:
            write_json(audit, paths["DATA_CHECKS_PATH"])
        elif action == ["src.ml.reproduce"]:
            copy(seed_model, paths["FINAL_MODEL"])
            copy(seed_cache, cache_path)
        elif action == ["src.ml.handoff", "verify"]:
            pass
        else:
            pytest.fail(f"unexpected preparation command: {action}")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(prepare, "load_xy", load_validation)
    monkeypatch.setattr(prepare.subprocess, "run", run)
    return {"paths": paths, "commands": commands, "requests": requested_splits,
            "X": X, "y": y, "metadata": metadata, "model": model, "cache_path": cache_path}


def test_fresh_preparation_uses_selected_recipe_and_saves_four_validation_outcomes(synthetic_demo):
    fixture = synthetic_demo
    result = prepare.prepare_demo()
    assert result["status"] == "prepared" and result["test_evaluated"] is False
    assert fixture["commands"] == [
        ["src.data.cli", "download"], ["src.data.cli", "optimize"], ["src.data.cli", "split"],
        ["src.ml.data_checks", "--review-recurring"], ["src.ml.reproduce"], ["src.ml.handoff", "verify"],
    ]
    document = read_json(fixture["paths"]["EXAMPLES_PATH"])
    assert document["source_split"] == "validation"
    assert document["selection"] == "One example per confusion-matrix outcome"
    assert {(item["actual_class"], item["expected_prediction"]) for item in document["examples"]} == {(0, 0), (0, 1), (1, 0), (1, 1)}
    for item in document["examples"]:
        assert set(item["features"]) == set(FEATURE_COLUMNS)
        assert all(type(value) in (float, int) and np.isfinite(value) for value in item["features"].values())
        assert item["actual_class"] == int(fixture["y"].loc[item[ID_COLUMN]])
    assert fixture["requests"] == ["validation"]
    assert read_json(fixture["paths"]["PREPARATION_STATE"])["hashes"] == result["hashes"]


def test_cached_preparation_is_offline_and_does_not_refit_or_reload_validation(synthetic_demo, monkeypatch):
    fixture = synthetic_demo
    prepare.prepare_demo()
    before = list(fixture["commands"])
    monkeypatch.setattr(prepare, "run_module", lambda *args: pytest.fail("cached preparation must not run subprocesses"))
    result = prepare.prepare_demo()
    assert result["status"] == "already_prepared" and result["steps_run"] == []
    assert fixture["commands"] == before
    assert fixture["requests"] == ["validation"]


@pytest.mark.parametrize("field", ["RAW_CSV_PATH", "TRANSACTIONS_PARQUET_PATH", "SPLIT_MANIFEST_PATH", "FINAL_MODEL", "EXAMPLES_PATH"])
def test_cached_preparation_detects_changed_files_without_network_or_fit(synthetic_demo, monkeypatch, field):
    fixture = synthetic_demo
    prepare.prepare_demo()
    with fixture["paths"][field].open("ab") as file:
        file.write(b"synthetic tampering")
    monkeypatch.setattr(prepare, "run_module", lambda *args: pytest.fail("tampered prepared data must not trigger implicit rebuilding"))
    with pytest.raises(ValueError, match="изменились|повреждена"):
        prepare.prepare_demo()


@pytest.mark.parametrize("problem", ["schema", "ids", "labels", "cache_checksum"])
def test_example_generation_rejects_inconsistent_validation_inputs(synthetic_demo, monkeypatch, problem):
    fixture = synthetic_demo
    prepare.prepare_demo()
    X, y = fixture["X"].copy(), fixture["y"].copy()
    if problem == "schema":
        X = X.drop(columns=["V1"])
    elif problem == "ids":
        X.index = X.index + 1000
    elif problem == "labels":
        y.iloc[0] = 0
    else:
        with fixture["cache_path"].open("ab") as file:
            file.write(b"synthetic changed cache")

    def validation_only(split, *, path):
        assert split == "validation"
        return X, y

    monkeypatch.setattr(prepare, "load_xy", validation_only)
    with pytest.raises(ValueError):
        prepare.build_demo_examples(fixture["model"], fixture["metadata"])


def test_missing_confusion_outcome_is_not_replaced_with_a_fake_example(synthetic_demo, monkeypatch):
    fixture = synthetic_demo
    prepare.prepare_demo()
    cache = pd.read_parquet(fixture["cache_path"])
    cache.loc[cache.index[-1], "y_true"] = 1  # The synthetic FP becomes TP.
    cache.to_parquet(fixture["cache_path"], index=False)
    y = fixture["y"].copy()
    y.iloc[-1] = 1
    metadata = dict(fixture["metadata"], scores_sha256=file_sha256(fixture["cache_path"]))
    monkeypatch.setattr(prepare, "load_xy", lambda split, *, path: (fixture["X"], y))
    with pytest.raises(ValueError, match="нет примера FP"):
        prepare.build_demo_examples(fixture["model"], metadata)
