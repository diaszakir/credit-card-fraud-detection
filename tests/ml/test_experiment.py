"""Synthetic worker integration; no real transactions or saved project runs are used."""

import hashlib
import json
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest

import src.config as project_config
import src.ml.experiment as experiment
from src.config import FEATURE_COLUMNS, ID_COLUMN, PROCESSED_COLUMNS, SPLIT_NAMES, V_COLUMNS
from src.data.io_utils import read_json, write_json
from src.ml.artifacts import file_sha256, load_artifact
from src.ml.data_checks import make_time_folds, summarize_time_folds
from src.ml.prediction import predict_transactions


def make_synthetic_frame(split, n_rows, first_id, first_time, rng):
    """Clearly separated synthetic classes make a tiny LR run deterministic."""
    labels = (np.arange(n_rows) % 5 == 0).astype("int8")
    data = {
        ID_COLUMN: np.arange(first_id, first_id + 2 * n_rows, 2, dtype="int32"),
        "split": pd.Categorical([split] * n_rows, categories=list(SPLIT_NAMES), ordered=True),
        "Time": np.arange(first_time, first_time + n_rows, dtype="int32"),
    }
    for column in V_COLUMNS:
        data[column] = rng.normal(0, 0.05, n_rows).astype("float32")
    data["V1"] = ((2 * labels.astype(int) - 1) * 4 + rng.normal(0, 0.05, n_rows)).astype("float32")
    data["Amount"] = rng.uniform(1, 10, n_rows).astype("float64")
    data["Class"] = labels
    return pd.DataFrame(data)[PROCESSED_COLUMNS]


@pytest.fixture
def isolated_worker(tmp_path, monkeypatch):
    """All metadata, pipelines and caches stay in a temporary synthetic project."""
    rng = np.random.default_rng(7)
    frames = {
        "train": make_synthetic_frame("train", 160, 0, 0, rng),
        "validation": make_synthetic_frame("validation", 40, 1000, 1000, rng),
    }
    config = read_json(experiment.CONFIG_PATH)
    requirements = (project_config.ROOT_DIR / "requirements.txt").read_text(encoding="utf-8")
    (tmp_path / "requirements.txt").write_text(requirements, encoding="utf-8")
    config_path = write_json(config, tmp_path / "experiment_config.json")
    experiments_dir = tmp_path / "reports" / "experiments"
    processed_path = tmp_path / "data" / "transactions.parquet"
    manifest_path = tmp_path / "data" / "manifest.parquet"
    requests = []

    def write_inputs():
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        all_rows = pd.concat([frames["train"], frames["validation"]], ignore_index=True)
        all_rows.to_parquet(processed_path, index=False)
        all_rows[[ID_COLUMN, "split"]].to_parquet(manifest_path, index=False)
        folds = make_time_folds(frames["train"])
        cv_summary = summarize_time_folds(frames["train"], folds)
        write_json({
            "status": "ok",
            "raw_sha256": hashlib.sha256(b"SYNTHETIC source for worker integration only").hexdigest(),
            "processed_sha256": file_sha256(processed_path),
            "split_manifest_sha256": file_sha256(manifest_path),
            "cv": {
                "n_splits": 3,
                "smote_k_neighbors": 5,
                "folds": cv_summary,
                "folds_sha256": hashlib.sha256(json.dumps(cv_summary, sort_keys=True).encode()).hexdigest(),
            },
        }, experiments_dir / "data_checks.json")

    def load_only_train_validation(*, split):
        assert split in ("train", "validation"), "test rows must never be requested by the ML worker"
        requests.append(split)
        return frames[split].copy()

    monkeypatch.setattr(experiment, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(experiment, "CONFIG_PATH", config_path)
    monkeypatch.setattr(experiment, "EXPERIMENTS_DIR", experiments_dir)
    monkeypatch.setattr(experiment, "RESULTS_DIR", experiments_dir / "results")
    monkeypatch.setattr(experiment, "CACHE_DIR", experiments_dir / "cache")
    monkeypatch.setattr(experiment, "MODELS_DIR", tmp_path / "models" / "runs")
    monkeypatch.setattr(project_config, "TRANSACTIONS_PARQUET_PATH", processed_path)
    monkeypatch.setattr(project_config, "SPLIT_MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(experiment, "load_transactions", load_only_train_validation)
    # Source files may be edited by another stage while this synthetic test runs.
    monkeypatch.setattr(experiment, "source_sha256", lambda: "synthetic-worker-source")
    monkeypatch.setattr(experiment, "peak_memory", lambda: (None, "synthetic test; RSS is not compared"))
    monkeypatch.setattr(experiment, "subprocess", SimpleNamespace(
        run=lambda *args, **kwargs: SimpleNamespace(stdout="synthetic-revision\n", returncode=0),
    ))
    write_inputs()
    return {
        "root": tmp_path,
        "frames": frames,
        "requests": requests,
        "processed": processed_path,
        "manifest": manifest_path,
        "write_inputs": write_inputs,
    }


def test_dummy_run_saves_complete_artifact_and_reuses_cache_without_fit(isolated_worker, monkeypatch):
    fixture = isolated_worker
    result = experiment.run_experiment("dummy")
    root = fixture["root"]
    model_path = root / result["model_path"]
    scores_path = root / result["scores_file"]
    metadata_path = root / result["metadata_path"]
    catalogue_path = experiment.RESULTS_DIR / (result["run_id"] + ".json")

    assert result["status"] == "no_feasible_threshold"
    assert result["threshold"] is None and result["test_evaluated"] is False
    assert result["selection_population"] == "full_validation"
    assert result["min_precision"] == 0.85
    assert result["train_rows"] == 160 and result["validation_rows"] == 40
    assert result["artifact_sha256"] == file_sha256(model_path)
    assert result["scores_sha256"] == file_sha256(scores_path)
    assert read_json(metadata_path) == read_json(catalogue_path) == result
    pipeline = joblib.load(model_path)  # This artifact was produced by this test only.
    assert list(pipeline.classes_) == [0, 1]
    scores = pd.read_parquet(scores_path)
    assert scores[ID_COLUMN].tolist() == fixture["frames"]["validation"][ID_COLUMN].tolist()
    assert scores["split"].eq("validation").all()
    assert scores["y_pred"].isna().all()
    assert scores["score"].between(0, 1).all()
    assert fixture["requests"] == ["train", "validation"]

    def unexpected_fit(*args, **kwargs):
        pytest.fail("a completed, verified run must be reused without fitting again")

    monkeypatch.setattr(experiment, "fit_with_cv", unexpected_fit)
    assert experiment.run_experiment("dummy") == result
    assert fixture["requests"] == ["train", "validation"]


@pytest.mark.parametrize("changed_input", ["processed", "manifest"])
def test_changed_input_is_refused_before_returning_a_cached_run(isolated_worker, monkeypatch, changed_input):
    fixture = isolated_worker
    experiment.run_experiment("dummy")
    with fixture[changed_input].open("ab") as file:
        file.write(b"synthetic change after the recorded data audit")
    monkeypatch.setattr(experiment, "fit_with_cv", lambda *args, **kwargs: pytest.fail("changed inputs must be rejected before fit"))
    with pytest.raises(ValueError, match="Data changed"):
        experiment.run_experiment("dummy")
    assert fixture["requests"] == ["train", "validation"]


def test_corrupt_cached_model_is_refused_without_refitting(isolated_worker, monkeypatch):
    fixture = isolated_worker
    result = experiment.run_experiment("dummy")
    with (fixture["root"] / result["model_path"]).open("ab") as file:
        file.write(b"synthetic cache corruption")
    monkeypatch.setattr(experiment, "fit_with_cv", lambda *args, **kwargs: pytest.fail("a corrupt cache must not trigger an implicit fit"))
    with pytest.raises(ValueError, match="Cached files are inconsistent"):
        experiment.run_experiment("dummy")


def test_single_class_novel_subset_is_unavailable_without_failing_full_validation(isolated_worker):
    fixture = isolated_worker
    train = fixture["frames"]["train"]
    validation = fixture["frames"]["validation"]
    known_fraud = train.loc[train["Class"] == 1, FEATURE_COLUMNS].iloc[0]
    validation.loc[validation["Class"] == 1, FEATURE_COLUMNS] = known_fraud.to_numpy()
    fixture["write_inputs"]()

    result = experiment.run_experiment("dummy")
    assert result["status"] == "no_feasible_threshold"
    assert result["novel_validation_rows"] == 32
    assert result["novel_validation_fraud"] == 0
    assert result["novel_validation_metrics"]["status"] == "unavailable"
    assert "average_precision" in result["validation_metrics"]
    assert result["test_evaluated"] is False


def test_lr_artifact_uses_shared_prediction_function_and_saved_threshold(isolated_worker):
    fixture = isolated_worker
    result = experiment.run_experiment("logistic_regression")
    assert result["status"] == "ok"
    assert result["validation_metrics"]["precision_constraint_met"] is True
    pipeline, metadata = load_artifact(fixture["root"] / result["model_path"])
    X = fixture["frames"]["validation"].set_index(ID_COLUMN)[FEATURE_COLUMNS]
    predictions = predict_transactions(X, pipeline, metadata)
    scores = pd.read_parquet(fixture["root"] / result["scores_file"])
    assert np.array_equal(predictions["score"].to_numpy(), scores["score"].to_numpy())
    assert np.array_equal(predictions["y_pred"].to_numpy(), scores["y_pred"].to_numpy())
    assert predictions.index.tolist() == X.index.tolist()
    assert predictions["threshold"].eq(metadata["threshold"]).all()
    assert fixture["requests"] == ["train", "validation"]
