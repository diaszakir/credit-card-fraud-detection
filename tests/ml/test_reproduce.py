"""Восстановление из рецепта на синтетических данных, без чтения настоящего test."""

import errno
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from src.config import FEATURE_COLUMNS
from src.ml import reproduce
from src.ml.artifacts import file_sha256, load_artifact, save_artifact
from src.ml.experiment import library_versions, write_atomic_json
from src.ml.metrics import compute_metrics
from src.ml.pipelines import make_pipeline
from src.ml.prediction import predict_transactions


@pytest.fixture
def recipe(tmp_path, monkeypatch):
    paths = {
        "ROOT_DIR": tmp_path,
        "FINAL_MODEL": tmp_path / "models" / "fraud_pipeline.joblib",
        "FINAL_METADATA": tmp_path / "models" / "fraud_pipeline.json",
        "SELECTION_LOCK": tmp_path / "models" / "selection_lock.json",
        "RAW_CSV_PATH": tmp_path / "data" / "raw.csv",
        "TRANSACTIONS_PARQUET_PATH": tmp_path / "data" / "processed.parquet",
        "SPLIT_MANIFEST_PATH": tmp_path / "data" / "manifest.parquet",
        "CONFIG_PATH": tmp_path / "config.json",
        "DATA_CHECKS_PATH": tmp_path / "data_checks.json",
        "REPORT_PATH": tmp_path / "reports" / "reproduction_check.json",
    }
    for name, path in paths.items():
        monkeypatch.setattr(reproduce, name, path)
    for name in ("RAW_CSV_PATH", "TRANSACTIONS_PARQUET_PATH", "SPLIT_MANIFEST_PATH"):
        paths[name].parent.mkdir(parents=True, exist_ok=True)
        paths[name].write_bytes((name + " synthetic provenance").encode())
    config = {
        "seed": 42, "min_precision": 0.85, "positive_label": 1, "comparison": ">=",
        "protocol_version": "synthetic-reproduction-v1",
        "resource_policy": {"n_jobs": 1, "blas_threads": 1},
    }
    write_atomic_json(config, paths["CONFIG_PATH"])
    monkeypatch.setattr(reproduce, "source_sha256", lambda: "f" * 64)

    rng = np.random.default_rng(42)
    dtypes = {name: "float32" for name in FEATURE_COLUMNS}
    dtypes["Amount"] = "float64"
    frames, targets = {}, {}
    for split, n_rows, first_id in (("train", 160, 100), ("validation", 60, 500)):
        y = pd.Series((np.arange(n_rows) % 10 == 0).astype("int8"),
                      index=pd.Index(np.arange(first_id, first_id + n_rows), name="transaction_id"))
        X = pd.DataFrame(rng.normal(size=(n_rows, 29)), columns=FEATURE_COLUMNS, index=y.index)
        X["V1"] = np.where(y == 1, 5.0, -5.0)
        X["Amount"] = rng.uniform(1, 100, size=n_rows)
        frames[split], targets[split] = X.astype(dtypes), y
    params = {"clf__C": 0.1}
    pipeline = make_pipeline("logistic_regression", seed=42, params=params)
    with threadpool_limits(limits=1):
        pipeline.fit(frames["train"], targets["train"])
    metadata = {
        "schema_version": 1, "protocol_version": config["protocol_version"],
        "run_id": "synthetic_frozen", "model_version": "synthetic_frozen",
        "model_name": "logistic_regression", "imbalance_strategy": "none", "params": params,
        "seed": 42, "min_precision": 0.85, "positive_label": 1, "comparison": ">=", "threshold": 0.5,
        "feature_names": list(FEATURE_COLUMNS), "feature_dtypes": dtypes,
        "dataset_sha256": file_sha256(paths["RAW_CSV_PATH"]),
        "processed_sha256": file_sha256(paths["TRANSACTIONS_PARQUET_PATH"]),
        "split_sha256": file_sha256(paths["SPLIT_MANIFEST_PATH"]),
        "config_sha256": file_sha256(paths["CONFIG_PATH"]), "source_sha256": "f" * 64,
        "library_versions": library_versions(), "frozen": True, "frozen_at": "synthetic timestamp",
        "test_evaluated": False, "train_rows": len(frames["train"]),
        "validation_rows": len(frames["validation"]),
        "source_paths": {"validation_scores": "reports/cache/synthetic_frozen/validation.parquet"},
        "model_path": "models/fraud_pipeline.joblib", "metadata_path": "models/fraud_pipeline.json",
        "scores_sha256": "a" * 64,
    }
    predictions = predict_transactions(frames["validation"], pipeline, metadata)
    metadata.update({
        "validation_scores_values_sha256": reproduce.array_sha256(predictions["score"], "<f8"),
        "validation_ids_sha256": reproduce.array_sha256(frames["validation"].index, "<i8"),
        "validation_labels_sha256": reproduce.array_sha256(targets["validation"], "i1"),
        "validation_metrics": compute_metrics(targets["validation"], predictions["score"], 0.5, 0.85),
    })
    saved = save_artifact(pipeline, metadata, paths["FINAL_MODEL"])
    lock = {
        "selected_run_id": metadata["run_id"],
        **{key: saved[key] for key in (
            "threshold", "comparison", "min_precision", "artifact_sha256",
            "dataset_sha256", "processed_sha256", "split_sha256", "source_sha256", "config_sha256",
        )},
        "metadata_sha256": file_sha256(paths["FINAL_METADATA"]),
        "frozen_at": "synthetic timestamp", "test_evaluated": False,
    }
    write_atomic_json(lock, paths["SELECTION_LOCK"])
    write_atomic_json({
        "status": "ok", "raw_sha256": metadata["dataset_sha256"],
        "processed_sha256": metadata["processed_sha256"],
        "split_manifest_sha256": metadata["split_sha256"],
    }, paths["DATA_CHECKS_PATH"])
    paths["FINAL_MODEL"].unlink()  # Новый клон содержит рецепт, но не ignored joblib.
    calls = []

    def load_synthetic(split):
        assert split in ("train", "validation"), "test must not be opened"
        calls.append(split)
        return frames[split].copy(), targets[split].copy()

    monkeypatch.setattr(reproduce, "load_xy", load_synthetic)
    return {"paths": paths, "metadata": saved, "lock": lock, "calls": calls,
            "frames": frames, "targets": targets, "predictions": predictions}


def update_recipe(recipe, change):
    paths = recipe["paths"]
    metadata = json.loads(paths["FINAL_METADATA"].read_text())
    change(metadata)
    write_atomic_json(metadata, paths["FINAL_METADATA"])
    lock = json.loads(paths["SELECTION_LOCK"].read_text())
    lock["metadata_sha256"] = file_sha256(paths["FINAL_METADATA"])
    write_atomic_json(lock, paths["SELECTION_LOCK"])


def test_missing_artifact_reproduces_only_train_and_keeps_frozen_threshold(recipe):
    report = reproduce.reproduce_selected()
    paths = recipe["paths"]
    saved = json.loads(paths["FINAL_METADATA"].read_text())
    lock = json.loads(paths["SELECTION_LOCK"].read_text())
    assert recipe["calls"] == ["train", "validation"]
    assert report["status"] == "reproduced"
    assert report["fitted_splits"] == ["train"]
    assert report["evaluated_splits"] == ["validation"]
    assert not report["cv_performed"] and not report["threshold_retuned"] and not report["test_evaluated"]
    assert all(report[name] for name in (
        "scores_values_match", "validation_ids_match", "validation_labels_match", "validation_metrics_match",
    ))
    assert saved["threshold"] == recipe["metadata"]["threshold"]
    assert saved["reproduced_from_artifact_sha256"] == recipe["metadata"]["artifact_sha256"]
    assert saved["artifact_sha256"] == file_sha256(paths["FINAL_MODEL"])
    assert lock["metadata_sha256"] == file_sha256(paths["FINAL_METADATA"])
    for name, value in recipe["lock"].items():
        if name not in ("artifact_sha256", "metadata_sha256"):
            assert lock[name] == value
    cache = paths["ROOT_DIR"] / saved["source_paths"]["validation_scores"]
    assert saved["scores_sha256"] == file_sha256(cache)
    stored = pd.read_parquet(cache)
    assert stored["split"].eq("validation").all()
    assert np.array_equal(stored["score"], recipe["predictions"]["score"])
    loaded, loaded_metadata = load_artifact(paths["FINAL_MODEL"])
    pd.testing.assert_frame_equal(
        predict_transactions(recipe["frames"]["validation"], loaded, loaded_metadata), recipe["predictions"],
    )
    assert json.loads(paths["REPORT_PATH"].read_text()) == report


def test_existing_artifact_is_verified_without_refitting(recipe, monkeypatch):
    reproduce.reproduce_selected()
    paths = recipe["paths"]
    before = {name: paths[name].read_bytes() for name in ("FINAL_MODEL", "FINAL_METADATA", "SELECTION_LOCK", "REPORT_PATH")}
    monkeypatch.setattr(reproduce, "load_xy", lambda _: pytest.fail("existing artifact must not be refitted"))
    report = reproduce.reproduce_selected()
    assert report["status"] == "already_present" and not report["fit_performed"]
    assert all(paths[name].read_bytes() == content for name, content in before.items())


def test_reproduction_publishes_cache_with_separate_models_and_reports_volumes(recipe, monkeypatch):
    paths = recipe["paths"]
    models_dir = paths["FINAL_MODEL"].parent.resolve()
    reports_dir = (paths["ROOT_DIR"] / "reports").resolve()
    cache = paths["ROOT_DIR"] / recipe["metadata"]["source_paths"]["validation_scores"]
    original_replace = Path.replace
    replacements = []

    def replace_on_same_volume(source, destination):
        destination = Path(destination)
        replacements.append((source.resolve(), destination.resolve()))
        if source.resolve().is_relative_to(models_dir) and destination.resolve().is_relative_to(reports_dir):
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        return original_replace(source, destination)

    monkeypatch.setattr(Path, "replace", replace_on_same_volume)
    report = reproduce.reproduce_selected()

    assert report["status"] == "reproduced"
    assert recipe["calls"] == ["train", "validation"]
    assert report["scores_values_match"] and report["validation_metrics_match"]
    assert not report["threshold_retuned"] and not report["test_evaluated"]
    cache_sources = [source for source, destination in replacements if destination == cache.resolve()]
    assert len(cache_sources) == 1 and cache_sources[0].is_relative_to(cache.parent.resolve())
    loaded, metadata = load_artifact(paths["FINAL_MODEL"])
    assert metadata["threshold"] == recipe["metadata"]["threshold"]
    assert metadata["scores_sha256"] == file_sha256(cache)
    assert metadata["validation_scores_values_sha256"] == recipe["metadata"]["validation_scores_values_sha256"]
    pd.testing.assert_frame_equal(
        predict_transactions(recipe["frames"]["validation"], loaded, metadata), recipe["predictions"],
    )


@pytest.mark.parametrize("field", list(reproduce.VALIDATION_DIGESTS))
def test_changed_validation_values_do_not_publish_any_artifact(recipe, field):
    update_recipe(recipe, lambda metadata: metadata.update({field: "0" * 64}))
    paths = recipe["paths"]
    before = {name: paths[name].read_bytes() for name in ("FINAL_METADATA", "SELECTION_LOCK")}
    cache = paths["ROOT_DIR"] / recipe["metadata"]["source_paths"]["validation_scores"]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(b"existing cache must not change")
    with pytest.raises(ValueError, match=field):
        reproduce.reproduce_selected()
    assert not paths["FINAL_MODEL"].exists()
    assert not paths["REPORT_PATH"].exists()
    assert cache.read_bytes() == b"existing cache must not change"
    assert all(paths[name].read_bytes() == content for name, content in before.items())


def test_changed_validation_metrics_are_refused_before_publication(recipe):
    update_recipe(recipe, lambda metadata: metadata["validation_metrics"].update({"recall": 0.0}))
    with pytest.raises(ValueError, match="metrics differ"):
        reproduce.reproduce_selected()
    assert not recipe["paths"]["FINAL_MODEL"].exists()


@pytest.mark.parametrize("name", ["RAW_CSV_PATH", "TRANSACTIONS_PARQUET_PATH", "SPLIT_MANIFEST_PATH", "CONFIG_PATH"])
def test_changed_provenance_is_rejected_before_loading_train(recipe, name):
    path = recipe["paths"][name]
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError):
        reproduce.reproduce_selected()
    assert recipe["calls"] == []


def test_changed_source_is_refused_before_fit(recipe, monkeypatch):
    monkeypatch.setattr(reproduce, "source_sha256", lambda: "0" * 64)
    with pytest.raises(ValueError, match="source differs"):
        reproduce.reproduce_selected()
    assert recipe["calls"] == []


def test_incompatible_environment_is_refused_before_fit(recipe):
    update_recipe(recipe, lambda metadata: metadata["library_versions"].update({"numpy": "0.0.0"}))
    with pytest.raises(ValueError, match="numpy==0.0.0"):
        reproduce.reproduce_selected()
    assert recipe["calls"] == []


def test_missing_environment_versions_are_refused_before_fit(recipe):
    update_recipe(recipe, lambda metadata: metadata["library_versions"].pop("pandas"))
    with pytest.raises(ValueError, match="complete library_versions"):
        reproduce.reproduce_selected()
    assert recipe["calls"] == []


def test_missing_validation_digest_does_not_trigger_new_selection(recipe):
    update_recipe(recipe, lambda metadata: metadata.pop("validation_scores_values_sha256"))
    with pytest.raises(ValueError, match="requires validation_scores_values_sha256"):
        reproduce.reproduce_selected()
    assert recipe["calls"] == []


def test_unresolved_data_check_is_refused_before_fit(recipe):
    path = recipe["paths"]["DATA_CHECKS_PATH"]
    audit = json.loads(path.read_text())
    write_atomic_json({**audit, "status": "needs_duplicate_review"}, path)
    with pytest.raises(ValueError, match="resolve their findings"):
        reproduce.reproduce_selected()
    assert recipe["calls"] == []


def test_corrupt_existing_artifact_is_not_overwritten_or_refitted(recipe, monkeypatch):
    path = recipe["paths"]["FINAL_MODEL"]
    path.write_bytes(b"corrupt existing model")
    monkeypatch.setattr(reproduce, "load_xy", lambda _: pytest.fail("corrupt model must not trigger refit"))
    with pytest.raises(ValueError, match="will not overwrite"):
        reproduce.reproduce_selected()
    assert path.read_bytes() == b"corrupt existing model"


def test_source_digest_uses_the_fixed_posix_names(tmp_path, monkeypatch):
    names = (
        "src/ml/experiment.py", "src/ml/pipelines.py", "src/ml/metrics.py",
        "src/ml/prediction.py", "src/ml/data_checks.py", "src/ml/artifacts.py",
        "src/ml/cli.py", "src/config.py", "src/data/api.py",
    )
    expected = hashlib.sha256()
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = ("frozen bytes for " + name + "\n").encode()
        path.write_bytes(payload)
        # Literal names retain slash even when pathlib is native WindowsPath.
        expected.update(name.encode())
        expected.update(payload)
    monkeypatch.setattr(reproduce, "ROOT_DIR", tmp_path)
    assert reproduce.source_sha256() == expected.hexdigest()


@pytest.mark.skipif(os.name == "nt", reason="The original worker uses native Windows separators")
def test_reproduction_digest_matches_the_original_worker_on_posix():
    from src.ml.experiment import source_sha256 as worker_source_sha256

    assert reproduce.source_sha256() == worker_source_sha256()
