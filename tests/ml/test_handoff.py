"""Пакет передачи проверяется только на синтетических данных во временной папке."""

import json
import platform
from importlib.metadata import version

import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS
from src.ml.artifacts import file_sha256, load_artifact, save_artifact
from src.ml.metrics import compute_metrics
from src.ml.pipelines import make_pipeline
from src.ml.prediction import predict_transactions


@pytest.fixture
def package(tmp_path, monkeypatch):
    import src.ml.handoff as handoff

    directory = tmp_path / "reports" / "experiments"
    directory.mkdir(parents=True)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"min_precision": 0.85, "model_ties": ["recall", "precision"]}))
    monkeypatch.setattr(handoff, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(handoff, "EXPERIMENTS_DIR", directory)
    monkeypatch.setattr(handoff, "CONFIG_PATH", config_path)
    monkeypatch.setattr(handoff, "FINAL_MODEL", tmp_path / "models" / "fraud_pipeline.joblib")
    monkeypatch.setattr(handoff, "FINAL_METADATA", tmp_path / "models" / "fraud_pipeline.json")
    monkeypatch.setattr(handoff, "SELECTION_LOCK", tmp_path / "models" / "selection_lock.json")

    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(scale=0.1, size=(24, 29)), columns=FEATURE_COLUMNS,
                     index=pd.Index(np.arange(1000, 1024), name="transaction_id"))
    y = np.tile([0, 1], 12)
    X["V1"] = np.where(y == 1, 4.0, -4.0)
    X["Amount"] = np.linspace(0, 100, len(X))
    dtypes = {name: "float32" if name != "Amount" else "float64" for name in FEATURE_COLUMNS}
    X = X.astype(dtypes)
    pipeline = make_pipeline("logistic_regression")
    pipeline.fit(X.iloc[:16], y[:16])
    X_validation, y_validation = X.iloc[16:].copy(), y[16:].copy()

    model_path = tmp_path / "models" / "runs" / "synthetic_logistic.joblib"
    scores_path = directory / "cache" / "synthetic_validation.parquet"
    metadata = {
        "schema_version": 1, "protocol_version": "synthetic-test-only", "run_id": "synthetic_logistic",
        "feature_names": list(FEATURE_COLUMNS), "feature_dtypes": dtypes,
        "positive_label": 1, "threshold": 0.5, "comparison": ">=", "min_precision": 0.85,
        "dataset_sha256": "a" * 64, "processed_sha256": "f" * 64,
        "split_sha256": "b" * 64, "source_sha256": "c" * 64,
        "config_sha256": file_sha256(config_path), "synthetic_fixture": True,
        "model_path": str(model_path.relative_to(tmp_path)),
        "metadata_path": str(model_path.with_suffix(".json").relative_to(tmp_path)),
        "scores_file": str(scores_path.relative_to(tmp_path)),
        "library_versions": {
            "python": platform.python_version(),
            **{name: version(name) for name in ("scikit-learn", "imbalanced-learn", "joblib", "numpy", "scipy")},
        },
    }
    predictions = predict_transactions(X_validation, pipeline, metadata)
    metadata["validation_metrics"] = compute_metrics(y_validation, predictions["score"], metadata["threshold"])
    cache = predictions.copy()
    cache.insert(0, "y_true", y_validation)
    cache.insert(0, "split", "validation")
    scores_path.parent.mkdir(parents=True)
    cache.reset_index().to_parquet(scores_path, index=False)
    metadata["scores_sha256"] = file_sha256(scores_path)
    selected = save_artifact(pipeline, metadata, model_path)
    (directory / "selected_run.json").write_text(json.dumps(selected))

    def load_synthetic_results(path):
        assert path == directory / "results"
        return [selected]

    def choose_synthetic_result(results, config):
        assert config["min_precision"] == 0.85
        assert config["config_sha256"] == file_sha256(config_path)
        return results[0]

    calls = []

    def load_validation_only(split):
        calls.append(split)
        assert split == "validation", "handoff must never request test rows"
        return X_validation.copy(), y_validation.copy()

    monkeypatch.setattr(handoff, "load_results", load_synthetic_results)
    monkeypatch.setattr(handoff, "choose_best", choose_synthetic_result)
    monkeypatch.setattr(handoff, "load_xy", load_validation_only)
    return {
        "handoff": handoff, "pipeline": pipeline, "selected": selected,
        "X_validation": X_validation, "y_validation": y_validation,
        "predictions": predictions, "calls": calls, "source_model": model_path,
        "source_metadata": model_path.with_suffix(".json"), "scores_path": scores_path,
        "experiments": directory,
    }


def test_freeze_reuses_fitted_pipeline_and_is_idempotent(package, monkeypatch):
    handoff = package["handoff"]

    def forbidden_fit(*args, **kwargs):
        pytest.fail("freeze must preserve the fitted train pipeline, not fit it again")

    monkeypatch.setattr(type(package["pipeline"]), "fit", forbidden_fit)
    lock = handoff.freeze_model()
    assert handoff.FINAL_MODEL.is_file() and handoff.FINAL_METADATA.is_file() and handoff.SELECTION_LOCK.is_file()
    assert lock["selected_run_id"] == package["selected"]["run_id"]
    assert lock["artifact_sha256"] == file_sha256(handoff.FINAL_MODEL)
    assert lock["metadata_sha256"] == file_sha256(handoff.FINAL_METADATA)
    assert lock["processed_sha256"] == package["selected"]["processed_sha256"]
    assert lock["threshold"] == 0.5 and lock["min_precision"] == 0.85
    assert lock["comparison"] == ">=" and lock["evaluation_owner"] == "Adam"
    assert not lock["test_evaluated"]

    loaded, metadata = load_artifact(handoff.FINAL_MODEL)
    pd.testing.assert_frame_equal(
        predict_transactions(package["X_validation"], loaded, metadata), package["predictions"],
    )
    np.testing.assert_array_equal(loaded.named_steps["clf"].coef_, package["pipeline"].named_steps["clf"].coef_)
    assert metadata["frozen"] and not metadata["refit_on_train_validation"]
    assert metadata["source_paths"]["model"] == package["selected"]["model_path"]
    before = {path: path.read_bytes() for path in (handoff.FINAL_MODEL, handoff.FINAL_METADATA, handoff.SELECTION_LOCK)}
    monkeypatch.setattr(handoff, "save_artifact", lambda *args: pytest.fail("unchanged selection was frozen twice"))
    assert handoff.freeze_model() == lock
    assert all(path.read_bytes() == contents for path, contents in before.items())
    assert package["calls"] == []


def test_freeze_refuses_an_inconsistent_saved_selection(package):
    handoff = package["handoff"]
    (package["experiments"] / "selected_run.json").write_text(json.dumps({"run_id": "another_run"}))
    with pytest.raises(ValueError, match="selection is inconsistent"):
        handoff.freeze_model()
    assert not handoff.FINAL_MODEL.exists()


def test_freeze_refuses_to_replace_a_different_frozen_run(package):
    handoff = package["handoff"]
    handoff.freeze_model()
    package["selected"]["run_id"] = "another_run"
    (package["experiments"] / "selected_run.json").write_text(json.dumps(package["selected"]))
    with pytest.raises(ValueError, match="different model is already frozen"):
        handoff.freeze_model()


@pytest.mark.parametrize("target", ["source_model", "source_metadata"])
def test_freeze_refuses_changed_source_artifact_or_metadata(package, target):
    path = package[target]
    if target == "source_model":
        with path.open("ab") as file:
            file.write(b"changed synthetic artifact")
    else:
        metadata = json.loads(path.read_text())
        metadata["threshold"] = 0.6
        path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="changed|disagrees"):
        package["handoff"].freeze_model()


def test_freeze_rejects_processed_hash_mismatch_between_sidecar_and_selected_run(package):
    path = package["source_metadata"]
    metadata = json.loads(path.read_text())
    metadata["processed_sha256"] = "e" * 64
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="disagrees.*processed_sha256"):
        package["handoff"].freeze_model()
    assert not package["handoff"].FINAL_MODEL.exists()


@pytest.mark.parametrize("target", ["FINAL_MODEL", "FINAL_METADATA"])
def test_frozen_file_tampering_is_rejected_by_freeze_and_verify(package, target):
    handoff = package["handoff"]
    handoff.freeze_model()
    with getattr(handoff, target).open("ab") as file:
        file.write(b"changed frozen file")
    with pytest.raises(ValueError, match="Frozen package changed"):
        handoff.freeze_model()
    with pytest.raises(ValueError, match="does not match the selection lock"):
        handoff.verify_package()


@pytest.mark.parametrize("field,value", [
    ("threshold", 0.6), ("selected_run_id", "wrong_run"),
    ("source_sha256", "e" * 64), ("processed_sha256", "e" * 64),
])
def test_semantic_lock_changes_are_rejected(package, field, value):
    handoff = package["handoff"]
    handoff.freeze_model()
    lock = json.loads(handoff.SELECTION_LOCK.read_text())
    lock[field] = value
    handoff.SELECTION_LOCK.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="disagrees"):
        handoff.verify_package()
    with pytest.raises(ValueError, match="disagrees|different model"):
        handoff.freeze_model()


def test_verify_reads_only_validation_and_saves_labeled_synthetic_example(package):
    handoff = package["handoff"]
    handoff.freeze_model()
    report = handoff.verify_package()
    assert package["calls"] == ["validation"]
    assert report["validation_rows_checked"] == len(package["X_validation"])
    assert report["scores_identical_after_reload"] and report["classes_identical_after_reload"]
    assert report["metrics_identical_after_reload"] and report["invalid_schema_rejected"]
    assert "missing" in report["schema_error"]
    assert not report["test_evaluated"]
    assert report["validation_metrics"] == package["selected"]["validation_metrics"]
    saved = json.loads((package["experiments"] / "handoff_verification.json").read_text())
    assert saved == report
    example = json.loads((package["experiments"] / "synthetic_prediction_check.json").read_text())
    assert example["synthetic"] is True
    assert "not observed transactions" in example["purpose"]
    assert len(example["input"]) == len(example["predictions"]) == 3
    assert [row["example_id"] for row in example["input"]] == ["synthetic_1", "synthetic_2", "synthetic_3"]


def test_changed_validation_scores_cache_is_rejected_before_loading_data(package):
    handoff = package["handoff"]
    handoff.freeze_model()
    with package["scores_path"].open("ab") as file:
        file.write(b"changed validation cache")
    with pytest.raises(ValueError, match="validation predictions changed"):
        handoff.verify_package()
    assert package["calls"] == []


@pytest.mark.parametrize("problem", ["ids", "labels", "features"])
def test_changed_validation_population_or_predictions_are_rejected(package, monkeypatch, problem):
    handoff = package["handoff"]
    handoff.freeze_model()
    X, y = package["X_validation"].copy(), package["y_validation"].copy()
    if problem == "ids":
        X.index = X.index + 100
    elif problem == "labels":
        y[0] = 1 - y[0]
    else:
        X.iloc[0, X.columns.get_loc("V1")] += 0.1

    def changed_validation(split):
        assert split == "validation"
        return X, y

    monkeypatch.setattr(handoff, "load_xy", changed_validation)
    with pytest.raises(ValueError, match="ids or labels|Probabilities or classes"):
        handoff.verify_package()


def test_verify_detects_a_broken_schema_guard(package, monkeypatch):
    handoff = package["handoff"]
    handoff.freeze_model()
    original = handoff.predict_transactions

    def broken_guard(X, pipeline, metadata):
        X = X.copy()
        for name in metadata["feature_names"]:
            if name not in X:
                X[name] = 0.0
        return original(X, pipeline, metadata)

    monkeypatch.setattr(handoff, "predict_transactions", broken_guard)
    with pytest.raises(ValueError, match="Invalid feature schema was unexpectedly accepted"):
        handoff.verify_package()
