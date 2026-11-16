"""Синтетическая проверка передачи Pipeline и метаданных без реальных транзакций."""

import json
import platform
from importlib.metadata import version

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.config import FEATURE_COLUMNS
from src.ml.artifacts import file_sha256, load_artifact, save_artifact
from src.ml.pipelines import make_pipeline
from src.ml.prediction import predict_transactions


@pytest.fixture
def trained_artifact():
    X, y = make_classification(
        n_samples=120, n_features=29, n_informative=5, n_redundant=0,
        weights=[0.9, 0.1], random_state=42,
    )
    X = pd.DataFrame(X, columns=FEATURE_COLUMNS, index=np.arange(1000, 1120))
    dtypes = {name: "float32" for name in FEATURE_COLUMNS}
    dtypes["Amount"] = "float64"
    X = X.astype(dtypes)
    pipeline = make_pipeline("logistic_regression")
    pipeline.fit(X, y)
    metadata = {
        "schema_version": 1,
        "protocol_version": "1.0",
        "feature_names": list(FEATURE_COLUMNS),
        "feature_dtypes": dtypes,
        "positive_label": np.int64(1),
        "threshold": np.float64(0.5),
        "comparison": ">=",
        "min_precision": np.float64(0.85),
        "library_versions": {
            "python": platform.python_version(),
            **{name: version(name) for name in ("scikit-learn", "imbalanced-learn", "joblib", "numpy", "scipy")},
        },
    }
    return X, pipeline, metadata


def test_saved_pipeline_has_identical_predictions_after_loading(tmp_path, trained_artifact):
    X, pipeline, metadata = trained_artifact
    path = tmp_path / "nested" / "model.joblib"
    before = predict_transactions(X, pipeline, metadata)
    saved = save_artifact(pipeline, metadata, path)
    loaded, loaded_metadata = load_artifact(path)
    after = predict_transactions(X, loaded, loaded_metadata)

    pd.testing.assert_frame_equal(before, after)
    assert saved == loaded_metadata
    assert saved["artifact_sha256"] == file_sha256(path)
    assert saved["artifact_file"] == path.name
    assert type(saved["positive_label"]) is int
    assert type(saved["threshold"]) is float
    assert "artifact_sha256" not in metadata
    assert not list(tmp_path.rglob("*.tmp"))


def test_checksum_is_checked_before_joblib_load(tmp_path, trained_artifact, monkeypatch):
    _, pipeline, metadata = trained_artifact
    path = tmp_path / "model.joblib"
    save_artifact(pipeline, metadata, path)
    with path.open("ab") as file:
        file.write(b"damaged")
    monkeypatch.setattr("src.ml.artifacts.joblib.load", lambda _: pytest.fail("unsafe load was attempted"))
    with pytest.raises(ValueError, match="SHA-256"):
        load_artifact(path)


@pytest.mark.parametrize("field,value", [
    ("schema_version", 2), ("schema_version", True),
    ("protocol_version", ""), ("positive_label", 0), ("comparison", ">"),
    ("threshold", float("nan")), ("threshold", 1.1), ("threshold", True),
    ("min_precision", 0.9),
])
def test_invalid_metadata_is_refused_before_writing(tmp_path, trained_artifact, field, value):
    _, pipeline, metadata = trained_artifact
    invalid = {**metadata, field: value}
    path = tmp_path / "model.joblib"
    with pytest.raises(ValueError):
        save_artifact(pipeline, invalid, path)
    assert not path.exists() and not path.with_suffix(".json").exists()


def test_saved_feature_order_must_match_fitted_pipeline(tmp_path, trained_artifact):
    _, pipeline, metadata = trained_artifact
    metadata["feature_names"] = list(reversed(FEATURE_COLUMNS))
    with pytest.raises(ValueError, match="feature order"):
        save_artifact(pipeline, metadata, tmp_path / "model.joblib")


def test_threshold_field_is_required_even_when_value_can_be_none(tmp_path, trained_artifact):
    _, pipeline, metadata = trained_artifact
    metadata.pop("threshold")
    with pytest.raises(ValueError, match="missing fields.*threshold"):
        save_artifact(pipeline, metadata, tmp_path / "model.joblib")


def test_no_feasible_threshold_is_saved_but_not_loaded_for_classification(tmp_path, trained_artifact, monkeypatch):
    _, pipeline, metadata = trained_artifact
    metadata.update({"threshold": None, "status": "no_feasible_threshold"})
    path = tmp_path / "baseline.joblib"
    saved = save_artifact(pipeline, metadata, path)
    assert saved["threshold"] is None and path.exists()
    monkeypatch.setattr("src.ml.artifacts.joblib.load", lambda _: pytest.fail("classification load was attempted"))
    with pytest.raises(ValueError, match="no feasible threshold"):
        load_artifact(path)


def test_version_mismatch_is_refused_before_loading(tmp_path, trained_artifact, monkeypatch):
    _, pipeline, metadata = trained_artifact
    path = tmp_path / "model.joblib"
    save_artifact(pipeline, metadata, path)
    json_path = path.with_suffix(".json")
    saved = json.loads(json_path.read_text())
    saved["library_versions"]["scikit-learn"] = "0.0.0"
    json_path.write_text(json.dumps(saved))
    monkeypatch.setattr("src.ml.artifacts.joblib.load", lambda _: pytest.fail("incompatible load was attempted"))
    with pytest.raises(ValueError, match="scikit-learn==0.0.0"):
        load_artifact(path)


def test_python_patch_version_can_differ(tmp_path, trained_artifact):
    _, pipeline, metadata = trained_artifact
    metadata["library_versions"]["python"] = ".".join(platform.python_version().split(".")[:2]) + ".999"
    path = tmp_path / "model.joblib"
    save_artifact(pipeline, metadata, path)
    load_artifact(path)


def test_missing_and_corrupt_files_have_clear_errors(tmp_path, trained_artifact):
    _, pipeline, metadata = trained_artifact
    path = tmp_path / "missing.joblib"
    with pytest.raises(FileNotFoundError, match="artifact file not found"):
        load_artifact(path)
    save_artifact(pipeline, metadata, path)
    path.with_suffix(".json").write_text("{")
    with pytest.raises(ValueError, match="valid JSON"):
        load_artifact(path)
    path.with_suffix(".json").unlink()
    with pytest.raises(FileNotFoundError, match="missing.json"):
        load_artifact(path)
