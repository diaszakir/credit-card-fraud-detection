"""Сохранение всего пайплайна и проверка доверенного joblib перед загрузкой."""

import hashlib
import json
import platform
from importlib.metadata import PackageNotFoundError, version
from numbers import Real
from pathlib import Path

import joblib
import numpy as np

from src.ml.prediction import feature_schema


def file_sha256(path):
    """Контрольная сумма файла без загрузки всего содержимого в память."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"metadata value {type(value).__name__} is not JSON serializable")


def _validate_metadata(metadata, require_threshold=False):
    if not isinstance(metadata, dict):
        raise ValueError("artifact metadata must be a dictionary")
    required = {
        "schema_version", "protocol_version", "feature_names", "feature_dtypes",
        "positive_label", "threshold", "comparison", "min_precision",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"artifact metadata is missing fields: {missing}")
    if type(metadata.get("schema_version")) is not int or metadata["schema_version"] != 1:
        raise ValueError("artifact schema_version must be 1")
    if not isinstance(metadata.get("protocol_version"), str) or not metadata["protocol_version"].strip():
        raise ValueError("artifact protocol_version must be a non-empty string")
    feature_schema(metadata)
    if type(metadata.get("positive_label")) is not int or metadata["positive_label"] != 1:
        raise ValueError("artifact positive_label must be 1")
    if metadata.get("comparison") != ">=":
        raise ValueError("artifact comparison must be >=")

    minimum = metadata.get("min_precision")
    if isinstance(minimum, bool) or not isinstance(minimum, Real) or minimum != 0.85:
        raise ValueError("artifact min_precision must be the agreed value 0.85")
    threshold = metadata.get("threshold")
    if threshold is None:
        if require_threshold:
            raise ValueError("artifact has no feasible threshold and cannot be used for class prediction")
    elif (
        isinstance(threshold, bool)
        or not isinstance(threshold, Real)
        or not np.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError("artifact threshold must be finite and in [0, 1]")


def _validate_pipeline(pipeline, metadata):
    names, _ = feature_schema(metadata)
    classes = np.asarray(getattr(pipeline, "classes_", []))
    if classes.ndim != 1 or classes.size != 2 or set(classes.tolist()) != {0, 1}:
        raise ValueError("artifact pipeline must be fitted with both classes 0 and 1")
    if not callable(getattr(pipeline, "predict_proba", None)):
        raise ValueError("artifact pipeline does not provide predict_proba")
    if getattr(pipeline, "n_features_in_", len(names)) != len(names):
        raise ValueError("saved feature count disagrees with the fitted pipeline")
    fitted_names = getattr(pipeline, "feature_names_in_", None)
    if fitted_names is not None and list(fitted_names) != names:
        raise ValueError("saved feature order disagrees with the fitted pipeline")


def _validate_versions(metadata):
    saved = metadata.get("library_versions", {})
    if not isinstance(saved, dict):
        raise ValueError("artifact library_versions must be a dictionary")
    python_version = saved.get("python", metadata.get("python_version"))
    if python_version is not None:
        if str(python_version).split(".")[:2] != platform.python_version().split(".")[:2]:
            raise ValueError(f"artifact requires Python {python_version}; current Python is {platform.python_version()}")
    packages = {
        "sklearn": "scikit-learn", "scikit-learn": "scikit-learn",
        "imblearn": "imbalanced-learn", "imbalanced-learn": "imbalanced-learn",
        "joblib": "joblib", "numpy": "numpy", "scipy": "scipy",
    }
    for name, package in packages.items():
        if name not in saved:
            continue
        try:
            installed = version(package)
        except PackageNotFoundError as exc:
            raise ValueError(f"artifact requires missing package {package}") from exc
        if saved[name] != installed:
            raise ValueError(f"artifact requires {package}=={saved[name]}; installed version is {installed}")


def save_artifact(pipeline, metadata, model_path):
    """Сохранить Pipeline и соседний JSON; каждый файл заменяется атомарно.

    threshold=None разрешён для baseline без допустимого порога. Такой файл можно
    хранить как результат запуска, но load_artifact не допускает его к классификации.
    """
    try:
        saved = json.loads(json.dumps(metadata, default=_json_default, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("artifact metadata must contain finite JSON-serializable values") from exc
    _validate_metadata(saved)
    _validate_pipeline(pipeline, saved)
    _validate_versions(saved)

    model_path = Path(model_path)
    metadata_path = model_path.with_suffix(".json")
    if model_path == metadata_path:
        raise ValueError("model_path must have a model extension such as .joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_temp = model_path.with_name(model_path.name + ".tmp")
    metadata_temp = metadata_path.with_name(metadata_path.name + ".tmp")
    try:
        joblib.dump(pipeline, model_temp, compress=3)
        saved["artifact_file"] = model_path.name
        saved["artifact_sha256"] = file_sha256(model_temp)
        metadata_temp.write_text(
            json.dumps(saved, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        # JSON публикуется последним: неполная пара не пройдёт проверку SHA.
        model_temp.replace(model_path)
        metadata_temp.replace(metadata_path)
    finally:
        model_temp.unlink(missing_ok=True)
        metadata_temp.unlink(missing_ok=True)
    return saved


def load_artifact(model_path, metadata_path=None):
    """Загрузить только доверенные joblib: SHA не защищает от вредоносного автора.

    До joblib.load проверяются JSON, контрольная сумма и совместимость версий.
    Не загружайте через эту функцию файлы из неизвестного источника.
    """
    model_path = Path(model_path)
    metadata_path = Path(metadata_path) if metadata_path is not None else model_path.with_suffix(".json")
    for path in (model_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(f"artifact file not found: {path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f"artifact metadata is not valid JSON: {metadata_path}") from exc
    _validate_metadata(metadata, require_threshold=True)
    if metadata.get("artifact_file") != model_path.name:
        raise ValueError("artifact filename disagrees with metadata")
    expected = metadata.get("artifact_sha256")
    if not isinstance(expected, str) or len(expected) != 64 or file_sha256(model_path) != expected:
        raise ValueError("artifact SHA-256 mismatch; the model file is damaged or does not match its metadata")
    _validate_versions(metadata)
    try:
        pipeline = joblib.load(model_path)
    except Exception as exc:
        raise ValueError(f"could not load the trusted artifact: {model_path}") from exc
    _validate_pipeline(pipeline, metadata)
    return pipeline, metadata
