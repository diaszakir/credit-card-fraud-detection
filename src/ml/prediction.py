"""Единое предсказание для экспериментов, итоговой оценки и Streamlit."""

from numbers import Real

import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS


def feature_schema(metadata):
    """Получить сохранённый порядок признаков и их числовые типы."""
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a dictionary")
    names = metadata.get("feature_names")
    if (
        not isinstance(names, list)
        or not all(isinstance(name, str) for name in names)
        or len(names) != len(FEATURE_COLUMNS)
        or len(set(names)) != len(names)
        or set(names) != set(FEATURE_COLUMNS)
    ):
        raise ValueError("feature_names must contain exactly the 29 project features")

    dtypes = metadata.get("feature_dtypes")
    if isinstance(dtypes, list) and len(dtypes) == len(names):
        dtypes = dict(zip(names, dtypes))
    if not isinstance(dtypes, dict) or set(dtypes) != set(names):
        raise ValueError("feature_dtypes must describe every saved feature")
    for name in names:
        try:
            dtype = np.dtype(dtypes[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid saved dtype for {name}") from exc
        if dtype.kind not in "iuf":
            raise ValueError(f"saved dtype for {name} must be a real numeric type")
    return names, dtypes


def validate_features(X, metadata):
    """Проверить вход без изменения оригинала и выстроить сохранённый порядок."""
    names, dtypes = feature_schema(metadata)
    if not isinstance(X, pd.DataFrame) or X.empty:
        raise ValueError("X must be a non-empty pandas DataFrame")
    if not X.columns.is_unique:
        raise ValueError("X contains duplicate column names")
    missing = [name for name in names if name not in X.columns]
    extra = [name for name in X.columns if name not in names]
    if missing or extra:
        raise ValueError(f"invalid input columns: missing={missing}, extra={extra}")

    ordered = X[names].copy()
    for name in names:
        column = ordered[name]
        if (
            not pd.api.types.is_numeric_dtype(column.dtype)
            or pd.api.types.is_bool_dtype(column.dtype)
            or pd.api.types.is_complex_dtype(column.dtype)
        ):
            raise ValueError(f"feature {name} must contain real numeric values")
        values = column.to_numpy(dtype="float64", na_value=np.nan)
        if not np.isfinite(values).all():
            raise ValueError(f"feature {name} contains missing or non-finite values")
        target = np.dtype(dtypes[name])
        if target.kind in "iu":
            limits = np.iinfo(target)
            if not np.equal(values, np.floor(values)).all() or (values < limits.min).any() or (values > limits.max).any():
                raise ValueError(f"feature {name} cannot be represented as {target}")
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            ordered = ordered.astype(dtypes)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("input cannot be converted to the saved feature types") from exc
    if not np.isfinite(ordered.to_numpy(dtype="float64")).all():
        raise ValueError("input exceeds the range of the saved feature types")
    return ordered


def fraud_scores(pipeline, X, positive_label=1):
    """Выбрать fraud из predict_proba по classes_, сохранив порядок строк."""
    if isinstance(positive_label, (bool, np.bool_)) or positive_label != 1:
        raise ValueError("the project positive_label must be 1")
    classes = np.asarray(getattr(pipeline, "classes_", []))
    if classes.ndim != 1 or len(classes) != 2 or set(classes.tolist()) != {0, 1}:
        raise ValueError("pipeline must be fitted with both classes 0 and 1")
    if not callable(getattr(pipeline, "predict_proba", None)):
        raise ValueError("pipeline does not provide predict_proba")
    probabilities = np.asarray(pipeline.predict_proba(X))
    if probabilities.shape != (len(X), len(classes)) or probabilities.dtype.kind not in "iuf":
        raise ValueError("predict_proba returned an invalid probability matrix")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("predict_proba must return finite probabilities in [0, 1]")
    positive_index = int(np.flatnonzero(classes == positive_label)[0])
    return probabilities[:, positive_index].astype("float64")


def predict_transactions(X, pipeline, metadata):
    """Вернуть score, сохранённый threshold и y_pred с исходным индексом."""
    ordered = validate_features(X, metadata)
    if metadata.get("comparison") != ">=":
        raise ValueError("metadata comparison must be >=")
    threshold = metadata.get("threshold")
    if (
        isinstance(threshold, (bool, np.bool_))
        or not isinstance(threshold, Real)
        or not np.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError("metadata threshold must be a finite number in [0, 1]")
    score = fraud_scores(pipeline, ordered, metadata.get("positive_label"))
    return pd.DataFrame({
        "score": score,
        "threshold": float(threshold),
        "y_pred": (score >= threshold).astype("int8"),
    }, index=X.index)
