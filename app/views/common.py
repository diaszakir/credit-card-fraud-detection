"""Общие подписи, кэш модели и вывод «нет данных» для всех разделов."""

from __future__ import annotations

import streamlit as st

from src.evaluation.contract import PredictorStatus, load_predictor, stub_enabled
from src.evaluation.sources import ProjectPaths, Source, project_paths

MODEL_NAMES = {
    "dummy": "Dummy baseline",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "stub": "Заглушка",
}
STRATEGY_NAMES = {
    "none": "Без балансировки",
    "class_weight": "Веса классов",
    "smote": "SMOTE",
    "undersampling": "Undersampling",
}
SPLIT_NAMES = {"train": "Train", "validation": "Validation", "test": "Test"}


def paths() -> ProjectPaths:
    return project_paths()


def show_missing(source: Source, level: str = "info") -> None:
    """Показать причину отсутствия данных вместо падения страницы."""
    text = source.message or f"Источник {source.name} недоступен."
    getattr(st, level)(text)


def _mtime(path) -> int:
    return path.stat().st_mtime_ns if path.exists() else 0


@st.cache_resource(show_spinner="Загрузка модели…")
def _cached_predictor(root: str, model_mtime: int, lock_mtime: int, allow_stub: bool) -> PredictorStatus:
    project = project_paths(root)
    return load_predictor(project.model, project.selection_lock, allow_stub=allow_stub)


def predictor_status() -> PredictorStatus:
    """Модель загружается один раз и перечитывается при изменении файлов."""
    project = paths()
    return _cached_predictor(str(project.root), _mtime(project.model), _mtime(project.selection_lock), stub_enabled())


def model_label(model_name: str, strategy: str) -> str:
    return f"{MODEL_NAMES.get(model_name, model_name)} · {STRATEGY_NAMES.get(strategy, strategy)}"


def fmt(value, digits: int = 4) -> str:
    """Число для карточки метрики; отсутствующее значение явно подписано."""
    if value is None:
        return "нет"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:  # NaN
        return "нет"
    return f"{number:.{digits}f}"
