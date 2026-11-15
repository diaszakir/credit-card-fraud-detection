"""Единая точка чтения результатов команды для приложения и модуля оценки.

Каждая функция возвращает ``Source``: либо данные, либо понятную причину отсутствия.
Модуль ничего не записывает и не вычисляет метрики — только читает то, что уже
сохранили Жандарбек (данные, SQLite, SQL-результаты) и Диас (эксперименты, модель).
Корень проекта можно подменить переменной ``FRAUD_PROJECT_ROOT`` (тесты, Docker).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import ROOT_DIR

ROOT_ENV_VAR = "FRAUD_PROJECT_ROOT"
COMPARISON_REQUIRED = (
    "run_id", "model_name", "imbalance_strategy", "tuning", "status",
    "threshold", "precision", "recall", "pr_auc", "average_precision",
)


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def processed_dir(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def transactions(self) -> Path:
        return self.processed_dir / "transactions.parquet"

    @property
    def sqlite(self) -> Path:
        return self.processed_dir / "fraud.db"

    @property
    def split_summary(self) -> Path:
        return self.root / "reports" / "split_summary.json"

    @property
    def data_audit(self) -> Path:
        return self.root / "reports" / "data_audit.json"

    @property
    def eda_summary(self) -> Path:
        return self.root / "reports" / "eda" / "eda_summary.json"

    @property
    def eda_figures_dir(self) -> Path:
        return self.root / "reports" / "figures"

    @property
    def sql_results_dir(self) -> Path:
        return self.root / "reports" / "sql_results"

    @property
    def experiments_dir(self) -> Path:
        return self.root / "reports" / "experiments"

    @property
    def comparison(self) -> Path:
        return self.experiments_dir / "comparison.csv"

    @property
    def runs(self) -> Path:
        return self.experiments_dir / "runs.parquet"

    @property
    def metrics_long(self) -> Path:
        return self.experiments_dir / "metrics.parquet"

    @property
    def predictions(self) -> Path:
        return self.experiments_dir / "predictions.parquet"

    @property
    def evaluation_dir(self) -> Path:
        """Результаты модуля оценки (Адам) внутри общей папки экспериментов."""
        return self.experiments_dir / "evaluation"

    @property
    def ml_figures_dir(self) -> Path:
        return self.experiments_dir / "figures"

    @property
    def model(self) -> Path:
        return self.root / "models" / "fraud_pipeline.joblib"

    @property
    def model_metadata(self) -> Path:
        return self.model.with_suffix(".json")

    @property
    def selection_lock(self) -> Path:
        return self.root / "models" / "selection_lock.json"


def project_paths(root: Path | str | None = None) -> ProjectPaths:
    if root is None:
        root = os.environ.get(ROOT_ENV_VAR) or ROOT_DIR
    return ProjectPaths(Path(root).resolve())


@dataclass(frozen=True)
class Source:
    """Результат чтения: ``available`` и ``data`` либо ``message`` с причиной."""

    name: str
    available: bool
    data: Any = None
    message: str = ""
    path: Path | None = None


def _missing(name: str, path: Path, hint: str) -> Source:
    return Source(name, False, None, f"Нет файла {path.name}. {hint}", path)


def _read_json(name: str, path: Path, hint: str) -> Source:
    if not path.is_file():
        return _missing(name, path, hint)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as exc:
        return Source(name, False, None, f"Файл {path.name} повреждён: {exc}", path)
    if not isinstance(data, dict):
        return Source(name, False, None, f"Файл {path.name} должен содержать JSON-объект.", path)
    return Source(name, True, data, "", path)


def _read_table(name: str, path: Path, hint: str, required=()) -> Source:
    if not path.is_file():
        return _missing(name, path, hint)
    try:
        frame = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
    except (OSError, ValueError, ImportError) as exc:
        return Source(name, False, None, f"Не удалось прочитать {path.name}: {exc}", path)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        return Source(name, False, None, f"В {path.name} нет колонок {missing}.", path)
    return Source(name, True, frame, "", path)


HINT_DATA = "Подготовьте данные: python -m src.data.cli run-all."
HINT_ML = "Результаты экспериментов создаёт Диас: python -m src.ml.cli compare."
HINT_MODEL = "Восстановите модель: python -m src.ml.reproduce (см. README)."


# --------------------------- Data Overview (Жандарбек) --------------------------- #
def load_split_summary(paths: ProjectPaths) -> Source:
    source = _read_json("split_summary", paths.split_summary, HINT_DATA)
    if not source.available:
        return source
    rows = []
    for split, values in source.data.items():
        if isinstance(values, dict):
            rows.append({"split": split, **values})
    if not rows:
        return Source(source.name, False, None, "split_summary.json не содержит разбиений.", source.path)
    return Source(source.name, True, pd.DataFrame(rows), "", source.path)


def load_data_audit(paths: ProjectPaths) -> Source:
    return _read_json("data_audit", paths.data_audit, HINT_DATA)


def list_sql_results(paths: ProjectPaths) -> Source:
    folder = paths.sql_results_dir
    files = sorted(folder.glob("q*.csv")) if folder.is_dir() else []
    if not files:
        return Source("sql_results", False, None, f"В {folder} нет результатов SQL. Выполните python -m src.data.cli query.", folder)
    return Source("sql_results", True, {path.stem: path for path in files}, "", folder)


def load_sql_result(paths: ProjectPaths, name: str) -> Source:
    return _read_table(name, paths.sql_results_dir / f"{name}.csv", "Выполните python -m src.data.cli query.")


def describe_sql_queries() -> Source:
    """Описание запросов (заголовок, вопрос, конструкции SQL) из файлов Жандарбека."""
    try:
        from src.db.queries import describe_queries

        return Source("sql_catalog", True, describe_queries())
    except (OSError, ValueError, ImportError) as exc:
        return Source("sql_catalog", False, None, f"Не удалось прочитать каталог запросов: {exc}")


def run_live_query(paths: ProjectPaths, name: str) -> Source:
    """Выполнить запрос на fraud.db, если база построена; иначе вернуть причину."""
    if not paths.sqlite.is_file():
        return _missing(name, paths.sqlite, "Постройте базу: python -m src.data.cli build-db.")
    from src.db.engine import get_engine
    from src.db.queries import run_query

    engine = get_engine(paths.sqlite)
    try:
        return Source(name, True, run_query(engine, name), "", paths.sqlite)
    except Exception as exc:  # ошибка SQL не должна ронять приложение
        return Source(name, False, None, f"Запрос {name} не выполнен: {exc}", paths.sqlite)
    finally:
        engine.dispose()


def list_figures(folder: Path, pattern: str = "*.png") -> list[Path]:
    return sorted(folder.glob(pattern)) if folder.is_dir() else []


# --------------------------- Model Comparison (Диас) --------------------------- #
def load_comparison(paths: ProjectPaths) -> Source:
    return _read_table("comparison", paths.comparison, HINT_ML, COMPARISON_REQUIRED)


def load_runs(paths: ProjectPaths) -> Source:
    return _read_table("runs", paths.runs, HINT_ML, ("run_id", "model_name", "imbalance_strategy"))


def load_metrics_long(paths: ProjectPaths) -> Source:
    return _read_table("metrics", paths.metrics_long, HINT_ML, ("run_id", "split", "metric", "value"))


def load_model_metadata(paths: ProjectPaths) -> Source:
    """JSON выбранной модели с проверкой SHA по selection_lock.json (без загрузки joblib)."""
    source = _read_json("model_metadata", paths.model_metadata, HINT_ML)
    if not source.available:
        return source
    lock = _read_json("selection_lock", paths.selection_lock, HINT_ML)
    if not lock.available:
        return Source(source.name, False, None, lock.message, lock.path)
    from src.ml.artifacts import file_sha256
    from src.ml.handoff import check_lock_metadata

    if file_sha256(paths.model_metadata) != lock.data.get("metadata_sha256"):
        return Source(source.name, False, None, "fraud_pipeline.json не совпадает с selection_lock.json.", source.path)
    try:
        check_lock_metadata(lock.data, source.data)
    except ValueError as exc:
        return Source(source.name, False, None, str(exc), source.path)
    return source


# --------------------------- Error Analysis (построчные прогнозы) --------------------------- #
def selected_validation_scores_path(paths: ProjectPaths, metadata: dict) -> Path | None:
    relative = (metadata.get("source_paths") or {}).get("validation_scores")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        return None
    path = (paths.root / relative).resolve()
    return path if path.is_relative_to(paths.root) else None


def load_validation_predictions(paths: ProjectPaths, metadata: dict | None = None) -> Source:
    """Validation-прогнозы выбранного запуска: сначала predictions.parquet, затем кэш Диаса."""
    from src.evaluation.contract import PredictionContractError, validate_predictions_table

    run_id = (metadata or {}).get("run_id")
    threshold = (metadata or {}).get("threshold")
    candidates = []
    if paths.predictions.is_file():
        candidates.append(paths.predictions)
    cache = selected_validation_scores_path(paths, metadata or {})
    if cache is not None and cache.is_file():
        candidates.append(cache)
    if not candidates:
        return Source("validation_predictions", False, None,
                      "Построчных validation-прогнозов нет на этом компьютере. "
                      "Они не хранятся в Git: python -m src.ml.reproduce и python -m src.ml.cli compare.", paths.predictions)
    errors = []
    for path in candidates:
        try:
            frame = pd.read_parquet(path)
        except (OSError, ValueError, ImportError) as exc:
            errors.append(f"{path.name}: {exc}")
            continue
        if "run_id" not in frame.columns and run_id is not None:
            frame.insert(0, "run_id", run_id)
        if run_id is not None:
            frame = frame[frame["run_id"] == run_id]
        if frame.empty:
            errors.append(f"{path.name}: нет строк запуска {run_id}")
            continue
        try:
            validate_predictions_table(frame, threshold=threshold, split="validation")
        except PredictionContractError as exc:
            errors.append(f"{path.name}: {exc}")
            continue
        return Source("validation_predictions", True, frame.reset_index(drop=True), "", path)
    return Source("validation_predictions", False, None, "; ".join(errors), candidates[0])


# --------------------------- Сводка состояния --------------------------- #
def availability(paths: ProjectPaths) -> pd.DataFrame:
    """Таблица «что есть на этом компьютере» для боковой панели и проверки запуска."""
    items = [
        ("Разбиение train/validation/test", paths.split_summary, "Жандарбек"),
        ("Обработанные транзакции (Parquet)", paths.transactions, "Жандарбек"),
        ("SQLite fraud.db", paths.sqlite, "Жандарбек"),
        ("Результаты SQL-запросов", paths.sql_results_dir, "Жандарбек"),
        ("Сравнение экспериментов", paths.comparison, "Диас"),
        ("Метаданные выбранной модели", paths.model_metadata, "Диас"),
        ("Файл модели .joblib", paths.model, "Диас"),
        ("Построчные validation-прогнозы", paths.predictions, "Диас"),
    ]
    rows = [{"Артефакт": label, "Владелец": owner, "Есть": path.exists(), "Путь": str(path.relative_to(paths.root))}
            for label, path, owner in items]
    return pd.DataFrame(rows)
