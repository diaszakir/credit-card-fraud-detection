"""Контракт прогнозов: проверка входа, единый Predictor и проверка результата.

Приложение и модуль оценки получают прогнозы только через объект с методом
``predict(X) -> DataFrame[score, threshold, y_pred]``. Реальная модель обёрнута в
``FrozenPipelinePredictor`` и использует функцию Диаса ``predict_transactions``;
``StubPredictor`` нужен только для разработки интерфейса и никогда не выдаётся за модель.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from src.config import AMOUNT_COLUMN, FEATURE_COLUMNS, ID_COLUMN, TARGET_COLUMN

PREDICTION_COLUMNS = ("score", "threshold", "y_pred")
OPTIONAL_INPUT_COLUMNS = (ID_COLUMN, TARGET_COLUMN)
PREDICTIONS_TABLE_COLUMNS = ("run_id", ID_COLUMN, "split", "y_true", "score", "y_pred")
PROVENANCE_REAL = "real"
PROVENANCE_STUB = "stub"
STUB_ENV_VAR = "FRAUD_APP_STUB"


class PredictionContractError(ValueError):
    """Нарушен контракт входа или выхода прогноза."""


# --------------------------------------------------------------------------- #
# Описание модели
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelInfo:
    """То, что интерфейсу нужно знать о модели, без доступа к joblib."""

    run_id: str
    model_name: str
    imbalance_strategy: str
    threshold: float
    min_precision: float
    feature_names: tuple[str, ...]
    provenance: str
    frozen: bool
    test_evaluated: bool
    validation_metrics: dict = field(default_factory=dict)

    @property
    def is_real(self) -> bool:
        return self.provenance == PROVENANCE_REAL

    @classmethod
    def from_metadata(cls, metadata: dict) -> "ModelInfo":
        if not isinstance(metadata, dict):
            raise PredictionContractError("метаданные модели должны быть словарём")
        threshold = metadata.get("threshold")
        if (
            isinstance(threshold, (bool, np.bool_))
            or not isinstance(threshold, Real)
            or not np.isfinite(threshold)
            or not 0 <= threshold <= 1
        ):
            raise PredictionContractError("в метаданных нет допустимого порога в [0, 1]")
        names = metadata.get("feature_names")
        if not isinstance(names, list) or sorted(names) != sorted(FEATURE_COLUMNS):
            raise PredictionContractError("feature_names должны совпадать с 29 признаками проекта")
        return cls(
            run_id=str(metadata.get("run_id", "unknown")),
            model_name=str(metadata.get("model_name", "unknown")),
            imbalance_strategy=str(metadata.get("imbalance_strategy", "unknown")),
            threshold=float(threshold),
            min_precision=float(metadata.get("min_precision", float("nan"))),
            feature_names=tuple(names),
            provenance=str(metadata.get("provenance", "unknown")),
            frozen=metadata.get("frozen") is True,
            test_evaluated=metadata.get("test_evaluated") is True,
            validation_metrics=dict(metadata.get("validation_metrics") or {}),
        )


class Predictor(Protocol):
    """Любая реализация обязана вернуть таблицу по контракту PREDICTION_COLUMNS."""

    info: ModelInfo

    def predict(self, X: pd.DataFrame) -> pd.DataFrame: ...


# --------------------------------------------------------------------------- #
# Проверка входа (дружелюбная, собирает все проблемы сразу)
# --------------------------------------------------------------------------- #
@dataclass
class SchemaReport:
    n_rows: int
    missing: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    duplicated: list[str] = field(default_factory=list)
    non_numeric: list[str] = field(default_factory=list)
    non_finite: dict[str, int] = field(default_factory=dict)
    id_problems: list[str] = field(default_factory=list)
    label_problems: list[str] = field(default_factory=list)
    has_id: bool = False
    has_label: bool = False

    @property
    def ok(self) -> bool:
        return self.n_rows > 0 and not any([
            self.missing, self.extra, self.duplicated, self.non_numeric,
            self.non_finite, self.id_problems, self.label_problems,
        ])

    def messages(self) -> list[str]:
        """Сообщения для пользователя на русском, по одному на тип проблемы."""
        result = []
        if self.n_rows == 0:
            result.append("В файле нет ни одной строки с транзакцией.")
        if self.duplicated:
            result.append(f"Повторяющиеся названия колонок: {', '.join(self.duplicated)}.")
        if self.missing:
            result.append(f"Не хватает признаков: {', '.join(self.missing)}.")
        if self.extra:
            allowed = ", ".join(OPTIONAL_INPUT_COLUMNS)
            result.append(f"Лишние колонки: {', '.join(self.extra)}. Кроме признаков допустимы только {allowed}.")
        if self.non_numeric:
            result.append(f"Нечисловые значения в колонках: {', '.join(self.non_numeric)}.")
        if self.non_finite:
            details = ", ".join(f"{name} ({count})" for name, count in self.non_finite.items())
            result.append(f"Пропуски или бесконечные значения: {details}.")
        result.extend(self.id_problems)
        result.extend(self.label_problems)
        return result


def _is_real_numeric(series: pd.Series) -> bool:
    dtype = series.dtype
    return (
        pd.api.types.is_numeric_dtype(dtype)
        and not pd.api.types.is_bool_dtype(dtype)
        and not pd.api.types.is_complex_dtype(dtype)
    )


def check_input_schema(frame: pd.DataFrame, feature_names=FEATURE_COLUMNS) -> SchemaReport:
    """Проверить таблицу до модели и вернуть полный отчёт, не бросая исключений."""
    if not isinstance(frame, pd.DataFrame):
        raise PredictionContractError("ожидается pandas DataFrame")
    names = list(feature_names)
    columns = [str(column) for column in frame.columns]
    report = SchemaReport(n_rows=len(frame))
    report.duplicated = sorted({c for c in columns if columns.count(c) > 1})
    report.missing = [name for name in names if name not in columns]
    report.extra = [c for c in dict.fromkeys(columns) if c not in names and c not in OPTIONAL_INPUT_COLUMNS]
    report.has_id = ID_COLUMN in columns
    report.has_label = TARGET_COLUMN in columns
    if report.duplicated:
        return report  # по неоднозначным колонкам дальше проверять нельзя

    for name in names:
        if name not in frame.columns:
            continue
        column = frame[name]
        if not _is_real_numeric(column):
            report.non_numeric.append(name)
            continue
        bad = int((~np.isfinite(column.to_numpy(dtype="float64", na_value=np.nan))).sum())
        if bad:
            report.non_finite[name] = bad

    if report.has_id:
        ids = frame[ID_COLUMN]
        if not pd.api.types.is_integer_dtype(ids.dtype) or ids.isna().any() or (ids < 0).any():
            report.id_problems.append("transaction_id должен содержать целые числа ≥ 0 без пропусков.")
        elif ids.duplicated().any():
            report.id_problems.append("transaction_id содержит повторы.")
    if report.has_label:
        labels = frame[TARGET_COLUMN]
        if labels.isna().any() or not labels.isin([0, 1]).all():
            report.label_problems.append("Class может содержать только 0 и 1.")
    return report


def split_input(frame: pd.DataFrame, feature_names=FEATURE_COLUMNS) -> tuple[pd.DataFrame, pd.Series | None]:
    """Отделить признаки от необязательных transaction_id и Class.

    Возвращает ``X`` (только признаки, индекс = transaction_id, если он был) и ``y`` или None.
    Перед вызовом схема должна пройти ``check_input_schema``.
    """
    report = check_input_schema(frame, feature_names)
    if not report.ok:
        raise PredictionContractError(" ".join(report.messages()))
    data = frame.copy()
    if report.has_id:
        data = data.set_index(ID_COLUMN)
    y = data.pop(TARGET_COLUMN).astype("int8") if report.has_label else None
    return data[list(feature_names)], y


# --------------------------------------------------------------------------- #
# Проверка выхода
# --------------------------------------------------------------------------- #
def validate_prediction_frame(predictions: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """Проверить, что прогноз соответствует контракту; вернуть его без изменений."""
    if not isinstance(predictions, pd.DataFrame):
        raise PredictionContractError("прогноз должен быть DataFrame")
    if tuple(predictions.columns) != PREDICTION_COLUMNS:
        raise PredictionContractError(f"колонки прогноза должны быть {PREDICTION_COLUMNS}, получено {tuple(predictions.columns)}")
    if not predictions.index.equals(X.index):
        raise PredictionContractError("индекс или порядок строк прогноза не совпадает со входом")
    score = predictions["score"].to_numpy(dtype="float64", na_value=np.nan)
    if not np.isfinite(score).all() or (score < 0).any() or (score > 1).any():
        raise PredictionContractError("score должен быть конечным числом в [0, 1]")
    threshold = predictions["threshold"].to_numpy(dtype="float64", na_value=np.nan)
    if not np.isfinite(threshold).all() or len(np.unique(threshold)) != 1 or not 0 <= threshold[0] <= 1:
        raise PredictionContractError("threshold должен быть одинаковым числом в [0, 1] во всех строках")
    y_pred = predictions["y_pred"]
    if not pd.api.types.is_integer_dtype(y_pred.dtype) or not y_pred.isin([0, 1]).all():
        raise PredictionContractError("y_pred должен содержать целые метки 0 и 1")
    if not np.array_equal(y_pred.to_numpy(), (score >= threshold[0]).astype("int8")):
        raise PredictionContractError("y_pred должен быть ровно score >= threshold")
    return predictions


def validate_predictions_table(table: pd.DataFrame, threshold: float | None = None, split: str | None = None) -> pd.DataFrame:
    """Проверить построчные прогнозы (predictions.parquet или кэш validation) для анализа ошибок."""
    if not isinstance(table, pd.DataFrame) or table.empty:
        raise PredictionContractError("таблица прогнозов пуста")
    missing = [name for name in PREDICTIONS_TABLE_COLUMNS if name not in table.columns]
    if missing:
        raise PredictionContractError(f"в таблице прогнозов нет колонок {missing}")
    if table.duplicated(["run_id", ID_COLUMN]).any():
        raise PredictionContractError("transaction_id повторяется внутри одного run_id")
    if split is not None and not table["split"].astype(str).eq(split).all():
        raise PredictionContractError(f"таблица должна содержать только split={split!r}")
    for name in ("y_true", "y_pred"):
        if table[name].isna().any() or not table[name].isin([0, 1]).all():
            raise PredictionContractError(f"{name} должен содержать только 0 и 1")
    score = table["score"].to_numpy(dtype="float64", na_value=np.nan)
    if not np.isfinite(score).all() or (score < 0).any() or (score > 1).any():
        raise PredictionContractError("score должен быть конечным числом в [0, 1]")
    if threshold is not None:
        if table["run_id"].nunique() != 1:
            raise PredictionContractError("проверка порога возможна только для одного run_id")
        if not np.array_equal(table["y_pred"].to_numpy(dtype="int8"), (score >= threshold).astype("int8")):
            raise PredictionContractError("y_pred в таблице не равен score >= сохранённого порога")
    return table


# --------------------------------------------------------------------------- #
# Реализации Predictor
# --------------------------------------------------------------------------- #
class FrozenPipelinePredictor:
    """Замороженный Pipeline Диаса + сохранённый порог. Ничего не переобучает."""

    def __init__(self, pipeline, metadata: dict):
        self._pipeline = pipeline
        self._metadata = metadata
        self.info = ModelInfo.from_metadata(metadata)

    @classmethod
    def load(cls, model_path: Path, lock_path: Path) -> "FrozenPipelinePredictor":
        """Проверить selection_lock и SHA, затем загрузить доверенный joblib."""
        # Импорт здесь: модуль контракта можно использовать без установленного sklearn.
        from src.data.io_utils import read_json
        from src.ml.artifacts import file_sha256, load_artifact
        from src.ml.handoff import check_lock_metadata

        model_path, lock_path = Path(model_path), Path(lock_path)
        metadata_path = model_path.with_suffix(".json")
        for path, hint in ((metadata_path, "метаданные модели"), (lock_path, "selection_lock.json"), (model_path, "файл модели .joblib")):
            if not path.is_file():
                raise FileNotFoundError(f"Не найден {hint}: {path}")
        lock = read_json(lock_path)
        if file_sha256(metadata_path) != lock.get("metadata_sha256"):
            raise PredictionContractError("метаданные модели не совпадают с selection_lock.json")
        pipeline, metadata = load_artifact(model_path, metadata_path)
        check_lock_metadata(lock, metadata)
        return cls(pipeline, metadata)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        from src.ml.prediction import predict_transactions

        predictions = predict_transactions(X, self._pipeline, self._metadata)
        return validate_prediction_frame(predictions, X)


def stub_metadata() -> dict:
    """Метаданные заглушки: схема проекта, условный порог 0.5, provenance='stub'."""
    dtypes = {name: "float64" if name == AMOUNT_COLUMN else "float32" for name in FEATURE_COLUMNS}
    return {
        "run_id": "stub_for_ui_development",
        "model_name": "stub",
        "imbalance_strategy": "none",
        "threshold": 0.5,
        "min_precision": 0.85,
        "comparison": ">=",
        "positive_label": 1,
        "feature_names": list(FEATURE_COLUMNS),
        "feature_dtypes": dtypes,
        "provenance": PROVENANCE_STUB,
        "frozen": False,
        "test_evaluated": False,
        "validation_metrics": {},
    }


class StubPredictor:
    """ЗАГЛУШКА для разработки интерфейса. Это не модель и не оценка качества.

    score — детерминированный хеш значений признаков, отображённый в [0, 1).
    Одинаковые строки дают одинаковый score, порядок строк сохраняется.
    """

    def __init__(self):
        self._metadata = stub_metadata()
        self.info = ModelInfo.from_metadata(self._metadata)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        from src.ml.prediction import validate_features

        ordered = validate_features(X, self._metadata)
        hashes = pd.util.hash_pandas_object(ordered, index=False).to_numpy(dtype="uint64")
        score = hashes.astype("float64") / float(2**64)
        threshold = self.info.threshold
        predictions = pd.DataFrame({
            "score": score,
            "threshold": threshold,
            "y_pred": (score >= threshold).astype("int8"),
        }, index=X.index)
        return validate_prediction_frame(predictions, X)


@dataclass(frozen=True)
class PredictorStatus:
    predictor: Predictor | None
    mode: str  # "frozen", "stub" или "unavailable"
    message: str


def stub_enabled() -> bool:
    return os.environ.get(STUB_ENV_VAR, "").strip() == "1"


def load_predictor(model_path: Path, lock_path: Path, allow_stub: bool | None = None) -> PredictorStatus:
    """Выбрать реализацию: замороженная модель, иначе заглушка только по явному разрешению."""
    allow_stub = stub_enabled() if allow_stub is None else allow_stub
    try:
        predictor = FrozenPipelinePredictor.load(model_path, lock_path)
    except FileNotFoundError as exc:
        reason = str(exc)
    except ValueError as exc:  # SHA, версии библиотек, схема
        return PredictorStatus(None, "unavailable", f"Модель найдена, но не прошла проверку: {exc}")
    else:
        return PredictorStatus(predictor, "frozen", f"Загружена замороженная модель {predictor.info.run_id}.")
    if allow_stub:
        return PredictorStatus(StubPredictor(), "stub", f"{reason}. Включена заглушка ({STUB_ENV_VAR}=1): прогнозы не являются результатом модели.")
    return PredictorStatus(None, "unavailable", reason)
