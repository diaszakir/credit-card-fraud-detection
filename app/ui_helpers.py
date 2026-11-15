"""Чтение сохранённых результатов и входных данных для простого экрана Streamlit."""

from numbers import Real

import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS, ROOT_DIR
from src.data.io_utils import read_json
from src.ml.artifacts import file_sha256, load_artifact
from src.ml.handoff import check_lock_metadata


def load_saved_metadata():
    metadata_path = ROOT_DIR / "models" / "fraud_pipeline.json"
    lock_path = ROOT_DIR / "models" / "selection_lock.json"
    if not metadata_path.is_file() or not lock_path.is_file():
        raise FileNotFoundError("Метаданные выбранной модели пока недоступны. Проверьте файлы модели по README.")
    metadata, lock = read_json(metadata_path), read_json(lock_path)
    if not isinstance(metadata, dict) or not isinstance(lock, dict):
        raise ValueError("Метаданные модели или файл фиксации имеют неверный формат.")
    if file_sha256(metadata_path) != lock.get("metadata_sha256"):
        raise ValueError("Метаданные модели изменены или повреждены; проверка контрольной суммы не пройдена.")
    check_lock_metadata(lock, metadata)
    if metadata.get("frozen") is not True or metadata.get("split") != "validation" or metadata.get("test_evaluated") is not False:
        raise ValueError("Экран рассчитан на замороженную модель с результатами validation до итоговой оценки test.")
    metrics = metadata.get("validation_metrics", {})
    for name in ("precision", "recall", "pr_auc", "average_precision"):
        value = metrics.get(name)
        if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"В метаданных нет корректной validation-метрики {name}.")
    return metadata


def load_frozen_model():
    load_saved_metadata()
    model_path = ROOT_DIR / "models" / "fraud_pipeline.joblib"
    if not model_path.is_file():
        raise FileNotFoundError("Файл обученной модели не найден. Восстановите модель по инструкции в README.")
    return load_artifact(model_path)


def load_comparison():
    path = ROOT_DIR / "reports" / "experiments" / "comparison.csv"
    if not path.is_file():
        raise FileNotFoundError("Таблица экспериментов пока недоступна.")
    frame = pd.read_csv(path)
    required = ["run_id", "model_name", "imbalance_strategy", "tuning", "status", "precision", "recall", "pr_auc", "average_precision", "threshold"]
    if frame.empty or any(name not in frame.columns for name in required):
        raise ValueError("Таблица экспериментов имеет неверный формат.")
    return frame


def load_demo_examples():
    path = ROOT_DIR / "reports" / "experiments" / "demo_examples.json"
    document = read_json(path)
    if not isinstance(document, dict) or document.get("source_split") != "validation":
        raise ValueError("Готовые примеры должны относиться к validation.")
    examples = document.get("examples")
    if not isinstance(examples, list) or len(examples) != 4:
        raise ValueError("Ожидаются четыре готовых примера TP, TN, FN и FP.")
    outcomes, names = set(), set()
    for example in examples:
        if not isinstance(example, dict) or not isinstance(example.get("name"), str) or not example["name"]:
            raise ValueError("У готового примера должно быть понятное название.")
        if example["name"] in names:
            raise ValueError("Названия готовых примеров должны различаться.")
        names.add(example["name"])
        actual, expected = example.get("actual_class"), example.get("expected_prediction")
        if type(actual) is not int or type(expected) is not int or actual not in (0, 1) or expected not in (0, 1):
            raise ValueError("Классы готового примера должны быть 0 или 1.")
        outcomes.add((actual, expected))
        example_frame(example)
    if outcomes != {(0, 0), (0, 1), (1, 0), (1, 1)}:
        raise ValueError("Готовые примеры должны включать все четыре исхода: TP, TN, FN и FP.")
    return examples


def example_frame(example):
    features = example.get("features")
    transaction_id = example.get("transaction_id")
    if not isinstance(features, dict) or set(features) != set(FEATURE_COLUMNS):
        raise ValueError("В готовом примере должны быть все 29 признаков: V1–V28 и Amount.")
    if type(transaction_id) is not int or transaction_id < 0:
        raise ValueError("В готовом примере неверный идентификатор транзакции.")
    return pd.DataFrame([features], columns=FEATURE_COLUMNS,
                        index=pd.Index([transaction_id], name="transaction_id"))


def read_prediction_csv(source):
    try:
        if hasattr(source, "seek"):
            source.seek(0)
        return pd.read_csv(source)
    except (OSError, ValueError, UnicodeError) as exc:
        raise ValueError("Не удалось прочитать CSV. Проверьте файл, заголовки и разделитель-запятую.") from exc


def example_csv(example=None):
    frame = example_frame(example) if example is not None else pd.DataFrame(columns=FEATURE_COLUMNS)
    return frame.to_csv(index=False).encode("utf-8")


def prediction_table(predictions):
    table = predictions.rename(columns={"score": "Оценка fraud", "threshold": "Порог", "y_pred": "Метка"}).copy()
    table["Класс"] = table["Метка"].map({0: "Обычная транзакция", 1: "Fraud"})
    return table


def prediction_error(error):
    message = str(error)
    if "non-empty pandas DataFrame" in message:
        return "В CSV нет транзакций. Добавьте хотя бы одну строку данных."
    if "invalid input columns" in message or "duplicate column" in message:
        return "Неверные колонки CSV. Нужны только V1–V28 и Amount, без дополнительных полей."
    if "real numeric" in message or "missing or non-finite" in message:
        return "Все 29 признаков должны быть заполнены конечными числами, без пропусков и текста."
    if "range of the saved feature types" in message or "cannot be represented" in message:
        return "Значения признаков выходят за допустимый числовой диапазон. Проверьте CSV."
    return message
