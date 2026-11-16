"""Подготовить данные и выбранную модель один раз, затем проверять их офлайн."""

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    AUDIT_JSON_PATH, EXPERIMENTS_DIR, FEATURE_COLUMNS, ID_COLUMN, RAW_CSV_PATH,
    ROOT_DIR, SPLIT_MANIFEST_PATH, SPLIT_SUMMARY_JSON_PATH, TRANSACTIONS_PARQUET_PATH,
)
from src.data.api import load_xy
from src.data.io_utils import read_json
from src.ml.artifacts import _validate_metadata, file_sha256, load_artifact
from src.ml.experiment import CONFIG_PATH, write_atomic_json
from src.ml.handoff import FINAL_METADATA, FINAL_MODEL, SELECTION_LOCK, check_lock_metadata
from src.ml.prediction import predict_transactions, validate_features
from src.ml.reproduce import _check_versions, source_sha256

DATA_CHECKS_PATH = EXPERIMENTS_DIR / "data_checks.json"
EXAMPLES_PATH = EXPERIMENTS_DIR / "demo_examples.json"
PREPARATION_STATE = EXPERIMENTS_DIR / "demo_preparation.json"
EXAMPLE_SELECTION = "One example per confusion-matrix outcome"
OUTCOMES = [
    ("TP", "Мошенничество обнаружено (TP)", 1, 1),
    ("TN", "Обычная транзакция распознана (TN)", 0, 0),
    ("FN", "Пропущенное мошенничество (FN)", 1, 0),
    ("FP", "Ложное срабатывание (FP)", 0, 1),
]
EVALUATION_SEED = Path("/opt/fraud/evaluation")


def run_module(module, *arguments):
    """Запускать существующие команды тем же Python и из корня проекта."""
    command = [sys.executable, "-m", module, *arguments]
    print("Подготовка:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT_DIR, check=True)


def read_frozen_recipe():
    """Seed JSON из образа должен оставаться доступным внутри named volumes."""
    for path in (FINAL_METADATA, SELECTION_LOCK):
        if not Path(path).is_file():
            raise ValueError(
                f"Нет сохранённого рецепта {path}. Используйте named volumes с файлами из образа; "
                "пустой bind mount скрывает эти JSON."
            )
    metadata = read_json(FINAL_METADATA)
    lock = read_json(SELECTION_LOCK)
    if not isinstance(metadata, dict) or not isinstance(lock, dict):
        raise ValueError("Метаданные и selection lock должны содержать JSON-объекты")
    if file_sha256(FINAL_METADATA) != lock.get("metadata_sha256"):
        raise ValueError("Метаданные модели не совпадают с selection_lock.json")
    check_lock_metadata(lock, metadata)
    _validate_metadata(metadata, require_threshold=True)
    _check_versions(metadata)
    if metadata.get("frozen") is not True or metadata.get("test_evaluated") is not False:
        raise ValueError("Нужен замороженный рецепт модели до итоговой оценки test")
    if file_sha256(CONFIG_PATH) != metadata.get("config_sha256") or source_sha256() != metadata.get("source_sha256"):
        raise ValueError("Код или конфигурация отличаются от замороженного рецепта модели")
    if Path(FINAL_MODEL).exists() and not Path(FINAL_MODEL).is_file():
        raise ValueError("Путь модели занят каталогом; проверьте настройки Docker volumes")
    return metadata, lock


def validation_cache_path(metadata):
    source = metadata.get("source_paths", {}).get("validation_scores")
    if not isinstance(source, str) or not source or Path(source).is_absolute():
        raise ValueError("В рецепте отсутствует относительный путь validation cache")
    path = (Path(ROOT_DIR) / source).resolve()
    if not path.is_relative_to(Path(ROOT_DIR).resolve()):
        raise ValueError("Validation cache должен находиться внутри проекта")
    return path


def matching_file(path, expected_sha256):
    return Path(path).is_file() and file_sha256(path) == expected_sha256


def matching_data_checks(metadata):
    if not Path(DATA_CHECKS_PATH).is_file():
        return False
    audit = read_json(DATA_CHECKS_PATH)
    fields = {"raw_sha256": "dataset_sha256", "processed_sha256": "processed_sha256", "split_manifest_sha256": "split_sha256"}
    return audit.get("status") in ("ok", "ok_with_recurring_features") and all(
        audit.get(audit_name) == metadata.get(metadata_name) for audit_name, metadata_name in fields.items()
    )


def validate_demo_examples(document, pipeline, metadata):
    """Проверить четыре специально подобранных validation-примера."""
    if not isinstance(document, dict):
        raise ValueError("Файл примеров должен содержать JSON-объект")
    if document.get("source_split") != "validation" or document.get("selection") != EXAMPLE_SELECTION:
        raise ValueError("Примеры должны быть явно отобраны из validation по исходам матрицы ошибок")
    if document.get("run_id") != metadata["run_id"] or document.get("artifact_sha256") != metadata["artifact_sha256"]:
        raise ValueError("Примеры относятся к другой модели")
    examples = document.get("examples")
    if not isinstance(examples, list) or len(examples) != 4:
        raise ValueError("Для демонстрации нужны ровно четыре примера: TP, TN, FN и FP")
    expected_pairs = {(actual, predicted) for _, _, actual, predicted in OUTCOMES}
    pairs, ids, features, scores, expected_predictions = [], [], [], [], []
    for example in examples:
        for field in ("transaction_id", "actual_class", "expected_prediction"):
            if type(example.get(field)) is not int:
                raise ValueError(f"Поле примера {field} должно быть целым числом")
        if not isinstance(example.get("name"), str) or not example["name"].strip():
            raise ValueError("У каждого примера должно быть понятное название")
        pairs.append((example["actual_class"], example["expected_prediction"]))
        ids.append(example["transaction_id"])
        features.append(example.get("features", {}))
        scores.append(example.get("expected_score"))
        expected_predictions.append(example["expected_prediction"])
    if set(pairs) != expected_pairs or len(set(ids)) != 4:
        raise ValueError("Примеры должны содержать разные транзакции и все четыре исхода")
    X = pd.DataFrame(features, index=pd.Index(ids, name=ID_COLUMN))
    X = validate_features(X, metadata)
    predictions = predict_transactions(X, pipeline, metadata)
    if not np.array_equal(predictions["y_pred"], expected_predictions) or not np.array_equal(predictions["score"], scores):
        raise ValueError("Предсказания модели не совпадают с сохранёнными примерами")
    return document


def build_demo_examples(pipeline, metadata):
    """Взять по одному TP/TN/FN/FP, не загружая ни одного test-примера."""
    cache_path = validation_cache_path(metadata)
    if not matching_file(cache_path, metadata.get("scores_sha256")):
        raise ValueError("Validation cache отсутствует или не совпадает с контрольной суммой")
    cache = pd.read_parquet(cache_path)
    required = {ID_COLUMN, "split", "y_true", "score", "y_pred"}
    if not required.issubset(cache.columns) or cache.empty or cache[ID_COLUMN].duplicated().any():
        raise ValueError("Неверная схема validation cache")
    if not cache["split"].eq("validation").all() or not cache["y_true"].isin([0, 1]).all() or not cache["y_pred"].isin([0, 1]).all():
        raise ValueError("Cache должен содержать только validation и бинарные классы")
    if not np.isfinite(cache["score"].to_numpy(dtype="float64")).all():
        raise ValueError("Validation cache содержит некорректные scores")
    if not np.array_equal((cache["score"] >= metadata["threshold"]).astype(int), cache["y_pred"]):
        raise ValueError("Решения в cache не соответствуют сохранённому порогу")
    X, y = load_xy("validation", path=TRANSACTIONS_PARQUET_PATH)
    X = validate_features(X, metadata)
    if len(X) != metadata.get("validation_rows") or not np.array_equal(X.index, cache[ID_COLUMN]) or not np.array_equal(y, cache["y_true"]):
        raise ValueError("ID или истинные классы validation отличаются от выбранного запуска")
    examples = []
    for outcome, name, actual, predicted in OUTCOMES:
        candidates = cache[(cache["y_true"] == actual) & (cache["y_pred"] == predicted)]
        if candidates.empty:
            raise ValueError(f"В validation нет примера {outcome}; нельзя подменять его выдуманной транзакцией")
        row = candidates.iloc[0]
        transaction_id = int(row[ID_COLUMN])
        examples.append({
            "name": name,
            "outcome": outcome,
            "transaction_id": transaction_id,
            "actual_class": actual,
            "expected_prediction": predicted,
            "expected_score": float(row["score"]),
            "features": {column: float(X.loc[transaction_id, column]) for column in FEATURE_COLUMNS},
        })
    document = {
        "source_split": "validation",
        "selection": EXAMPLE_SELECTION,
        "selection_note": "Примеры специально выбраны по исходам TP/TN/FN/FP; это демонстрация, а не отдельная оценка качества.",
        "run_id": metadata["run_id"],
        "artifact_sha256": metadata["artifact_sha256"],
        "test_evaluated": False,
        "examples": examples,
    }
    validate_demo_examples(document, pipeline, metadata)
    write_atomic_json(document, EXAMPLES_PATH)
    return document


def prepared_hashes(metadata):
    files = {
        "raw": RAW_CSV_PATH, "processed": TRANSACTIONS_PARQUET_PATH,
        "manifest": SPLIT_MANIFEST_PATH, "config": CONFIG_PATH,
        "raw_audit": AUDIT_JSON_PATH, "split_summary": SPLIT_SUMMARY_JSON_PATH,
        "model": FINAL_MODEL, "metadata": FINAL_METADATA, "lock": SELECTION_LOCK,
        "validation_cache": validation_cache_path(metadata), "examples": EXAMPLES_PATH,
        "data_checks": DATA_CHECKS_PATH,
    }
    return {name: file_sha256(path) for name, path in files.items()}


def prepare_demo():
    """Готовый комплект проверяется по SHA и четырём примерам без повторного fit."""
    metadata, lock = read_frozen_recipe()
    if Path(PREPARATION_STATE).is_file():
        state = read_json(PREPARATION_STATE)
        if not isinstance(state, dict) or state.get("status") != "prepared" or state.get("test_evaluated") is not False:
            raise ValueError("Неверный файл состояния подготовки демонстрации")
        try:
            hashes = prepared_hashes(metadata)
        except OSError as exc:
            raise ValueError("Подготовленный комплект неполон; проверьте сохранённые Docker volumes") from exc
        if state.get("run_id") != metadata["run_id"] or state.get("hashes") != hashes:
            raise ValueError("Файлы подготовленной демонстрации изменились; повторного обучения не выполнялось")
        if not matching_file(FINAL_MODEL, lock["artifact_sha256"]):
            raise ValueError("Сохранённая модель повреждена; повторного обучения не выполнялось")
        pipeline, loaded_metadata = load_artifact(FINAL_MODEL, FINAL_METADATA)
        check_lock_metadata(lock, loaded_metadata)
        validate_demo_examples(read_json(EXAMPLES_PATH), pipeline, loaded_metadata)
        return {"status": "already_prepared", "run_id": metadata["run_id"], "steps_run": [], "test_evaluated": False}

    steps = []

    def run(module, *arguments):
        run_module(module, *arguments)
        steps.append([module, *arguments])

    if not Path(RAW_CSV_PATH).is_file():
        run("src.data.cli", "download")
    if not matching_file(RAW_CSV_PATH, metadata.get("dataset_sha256")):
        raise ValueError("Исходный CSV не соответствует замороженной модели")
    if not Path(AUDIT_JSON_PATH).is_file():
        run("src.data.cli", "audit")
    rebuild_data = (
        not matching_file(TRANSACTIONS_PARQUET_PATH, metadata.get("processed_sha256"))
        or not matching_file(SPLIT_MANIFEST_PATH, metadata.get("split_sha256"))
        or not Path(SPLIT_SUMMARY_JSON_PATH).is_file()
    )
    if rebuild_data:
        run("src.data.cli", "optimize")
        run("src.data.cli", "split")
    if not matching_file(TRANSACTIONS_PARQUET_PATH, metadata.get("processed_sha256")) or not matching_file(SPLIT_MANIFEST_PATH, metadata.get("split_sha256")):
        raise ValueError("Подготовленные данные или manifest отличаются от замороженного рецепта")
    if rebuild_data or not matching_data_checks(metadata):
        run("src.ml.data_checks", "--review-recurring")
    if not matching_data_checks(metadata):
        raise ValueError("Проверка данных не подтвердила выбранные версии входных файлов")
    if not Path(FINAL_MODEL).is_file():
        run("src.ml.reproduce")
    metadata, lock = read_frozen_recipe()
    if not matching_file(FINAL_MODEL, lock["artifact_sha256"]):
        raise ValueError("Модель не совпадает с selection lock; существующий файл не перезаписывается")
    if not matching_file(validation_cache_path(metadata), metadata.get("scores_sha256")):
        raise ValueError("Нет корректного validation cache. Восстановите полный комплект, не заменяя существующую модель")
    run("src.ml.handoff", "verify")
    pipeline, metadata = load_artifact(FINAL_MODEL, FINAL_METADATA)
    check_lock_metadata(lock, metadata)
    build_demo_examples(pipeline, metadata)
    state = {
        "status": "prepared", "run_id": metadata["run_id"],
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "hashes": prepared_hashes(metadata), "test_evaluated": False,
        "steps_run": steps,
    }
    write_atomic_json(state, PREPARATION_STATE)
    return state


def restore_evaluation_reports():
    """Populate old and new Docker volumes from the immutable image seed.

    Never run test during startup. Only the exact legacy document preserved by
    the verified migration may be replaced; other conflicts fail closed.
    """
    if not EVALUATION_SEED.is_dir():
        return
    from src.evaluation.contract import FrozenPipelinePredictor
    from src.evaluation.final_test import load_result, validate_result
    from src.evaluation.sources import project_paths

    paths = project_paths(ROOT_DIR)
    predictor = FrozenPipelinePredictor.load(paths.model, paths.selection_lock)
    source = EVALUATION_SEED / "test_final.json"
    seed = read_json(source)
    validate_result(seed, paths, predictor)
    existing = load_result(ROOT_DIR)
    if existing is not None:
        try:
            validate_result(existing, paths, predictor)
        except ValueError:
            if existing != seed.get("provenance_verification", {}).get("original_result"):
                raise ValueError("Docker volume содержит другую итоговую оценку; сохраните её и проверьте происхождение")
            shutil.copyfile(source, paths.evaluation_dir / "test_final.json")
    for file in EVALUATION_SEED.rglob("*"):
        if file.is_file() and file.suffix in (".json", ".csv", ".png"):
            destination = paths.evaluation_dir / file.relative_to(EVALUATION_SEED)
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(file, destination)


def prepare_full_app():
    """Create the additional sources used by all four application sections."""
    restore_evaluation_reports()
    steps = [
        (Path(ROOT_DIR) / "data/processed/fraud.db", "src.data.cli", ("build-db",)),
        (Path(ROOT_DIR) / "reports/figures/eda_01_class_balance.png", "src.data.cli", ("eda",)),
        (Path(ROOT_DIR) / "reports/sql_results/q01_split_overview.csv", "src.data.cli", ("sql",)),
        (Path(ROOT_DIR) / "reports/experiments/evaluation/validation_selected.json", "src.evaluation.metrics", ()),
    ]
    for output, module, arguments in steps:
        if not output.is_file():
            run_module(module, *arguments)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-app", action="store_true", help="Также подготовить SQLite, EDA, SQL и оценку validation")
    parser.add_argument("--check", action="store_true", help="Проверить готовность комплекта перед запуском сервера")
    args = parser.parse_args()
    try:
        result = prepare_demo()
        if args.full_app:
            prepare_full_app()
        if args.check:
            run_module("src.evaluation.check", "--strict", "--no-app")
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Не удалось подготовить демонстрацию: {exc}\n")
    print(json.dumps({key: result[key] for key in ("status", "run_id", "steps_run", "test_evaluated")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
