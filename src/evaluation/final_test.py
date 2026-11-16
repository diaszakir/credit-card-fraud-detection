"""Итоговая оценка замороженной модели и явная проверка её происхождения.

Правила протокола (docs/CONTRACTS.md, раздел 4; соглашения проекта 6–7):
* модель и порог берутся из замороженного пакета Диаса и не меняются;
* test читается только здесь; результат пишется в отдельный файл, ``models/`` не трогается;
* повторный запуск разрешён только для того же самого пакета модели и тех же данных
  (тогда результат идентичен); если пакет изменился после оценки test — отказ.

    python -m src.evaluation.final_test            # показать, что будет сделано, и выйти
    python -m src.evaluation.final_test --confirm  # выполнить оценку на test
    python -m src.evaluation.final_test --show     # показать сохранённый результат
    python -m src.evaluation.final_test --verify-existing  # явная проверка legacy с сохранением истории
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

RESULT_JSON = "test_final.json"
PREDICTIONS_PARQUET = "test_predictions.parquet"
COMPARED = ("average_precision", "pr_auc", "roc_auc", "precision", "recall", "f1", "tp", "fp", "fn", "tn")


def _fingerprint(paths, metadata: dict) -> dict:
    """Что должно остаться неизменным между повторными запусками."""
    from src.ml.artifacts import file_sha256

    return {
        "run_id": metadata.get("run_id"),
        "artifact_sha256": file_sha256(paths.model),
        "metadata_sha256": file_sha256(paths.model_metadata),
        "threshold": metadata.get("threshold"),
        "processed_sha256": file_sha256(paths.transactions),
        "split_sha256": metadata.get("split_sha256"),
    }


def load_result(root=None) -> dict | None:
    from src.evaluation.sources import project_paths

    path = project_paths(root).evaluation_dir / RESULT_JSON
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def validate_result(document, paths, predictor) -> None:
    """Accept identical bytes, or a verified reproduction of identical fitted state."""
    from src.evaluation.identity import model_identity

    if not isinstance(document, dict):
        raise ValueError("Итоговая оценка должна содержать JSON-объект")
    current = _fingerprint(paths, predictor.metadata)
    previous = document.get("fingerprint")
    identity = document.get("model_identity")
    if identity is not None and identity != model_identity(predictor):
        raise ValueError("test уже оценён другой обученной моделью или по другому рецепту")
    if previous != current:
        metadata = predictor.metadata
        ancestor = metadata.get("recipe_artifact_sha256", metadata.get("reproduced_from_artifact_sha256"))
        invariants = ("run_id", "threshold", "processed_sha256", "split_sha256")
        if (not isinstance(previous, dict) or identity is None
                or not metadata.get("reproduced_from_artifact_sha256")
                or ancestor != document.get("recipe_artifact_sha256")
                or any(previous.get(key) != current[key] for key in invariants)):
            raise ValueError("test уже оценён другим пакетом модели или на других данных; "
                             "повторная оценка изменённой модели нарушает протокол")
    expected = {"run_id": current["run_id"], "split": "test", "threshold": current["threshold"],
                "comparison": predictor.metadata.get("comparison"),
                "min_precision": predictor.metadata.get("min_precision")}
    if any(document.get(key) != value for key, value in expected.items()):
        raise ValueError("Сохранённая итоговая оценка имеет другой порог, запуск или протокол")


def preflight(root=None, allow_non_real: bool = False, *, verify_legacy: bool = False):
    """Все проверки до чтения test. Возвращает (paths, predictor, validation_result)."""
    from src.evaluation import sources
    from src.evaluation.contract import FrozenPipelinePredictor
    from src.evaluation.metrics import evaluate_selected_validation

    paths = sources.project_paths(root)
    predictor = FrozenPipelinePredictor.load(paths.model, paths.selection_lock)
    metadata = predictor.metadata
    if metadata.get("provenance") != "real" and not allow_non_real:
        raise ValueError("итоговая оценка выполняется только для модели с provenance='real'")
    if metadata.get("frozen") is not True:
        raise ValueError("модель не заморожена: python -m src.ml.handoff verify")
    if metadata.get("test_evaluated") is not False:
        raise ValueError("метаданные модели не подтверждают, что test ещё не использовался")
    if not paths.transactions.is_file():
        raise FileNotFoundError(f"Нет {paths.transactions}. Подготовьте данные: python -m src.data.cli split.")
    # Validation сверяется с Диасом до test: если здесь расхождение, test не открывается.
    validation = evaluate_selected_validation(root)
    previous = load_result(root)
    if previous is not None and not isinstance(previous, dict):
        raise ValueError("Итоговая оценка должна содержать JSON-объект")
    if verify_legacy and previous is None:
        raise ValueError("Нет legacy test-результата для проверки")
    if previous is not None:
        if verify_legacy:
            current = _fingerprint(paths, metadata)
            old = previous.get("fingerprint", {})
            if previous.get("model_identity") is not None or any(
                old.get(key) != current[key] for key in ("run_id", "threshold", "processed_sha256", "split_sha256")
            ):
                raise ValueError("Проверка legacy разрешена только для того же запуска, данных и порога")
            if previous.get("validation_vs_test") != {
                key: {"validation": validation[key], "test": previous.get(key)} for key in COMPARED
            }:
                raise ValueError("Legacy-оценка не совпадает с текущими validation-метриками")
        else:
            validate_result(previous, paths, predictor)
    return paths, predictor, validation


def evaluate_test(root=None, allow_non_real: bool = False, *, verify_legacy: bool = False) -> dict:
    """Прочитать test, предсказать сохранённой моделью и посчитать метрики при сохранённом пороге."""
    from src.data.api import load_xy
    from src.evaluation.metrics import evaluate_scores

    from src.evaluation.identity import model_identity

    paths, predictor, validation = preflight(root, allow_non_real, verify_legacy=verify_legacy)
    metadata = predictor.metadata
    X_test, y_test = load_xy("test", path=paths.transactions)
    predictions = predictor.predict(X_test)  # контракт проверяется внутри
    result = evaluate_scores(
        y_test.to_numpy(), predictions["score"].to_numpy(), metadata["threshold"],
        run_id=metadata["run_id"], split="test", provenance=str(metadata.get("provenance")),
        min_precision=float(metadata["min_precision"]),
    )
    comparison = {key: {"validation": validation.get(key), "test": result.get(key)} for key in COMPARED}
    document = {
        **result,
        "fingerprint": _fingerprint(paths, metadata),
        "model_identity": model_identity(predictor),
        "recipe_artifact_sha256": metadata.get("recipe_artifact_sha256",
            metadata.get("reproduced_from_artifact_sha256", metadata["artifact_sha256"])),
        "validation_vs_test": comparison,
        "threshold_retuned": False,
        "model_retrained": False,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    if verify_legacy:
        previous = load_result(root)
        keys = (*COMPARED, "n", "n_positive", "predicted_positive", "threshold", "min_precision",
                "precision_ci", "recall_ci", "precision_constraint_met", "comparison", "provenance")
        if any(json.dumps(previous.get(key), sort_keys=True) != json.dumps(document.get(key), sort_keys=True) for key in keys):
            raise ValueError("Legacy test-метрики не воспроизвелись; исходный результат не изменён")
        document["provenance_verification"] = {
            "reason": "Explicit fixed-model verification of a legacy result with different serialization hashes",
            "verified_at": document["evaluated_at"], "all_reported_metrics_identical": True,
            "threshold_retuned": False, "model_retrained": False, "original_result": previous,
        }
    else:
        previous = load_result(root)
        if previous is not None and "provenance_verification" in previous:
            document["provenance_verification"] = previous["provenance_verification"]
    table = pd.DataFrame({
        "run_id": metadata["run_id"], "transaction_id": X_test.index.astype("int64"), "split": "test",
        "y_true": y_test.to_numpy().astype("int8"), "score": predictions["score"].to_numpy(),
        "y_pred": predictions["y_pred"].to_numpy(),
    })
    return {"result": document, "predictions": table}


def save(output: dict, root=None) -> Path:
    from src.evaluation.metrics import save_evaluation
    from src.evaluation.sources import project_paths

    out_dir = project_paths(root).evaluation_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    output["predictions"].to_parquet(out_dir / PREDICTIONS_PARQUET, index=False)
    return save_evaluation(output["result"], out_dir / RESULT_JSON)


def _print_summary(document: dict) -> None:
    rows = []
    for key, values in document["validation_vs_test"].items():
        rows.append({"metric": key, "validation": values["validation"], "test": values["test"]})
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"precision 95% ДИ на test: {document.get('precision_ci')}")
    print(f"recall 95% ДИ на test: {document.get('recall_ci')}")
    print(f"Ограничение precision ≥ {document['min_precision']} на test выполнено: {document.get('precision_constraint_met')}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Однократная итоговая оценка замороженной модели на test")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--confirm", action="store_true", help="выполнить оценку на test")
    group.add_argument("--show", action="store_true", help="показать сохранённый результат")
    group.add_argument("--verify-existing", action="store_true",
                       help="явно перепроверить legacy test-результат без изменения модели и порога; сохранить историю")
    args = parser.parse_args(argv)

    if args.show:
        document = load_result()
        if document is None:
            print("Итоговая оценка на test ещё не выполнялась.")
            return 1
        preflight()  # show only a result verified against the current package
        _print_summary(document)
        return 0

    if args.verify_existing:
        if load_result() is None:
            raise ValueError("Нет legacy test-результата для проверки")
        output = evaluate_test(verify_legacy=True)
        path = save(output)
        _print_summary(output["result"])
        print(f"Legacy-метрики воспроизвелись полностью; исходный результат сохранён в истории: {path}")
        return 0

    paths, predictor, _ = preflight()
    previous = load_result()
    print(f"Модель: {predictor.info.run_id}, порог {predictor.info.threshold} (не меняется).")
    print("Проверки пройдены: пакет заморожен, validation совпадает с Диасом, test ещё не использовался в модели.")
    if previous is not None:
        print("Результат на test уже есть; повтор с тем же пакетом даст тот же результат.")
    if not args.confirm:
        print("\nЭто итоговая оценка: после неё нельзя менять модель или порог по результатам test.\n"
              "Согласуйте запуск с командой, затем выполните с флагом --confirm.")
        return 2
    output = evaluate_test()
    path = save(output)
    _print_summary(output["result"])
    print(f"Записано: {path}")
    return 0


if __name__ == "__main__":
    from src.evaluation.cli_utils import run_cli

    raise SystemExit(run_cli(main))
