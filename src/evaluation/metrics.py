"""Единый модуль оценки: ранговые и пороговые метрики, интервалы и сохранение результата.

Основа — общая функция Диаса ``src.ml.metrics.compute_metrics`` (PR-AUC трапециями, AP,
precision, recall, accuracy, confusion matrix). Здесь добавляются ROC-AUC, F1, specificity,
доверительные интервалы Уилсона и единый формат результата для отчёта, SQLite и приложения.
Порог никогда не подбирается: он всегда передаётся извне (сохранённый на validation).

Командная строка (оценивает выбранную модель по сохранённым validation-прогнозам):
    python -m src.evaluation.metrics
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from src.ml.metrics import compute_metrics, validate_y_scores

PROVENANCE_REAL = "real"
COUNT_KEYS = ("tn", "fp", "fn", "tp")
# Метрики, которые сверяются с сохранёнными результатами Диаса (должны совпасть точно).
SHARED_KEYS = ("pr_auc", "average_precision", "precision", "recall", "accuracy", *COUNT_KEYS)


# --------------------------------------------------------------------------- #
# Базовые вычисления
# --------------------------------------------------------------------------- #
def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float] | None:
    """Доверительный интервал Уилсона для доли; None, если знаменатель равен нулю.

    Подходит для малого числа fraud (55 в validation, 52 в test), где нормальное
    приближение даёт границы за пределами [0, 1].
    """
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("нужно 0 <= successes <= total")
    if not 0 < confidence < 1:
        raise ValueError("confidence должен быть в (0, 1)")
    if total == 0:
        return None
    z = float(norm.ppf(1 - (1 - confidence) / 2))
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    low = 0.0 if successes == 0 else max(0.0, centre - half)
    high = 1.0 if successes == total else min(1.0, centre + half)  # точные края без ошибки округления
    return low, high


def _ratio(numerator: int, denominator: int) -> float | None:
    """Доля или None, если она не определена (а не 0)."""
    return float(numerator / denominator) if denominator else None


def metrics_from_counts(tn: int, fp: int, fn: int, tp: int, confidence: float = 0.95) -> dict:
    """Пороговые метрики только из матрицы ошибок.

    Позволяет посчитать F1 и интервалы для всех 17 запусков по сохранённым счётчикам,
    даже если построчных прогнозов этих запусков на компьютере нет.
    """
    counts = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}
    for name, value in counts.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"{name} должен быть целым числом >= 0")
    tn, fp, fn, tp = (int(counts[k]) for k in COUNT_KEYS)
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = None
    if precision is not None and recall is not None and precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    elif precision is not None and recall is not None:
        f1 = 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": _ratio(tn, tn + fp),
        "false_positive_rate": _ratio(fp, tn + fp),
        "accuracy": _ratio(tn + tp, tn + fp + fn + tp),
        "predicted_positive": tp + fp,
        "precision_ci": wilson_interval(tp, tp + fp, confidence),
        "recall_ci": wilson_interval(tp, tp + fn, confidence),
        **counts,
    }


# --------------------------------------------------------------------------- #
# Полная оценка одного набора прогнозов
# --------------------------------------------------------------------------- #
def evaluate_scores(
    y_true,
    scores,
    threshold: float | None,
    *,
    run_id: str,
    split: str,
    provenance: str,
    min_precision: float = 0.85,
    confidence: float = 0.95,
) -> dict:
    """Все метрики для одного запуска и одной части данных.

    threshold=None означает «нет допустимого порога» (например, Dummy): тогда считаются
    только ранговые метрики, а пороговые поля отсутствуют, а не равны нулю.
    """
    y, score = validate_y_scores(y_true, scores)
    base = compute_metrics(y, score, threshold, min_precision)  # общий код Диаса
    result = {
        "run_id": run_id,
        "split": split,
        "provenance": provenance,
        "threshold": None if threshold is None else float(threshold),
        "comparison": ">=",
        "min_precision": float(min_precision),
        "confidence": float(confidence),
        "n": int(len(y)),
        "n_positive": int(y.sum()),
        "prevalence": float(y.mean()),
        "pr_auc": base["pr_auc"],
        "average_precision": base["average_precision"],
        "roc_auc": float(roc_auc_score(y, score)),
        "status": "ok" if threshold is not None else "no_feasible_threshold",
    }
    if threshold is None:
        return result
    from_counts = metrics_from_counts(*(base[k] for k in COUNT_KEYS), confidence=confidence)
    # Значения precision/recall/accuracy берутся из общего кода, чтобы совпадать побитно.
    from_counts.update({key: base[key] for key in ("precision", "recall", "accuracy")})
    result.update(from_counts)
    result["precision_constraint_met"] = base["precision_constraint_met"]
    return result


def evaluate_predictions_table(table: pd.DataFrame, metadata: dict, split: str = "validation") -> dict:
    """Оценка по построчным прогнозам (кэш validation / predictions.parquet) с проверкой контракта."""
    from src.evaluation.contract import validate_predictions_table

    threshold = metadata.get("threshold")
    validate_predictions_table(table, threshold=threshold, split=split)
    return evaluate_scores(
        table["y_true"].to_numpy(), table["score"].to_numpy(), threshold,
        run_id=str(metadata.get("run_id")), split=split,
        provenance=str(metadata.get("provenance", "unknown")),
        min_precision=float(metadata.get("min_precision", 0.85)),
    )


def compare_with_saved(result: dict, saved: dict, tolerance: float = 1e-12) -> list[str]:
    """Сверить наш результат с метриками, сохранёнными Диасом. Пустой список — всё совпало."""
    problems = []
    for key in SHARED_KEYS:
        if key not in saved:
            continue
        ours, theirs = result.get(key), saved[key]
        if ours is None or theirs is None:
            if ours is not theirs:
                problems.append(f"{key}: {ours} != {theirs}")
        elif key in COUNT_KEYS:
            if int(ours) != int(theirs):
                problems.append(f"{key}: {ours} != {theirs}")
        elif abs(float(ours) - float(theirs)) > tolerance:
            problems.append(f"{key}: {ours} != {theirs}")
    return problems


# --------------------------------------------------------------------------- #
# Таблица сравнения всех запусков (по сохранённым счётчикам)
# --------------------------------------------------------------------------- #
def add_count_metrics(comparison: pd.DataFrame, confidence: float = 0.95) -> pd.DataFrame:
    """Добавить F1, specificity и интервалы precision/recall к comparison.csv.

    Строки без порога (нет tp/fp/fn/tn) остаются пустыми — значения не придумываются.
    """
    frame = comparison.copy()
    columns = ["f1", "specificity", "precision_low", "precision_high", "recall_low", "recall_high"]
    for column in columns:
        frame[column] = np.nan
    for index, row in frame.iterrows():
        if any(pd.isna(row.get(key)) for key in COUNT_KEYS):
            continue
        values = metrics_from_counts(*(int(row[key]) for key in COUNT_KEYS), confidence=confidence)
        frame.at[index, "f1"] = values["f1"]
        frame.at[index, "specificity"] = values["specificity"]
        if values["precision_ci"]:
            frame.at[index, "precision_low"], frame.at[index, "precision_high"] = values["precision_ci"]
        if values["recall_ci"]:
            frame.at[index, "recall_low"], frame.at[index, "recall_high"] = values["recall_ci"]
    return frame


# --------------------------------------------------------------------------- #
# Сохранение
# --------------------------------------------------------------------------- #
def _json_ready(value):
    if isinstance(value, tuple):
        return [_json_ready(v) for v in value]
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Real) and not math.isfinite(float(value)):
        raise ValueError("в результате есть NaN или бесконечность")
    return value


def save_evaluation(result: dict, path: Path) -> Path:
    """Записать JSON атомарно; результаты заглушек и тестовых фикстур не сохраняются."""
    if result.get("provenance") != PROVENANCE_REAL:
        raise ValueError("сохраняются только результаты с provenance='real'")
    document = dict(_json_ready(result), created_at=datetime.now(timezone.utc).isoformat())
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def to_long(result: dict) -> pd.DataFrame:
    """Формат metrics.parquet: run_id, split, metric, value (только числовые метрики)."""
    rows = []
    for key, value in result.items():
        if key in {"run_id", "split", "provenance", "comparison", "status", "created_at"}:
            continue
        if isinstance(value, tuple) or isinstance(value, list):
            if value:
                rows.append((f"{key}_low", float(value[0])))
                rows.append((f"{key}_high", float(value[1])))
        elif isinstance(value, (bool, np.bool_)):
            rows.append((key, float(value)))
        elif isinstance(value, Real):
            rows.append((key, float(value)))
    return pd.DataFrame([{"run_id": result["run_id"], "split": result["split"], "metric": k, "value": v}
                         for k, v in rows])


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def evaluate_selected_validation(root=None) -> dict:
    """Оценить выбранную модель на validation и сверить с сохранёнными метриками Диаса."""
    from src.evaluation import sources

    paths = sources.project_paths(root)
    metadata = sources.load_model_metadata(paths)
    if not metadata.available:
        raise FileNotFoundError(metadata.message)
    predictions = sources.load_validation_predictions(paths, metadata.data)
    if not predictions.available:
        raise FileNotFoundError(predictions.message)
    result = evaluate_predictions_table(predictions.data, metadata.data, split="validation")
    problems = compare_with_saved(result, metadata.data.get("validation_metrics", {}))
    if problems:
        raise ValueError("метрики не совпали с сохранёнными: " + "; ".join(problems))
    result["source"] = str(predictions.path.relative_to(paths.root))
    result["matches_saved_validation_metrics"] = True
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Оценка выбранной модели на validation")
    parser.add_argument("--output", type=Path, default=None, help="куда сохранить JSON")
    args = parser.parse_args(argv)
    from src.evaluation.sources import project_paths

    result = evaluate_selected_validation()
    output = args.output or project_paths().evaluation_dir / "validation_selected.json"
    save_evaluation(result, output)
    shown = {k: result[k] for k in ("run_id", "n", "n_positive", "threshold", "average_precision", "pr_auc",
                                     "roc_auc", "precision", "recall", "f1", "tn", "fp", "fn", "tp")}
    shown["precision_ci_95"] = result["precision_ci"]
    shown["recall_ci_95"] = result["recall_ci"]
    print(json.dumps(_json_ready(shown), ensure_ascii=False, indent=2))
    print(f"Совпадает с сохранёнными метриками Диаса. Записано: {output}")
    return 0


if __name__ == "__main__":
    from src.evaluation.cli_utils import run_cli

    raise SystemExit(run_cli(main))
