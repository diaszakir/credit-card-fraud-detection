"""Данные для кривых: PR, ROC, precision/recall/F1 от порога и рабочие точки запусков.

Модуль только считает таблицы; рисованием занимается ``plots.py``. Все кривые строятся по
сохранённым validation-прогнозам. Порог не выбирается: сохранённый отмечается на кривых.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from src.ml.metrics import validate_y_scores

MAX_POINTS = 2000


def _thin(frame: pd.DataFrame, max_points: int = MAX_POINTS, keep=()) -> pd.DataFrame:
    """Оставить не больше max_points строк равномерно, сохранив края и указанные индексы."""
    if len(frame) <= max_points:
        return frame.reset_index(drop=True)
    positions = set(np.linspace(0, len(frame) - 1, max_points).round().astype(int).tolist())
    positions.update(int(i) for i in keep)
    return frame.iloc[sorted(positions)].reset_index(drop=True)


def pr_curve(y_true, scores, max_points: int = MAX_POINTS) -> pd.DataFrame:
    """Точки PR-кривой. Последняя точка (recall=0, precision=1) не имеет порога."""
    y, score = validate_y_scores(y_true, scores)
    precision, recall, thresholds = precision_recall_curve(y, score, pos_label=1, drop_intermediate=False)
    frame = pd.DataFrame({
        "precision": precision,
        "recall": recall,
        "threshold": np.r_[thresholds, np.nan],
    })
    return _thin(frame, max_points, keep=[0, len(frame) - 1])


def roc_points(y_true, scores, max_points: int = MAX_POINTS) -> pd.DataFrame:
    y, score = validate_y_scores(y_true, scores)
    fpr, tpr, thresholds = roc_curve(y, score, pos_label=1, drop_intermediate=True)
    frame = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})
    return _thin(frame, max_points, keep=[0, len(frame) - 1])


def threshold_curve(y_true, scores, max_points: int = MAX_POINTS) -> pd.DataFrame:
    """Precision, recall и F1 для каждого возможного порога ``score >= t``.

    Строится из той же PR-кривой sklearn, поэтому в сохранённом пороге значения
    совпадают с метриками модуля ``metrics``. Порог по возрастанию.
    """
    y, score = validate_y_scores(y_true, scores)
    precision, recall, thresholds = precision_recall_curve(y, score, pos_label=1, drop_intermediate=False)
    precision, recall = precision[:-1], recall[:-1]  # последняя точка без порога
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    order = np.argsort(thresholds)
    ordered = thresholds[order]
    # Число строк со score >= t для каждого порога за один проход по отсортированным оценкам.
    predicted = len(score) - np.searchsorted(np.sort(score), ordered, side="left")
    frame = pd.DataFrame({
        "threshold": ordered,
        "precision": precision[order],
        "recall": recall[order],
        "f1": f1[order],
        "predicted_positive": predicted,
    })
    return _thin(frame, max_points, keep=[0, len(frame) - 1])


def operating_point(y_true, scores, threshold: float) -> dict:
    """Precision и recall в сохранённом пороге (точка на кривых)."""
    y, score = validate_y_scores(y_true, scores)
    predicted = score >= threshold
    tp = int((predicted & (y == 1)).sum())
    fp = int((predicted & (y == 0)).sum())
    fn = int((~predicted & (y == 1)).sum())
    return {
        "threshold": float(threshold),
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "fpr": fp / int((y == 0).sum()),
    }


def run_operating_points(comparison: pd.DataFrame) -> pd.DataFrame:
    """Рабочие точки всех запусков с допустимым порогом (из сохранённых счётчиков)."""
    from src.evaluation.metrics import add_count_metrics

    table = add_count_metrics(comparison)
    feasible = table[(table["status"] == "ok") & table["precision"].notna() & table["recall"].notna()]
    columns = ["run_id", "model_name", "imbalance_strategy", "tuning", "precision", "recall", "f1",
               "precision_low", "precision_high", "recall_low", "recall_high", "average_precision", "threshold"]
    return feasible[[c for c in columns if c in feasible.columns]].reset_index(drop=True)


def available_run_scores(experiments_dir: Path) -> dict[str, Path]:
    """Запуски, у которых на этом компьютере есть построчные validation-прогнозы."""
    cache = Path(experiments_dir) / "cache"
    if not cache.is_dir():
        return {}
    return {path.parent.name: path for path in sorted(cache.glob("*/validation.parquet"))}
