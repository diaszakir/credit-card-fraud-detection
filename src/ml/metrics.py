"""Общие метрики fraud и выбор порога только на validation."""

from numbers import Real

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)


def validate_y_scores(y_true, scores):
    """Проверить бинарные метки и непрерывные оценки одинаковой длины."""
    y = np.asarray(y_true)
    score = np.asarray(scores)
    if y.ndim != 1 or score.ndim != 1 or len(y) == 0 or len(y) != len(score):
        raise ValueError("y_true and scores must be non-empty one-dimensional arrays of equal length")
    if y.dtype.kind not in "biuf" or not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("y_true must contain only binary labels 0 and 1")
    if len(np.unique(y)) != 2:
        raise ValueError("evaluation requires both classes 0 and 1")
    if score.dtype.kind not in "iuf" or not np.isfinite(score).all():
        raise ValueError("scores must contain finite real numbers")
    return y.astype("int8"), score.astype("float64")


def validate_min_precision(min_precision):
    if isinstance(min_precision, (bool, np.bool_)) or not isinstance(min_precision, Real):
        raise ValueError("min_precision must be a number in (0, 1]")
    if not np.isfinite(min_precision) or not 0 < min_precision <= 1:
        raise ValueError("min_precision must be a number in (0, 1]")
    return float(min_precision)


def precision_recall_points(y_true, scores):
    """Точки PR-кривой для сохранения и последующего построения графика."""
    y, score = validate_y_scores(y_true, scores)
    precision, recall, thresholds = precision_recall_curve(y, score, pos_label=1, drop_intermediate=False)
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": thresholds.tolist(),
    }


def select_threshold(y_true, scores, min_precision=0.85, *, split="validation"):
    """Максимальный recall при заданном precision; ничьи: precision, затем порог."""
    if split != "validation":
        raise ValueError("threshold selection is allowed only on validation")
    y, score = validate_y_scores(y_true, scores)
    minimum = validate_min_precision(min_precision)
    precision, recall, thresholds = precision_recall_curve(y, score, pos_label=1, drop_intermediate=False)

    # Последняя точка PR-кривой не имеет порога и не участвует в выборе.
    candidates = [i for i in range(len(thresholds)) if precision[i] >= minimum]
    if not candidates:
        return {
            "status": "no_feasible_threshold",
            "threshold": None,
            "precision": None,
            "recall": None,
            "predicted_positive": None,
        }

    best = max(candidates, key=lambda i: (recall[i], precision[i], thresholds[i]))
    threshold = float(thresholds[best])
    return {
        "status": "ok",
        "threshold": threshold,
        "precision": float(precision[best]),
        "recall": float(recall[best]),
        "predicted_positive": int((score >= threshold).sum()),
    }


def compute_metrics(y_true, scores, threshold=None, min_precision=0.85):
    """PR-AUC по трапециям и отдельная AP; пороговые метрики при заданном пороге."""
    y, score = validate_y_scores(y_true, scores)
    minimum = validate_min_precision(min_precision)
    precision, recall, _ = precision_recall_curve(y, score, pos_label=1, drop_intermediate=False)
    metrics = {
        "pr_auc": float(auc(recall, precision)),
        "average_precision": float(average_precision_score(y, score, pos_label=1)),
    }
    if threshold is None:
        return metrics
    if isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, Real) or not np.isfinite(threshold):
        raise ValueError("threshold must be a finite real number")

    y_pred = (score >= threshold).astype("int8")
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    # Нет положительных прогнозов: precision не определён, ограничение не выполнено.
    threshold_precision = float(tp / (tp + fp)) if tp + fp else None
    metrics.update({
        "precision": threshold_precision,
        "recall": float(tp / (tp + fn)),
        "accuracy": float(accuracy_score(y, y_pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision_constraint_met": threshold_precision is not None and threshold_precision >= minimum,
    })
    return metrics
