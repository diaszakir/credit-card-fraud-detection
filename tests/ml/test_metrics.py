"""Небольшие синтетические примеры: это проверки, а не качество fraud-модели."""

import numpy as np
import pytest
from sklearn.metrics import auc, average_precision_score, precision_recall_curve

from src.ml.metrics import compute_metrics, precision_recall_points, select_threshold


def test_selects_highest_recall_at_required_precision():
    y = [1, 0, 1, 1, 0]
    scores = [0.9, 0.8, 0.7, 0.6, 0.1]
    selected = select_threshold(y, scores, min_precision=0.7)
    assert selected == {
        "status": "ok", "threshold": 0.6, "precision": 0.75,
        "recall": 1.0, "predicted_positive": 4,
    }


def test_equal_recall_prefers_higher_precision():
    selected = select_threshold([1, 0, 0, 1], [0.9, 0.8, 0.7, 0.1], min_precision=0.6)
    assert selected["threshold"] == 0.9
    assert selected["precision"] == 1.0
    assert selected["recall"] == 0.5


def test_unattainable_precision_does_not_use_artificial_pr_endpoint():
    selected = select_threshold([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5])
    assert selected["status"] == "no_feasible_threshold"
    assert selected["threshold"] is None
    assert selected["precision"] is None
    assert selected["recall"] is None


def test_tied_scores_are_classified_together_and_boundary_is_inclusive():
    y = [1, 1, 0, 0]
    scores = [0.8, 0.8, 0.6, 0.1]
    selected = select_threshold(y, scores)
    assert selected["threshold"] == 0.8
    assert selected["predicted_positive"] == 2
    metrics = compute_metrics(y, scores, selected["threshold"])
    assert (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]) == (2, 0, 0, 2)
    assert metrics["precision_constraint_met"]


def test_pr_auc_and_ap_are_distinct_and_curve_retains_all_thresholds():
    y = [0, 0, 1, 1]
    scores = [0.1, 0.4, 0.35, 0.8]
    precision, recall, thresholds = precision_recall_curve(y, scores)
    metrics = compute_metrics(y, scores)
    assert metrics["pr_auc"] == pytest.approx(auc(recall, precision))
    assert metrics["average_precision"] == pytest.approx(average_precision_score(y, scores))
    assert metrics["pr_auc"] != pytest.approx(metrics["average_precision"])
    points = precision_recall_points(y, scores)
    assert points["thresholds"] == thresholds.tolist()
    assert len(points["precision"]) == len(points["thresholds"]) + 1


def test_no_positive_predictions_has_undefined_precision_and_failed_constraint():
    metrics = compute_metrics([0, 1], [0.1, 0.2], threshold=0.9)
    assert metrics["precision"] is None
    assert metrics["recall"] == 0.0
    assert not metrics["precision_constraint_met"]


def test_unrounded_precision_is_used_for_constraint():
    # 17 / 20 is exactly the agreed 85%; adding one FP makes it inadmissible.
    y = [1] * 17 + [0] * 3 + [0]
    scores = [0.8] * 20 + [0.2]
    assert select_threshold(y, scores)["threshold"] == 0.8
    assert select_threshold(y, scores, min_precision=0.85000001)["threshold"] is None


def test_threshold_matches_independent_counting_with_tied_scores():
    rng = np.random.default_rng(42)
    for _ in range(20):
        y = rng.integers(0, 2, size=30)
        y[:2] = [0, 1]
        scores = rng.choice([0.1, 0.2, 0.5, 0.8, 0.9], size=len(y))
        candidates = []
        for threshold in np.unique(scores):
            predictions = scores >= threshold
            tp = int(((y == 1) & predictions).sum())
            precision = tp / int(predictions.sum())
            recall = tp / int((y == 1).sum())
            if precision >= 0.6:
                candidates.append((recall, precision, float(threshold)))
        selected = select_threshold(y, scores, min_precision=0.6)
        if candidates:
            recall, precision, threshold = max(candidates)
            assert selected["threshold"] == threshold
            assert selected["precision"] == precision
            assert selected["recall"] == recall
        else:
            assert selected["status"] == "no_feasible_threshold"


@pytest.mark.parametrize("y,scores", [
    ([], []), ([0, 1], [0.1]), ([[0, 1]], [[0.1, 0.9]]),
    ([0, 0], [0.1, 0.2]), ([1, 1], [0.1, 0.2]),
    ([0, 2], [0.1, 0.9]), ([0, np.nan], [0.1, 0.9]),
    ([0, 1], [0.1, np.inf]), ([0, 1], [0.1, np.nan]),
    ([0, 1], [True, False]), ([0, 1], [0.1j, 0.9j]),
    (["0", "1"], [0.1, 0.9]),
])
def test_invalid_labels_or_scores_fail_clearly(y, scores):
    with pytest.raises(ValueError):
        select_threshold(y, scores)
    with pytest.raises(ValueError):
        compute_metrics(y, scores)


@pytest.mark.parametrize("minimum", [0, -0.1, 1.1, np.nan, np.inf, True, "0.85"])
def test_invalid_precision_minimum_fails(minimum):
    with pytest.raises(ValueError, match="min_precision"):
        select_threshold([0, 1], [0.1, 0.9], minimum)


def test_selection_cannot_be_requested_on_train_or_test():
    for split in ("train", "test"):
        with pytest.raises(ValueError, match="validation"):
            select_threshold([0, 1], [0.1, 0.9], split=split)


@pytest.mark.parametrize("threshold", [True, np.nan, np.inf, "0.5"])
def test_invalid_threshold_fails(threshold):
    with pytest.raises(ValueError, match="threshold"):
        compute_metrics([0, 1], [0.1, 0.9], threshold)
