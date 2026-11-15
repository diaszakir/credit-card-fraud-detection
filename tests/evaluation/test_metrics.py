"""Модуль метрик: согласованность с кодом Диаса, интервалы, отсутствие подстановок."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, roc_auc_score

from src.config import ROOT_DIR
from src.evaluation import metrics
from src.evaluation.sources import project_paths
from src.ml.metrics import compute_metrics

RNG = np.random.default_rng(0)
Y = np.r_[np.zeros(200, dtype=int), np.ones(20, dtype=int)]
SCORES = np.clip(np.r_[RNG.beta(2, 8, 200), RNG.beta(6, 3, 20)], 0, 1)  # синтетические оценки


def test_wilson_interval_matches_textbook_value_and_edges():
    low, high = metrics.wilson_interval(5, 10)
    assert low == pytest.approx(0.2366, abs=1e-4) and high == pytest.approx(0.7634, abs=1e-4)
    assert metrics.wilson_interval(0, 10)[0] == 0.0
    assert metrics.wilson_interval(10, 10)[1] == 1.0
    assert metrics.wilson_interval(0, 0) is None
    with pytest.raises(ValueError):
        metrics.wilson_interval(3, 2)


def test_counts_give_undefined_values_instead_of_zero():
    empty = metrics.metrics_from_counts(tn=10, fp=0, fn=3, tp=0)
    assert empty["precision"] is None and empty["f1"] is None and empty["precision_ci"] is None
    assert empty["recall"] == 0.0
    full = metrics.metrics_from_counts(tn=90, fp=2, fn=4, tp=6)
    assert full["f1"] == pytest.approx(2 * 6 / (2 * 6 + 2 + 4))
    assert full["specificity"] == pytest.approx(90 / 92)


def test_evaluate_scores_extends_shared_metrics_without_changing_them():
    result = metrics.evaluate_scores(Y, SCORES, 0.5, run_id="r", split="validation", provenance="synthetic")
    shared = compute_metrics(Y, SCORES, 0.5)
    assert metrics.compare_with_saved(result, shared) == []
    assert result["roc_auc"] == roc_auc_score(Y, SCORES)
    assert result["f1"] == pytest.approx(f1_score(Y, (SCORES >= 0.5).astype(int)))
    assert result["predicted_positive"] == int((SCORES >= 0.5).sum())
    assert result["n"] == 220 and result["n_positive"] == 20


def test_missing_threshold_keeps_only_ranking_metrics():
    result = metrics.evaluate_scores(Y, SCORES, None, run_id="dummy", split="validation", provenance="synthetic")
    assert result["status"] == "no_feasible_threshold"
    assert {"pr_auc", "average_precision", "roc_auc"} <= set(result)
    assert "precision" not in result and "tp" not in result


def test_compare_with_saved_reports_differences():
    result = metrics.evaluate_scores(Y, SCORES, 0.5, run_id="r", split="validation", provenance="synthetic")
    saved = dict(compute_metrics(Y, SCORES, 0.5), tp=999)
    assert metrics.compare_with_saved(result, saved) == [f"tp: {result['tp']} != 999"]


def test_only_real_results_are_saved(tmp_path):
    result = metrics.evaluate_scores(Y, SCORES, 0.5, run_id="r", split="validation", provenance="synthetic")
    with pytest.raises(ValueError, match="real"):
        metrics.save_evaluation(result, tmp_path / "x.json")
    saved = metrics.save_evaluation(dict(result, provenance="real"), tmp_path / "x.json")
    document = json.loads(saved.read_text(encoding="utf-8"))
    assert document["precision_ci"] == list(result["precision_ci"])
    long = metrics.to_long(result)
    assert {"precision_ci_low", "precision_ci_high", "roc_auc"} <= set(long["metric"])


def test_count_metrics_for_committed_comparison_are_consistent():
    """Реальный comparison.csv: F1 из счётчиков согласован с сохранёнными precision и recall."""
    comparison = pd.read_csv(ROOT_DIR / "reports" / "experiments" / "comparison.csv")
    table = metrics.add_count_metrics(comparison)
    feasible = table["status"] == "ok"
    p, r = table.loc[feasible, "precision"], table.loc[feasible, "recall"]
    np.testing.assert_allclose(table.loc[feasible, "f1"], 2 * p * r / (p + r))
    assert table.loc[~feasible, "f1"].isna().all()
    assert (table.loc[feasible, "precision_low"] <= p).all() and (p <= table.loc[feasible, "precision_high"]).all()


def test_selected_validation_is_evaluated_from_saved_predictions(project):
    result = metrics.evaluate_selected_validation(project["root"])
    assert result["matches_saved_validation_metrics"] is True
    assert (result["tn"], result["fp"], result["fn"], result["tp"]) == (1, 1, 1, 1)
    with pytest.raises(ValueError, match="real"):  # фикстура не сохраняется как результат
        metrics.save_evaluation(result, project_paths(project["root"]).evaluation_dir / "v.json")


def test_mismatch_with_saved_metrics_stops_the_cli(project):
    cache = project["root"] / project["metadata"]["source_paths"]["validation_scores"]
    frame = pd.read_parquet(cache)
    frame.loc[0, "score"] = 0.999  # TP остаётся TP, но ранговые метрики меняются
    frame.to_parquet(cache)
    with pytest.raises(ValueError, match="не совпали"):
        metrics.evaluate_selected_validation(project["root"])
