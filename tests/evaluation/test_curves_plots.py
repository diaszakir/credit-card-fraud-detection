"""Кривые согласованы с метриками; графики строятся и не сохраняются для фикстур без разрешения."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import precision_recall_curve

from src.config import ROOT_DIR
from src.evaluation import curves, plots
from src.ml.metrics import compute_metrics, select_threshold

RNG = np.random.default_rng(1)
Y = np.r_[np.zeros(3000, dtype=int), np.ones(60, dtype=int)]
SCORES = np.r_[RNG.beta(1, 12, 3000), RNG.beta(5, 2, 60)]  # синтетические оценки


def test_pr_curve_matches_sklearn_and_keeps_ends_when_thinned():
    full = curves.pr_curve(Y, SCORES, max_points=10**6)
    precision, recall, _ = precision_recall_curve(Y, SCORES, drop_intermediate=False)
    np.testing.assert_array_equal(full["precision"], precision)
    assert np.isnan(full["threshold"].iloc[-1])
    thin = curves.pr_curve(Y, SCORES, max_points=100)
    assert len(thin) <= 102
    assert thin.iloc[0].equals(full.iloc[0]) and thin.iloc[-1]["recall"] == 0.0


def test_threshold_curve_agrees_with_metrics_at_selected_threshold():
    choice = select_threshold(Y, SCORES, 0.85)
    curve = curves.threshold_curve(Y, SCORES, max_points=10**6)
    assert curve["threshold"].is_monotonic_increasing
    row = curve[curve["threshold"] == choice["threshold"]].iloc[0]
    shared = compute_metrics(Y, SCORES, choice["threshold"])
    assert row["precision"] == shared["precision"] and row["recall"] == shared["recall"]
    assert row["predicted_positive"] == shared["tp"] + shared["fp"]
    point = curves.operating_point(Y, SCORES, choice["threshold"])
    assert point["precision"] == shared["precision"] and point["recall"] == shared["recall"]


def test_roc_points_span_the_unit_square():
    roc = curves.roc_points(Y, SCORES)
    assert roc["fpr"].iloc[0] == 0 and roc["tpr"].iloc[-1] == 1


def test_operating_points_of_committed_runs_exclude_infeasible():
    comparison = pd.read_csv(ROOT_DIR / "reports" / "experiments" / "comparison.csv")
    points = curves.run_operating_points(comparison)
    assert len(points) == int((comparison["status"] == "ok").sum())
    assert "dummy" not in set(points["model_name"])
    assert (points["precision_low"] <= points["precision"]).all()


def test_figures_build_from_curves():
    point = curves.operating_point(Y, SCORES, 0.5)
    pr = plots.pr_figure(curves.pr_curve(Y, SCORES), point, 0.85, Y.mean(), 0.9)
    assert pr.data[0].name.startswith("PR-кривая")
    thr = plots.threshold_figure(curves.threshold_curve(Y, SCORES), 0.5, 0.85)
    assert [t.name for t in thr.data] == ["Precision", "Recall", "F1"]
    assert len(plots.roc_figure(curves.roc_points(Y, SCORES), point).data) == 3


def test_comparison_figures_highlight_selected_run():
    comparison = pd.read_csv(ROOT_DIR / "reports" / "experiments" / "comparison.csv")
    selected = "random_forest_class_weight_cv_9db7932bff3b"
    bars = plots.ap_figure(comparison, selected)
    assert list(bars.data[0].marker.color).count(plots.BLUE) == 1
    scatter = plots.operating_points_figure(curves.run_operating_points(comparison), selected, 0.85)
    assert {t.name for t in scatter.data} == {"Logistic Regression", "Random Forest"}


def test_report_figures_require_real_model(project):
    with pytest.raises(ValueError, match="real"):
        plots.save_report_figures(project["root"])
    written = plots.save_report_figures(project["root"], allow_non_real=True)
    assert len(written) == 4 and all(path.stat().st_size > 0 for path in written)


def test_available_run_scores_lists_local_caches(project):
    found = curves.available_run_scores(project["root"] / "reports" / "experiments")
    assert list(found) == ["synthetic_selected"]
