"""Графики используют только синтетические validation cache и временные файлы."""

import json

import pandas as pd
import pytest

from src.ml.metrics import compute_metrics
from tests.ml.test_comparison import make_result, save_scores


@pytest.fixture
def plotting_inputs(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    import src.ml.plots as plots

    directory = tmp_path / "reports" / "experiments"
    results = directory / "results"
    results.mkdir(parents=True)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "min_precision": 0.85, "seed": 42, "models": ["logistic_regression"],
        "model_order": ["logistic_regression", "random_forest", "dummy"],
        "strategy_order": ["none", "class_weight", "smote", "undersampling"],
    }))
    run = make_result("synthetic_plot")
    cache_path = save_scores(tmp_path, run)
    frame = pd.read_parquet(cache_path)
    run["validation_metrics"] = compute_metrics(frame["y_true"], frame["score"], run["threshold"])
    (results / "synthetic_plot.json").write_text(json.dumps(run))
    monkeypatch.setattr(plots, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(plots, "EXPERIMENTS_DIR", directory)
    monkeypatch.setattr(plots, "CONFIG_PATH", config_path)
    return plots, cache_path, directory


def test_plots_create_validation_figures_from_verified_cache(plotting_inputs):
    plots, _, directory = plotting_inputs
    paths = plots.draw_figures()
    expected = {
        "validation_average_precision.png", "validation_precision_recall.png",
        "validation_confusion_matrix.png", "selected_validation_pr_curve.csv",
    }
    assert {path.name for path in (directory / "figures").iterdir()} == expected
    assert len(paths) == len(expected)
    assert all((plots.ROOT_DIR / path).stat().st_size > 0 for path in paths)
    curve = pd.read_csv(directory / "figures" / "selected_validation_pr_curve.csv")
    assert len(curve) == 5 and curve.iloc[-1]["recall"] == 0
    assert pd.isna(curve.iloc[-1]["threshold"])


def test_plots_reject_changed_scores_before_reading_parquet(plotting_inputs, monkeypatch):
    plots, cache_path, _ = plotting_inputs
    frame = pd.read_parquet(cache_path)
    frame.loc[0, "score"] = 0.7
    frame.to_parquet(cache_path, index=False)
    monkeypatch.setattr(plots.pd, "read_parquet", lambda *args, **kwargs: pytest.fail("unverified cache was read"))
    with pytest.raises(ValueError, match="cache SHA-256 mismatch"):
        plots.draw_figures()
