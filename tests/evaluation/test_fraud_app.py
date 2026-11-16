"""Четыре раздела Streamlit на чистом клоне и на синтетическом проекте."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from tests.evaluation.conftest import feature_frame

APP_PATH = str(Path(__file__).resolve().parents[2] / "app" / "fraud_app.py")
SECTIONS = ["Data Overview", "Model Comparison", "Error Analysis", "Fraud Prediction"]


def result_table(app):
    """Таблица прогноза; редактор ручного ввода тоже является dataframe-элементом."""
    return next(df.value for df in reversed(app.main.dataframe) if "score" in df.value.columns)


def open_section(name):
    app = AppTest.from_file(APP_PATH, default_timeout=20).run()
    app.sidebar.radio(key="section").set_value(name).run()
    return app


@pytest.mark.parametrize("section", SECTIONS)
def test_sections_explain_missing_artifacts_on_empty_clone(empty_project, section):
    app = open_section(section)
    assert not app.exception
    assert not app.error, [e.value for e in app.error]
    assert app.warning or app.info


@pytest.mark.parametrize("section", SECTIONS)
def test_sections_render_fixture_project(project, section):
    app = open_section(section)
    assert not app.exception
    assert not app.error, [e.value for e in app.error]


def test_model_comparison_keeps_infeasible_run_empty(project):
    app = open_section("Model Comparison")
    table = app.main.dataframe[0].value
    assert len(table) == 2
    assert table["Precision"].isna().sum() == 1  # Dummy без порога: пусто, не 0


def test_error_analysis_counts_outcomes_from_saved_predictions(project):
    app = open_section("Error Analysis")
    values = {metric.label: metric.value for metric in app.main.metric}
    assert {k: values[k] for k in ("TP", "FP", "FN", "TN")} == {"TP": "1", "FP": "1", "FN": "1", "TN": "1"}
    assert {"AP", "PR-AUC", "ROC-AUC", "F1", "Precision", "Recall"} <= set(values)
    assert not app.warning  # метрики совпали с сохранёнными в фикстуре


def test_csv_prediction_uses_frozen_model_and_label_column(project):
    payload = project["X_val"].reset_index().assign(Class=project["y_val"]).to_csv(index=False).encode()
    app = open_section("Fraud Prediction")
    assert any("Модель:" in s.value for s in app.success)
    app.file_uploader(key="prediction_csv").set_value(("tx.csv", payload, "text/csv")).run()
    app.button(key="predict_csv").click().run()
    assert not app.exception and not app.error
    result = result_table(app)
    assert result["transaction_id"].tolist() == [201, 202, 203, 204]
    np.testing.assert_array_equal(result["score"], project["predicted"]["score"])
    assert result["Исход"].tolist() == ["TP", "TN", "FN", "FP"]


def test_schema_errors_are_listed_even_without_model(empty_project):
    payload = feature_frame([1.0]).drop(columns=["V2"]).assign(note="x").to_csv(index=False).encode()
    app = open_section("Fraud Prediction")
    app.file_uploader(key="prediction_csv").set_value(("bad.csv", payload, "text/csv")).run()
    app.button(key="predict_csv").click().run()
    assert not app.exception
    text = " ".join(m.value for m in app.markdown)
    assert "V2" in text and "note" in text


def test_stub_mode_is_explicit_and_clearly_marked(empty_project, monkeypatch):
    monkeypatch.setenv("FRAUD_APP_STUB", "1")
    app = open_section("Fraud Prediction")
    assert any("ЗАГЛУШКА" in e.value for e in app.error)
    app.button(key="predict_manual").click().run()
    assert not app.exception
    assert result_table(app)["score"].between(0, 1).all()


def test_model_comparison_shows_curves_and_comparison_charts(project):
    app = open_section("Model Comparison")
    assert not app.exception and not app.error, [e.value for e in app.error]
    charts = app.main.get("plotly_chart")
    assert len(charts) == 5  # PR, порог, ROC, рабочие точки, AP


def test_error_analysis_tabs_use_features_sql_and_saved_importance(data_project):
    from src.evaluation import importance as imp

    table, info = imp.compute_selected(data_project["root"], n_repeats=2, allow_non_real=True)
    imp.save(table, dict(info, provenance="real"), data_project["root"] / "reports" / "experiments" / "evaluation")
    app = open_section("Error Analysis")
    assert not app.exception and not app.error, [e.value for e in app.error]
    assert len(app.main.get("plotly_chart")) == 3  # FN vs TP, FP vs TN, importance
    captions = " ".join(c.value for c in app.main.caption)
    assert "метки совпадают" in captions


def test_model_comparison_shows_saved_test_result_only_when_it_exists(data_project):
    import json

    from src.evaluation import final_test

    app = open_section("Model Comparison")
    assert any("ещё не выполнялась" in i.value for i in app.info)
    output = final_test.evaluate_test(data_project["root"], allow_non_real=True)
    out_dir = data_project["root"] / "reports" / "experiments" / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / final_test.RESULT_JSON).write_text(json.dumps(output["result"], default=list))
    app = open_section("Model Comparison")
    assert not app.exception and not app.error
    assert not any("ещё не выполнялась" in i.value for i in app.info)
    assert any("Порог" in c.value and "не менялся" in c.value for c in app.main.caption)
