"""Streamlit проверяется на синтетических моделях и примерах в tmp, без реальных данных."""

from io import BytesIO
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

streamlit = pytest.importorskip("streamlit", reason="Streamlit is required for the optional demo screen")
from streamlit.testing.v1 import AppTest

from app import ui_helpers
from src.config import FEATURE_COLUMNS
from src.ml.artifacts import file_sha256, save_artifact
from src.ml.metrics import compute_metrics
from src.ml.pipelines import make_pipeline
from src.ml.prediction import predict_transactions


APP_PATH = Path(__file__).resolve().parents[2] / "app" / "streamlit_app.py"


@pytest.fixture
def demo_files(tmp_path, monkeypatch):
    monkeypatch.setattr(ui_helpers, "ROOT_DIR", tmp_path)
    dtypes = {name: "float32" if name != "Amount" else "float64" for name in FEATURE_COLUMNS}
    X_train = pd.DataFrame(0.0, index=range(16), columns=FEATURE_COLUMNS)
    y_train = np.tile([0, 1], 8)
    X_train["V1"] = np.where(y_train == 1, 4.0, -4.0)
    X_train["Amount"] = 20.0
    X_train = X_train.astype(dtypes)
    pipeline = make_pipeline("logistic_regression")
    pipeline.fit(X_train, y_train)

    X = X_train.iloc[:4].copy()
    X.index = pd.Index([101, 102, 103, 104], name="transaction_id")
    X["V1"] = [4.0, -4.0, -4.0, 4.0]
    X = X.astype(dtypes)
    actual = np.array([1, 0, 1, 0])
    metadata = {
        "schema_version": 1, "protocol_version": "synthetic-ui-test", "run_id": "synthetic_selected",
        "model_name": "logistic_regression", "imbalance_strategy": "none", "tuning": "fixed",
        "split": "validation", "frozen": True, "test_evaluated": False, "synthetic_fixture": True,
        "feature_names": list(FEATURE_COLUMNS), "feature_dtypes": dtypes,
        "positive_label": 1, "threshold": 0.5, "comparison": ">=", "min_precision": 0.85,
        "dataset_sha256": "a" * 64, "processed_sha256": "b" * 64,
        "split_sha256": "c" * 64, "source_sha256": "d" * 64, "config_sha256": "e" * 64,
        "library_versions": {
            "python": platform.python_version(),
            **{name: version(name) for name in ("scikit-learn", "imbalanced-learn", "joblib", "numpy", "scipy")},
        },
    }
    predictions = predict_transactions(X, pipeline, metadata)
    metadata["validation_metrics"] = compute_metrics(actual, predictions["score"], metadata["threshold"])
    model_path = tmp_path / "models" / "fraud_pipeline.joblib"
    metadata = save_artifact(pipeline, metadata, model_path)
    lock = {"selected_run_id": metadata["run_id"], "metadata_sha256": file_sha256(model_path.with_suffix(".json"))}
    for name in ("threshold", "comparison", "min_precision", "artifact_sha256", "dataset_sha256",
                 "processed_sha256", "split_sha256", "source_sha256", "config_sha256"):
        lock[name] = metadata[name]
    (tmp_path / "models" / "selection_lock.json").write_text(json.dumps(lock))

    reports = tmp_path / "reports" / "experiments"
    reports.mkdir(parents=True)
    comparison = pd.DataFrame([{
        "run_id": metadata["run_id"], "model_name": "logistic_regression", "imbalance_strategy": "none",
        "tuning": "fixed", "status": "ok", "threshold": 0.5, **metadata["validation_metrics"],
    }, {
        "run_id": "synthetic_dummy", "model_name": "dummy", "imbalance_strategy": "none", "tuning": "fixed",
        "status": "no_feasible_threshold", "threshold": None, "precision": None, "recall": None,
        "pr_auc": 0.75, "average_precision": 0.5,
    }])
    comparison.to_csv(reports / "comparison.csv", index=False)
    examples = []
    for position, name in enumerate(["TP — распознанный fraud", "TN — обычная транзакция", "FN — пропущенный fraud", "FP — ложная тревога"]):
        examples.append({
            "name": name, "transaction_id": int(X.index[position]), "actual_class": int(actual[position]),
            "expected_prediction": int(predictions.iloc[position]["y_pred"]),
            "features": {name: float(X.iloc[position][name]) for name in FEATURE_COLUMNS},
        })
    (reports / "demo_examples.json").write_text(json.dumps({
        "source_split": "validation", "selection": "One example per confusion-matrix outcome", "examples": examples,
    }, ensure_ascii=False))
    return {"root": tmp_path, "model_path": model_path, "X": X, "metadata": metadata,
            "pipeline": pipeline, "predictions": predictions, "examples": examples}


def test_screen_shows_saved_validation_metrics_and_explicit_scope(demo_files):
    app = AppTest.from_file(str(APP_PATH), default_timeout=15).run()
    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["Результаты", "Прогноз"]
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Precision fraud"] == f"{demo_files['metadata']['validation_metrics']['precision']:.4f}"
    assert set(metrics) == {"Precision fraud", "Recall fraud", "PR-AUC", "AP"}
    assert any("test ещё не выполнена" in message.value for message in app.info)
    assert len(app.dataframe[0].value) == 2
    assert any("не являются отдельной оценкой качества" in caption.value for caption in app.caption)


def test_ready_example_uses_the_saved_model_and_preserves_score_threshold_class(demo_files):
    app = AppTest.from_file(str(APP_PATH), default_timeout=15).run()
    app.selectbox(key="demo_example").set_value(demo_files["examples"][0]["name"])
    app.button(key="predict_example").click().run()
    assert not app.exception
    assert not app.error
    result = app.dataframe[-1].value
    assert result.index.tolist() == [101]
    assert result.iloc[0]["Оценка fraud"] == demo_files["predictions"].iloc[0]["score"]
    assert result.iloc[0]["Порог"] == 0.5
    assert result.iloc[0]["Метка"] == demo_files["examples"][0]["expected_prediction"]
    assert result.iloc[0]["Класс"] == "Fraud"


def test_csv_flow_uses_shared_prediction_on_uploaded_numeric_features(demo_files):
    payload = demo_files["X"].to_csv(index=False).encode()
    app = AppTest.from_file(str(APP_PATH), default_timeout=15).run()
    app.file_uploader(key="prediction_csv").set_value(("transactions.csv", payload, "text/csv")).run()
    app.button(key="predict_csv").click().run()
    assert not app.exception
    assert not app.error
    result = app.dataframe[-1].value
    assert result["Метка"].tolist() == demo_files["predictions"]["y_pred"].tolist()
    np.testing.assert_array_equal(result["Оценка fraud"], demo_files["predictions"]["score"])
    assert result["Порог"].tolist() == [0.5] * len(demo_files["X"])


def test_bad_csv_columns_show_a_friendly_message(demo_files):
    payload = demo_files["X"].drop(columns=["V1"]).to_csv(index=False).encode()
    app = AppTest.from_file(str(APP_PATH), default_timeout=15).run()
    app.file_uploader(key="prediction_csv").set_value(("bad.csv", payload, "text/csv")).run()
    app.button(key="predict_csv").click().run()
    assert not app.exception
    assert any("Неверные колонки CSV" in message.value for message in app.error)


def test_missing_model_is_reported_without_crashing_the_page(demo_files):
    demo_files["model_path"].unlink()
    app = AppTest.from_file(str(APP_PATH), default_timeout=15).run()
    app.button(key="predict_example").click().run()
    assert not app.exception
    assert any("Файл обученной модели не найден" in message.value for message in app.error)


def test_metadata_tampering_is_rejected_before_artifact_loading(demo_files, monkeypatch):
    path = demo_files["model_path"].with_suffix(".json")
    metadata = json.loads(path.read_text())
    metadata["threshold"] = 0.6
    path.write_text(json.dumps(metadata))
    monkeypatch.setattr(ui_helpers, "load_artifact", lambda *args: pytest.fail("changed metadata reached joblib loader"))
    with pytest.raises(ValueError, match="контрольной суммы"):
        ui_helpers.load_frozen_model()


def test_csv_parser_handles_empty_files_and_can_be_called_twice(demo_files):
    with pytest.raises(ValueError, match="прочитать CSV"):
        ui_helpers.read_prediction_csv(BytesIO(b""))
    uploaded = BytesIO(demo_files["X"].to_csv(index=False).encode())
    first = ui_helpers.read_prediction_csv(uploaded)
    second = ui_helpers.read_prediction_csv(uploaded)
    pd.testing.assert_frame_equal(first, second)


def test_downloadable_template_has_only_29_features_and_round_trips_through_csv(demo_files):
    payload = ui_helpers.example_csv(demo_files["examples"][0])
    frame = ui_helpers.read_prediction_csv(BytesIO(payload))
    assert frame.columns.tolist() == FEATURE_COLUMNS
    assert len(frame) == 1 and "Class" not in frame and "transaction_id" not in frame
    predictions = predict_transactions(frame, demo_files["pipeline"], demo_files["metadata"])
    assert predictions.iloc[0]["score"] == demo_files["predictions"].iloc[0]["score"]
    app = AppTest.from_file(str(APP_PATH), default_timeout=15).run()
    assert app.download_button(key="download_csv_template").label == "Скачать CSV-шаблон"


def test_examples_from_another_split_are_refused(demo_files):
    path = demo_files["root"] / "reports" / "experiments" / "demo_examples.json"
    document = json.loads(path.read_text())
    document["source_split"] = "test"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="validation"):
        ui_helpers.load_demo_examples()
