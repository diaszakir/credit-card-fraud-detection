"""Контракт прогнозов: вход, выход, реализации Predictor и выбор между ними."""

import json

import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS
from src.evaluation.contract import (
    PREDICTION_COLUMNS, FrozenPipelinePredictor, ModelInfo, PredictionContractError, StubPredictor,
    check_input_schema, load_predictor, split_input, validate_prediction_frame, validate_predictions_table,
)
from src.ml.prediction import predict_transactions
from tests.evaluation.conftest import feature_frame


def test_schema_report_collects_every_problem_at_once():
    frame = feature_frame([1.0, 2.0]).drop(columns=["V3"])
    frame["V5"] = ["x", "y"]
    frame["V7"] = [np.nan, np.inf]
    frame["comment"] = "extra"
    report = check_input_schema(frame)
    assert not report.ok
    assert report.missing == ["V3"]
    assert report.extra == ["comment"]
    assert report.non_numeric == ["V5"]
    assert report.non_finite == {"V7": 2}
    assert len(report.messages()) == 4


def test_optional_id_and_label_are_separated_from_features():
    frame = feature_frame([1.0, -1.0]).assign(transaction_id=[7, 9], Class=[1, 0])
    report = check_input_schema(frame)
    assert report.ok and report.has_id and report.has_label
    X, y = split_input(frame)
    assert list(X.columns) == FEATURE_COLUMNS
    assert X.index.tolist() == [7, 9]
    assert y.tolist() == [1, 0]


@pytest.mark.parametrize("column, values, text", [
    ("transaction_id", [1, 1], "повторы"),
    ("transaction_id", [-1, 2], "≥ 0"),
    ("Class", [0, 2], "только 0 и 1"),
])
def test_bad_optional_columns_are_reported(column, values, text):
    report = check_input_schema(feature_frame([1.0, 2.0]).assign(**{column: values}))
    assert not report.ok
    assert any(text in message for message in report.messages())


def test_empty_and_duplicate_columns_are_not_ok():
    assert not check_input_schema(feature_frame([]).iloc[:0]).ok
    frame = feature_frame([1.0])
    duplicated = pd.concat([frame, frame[["V1"]]], axis=1)
    assert check_input_schema(duplicated).duplicated == ["V1"]


def _good_predictions(X):
    score = np.linspace(0.1, 0.9, len(X))
    return pd.DataFrame({"score": score, "threshold": 0.5, "y_pred": (score >= 0.5).astype("int8")}, index=X.index)


def test_output_contract_accepts_valid_frame_and_rejects_violations():
    X = feature_frame([1.0, 2.0, 3.0], index=[5, 6, 7])
    good = _good_predictions(X)
    assert validate_prediction_frame(good, X) is good
    with pytest.raises(PredictionContractError, match="индекс"):
        validate_prediction_frame(good.iloc[::-1], X)
    with pytest.raises(PredictionContractError, match="score >= threshold"):
        validate_prediction_frame(good.assign(y_pred=np.int8(1)), X)
    with pytest.raises(PredictionContractError, match="одинаковым"):
        validate_prediction_frame(good.assign(threshold=[0.5, 0.4, 0.5]), X)
    with pytest.raises(PredictionContractError, match="колонки"):
        validate_prediction_frame(good[["score", "y_pred", "threshold"]], X)


def test_stub_is_deterministic_labelled_and_follows_contract():
    X = feature_frame([1.0, 2.0, 1.0], index=[3, 1, 2])
    stub = StubPredictor()
    first, second = stub.predict(X), stub.predict(X)
    pd.testing.assert_frame_equal(first, second)
    assert tuple(first.columns) == PREDICTION_COLUMNS
    assert first.index.tolist() == [3, 1, 2]
    assert first["score"].iloc[0] == first["score"].iloc[2]  # одинаковые строки
    assert not stub.info.is_real and stub.info.provenance == "stub"


def test_frozen_predictor_reuses_shared_prediction_function(project):
    predictor = FrozenPipelinePredictor.load(project["model_path"], project["root"] / "models" / "selection_lock.json")
    result = predictor.predict(project["X_val"])
    expected = predict_transactions(project["X_val"], project["pipeline"], project["metadata"])
    pd.testing.assert_frame_equal(result, expected)
    assert predictor.info.threshold == 0.5
    assert predictor.info.run_id == "synthetic_selected"


def test_load_predictor_never_silently_substitutes_stub(project):
    lock = project["root"] / "models" / "selection_lock.json"
    assert load_predictor(project["model_path"], lock).mode == "frozen"
    project["model_path"].unlink()
    missing = load_predictor(project["model_path"], lock, allow_stub=False)
    assert missing.mode == "unavailable" and missing.predictor is None
    assert ".joblib" in missing.message
    stub = load_predictor(project["model_path"], lock, allow_stub=True)
    assert stub.mode == "stub" and "заглушка" in stub.message.lower()


def test_tampered_metadata_is_reported_as_failed_check(project):
    path = project["model_path"].with_suffix(".json")
    metadata = json.loads(path.read_text())
    metadata["threshold"] = 0.1
    path.write_text(json.dumps(metadata))
    status = load_predictor(project["model_path"], project["root"] / "models" / "selection_lock.json", allow_stub=True)
    assert status.mode == "unavailable"  # подделка не заменяется заглушкой
    assert "не прошла проверку" in status.message


def test_model_info_requires_feasible_threshold():
    with pytest.raises(PredictionContractError, match="порога"):
        ModelInfo.from_metadata({"threshold": None, "feature_names": FEATURE_COLUMNS})


def test_predictions_table_checks_threshold_and_split():
    table = pd.DataFrame({"run_id": "r", "transaction_id": [1, 2], "split": "validation",
                          "y_true": [1, 0], "score": [0.9, 0.2], "y_pred": [1, 0]})
    assert validate_predictions_table(table, threshold=0.5, split="validation") is table
    with pytest.raises(PredictionContractError, match="порога"):
        validate_predictions_table(table, threshold=0.95)
    with pytest.raises(PredictionContractError, match="split"):
        validate_predictions_table(table, split="test")
    with pytest.raises(PredictionContractError, match="повторяется"):
        validate_predictions_table(pd.concat([table, table]))
