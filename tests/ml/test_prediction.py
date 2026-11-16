"""Синтетические входы и mock predict_proba: реальные транзакции не используются."""

import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS
from src.ml.prediction import fraud_scores, predict_transactions, validate_features


class SyntheticPipeline:
    """Минимальная тестовая заглушка с намеренно обратным порядком classes_."""

    classes_ = np.array([1, 0])

    def predict_proba(self, X):
        scores = X["V1"].to_numpy(dtype="float64")
        return np.column_stack([scores, 1 - scores])


@pytest.fixture
def metadata():
    return {
        "feature_names": list(FEATURE_COLUMNS),
        "feature_dtypes": {name: "float32" if name != "Amount" else "float64" for name in FEATURE_COLUMNS},
        "positive_label": 1, "comparison": ">=", "threshold": 0.5,
    }


@pytest.fixture
def features():
    X = pd.DataFrame(0.0, index=pd.Index([9, 3, 7], name="transaction_id"), columns=FEATURE_COLUMNS)
    X["V1"] = [0.25, 0.5, 0.75]
    X["Amount"] = [0.0, 12.34, 99.99]
    return X


def test_orders_casts_and_preserves_input(features, metadata):
    shuffled = features[list(reversed(features.columns))]
    original = shuffled.copy()
    ordered = validate_features(shuffled, metadata)
    assert ordered.columns.tolist() == FEATURE_COLUMNS
    assert str(ordered["V1"].dtype) == "float32"
    assert str(ordered["Amount"].dtype) == "float64"
    pd.testing.assert_index_equal(ordered.index, features.index)
    pd.testing.assert_frame_equal(shuffled, original)


def test_fraud_class_order_and_inclusive_saved_threshold(features, metadata):
    result = predict_transactions(features, SyntheticPipeline(), metadata)
    pd.testing.assert_index_equal(result.index, features.index)
    assert result.columns.tolist() == ["score", "threshold", "y_pred"]
    assert result["score"].tolist() == [0.25, 0.5, 0.75]
    assert result["y_pred"].tolist() == [0, 1, 1]
    assert result["threshold"].tolist() == [0.5, 0.5, 0.5]


@pytest.mark.parametrize("problem", ["missing", "extra", "duplicate", "string", "bool", "complex", "nan", "inf", "overflow", "empty"])
def test_invalid_input_schema_or_values_fail(features, metadata, problem):
    X = features.copy()
    if problem == "missing":
        X = X.drop(columns=["V2"])
    elif problem == "extra":
        X["Class"] = 0
    elif problem == "duplicate":
        X.columns = ["V1", "V1", *X.columns[2:]]
    elif problem == "string":
        X["V2"] = "1.0"
    elif problem == "bool":
        X["V2"] = True
    elif problem == "complex":
        X["V2"] = 1 + 2j
    elif problem == "nan":
        X.loc[9, "V2"] = np.nan
    elif problem == "inf":
        X.loc[9, "V2"] = np.inf
    elif problem == "overflow":
        X.loc[9, "V2"] = 1e100
    else:
        X = X.iloc[:0]
    with pytest.raises(ValueError):
        predict_transactions(X, SyntheticPipeline(), metadata)


@pytest.mark.parametrize("problem", ["names", "dtype_missing", "dtype_bool", "dtype_complex", "comparison", "threshold_none", "threshold_nan", "threshold_bool", "threshold_range", "label"])
def test_invalid_metadata_fails(features, metadata, problem):
    metadata = {**metadata, "feature_dtypes": dict(metadata["feature_dtypes"])}
    if problem == "names":
        metadata["feature_names"] = ["Class", *FEATURE_COLUMNS[1:]]
    elif problem == "dtype_missing":
        del metadata["feature_dtypes"]["V1"]
    elif problem == "dtype_bool":
        metadata["feature_dtypes"]["V1"] = "bool"
    elif problem == "dtype_complex":
        metadata["feature_dtypes"]["V1"] = "complex64"
    elif problem == "comparison":
        metadata["comparison"] = ">"
    elif problem == "threshold_none":
        metadata["threshold"] = None
    elif problem == "threshold_nan":
        metadata["threshold"] = np.nan
    elif problem == "threshold_bool":
        metadata["threshold"] = True
    elif problem == "threshold_range":
        metadata["threshold"] = 1.1
    else:
        metadata["positive_label"] = 0
    with pytest.raises(ValueError):
        predict_transactions(features, SyntheticPipeline(), metadata)


def test_dtype_list_is_supported(features, metadata):
    metadata["feature_dtypes"] = [metadata["feature_dtypes"][name] for name in FEATURE_COLUMNS]
    assert predict_transactions(features, SyntheticPipeline(), metadata)["y_pred"].tolist() == [0, 1, 1]


def test_bad_classes_and_probability_outputs_fail(features):
    pipeline = SyntheticPipeline()
    pipeline.classes_ = np.array([0])
    with pytest.raises(ValueError, match="both classes"):
        fraud_scores(pipeline, features)
    pipeline.classes_ = np.array([1, 0])
    for output in (np.array([0.5, 0.5, 0.5]), np.ones((2, 2)), np.full((3, 2), np.nan), np.full((3, 2), 1.1)):
        pipeline.predict_proba = lambda X, output=output: output
        with pytest.raises(ValueError):
            fraud_scores(pipeline, features)
