"""Синтетические проверки обучения и предсказания для всех стратегий."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.ml.pipelines import MODEL_NAMES, STRATEGIES, make_pipeline, parameter_grid


@pytest.fixture
def synthetic_data():
    X, y = make_classification(
        n_samples=800,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        weights=[0.96, 0.04],
        flip_y=0,
        random_state=42,
    )
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    # Оба split синтетические, и число fraud в train достаточно для SMOTE(k=5).
    return X.iloc[:600].copy(), y[:600], X.iloc[600:].copy(), y[600:]


CASES = [("dummy", "none")]
for model_name in ("logistic_regression", "random_forest"):
    for strategy in STRATEGIES:
        CASES.append((model_name, strategy))


@pytest.mark.parametrize("model_name,strategy", CASES)
def test_training_and_prediction_preserve_validation_rows(synthetic_data, model_name, strategy):
    X_train, y_train, X_val, y_val = synthetic_data
    before = X_val.copy()
    labels_before = y_val.copy()
    pipeline = make_pipeline(model_name, strategy)
    pipeline.fit(X_train, y_train)
    scores = pipeline.predict_proba(X_val)

    assert scores.shape == (len(X_val), 2)
    assert np.array_equal(pipeline.classes_, [0, 1])
    assert np.isfinite(scores).all()
    assert np.allclose(scores.sum(axis=1), 1.0)
    pd.testing.assert_frame_equal(X_val, before)
    assert np.array_equal(y_val, labels_before)

    if strategy == "undersampling":
        # fit выбрал подмножество train; predict не пересэмплировал validation.
        sampled = pipeline.named_steps["sampler"].sample_indices_
        assert len(sampled) < len(X_train)
        assert sampled.max() < len(X_train)


def test_scaler_and_imputer_learn_train_only(synthetic_data):
    X_train, y_train, X_val, _ = synthetic_data
    X_train.iloc[0, 0] = np.nan
    expected_median = X_train.median().to_numpy()
    expected_mean = X_train.fillna(X_train.median()).mean().to_numpy()
    pipeline = make_pipeline("logistic_regression", "smote")
    pipeline.fit(X_train, y_train)

    assert np.allclose(pipeline.named_steps["imputer"].statistics_, expected_median)
    assert np.allclose(pipeline.named_steps["scaler"].mean_, expected_mean)
    before_mean = pipeline.named_steps["scaler"].mean_.copy()
    pipeline.predict_proba(X_val + 10_000)
    assert np.array_equal(pipeline.named_steps["scaler"].mean_, before_mean)


def test_pipeline_and_sampler_are_reproducible(synthetic_data):
    X_train, y_train, X_val, _ = synthetic_data
    first = make_pipeline("random_forest", "smote", seed=7)
    second = make_pipeline("random_forest", "smote", seed=7)
    first.fit(X_train, y_train)
    second.fit(X_train, y_train)
    assert np.array_equal(first.predict_proba(X_val), second.predict_proba(X_val))


def test_parameters_and_invalid_combinations_are_explicit():
    pipeline = make_pipeline("logistic_regression", "none", params={"clf__C": 0.1})
    assert pipeline.named_steps["clf"].C == 0.1
    for model_name in MODEL_NAMES:
        assert isinstance(parameter_grid(model_name), dict)
    with pytest.raises(ValueError, match="Unknown model"):
        make_pipeline("unknown")
    with pytest.raises(ValueError, match="Unknown strategy"):
        make_pipeline("logistic_regression", "combined")
    with pytest.raises(ValueError, match="baseline"):
        make_pipeline("dummy", "smote")
    with pytest.raises(ValueError, match="Unknown model"):
        parameter_grid("unknown")
    with pytest.raises(ValueError, match="class_weight"):
        make_pipeline("random_forest", "smote", params={"clf__class_weight": "balanced"})
