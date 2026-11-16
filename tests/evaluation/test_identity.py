"""Fitted inference identity must survive serialization but detect model changes."""
import copy

import joblib
import pytest

from src.evaluation.contract import FrozenPipelinePredictor
from src.evaluation.identity import model_identity


def test_round_trip_preserves_identity(project, tmp_path):
    predictor = FrozenPipelinePredictor(project['pipeline'], project['metadata'])
    before = model_identity(predictor)
    path = tmp_path / 'compressed.joblib'
    joblib.dump(project['pipeline'], path, compress=3)
    after = FrozenPipelinePredictor(joblib.load(path), project['metadata'])
    assert model_identity(after) == before


def test_coefficients_and_threshold_change_identity(project):
    pipeline = copy.deepcopy(project['pipeline'])
    metadata = dict(project['metadata'])
    before = model_identity(FrozenPipelinePredictor(pipeline, metadata))
    pipeline.named_steps['clf'].coef_[0, 0] += 0.1
    assert model_identity(FrozenPipelinePredictor(pipeline, metadata))['state_sha256'] != before['state_sha256']
    metadata['threshold'] += 0.1
    assert model_identity(FrozenPipelinePredictor(pipeline, metadata))['recipe_sha256'] != before['recipe_sha256']


def test_unknown_pipeline_is_refused(project):
    pipeline = copy.deepcopy(project['pipeline'])
    pipeline.steps = pipeline.steps[1:]
    with pytest.raises(ValueError, match='Unsupported'):
        model_identity(FrozenPipelinePredictor(pipeline, project['metadata']))


def test_forest_round_trip_and_changed_tree(project, tmp_path):
    from src.ml.pipelines import make_pipeline
    pipeline = make_pipeline('random_forest', params={'clf__n_estimators': 3})
    X = project['X_val']
    pipeline.fit(X, project['y_val'])
    predictor = FrozenPipelinePredictor(pipeline, project['metadata'])
    identity = model_identity(predictor)
    path = tmp_path / 'forest.joblib'
    joblib.dump(pipeline, path, compress=3)
    restored = joblib.load(path)
    assert model_identity(FrozenPipelinePredictor(restored, project['metadata'])) == identity
    restored.named_steps['clf'].estimators_[0].tree_.value[0, 0, 0] += 0.1
    assert model_identity(FrozenPipelinePredictor(restored, project['metadata'])) != identity
