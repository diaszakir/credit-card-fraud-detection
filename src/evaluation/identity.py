"""Portable identity of fitted inference state, independent of pickle bytes/padding."""
from __future__ import annotations

import hashlib
import json

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

METHOD = "sha256-fitted-pipeline-v1"
RECIPE_FIELDS = (
    "run_id", "model_name", "imbalance_strategy", "params", "seed", "threshold", "comparison",
    "positive_label", "min_precision", "feature_names", "feature_dtypes", "dataset_sha256",
    "processed_sha256", "split_sha256", "config_sha256", "source_sha256", "library_versions",
)


def model_identity(predictor) -> dict:
    """Hash all state used by the project's supported predict_proba pipelines.

    Tree fields are hashed separately: numpy structured arrays contain padding bytes
    which can differ between identical fitted trees and pickle round trips.
    Unknown steps/models fail closed rather than claiming a partial identity.
    """
    digest = hashlib.sha256()

    def add(name, value):
        digest.update(name.encode() + b"\0")
        array = np.asarray(value)
        if array.dtype.kind in "biu":
            array = np.ascontiguousarray(array, dtype="<i8")
        elif array.dtype.kind == "f":
            array = np.ascontiguousarray(array, dtype="<f8")
        else:
            raise ValueError(f"Unsupported model identity field: {name}")
        digest.update(str(array.shape).encode() + array.dtype.str.encode() + array.tobytes())

    pipeline = predictor.pipeline
    if type(pipeline) is not Pipeline or [name for name, _ in pipeline.steps] != ["imputer", "scaler", "sampler", "clf"]:
        raise ValueError("Unsupported pipeline structure for model identity")
    imputer, scaler, sampler, clf = [step for _, step in pipeline.steps]
    if (type(imputer) is not SimpleImputer or imputer.add_indicator
            or not isinstance(imputer.missing_values, (float, np.floating))
            or not np.isnan(imputer.missing_values) or type(scaler) is not StandardScaler):
        raise ValueError("Unsupported preprocessing for model identity")
    if not (isinstance(sampler, str) and sampler == "passthrough") and type(sampler) not in (SMOTE, RandomUnderSampler):
        raise ValueError("Unsupported sampling step for model identity")
    add("imputer.statistics_", imputer.statistics_)
    for name in ("mean_", "var_", "scale_"):
        value = getattr(scaler, name)
        if value is not None:
            add("scaler." + name, value)
    digest.update(json.dumps({"with_mean": scaler.with_mean, "with_std": scaler.with_std}, sort_keys=True).encode())
    add("classes_", clf.classes_)
    add("n_features_in_", clf.n_features_in_)
    if type(clf) is RandomForestClassifier:
        digest.update(b"RandomForestClassifier")
        add("n_outputs_", clf.n_outputs_)
        add("n_estimators", len(clf.estimators_))
        for i, estimator in enumerate(clf.estimators_):
            tree = estimator.tree_
            for field in ("children_left", "children_right", "feature", "threshold", "value", "missing_go_to_left"):
                add(f"tree.{i}.{field}", getattr(tree, field))
    elif type(clf) is LogisticRegression:
        digest.update(b"LogisticRegression")
        add("coef_", clf.coef_)
        add("intercept_", clf.intercept_)
    elif type(clf) is DummyClassifier and clf.strategy == "prior":
        digest.update(b"DummyClassifier")
        digest.update(str(clf.strategy).encode())
        add("class_prior_", clf.class_prior_)
    else:
        raise ValueError("Unsupported classifier for model identity")
    recipe = {key: predictor.metadata.get(key) for key in RECIPE_FIELDS}
    # Binary serialization may move between supported Python patch versions.
    versions = dict(recipe.get("library_versions") or {})
    if "python" in versions:
        versions["python"] = ".".join(versions["python"].split(".")[:2])
    recipe["library_versions"] = versions
    return {"method": METHOD, "state_sha256": digest.hexdigest(),
            "recipe_sha256": hashlib.sha256(json.dumps(recipe, sort_keys=True, allow_nan=False).encode()).hexdigest()}
