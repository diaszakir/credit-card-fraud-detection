"""Все run dict и числа синтетические; provenance='real' тестирует рабочий контракт."""

from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from src.ml.comparison import choose_best, export_results, load_results
from src.ml.artifacts import file_sha256


@pytest.fixture
def config():
    return {
        "min_precision": 0.85, "seed": 42,
        "model_order": ["logistic_regression", "random_forest", "dummy"],
        "strategy_order": ["none", "class_weight", "smote", "undersampling"],
    }


def make_result(run_id, model="logistic_regression", strategy="none", recall=0.5, precision=1.0, pr_auc=0.8):
    return {
        "run_id": run_id, "model_name": model, "imbalance_strategy": strategy,
        "tuning": "fixed", "status": "ok", "provenance": "real", "split": "validation",
        "threshold": 0.8, "min_precision": 0.85, "seed": 42,
        "python_version": "3.12", "library_versions": {"scikit-learn": "test-version"},
        "dataset_sha256": "a" * 64, "processed_sha256": "f" * 64, "split_sha256": "b" * 64,
        "source_sha256": "c" * 64, "config_sha256": "d" * 64,
        "validation_metrics": {
            "pr_auc": pr_auc, "average_precision": 0.75, "precision": precision,
            "recall": recall, "accuracy": 0.75, "tn": 2, "fp": 0, "fn": 1, "tp": 1,
            "precision_constraint_met": precision >= 0.85,
        },
        "fit_seconds": 0.1, "cv_seconds": 0.0, "peak_memory_mb": None,
        "model_size_bytes": 100, "created_at": "2026-11-15T00:00:00+00:00",
        "params": {"synthetic_fixture": True}, "command": "synthetic test fixture, not a real run",
        "model_path": "models/synthetic.joblib", "metadata_path": "models/synthetic.json",
    }


def make_infeasible(run_id="dummy"):
    result = make_result(run_id, model="dummy")
    result["threshold"] = None
    result["status"] = "no_feasible_threshold"
    result["validation_metrics"] = {"pr_auc": 0.5, "average_precision": 0.01}
    return result


def save_scores(tmp_path, result, *, labels=(0, 1, 0, 1), ids=(1, 3, 6, 7)):
    path = tmp_path / (result["run_id"] + ".parquet")
    pd.DataFrame({
        "transaction_id": ids, "split": "validation", "y_true": labels,
        "score": [0.1, 0.9, 0.2, 0.3], "y_pred": [0, 1, 0, 0],
    }).to_parquet(path, index=False)
    result["scores_file"] = str(path)
    result["scores_sha256"] = file_sha256(path)
    return path


def test_choose_best_uses_recall_then_precision_then_pr_auc(config):
    worse_recall = make_result("low_recall", recall=0.4, precision=1.0, pr_auc=0.99)
    worse_precision = make_result("low_precision", recall=0.7, precision=0.9, pr_auc=0.99)
    worse_auc = make_result("low_auc", recall=0.7, precision=0.95, pr_auc=0.8)
    winner = make_result("winner", recall=0.7, precision=0.95, pr_auc=0.9)
    assert choose_best([worse_recall, worse_precision, worse_auc, winner], config) is winner


def test_fixed_tie_order_and_run_id_ignore_tuning_and_supplementary_metrics(config):
    forest = make_result("forest", model="random_forest")
    logistic_weight = make_result("weight", strategy="class_weight")
    logistic_later = make_result("z_run")
    winner = make_result("a_run")
    winner["tuning"] = "cv"
    forest["novel_validation_metrics"] = {"recall": 1.0}
    winner["novel_validation_metrics"] = {"recall": 0.0}
    assert choose_best([forest, logistic_weight, logistic_later, winner], config) is winner


def test_infeasible_runs_are_excluded_and_all_infeasible_fails(config):
    good = make_result("good")
    too_low = make_result("too_low", recall=1.0, precision=0.84)
    dishonest_flag = make_result("flag", recall=1.0)
    dishonest_flag["validation_metrics"]["precision_constraint_met"] = False
    missing_threshold = make_result("missing", recall=1.0)
    missing_threshold["threshold"] = None
    results = [make_infeasible(), too_low, dishonest_flag, missing_threshold]
    assert choose_best([*results, good], config) is good
    with pytest.raises(ValueError, match="no run satisfies"):
        choose_best(results, config)


@pytest.mark.parametrize("field,value", [
    ("provenance", "synthetic"), ("split", "test"), ("split", "train"),
    ("dataset_sha256", "e" * 64), ("processed_sha256", "e" * 64), ("split_sha256", "e" * 64),
    ("source_sha256", "e" * 64), ("config_sha256", "e" * 64),
    ("seed", 11), ("min_precision", 0.9),
    ("python_version", "3.12.99"), ("library_versions", {"scikit-learn": "other-test-version"}),
])
def test_mixed_provenance_or_configuration_is_rejected(config, field, value):
    first, other = make_result("first"), make_result("other")
    other[field] = value
    with pytest.raises(ValueError):
        choose_best([first, other], config)


@pytest.mark.parametrize("problem", ["hash", "missing_processed_hash", "nan", "duplicate_id", "unknown_model", "unknown_strategy", "invalid_threshold"])
def test_invalid_results_fail(config, problem):
    first, other = make_result("first"), make_result("other")
    if problem == "hash":
        first["source_sha256"] = "not a hash"
    elif problem == "missing_processed_hash":
        first.pop("processed_sha256")
    elif problem == "nan":
        first["validation_metrics"]["pr_auc"] = np.nan
    elif problem == "duplicate_id":
        other["run_id"] = first["run_id"]
    elif problem == "unknown_model":
        first["model_name"] = "unknown"
    elif problem == "unknown_strategy":
        first["imbalance_strategy"] = "combined"
    else:
        first["threshold"] = 1.1
    with pytest.raises(ValueError):
        choose_best([first, other], config)


def test_config_minimum_seed_and_optional_hash_are_checked(config):
    result = make_result("run")
    for field, value in (("min_precision", 0.9), ("seed", 7), ("config_sha256", "e" * 64)):
        changed = {**config, field: value}
        with pytest.raises(ValueError):
            choose_best([result], changed)


def test_load_results_is_deterministic_and_rejects_empty_or_bad_json(tmp_path):
    for run_id in ("z", "a"):
        (tmp_path / (run_id + ".json")).write_text(json.dumps(make_result(run_id)))
    assert [result["run_id"] for result in load_results(tmp_path)] == ["a", "z"]
    (tmp_path / "broken.json").write_text("{")
    with pytest.raises(ValueError, match="Malformed JSON"):
        load_results(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_results(tmp_path / "missing")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no experiment"):
        load_results(empty)


def test_export_keeps_unknown_thresholds_missing_and_preserves_provenance(tmp_path):
    good, unavailable = make_result("good"), make_infeasible()
    good["novel_validation_metrics"] = {"recall": 0.2}
    save_scores(tmp_path, good)
    original = deepcopy(good)
    paths = export_results([good, unavailable], tmp_path / "output")
    assert set(paths) == {"runs", "metrics", "predictions", "comparison", "summary"}
    assert all(path.is_file() for path in paths.values())
    runs = pd.read_parquet(paths["runs"]).set_index("run_id")
    assert pd.isna(runs.loc["dummy", "threshold"])
    assert json.loads(runs.loc["good", "params_json"]) == original
    metrics = pd.read_parquet(paths["metrics"])
    assert not metrics["value"].isna().any()
    assert set(metrics.loc[metrics["run_id"] == "dummy", "metric"]) == {"pr_auc", "average_precision"}
    predictions = pd.read_parquet(paths["predictions"])
    assert predictions["run_id"].unique().tolist() == ["good"]
    assert predictions["transaction_id"].tolist() == [1, 3, 6, 7]
    assert set(predictions["split"]) == {"validation"}
    table = pd.read_csv(paths["comparison"])
    assert pd.isna(table.loc[table["run_id"] == "dummy", "precision"]).all()
    summary = json.loads(paths["summary"].read_text())
    assert summary["n_runs"] == 2
    assert summary["processed_sha256"] == original["processed_sha256"]
    assert summary["feasible_run_ids"] == ["good"]
    assert summary["no_feasible_threshold_run_ids"] == ["dummy"]
    assert good == original


def test_export_all_infeasible_writes_empty_predictions_with_schema(tmp_path):
    paths = export_results([make_infeasible()], tmp_path / "output")
    frame = pd.read_parquet(paths["predictions"])
    assert frame.empty
    assert frame.columns.tolist() == ["run_id", "transaction_id", "split", "y_true", "score", "y_pred"]


@pytest.mark.parametrize("problem", ["test", "bad_decision", "nan_score", "bad_label", "duplicate_id", "missing_column"])
def test_bad_score_files_fail_before_export_files_are_written(tmp_path, problem):
    result = make_result("good")
    path = save_scores(tmp_path, result)
    frame = pd.read_parquet(path)
    if problem == "test":
        frame["split"] = "test"
    elif problem == "bad_decision":
        frame.loc[1, "y_pred"] = 0
    elif problem == "nan_score":
        frame.loc[1, "score"] = np.nan
    elif problem == "bad_label":
        frame.loc[1, "y_true"] = 2
    elif problem == "duplicate_id":
        frame.loc[1, "transaction_id"] = frame.loc[0, "transaction_id"]
    else:
        frame = frame.drop(columns=["score"])
    frame.to_parquet(path, index=False)
    result["scores_sha256"] = file_sha256(path)
    directory = tmp_path / "output"
    with pytest.raises(ValueError):
        export_results([result], directory)
    assert not directory.exists()


def test_export_rejects_mismatching_validation_rows_across_runs(tmp_path):
    first, second = make_result("first"), make_result("second")
    save_scores(tmp_path, first)
    save_scores(tmp_path, second, labels=(0, 0, 0, 1))
    with pytest.raises(ValueError, match="disagree"):
        export_results([first, second], tmp_path / "output")


def test_changed_cache_is_rejected_even_when_threshold_classes_stay_the_same(tmp_path):
    result = make_result("good")
    path = save_scores(tmp_path, result)
    frame = pd.read_parquet(path)
    frame.loc[0, "score"] = 0.7  # Below 0.8 as before, but changes the ranking.
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="cache SHA-256 mismatch"):
        export_results([result], tmp_path / "output")


def test_root_relative_scores_path_is_supported(tmp_path, monkeypatch):
    import src.ml.comparison as comparison

    result = make_result("good")
    path = save_scores(tmp_path, result)
    monkeypatch.setattr(comparison, "ROOT_DIR", tmp_path)
    result["scores_file"] = path.name
    assert export_results([result], tmp_path / "output")["predictions"].is_file()


def test_export_is_compatible_with_existing_sqlite_loader(tmp_path):
    from sqlalchemy import text

    from src.data.memory import optimize_dtypes
    from src.data.splitting import build_processed_frame
    from src.db.builder import build_database
    from src.db.engine import get_engine
    from src.db.experiments import load_experiments
    from src.ml.metrics import compute_metrics
    from tests.data_engineering.conftest import make_raw_frame

    raw = make_raw_frame(n=60, n_fraud=20)
    raw["Time"] = np.arange(60) * 600
    raw["Class"] = (np.arange(60) % 3 == 0).astype(int)
    processed = build_processed_frame(optimize_dtypes(raw))
    transactions = tmp_path / "transactions.parquet"
    processed.to_parquet(transactions, index=False)
    db_path = tmp_path / "synthetic.db"
    build_database(transactions, db_path)
    validation = processed[processed["split"] == "validation"]

    result = make_result("good")
    scores = np.where(validation["Class"] == 1, 0.9, 0.1)
    result["validation_metrics"] = compute_metrics(validation["Class"], scores, result["threshold"])
    score_path = tmp_path / "scores.parquet"
    pd.DataFrame({
        "transaction_id": validation["transaction_id"].to_numpy(), "split": "validation",
        "y_true": validation["Class"].to_numpy(), "score": scores,
        "y_pred": (scores >= result["threshold"]).astype("int8"),
    }).to_parquet(score_path, index=False)
    result["scores_file"] = str(score_path)
    result["scores_sha256"] = file_sha256(score_path)
    directory = tmp_path / "exports"
    export_results([result, make_infeasible()], directory)

    engine = get_engine(db_path)
    try:
        report = load_experiments(engine, directory)
        assert report["model_runs"] == 2
        assert report["predictions"] == len(validation)
        with engine.connect() as connection:
            assert connection.execute(text("SELECT threshold FROM model_runs WHERE run_id='dummy'")).scalar_one() is None
            assert connection.execute(text("SELECT COUNT(*) FROM predictions WHERE y_true <> y_pred")).scalar_one() == 0
            assert connection.execute(text("SELECT COUNT(*) FROM run_metrics WHERE value IS NULL")).scalar_one() == 0
    finally:
        engine.dispose()
