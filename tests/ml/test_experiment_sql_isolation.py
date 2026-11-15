"""Synthetic SQLite checks: validation export must not read unrelated test labels."""

from collections import Counter

import pandas as pd
import pytest
from sqlalchemy import event, text

from src.db.builder import create_schema, populate_dimensions
from src.db.engine import get_engine
from src.db.experiments import load_experiments


@pytest.fixture
def synthetic_database(tmp_path):
    engine = get_engine(tmp_path / "synthetic.db", must_exist=False)
    create_schema(engine)
    populate_dimensions(engine, max_hour=0)
    validation_ids = list(range(100, 601))
    test_ids = [10000, 10001]
    rows = [
        {"id": number, "split": 2, "label": number % 2}
        for number in validation_ids
    ] + [
        {"id": number, "split": 3, "label": number % 2}
        for number in test_ids
    ]
    with engine.begin() as connection:
        connection.execute(text(
            "INSERT INTO transactions (transaction_id, split_id, class_id, hour_index, time_sec, amount) "
            "VALUES (:id, :split, :label, 0, 1, 1.0)"
        ), rows)

    label_queries = []

    @event.listens_for(engine, "before_cursor_execute")
    def record_label_reads(connection, cursor, statement, parameters, context, executemany):
        normalized = " ".join(statement.lower().split())
        if normalized.startswith("select transaction_id, class_id from transactions"):
            label_queries.append((normalized, tuple(parameters)))

    directory = tmp_path / "experiments"
    directory.mkdir()
    runs = pd.DataFrame({
        "run_id": ["synthetic_first", "synthetic_second"],
        "model_name": ["synthetic", "synthetic"],
        "imbalance_strategy": ["none", "none"],
    })
    runs.to_parquet(directory / "runs.parquet", index=False)
    predictions = pd.DataFrame({
        "run_id": ["synthetic_first"] * len(validation_ids) + ["synthetic_second"],
        "transaction_id": validation_ids + [validation_ids[0]],
        "y_true": [number % 2 for number in validation_ids] + [validation_ids[0] % 2],
        "score": [0.5] * (len(validation_ids) + 1),
        "y_pred": [1] * (len(validation_ids) + 1),
    })
    predictions.to_parquet(directory / "predictions.parquet", index=False)
    yield {
        "engine": engine,
        "directory": directory,
        "validation_ids": validation_ids,
        "test_ids": test_ids,
        "label_queries": label_queries,
        "predictions": predictions,
    }
    engine.dispose()


def test_label_reads_are_bound_to_deduplicated_prediction_ids_in_small_chunks(synthetic_database):
    fixture = synthetic_database
    report = load_experiments(fixture["engine"], fixture["directory"])
    assert report["model_runs"] == 2 and report["predictions"] == 502

    queries = fixture["label_queries"]
    assert len(queries) == 2
    looked_up = []
    for sql, parameters in queries:
        assert "where transaction_id in (" in sql
        assert sql.count("?") == len(parameters)
        assert 0 < len(parameters) <= 500
        looked_up.extend(parameters)
    assert Counter(looked_up) == Counter(fixture["validation_ids"])
    assert not set(looked_up) & set(fixture["test_ids"])
    with fixture["engine"].connect() as connection:
        assert connection.execute(text("SELECT COUNT(*) FROM predictions")).scalar_one() == 502


@pytest.mark.parametrize("problem, message", [
    ("unknown_id", "not in the database"),
    ("missing_id", "not in the database"),
    ("wrong_label", "y_true disagrees"),
])
def test_unknown_ids_and_wrong_labels_still_fail_without_publishing_partial_runs(synthetic_database, problem, message):
    fixture = synthetic_database
    predictions = fixture["predictions"].copy()
    if problem == "unknown_id":
        predictions.loc[0, "transaction_id"] = 999999
    elif problem == "missing_id":
        predictions["transaction_id"] = predictions["transaction_id"].astype("Int64")
        predictions.loc[0, "transaction_id"] = pd.NA
    else:
        predictions.loc[0, "y_true"] = 1 - predictions.loc[0, "y_true"]
    predictions.to_parquet(fixture["directory"] / "predictions.parquet", index=False)
    with pytest.raises(ValueError, match=message):
        load_experiments(fixture["engine"], fixture["directory"])
    for sql, parameters in fixture["label_queries"]:
        assert "where transaction_id in (" in sql
        assert len(parameters) <= 500
        assert not set(parameters) & set(fixture["test_ids"])
    with fixture["engine"].connect() as connection:
        assert connection.execute(text("SELECT COUNT(*) FROM model_runs")).scalar_one() == 0


@pytest.mark.parametrize("untyped_empty", [False, True])
def test_empty_predictions_do_not_read_any_transaction_labels(synthetic_database, untyped_empty):
    fixture = synthetic_database
    predictions = fixture["predictions"].iloc[:0]
    if untyped_empty:
        predictions = pd.DataFrame(columns=predictions.columns)
    predictions.to_parquet(fixture["directory"] / "predictions.parquet", index=False)
    report = load_experiments(fixture["engine"], fixture["directory"])
    assert report["predictions"] == 0
    assert fixture["label_queries"] == []
    with fixture["engine"].connect() as connection:
        assert connection.execute(text("SELECT COUNT(*) FROM predictions")).scalar_one() == 0
