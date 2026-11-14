import numpy as np
import pandas as pd
import pytest

from src.data.storage import save_parquet
from src.db.builder import build_database
from src.db.crosscheck import crosscheck_with_pandas
from src.db.engine import get_engine
from src.db.experiments import create_experiment_tables, load_experiments
from src.db.queries import (
    REQUIRED_CONSTRUCTS,
    construct_coverage,
    describe_queries,
    detected_constructs,
    feature_separation,
    list_queries,
    load_query,
    parse_query_text,
    run_all,
    run_query,
    save_results,
)


@pytest.fixture
def engine(tmp_path, processed_frame):
    parquet = save_parquet(processed_frame, tmp_path / "transactions.parquet")
    build_database(parquet, tmp_path / "fraud.db")
    return get_engine(tmp_path / "fraud.db")


# ----------------------------------------------------------------------------- requirements
def test_at_least_eight_queries_and_every_required_construct_is_used():
    assert len(list_queries()) >= 8
    coverage = construct_coverage()
    for construct in REQUIRED_CONSTRUCTS:
        assert coverage[construct], f"no query uses {construct}"
    # the course asks for these together; make sure several queries combine them
    assert len(coverage["JOIN"]) >= 8 and len(coverage["window function (OVER)"]) >= 3


def test_declared_constructs_match_the_sql_text():
    catalogue = describe_queries()
    assert catalogue["title"].str.len().min() > 0 and catalogue["question"].str.len().min() > 0
    for name in list_queries():
        query = load_query(name)
        detected = set(detected_constructs(query.sql))
        declared = " ".join(query.uses).upper()
        for needle, construct in (("HAVING", "HAVING"), ("WINDOW", "window function (OVER)"), ("CTE", "CTE (WITH)")):
            if needle in declared:
                assert construct in detected, f"{name} declares {needle} but the SQL does not use it"


def test_header_parser_reads_metadata_and_defaults():
    query = parse_query_text(
        "-- name: qx\n-- title: T\n-- uses: JOIN, GROUP BY\n-- param: n=5\n-- param: ratio=0.5\n-- param: who=train\nSELECT :n;\n"
    )
    assert (query.name, query.title, query.uses) == ("qx", "T", ("JOIN", "GROUP BY"))
    assert query.params == {"n": 5, "ratio": 0.5, "who": "train"} and query.sql.strip() == "SELECT :n;"


# ----------------------------------------------------------------------------- execution
def test_every_query_runs_and_returns_a_dataframe(engine):
    results = run_all(engine)
    assert list(results) == list_queries()
    assert all(isinstance(frame, pd.DataFrame) for frame in results.values())
    assert len(results["q01_split_overview"]) == 3
    assert len(results["q05_rolling_fraud_rate"]) == results["q05_rolling_fraud_rate"]["hour_index"].nunique()
    assert (results["q11_integrity_anti_joins"].query("not check_name.str.contains('informational')")["n_violations"] == 0).all()


def test_sql_matches_independent_pandas_computation(engine, processed_frame):
    outcome = crosscheck_with_pandas(engine, processed_frame)
    assert len(outcome) >= 8 and all(outcome.values())


def test_parameters_override_defaults_and_errors_are_clear(engine):
    default = run_query(engine, "q09_top_fraud_amounts_per_split")
    one = run_query(engine, "q09_top_fraud_amounts_per_split", params={"top_n": 1})
    assert len(one) == 3 and len(default) == 9
    with_threshold = run_query(engine, "q03_hourly_fraud_hotspots", params={"min_transactions": 1, "top_n": 2})
    assert len(with_threshold) == 2
    with pytest.raises(KeyError, match="unknown query"):
        run_query(engine, "q99_nope")


def test_having_filters_out_low_traffic_hours(engine):
    everything = run_query(engine, "q03_hourly_fraud_hotspots", params={"min_transactions": 1, "top_n": 100})
    strict = run_query(engine, "q03_hourly_fraud_hotspots", params={"min_transactions": 10**9, "top_n": 100})
    assert len(everything) > 0 and strict.empty


def test_save_results_writes_csv(engine, tmp_path):
    paths = save_results(run_all(engine), tmp_path / "out")
    assert len(paths) == len(list_queries())
    assert pd.read_csv(paths[0]).shape[0] == 3


def test_feature_separation_ranks_by_standardised_difference(engine):
    moments = run_query(engine, "q10_class_feature_moments")
    ranked = feature_separation(moments)
    assert len(ranked) == 28 and ranked["abs_standardised_difference"].is_monotonic_decreasing
    with pytest.raises(ValueError, match="both"):
        feature_separation(moments.iloc[:1])


# ----------------------------------------------------------------------------- experiments (synthetic)
def _synthetic_experiment_files(directory, processed_frame):
    """SYNTHETIC predictions (random scores) used only to test the loader, never to report results."""
    rng = np.random.default_rng(0)
    test = processed_frame[processed_frame["split"].astype(str) == "test"]
    pd.DataFrame({"run_id": ["r1"], "model_name": ["dummy"], "imbalance_strategy": ["none"]}).to_parquet(directory / "runs.parquet")
    pd.DataFrame({"run_id": ["r1"] * 2, "split": ["validation", "test"], "metric": ["pr_auc"] * 2, "value": [0.1, 0.2]}).to_parquet(directory / "metrics.parquet")
    score = rng.random(len(test))
    pd.DataFrame({"run_id": "r1", "transaction_id": test["transaction_id"].to_numpy(), "y_true": test["Class"].to_numpy(),
                  "score": score, "y_pred": (score > 0.5).astype(int)}).to_parquet(directory / "predictions.parquet")
    return test


def test_experiment_loader_is_empty_by_default_and_validates_input(engine, processed_frame, tmp_path):
    folder = tmp_path / "experiments"
    folder.mkdir()
    assert "nothing loaded" in load_experiments(engine, folder)["status"]
    create_experiment_tables(engine)  # idempotent

    test = _synthetic_experiment_files(folder, processed_frame)
    report = load_experiments(engine, folder)
    assert report == {"model_runs": 1, "run_metrics": 2, "predictions": len(test)}
    assert load_experiments(engine, folder)["predictions"] == len(test)  # re-running replaces, no duplicates
    joined = pd.read_sql_query(
        "SELECT COUNT(*) AS n FROM predictions p JOIN transactions t USING (transaction_id) WHERE p.y_true <> t.class_id", engine)
    assert joined.loc[0, "n"] == 0

    bad = pd.read_parquet(folder / "predictions.parquet")
    bad.loc[bad.index[0], "y_true"] = 1 - bad.loc[bad.index[0], "y_true"]
    bad.to_parquet(folder / "predictions.parquet")
    with pytest.raises(ValueError, match="disagrees"):
        load_experiments(engine, folder)
