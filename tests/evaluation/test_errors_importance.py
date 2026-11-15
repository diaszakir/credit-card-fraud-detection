"""Анализ ошибок и permutation importance на синтетических данных в формате команды."""

import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.evaluation import errors, importance
from src.evaluation.sources import project_paths


def _outcome_counts(predictions):
    return errors.outcome_sizes(errors.label_outcomes(predictions))


def test_outcomes_cover_every_row(data_project):
    labelled = errors.label_outcomes(data_project["predictions"])
    assert sum(errors.outcome_sizes(labelled).values()) == len(labelled)
    tp = labelled[(labelled["y_true"] == 1) & (labelled["y_pred"] == 1)]
    assert (tp["outcome"] == "TP").all()


def test_feature_profile_finds_the_shifted_feature(data_project):
    paths = project_paths()
    features = errors.load_split_features(paths.transactions)
    joined = errors.join_features(data_project["predictions"], features)
    profile = errors.feature_profile(joined)
    assert set(profile["feature"]) == set(errors.FEATURE_COLUMNS)
    tp_mean = profile.set_index("feature").loc["V1", "mean_TP"]
    tn_mean = profile.set_index("feature").loc["V1", "mean_TN"]
    assert tp_mean > tn_mean + 1  # смещение V1 у fraud видно в средних


def test_small_groups_give_nan_not_numbers():
    joined = pd.DataFrame({"outcome": ["TP", "TP", "FN", "TN", "TN"], "V1": [1.0, 2.0, 3.0, 0.0, 0.5]})
    profile = errors.feature_profile(joined, features=["V1"])
    assert np.isnan(profile.loc[0, "smd_fn_vs_tp"])  # одна строка FN


def test_unknown_ids_are_rejected(data_project):
    features = errors.load_split_features(project_paths().transactions)
    bad = data_project["predictions"].copy()
    bad.loc[0, "transaction_id"] = 10**9
    with pytest.raises(ValueError, match="не найдена"):
        errors.join_features(bad, features)


def test_error_sql_matches_pandas_and_leaves_database_unchanged(data_project):
    db = project_paths().sqlite
    before = db.stat().st_mtime_ns, db.read_bytes()[:4096]
    results = errors.run_error_sql(db, data_project["predictions"])
    assert set(results) == {"e01_errors_by_amount_band", "e02_errors_by_hour", "e03_error_ranking", "e04_label_check"}
    counts = _outcome_counts(data_project["predictions"])
    hours = results["e02_errors_by_hour"]
    assert hours["fn"].sum() == counts["FN"] and hours["fp"].sum() == counts["FP"]
    bands = results["e01_errors_by_amount_band"]
    assert bands["fn"].sum() == counts["FN"] and bands["tp"].sum() == counts["TP"]
    ranking = results["e03_error_ranking"]
    assert len(ranking) == counts["FN"] + counts["FP"]
    check = results["e04_label_check"].iloc[0]
    assert check["missing_in_db"] == 0 and check["label_mismatch"] == 0
    assert (db.stat().st_mtime_ns, db.read_bytes()[:4096]) == before
    with sqlite3.connect(db) as connection:  # временная таблица не попала в файл
        names = {r[0] for r in connection.execute("SELECT name FROM sqlite_master")}
    assert "eval_predictions" not in names


def test_label_check_detects_wrong_labels(data_project):
    flipped = data_project["predictions"].copy()
    flipped.loc[0, "y_true"] = 1 - flipped.loc[0, "y_true"]
    check = errors.run_error_sql(project_paths().sqlite, flipped, ["e04_label_check"])["e04_label_check"].iloc[0]
    assert check["label_mismatch"] == 1


def test_error_queries_declare_required_sql_constructs():
    uses = " ".join(errors.describe_error_query(name)["uses"] for name in errors.list_error_queries())
    for construct in ("JOIN", "GROUP BY", "HAVING", "CTE", "window function", "subquery"):
        assert construct in uses


def test_permutation_importance_is_computed_and_only_real_results_saved(data_project, tmp_path):
    with pytest.raises(ValueError, match="real"):
        importance.compute_selected(data_project["root"], n_repeats=2)
    table, info = importance.compute_selected(data_project["root"], n_repeats=2, allow_non_real=True)
    assert len(table) == 29 and table.iloc[0]["feature"] == "V1"
    assert table["importance_mean"].is_monotonic_decreasing
    assert info["baseline_average_precision"] == pytest.approx(data_project["metadata"]["validation_metrics"]["average_precision"])
    with pytest.raises(ValueError, match="real"):
        importance.save(table, info, tmp_path)
    importance.save(table, dict(info, provenance="real"), tmp_path)
    loaded, saved_info = importance.load_saved(tmp_path)
    assert loaded["feature"].tolist() == table["feature"].tolist() and saved_info["n_repeats"] == 2


def test_importance_needs_two_repeats(data_project):
    with pytest.raises(ValueError, match="2 повтора"):
        importance.importance_table(data_project["pipeline"], pd.DataFrame({"V1": [0.0]}), [0], n_repeats=1)
