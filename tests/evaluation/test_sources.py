"""Слой чтения: отсутствие файлов не роняет код, подделка метаданных обнаруживается."""

import json

import pytest

from src.config import ROOT_DIR
from src.evaluation import sources


LOADERS = [
    sources.load_split_summary, sources.load_data_audit, sources.list_sql_results,
    sources.load_comparison, sources.load_runs, sources.load_metrics_long, sources.load_model_metadata,
]


@pytest.mark.parametrize("loader", LOADERS)
def test_every_loader_explains_missing_files(empty_project, loader):
    result = loader(sources.project_paths())
    assert not result.available and result.data is None
    assert result.message


def test_missing_database_and_predictions_are_explained(empty_project):
    paths = sources.project_paths()
    assert "build-db" in sources.run_live_query(paths, "q01_split_overview").message
    assert "Git" in sources.load_validation_predictions(paths, {"run_id": "x"}).message
    table = sources.availability(paths)
    assert len(table) == 8 and not table["Есть"].any()


def test_fixture_project_is_read_through_one_interface(project):
    paths = sources.project_paths()
    assert paths.root == project["root"].resolve()
    assert len(sources.load_comparison(paths).data) == 2
    assert set(sources.load_split_summary(paths).data["split"]) == {"train", "validation"}
    assert list(sources.list_sql_results(paths).data) == ["q01_split_overview"]
    metadata = sources.load_model_metadata(paths)
    assert metadata.available and metadata.data["run_id"] == "synthetic_selected"


def test_validation_predictions_come_from_selected_cache(project):
    paths = sources.project_paths()
    metadata = sources.load_model_metadata(paths).data
    result = sources.load_validation_predictions(paths, metadata)
    assert result.available
    assert result.data["transaction_id"].tolist() == [201, 202, 203, 204]
    assert set(result.data["run_id"]) == {"synthetic_selected"}


def test_predictions_with_wrong_threshold_are_refused(project):
    paths = sources.project_paths()
    metadata = dict(sources.load_model_metadata(paths).data, threshold=0.999999)
    result = sources.load_validation_predictions(paths, metadata)
    assert not result.available and "порога" in result.message


def test_metadata_change_after_freeze_is_detected(project):
    path = project["model_path"].with_suffix(".json")
    data = json.loads(path.read_text())
    data["validation_metrics"]["recall"] = 1.0
    path.write_text(json.dumps(data))
    result = sources.load_model_metadata(sources.project_paths())
    assert not result.available and "selection_lock" in result.message


def test_comparison_without_required_columns_is_refused(project):
    path = project["root"] / "reports" / "experiments" / "comparison.csv"
    path.write_text("run_id,model_name\nx,dummy\n")
    result = sources.load_comparison(sources.project_paths())
    assert not result.available and "колонок" in result.message


def test_committed_team_results_are_readable():
    """Реальные файлы из Git (без данных и joblib) читаются текущей версией модуля."""
    paths = sources.project_paths(ROOT_DIR)
    split = sources.load_split_summary(paths)
    assert split.available and set(split.data["split"]) == {"train", "validation", "test"}
    comparison = sources.load_comparison(paths)
    metadata = sources.load_model_metadata(paths)
    assert comparison.available and metadata.available
    assert metadata.data["run_id"] in set(comparison.data["run_id"])
