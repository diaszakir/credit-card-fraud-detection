"""Проверка запуска: чистый клон допустим, испорченные файлы и версии — ошибка."""

import json

import pytest

from src.evaluation import check
from src.evaluation.sources import project_paths


def by_name(checks):
    return {c.name: c for c in checks}


def test_installed_environment_matches_requirements():
    assert check.check_python().status == check.OK
    result = check.check_requirements()
    assert result.status == check.OK, result.detail


def test_wrong_pin_and_missing_package_are_failures(tmp_path):
    path = tmp_path / "requirements.txt"
    path.write_text("numpy==0.0.1\n    # via pandas\nnot-a-real-package-xyz==1.0\ncolorama==9.9 ; sys_platform == 'never'\n")
    result = check.check_requirements(path)
    assert result.status == check.FAIL
    assert "numpy" in result.detail and "not-a-real-package-xyz" in result.detail and "colorama" not in result.detail


def test_empty_clone_gives_warnings_but_no_failures(empty_project):
    checks = check.run_checks(with_app=False)
    assert check.exit_code(checks) == 0
    assert check.exit_code(checks, strict=True) == 1
    named = by_name(checks)
    assert named["Загрузка модели"].status == check.WARN
    assert "пропущено" in named["Воспроизведение прогнозов"].detail


def test_prepared_project_passes_including_prediction_reproduction(data_project):
    checks = check.run_checks(with_app=False)
    named = by_name(checks)
    assert named["Загрузка модели"].status == check.OK
    assert named["Воспроизведение прогнозов"].status == check.OK
    assert "совпали" in named["Воспроизведение прогнозов"].detail


def test_damaged_model_package_is_a_failure(data_project):
    path = project_paths().model_metadata
    metadata = json.loads(path.read_text())
    metadata["threshold"] = 0.9
    path.write_text(json.dumps(metadata))
    named = by_name(check.run_checks(with_app=False))
    assert named["Загрузка модели"].status == check.FAIL


def test_changed_saved_scores_are_detected(data_project):
    import pandas as pd

    cache = data_project["root"] / data_project["metadata"]["source_paths"]["validation_scores"]
    frame = pd.read_parquet(cache)
    below = frame[frame["y_pred"] == 0].index[0]
    frame.loc[below, "score"] = frame.loc[below, "score"] / 2  # класс тот же, score другой
    frame.to_parquet(cache)
    named = by_name(check.run_checks(with_app=False))
    assert named["Воспроизведение прогнозов"].status == check.FAIL


@pytest.mark.parametrize("fixture_name", ["empty_project", "data_project"])
def test_all_sections_render(fixture_name, request):
    request.getfixturevalue(fixture_name)
    sections = check.check_app_sections(timeout=60)
    assert [c.name for c in sections] == [f"Раздел {s}" for s in check.SECTIONS]
    assert all(c.status == check.OK for c in sections), [c.detail for c in sections]
