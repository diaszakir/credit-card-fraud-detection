"""Итоговая оценка test: только замороженная реальная модель, порог не меняется, повтор безопасен."""

import json

import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS
from src.evaluation import final_test
from src.evaluation.sources import project_paths
from src.ml.metrics import compute_metrics
from src.ml.prediction import predict_transactions
from tests.evaluation.conftest import DTYPES


def test_non_real_model_never_reaches_test(data_project):
    with pytest.raises(ValueError, match="real"):
        final_test.preflight(data_project["root"])


def test_cli_without_confirm_does_not_evaluate(data_project, monkeypatch, capsys):
    original = final_test.preflight
    # Фикстура не "real": разрешаем её только для проверки поведения командной строки.
    monkeypatch.setattr(final_test, "preflight", lambda root=None, allow_non_real=False: original(root, True))
    monkeypatch.setattr(final_test, "evaluate_test", lambda *a, **k: pytest.fail("test was read without --confirm"))
    assert final_test.main([]) == 2
    assert "--confirm" in capsys.readouterr().out
    assert final_test.load_result(data_project["root"]) is None


def test_test_metrics_use_saved_threshold_and_match_direct_computation(data_project):
    output = final_test.evaluate_test(data_project["root"], allow_non_real=True)
    result, table = output["result"], output["predictions"]
    processed = data_project["processed"]
    test = processed[processed["split"] == "test"].set_index("transaction_id")
    assert table["transaction_id"].tolist() == test.index.tolist() and set(table["split"]) == {"test"}
    direct = predict_transactions(test[FEATURE_COLUMNS].astype(DTYPES), data_project["pipeline"], data_project["metadata"])
    expected = compute_metrics(test["Class"].to_numpy(), direct["score"].to_numpy(), data_project["metadata"]["threshold"])
    for key in ("average_precision", "pr_auc", "precision", "recall", "tp", "fp", "fn", "tn"):
        assert result[key] == expected[key]
    assert result["threshold"] == data_project["metadata"]["threshold"]
    assert result["threshold_retuned"] is False and result["model_retrained"] is False
    assert result["validation_vs_test"]["tp"]["validation"] == data_project["metadata"]["validation_metrics"]["tp"]


def test_models_folder_is_not_modified(data_project):
    models = data_project["root"] / "models"
    before = {p.name: p.read_bytes() for p in models.iterdir()}
    final_test.evaluate_test(data_project["root"], allow_non_real=True)
    assert {p.name: p.read_bytes() for p in models.iterdir()} == before


def test_rerun_is_identical_but_changed_package_is_refused(data_project):
    root = data_project["root"]
    first = final_test.evaluate_test(root, allow_non_real=True)
    out_dir = project_paths(root).evaluation_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / final_test.RESULT_JSON).write_text(json.dumps(first["result"], default=list))
    second = final_test.evaluate_test(root, allow_non_real=True)
    pd.testing.assert_frame_equal(first["predictions"], second["predictions"])
    stored = json.loads((out_dir / final_test.RESULT_JSON).read_text())
    stored["fingerprint"]["artifact_sha256"] = "0" * 64  # как будто test оценивала другая модель
    (out_dir / final_test.RESULT_JSON).write_text(json.dumps(stored))
    with pytest.raises(ValueError, match="уже оценён"):
        final_test.evaluate_test(root, allow_non_real=True)


def test_saving_refuses_fixture_results(data_project):
    output = final_test.evaluate_test(data_project["root"], allow_non_real=True)
    with pytest.raises(ValueError, match="real"):
        final_test.save(output, data_project["root"])


def test_validation_mismatch_blocks_test(data_project):
    cache = data_project["root"] / data_project["metadata"]["source_paths"]["validation_scores"]
    frame = pd.read_parquet(cache)
    top = frame["score"].idxmax()
    frame.loc[top, "y_true"] = 1 - frame.loc[top, "y_true"]
    frame.to_parquet(cache)
    with pytest.raises(ValueError, match="не совпали"):
        final_test.evaluate_test(data_project["root"], allow_non_real=True)


def test_commands_report_missing_files_without_traceback(empty_project, capsys):
    from src.evaluation import importance, metrics, plots
    from src.evaluation.cli_utils import run_cli

    for module in (metrics, plots, importance, final_test):
        assert run_cli(lambda: module.main([])) == 1
        assert "Ошибка:" in capsys.readouterr().err
