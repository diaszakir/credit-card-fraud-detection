"""README: команды существуют, а числа результатов совпадают с сохранёнными файлами оценки."""

import importlib
import json
import re

import pytest

from src.config import ROOT_DIR

README = (ROOT_DIR / "README.md").read_text(encoding="utf-8")
EVALUATION_DIR = ROOT_DIR / "reports" / "experiments" / "evaluation"
ROUNDED = ("average_precision", "pr_auc", "roc_auc", "precision", "recall", "f1")


def readme_mismatches(text: str, document: dict) -> list[str]:
    """Метрики документа, которых нет в README в виде числа с 4 знаками (счётчики — как есть)."""
    problems = []
    for key in ROUNDED:
        if f"{document[key]:.4f}" not in text:
            problems.append(f"{key}={document[key]:.4f}")
    for key in ("tp", "fp", "fn"):
        if not re.search(rf"\b{document[key]}\b", text):
            problems.append(f"{key}={document[key]}")
    for key in ("precision_ci", "recall_ci"):
        low, high = document[key]
        if f"{low:.3f}–{high:.3f}" not in text:
            problems.append(f"{key}={low:.3f}–{high:.3f}")
    return problems


def test_every_module_command_in_readme_exists():
    modules = set(re.findall(r"PY -m (src\.[a-z_.]+)", README))
    assert {"src.evaluation.metrics", "src.evaluation.final_test", "src.evaluation.check"} <= modules
    for name in modules:
        module = importlib.import_module(name)  # опечатка в имени модуля даст ImportError
        if name.startswith("src.evaluation."):
            assert callable(getattr(module, "main", None)), name
    assert (ROOT_DIR / "app" / "fraud_app.py").is_file() and "app/fraud_app.py" in README


def test_helper_reports_a_wrong_number():
    document = {"average_precision": 0.5, "pr_auc": 0.5, "roc_auc": 0.5, "precision": 0.5, "recall": 0.5,
                "f1": 0.5, "tp": 1, "fp": 2, "fn": 3, "precision_ci": [0.1, 0.2], "recall_ci": [0.3, 0.4]}
    text = "0.5000 1 2 3 0.100–0.200 0.300–0.400"
    assert readme_mismatches(text, document) == []
    assert readme_mismatches(text, dict(document, recall=0.75)) == ["recall=0.7500"]


@pytest.mark.parametrize("filename", ["validation_selected.json", "test_final.json"])
def test_readme_results_match_saved_evaluation(filename):
    path = EVALUATION_DIR / filename
    if not path.is_file():
        pytest.skip(f"{filename} создаётся командами оценки на компьютере с моделью")
    assert readme_mismatches(README, json.loads(path.read_text(encoding="utf-8"))) == []
