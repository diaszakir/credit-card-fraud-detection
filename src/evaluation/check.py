"""Проверка запуска: окружение, файлы команды, модель и четыре раздела приложения.

    python -m src.evaluation.check            # приложение должно открываться и на чистом клоне
    python -m src.evaluation.check --strict   # перед защитой: нужны данные, база, модель и прогнозы

Статусы: OK — в порядке; WARN — раздел покажет подсказку вместо данных; FAIL — нужно чинить.
Код возврата 1, если есть FAIL (в режиме --strict предупреждения тоже считаются ошибками).
"""

from __future__ import annotations

import argparse
import platform
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from src.config import ROOT_DIR

OK, WARN, FAIL = "OK", "WARN", "FAIL"
REQUIRED_PYTHON = (3, 12)
APP_PATH = ROOT_DIR / "app" / "fraud_app.py"
SECTIONS = ("Data Overview", "Model Comparison", "Error Analysis", "Fraud Prediction")


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str


def check_python() -> Check:
    current = sys.version_info[:2]
    if current == REQUIRED_PYTHON:
        return Check("Python", OK, platform.python_version())
    return Check("Python", FAIL, f"нужен Python 3.12, запущен {platform.python_version()}; "
                                 "создайте окружение: uv venv -p 3.12 .venv")


def check_requirements(path: Path | None = None) -> Check:
    """Установленные версии должны совпадать с requirements.txt."""
    from packaging.requirements import InvalidRequirement, Requirement

    path = Path(path) if path is not None else ROOT_DIR / "requirements.txt"
    if not path.is_file():
        return Check("Зависимости", FAIL, f"нет файла {path.name}")
    problems, checked = [], 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            requirement = Requirement(line)
        except InvalidRequirement:
            problems.append(f"не разобрана строка: {line}")
            continue
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue  # пакет не нужен на этой платформе
        checked += 1
        try:
            installed = version(requirement.name)
        except PackageNotFoundError:
            problems.append(f"{requirement.name} не установлен")
            continue
        if not requirement.specifier.contains(installed, prereleases=True):
            problems.append(f"{requirement.name}: установлена {installed}, нужна {requirement.specifier}")
    if problems:
        shown = "; ".join(problems[:5]) + (f" и ещё {len(problems) - 5}" if len(problems) > 5 else "")
        return Check("Зависимости", FAIL, f"{shown}. Исправить: uv pip install -r requirements.txt")
    return Check("Зависимости", OK, f"{checked} пакетов совпадают с requirements.txt")


def check_artifacts(paths) -> list[Check]:
    from src.evaluation.sources import availability

    checks = []
    for row in availability(paths).to_dict("records"):
        if row["Есть"]:
            checks.append(Check(row["Артефакт"], OK, row["Путь"]))
        else:
            checks.append(Check(row["Артефакт"], WARN, f"нет {row['Путь']}; создать: {row['Команда']}"))
    return checks


def check_model(paths):
    """Загрузка замороженной модели с проверкой SHA и версий. Возвращает (Check, predictor | None)."""
    from src.evaluation.contract import load_predictor

    status = load_predictor(paths.model, paths.selection_lock, allow_stub=False)
    if status.mode == "frozen":
        return Check("Загрузка модели", OK, f"{status.predictor.info.run_id}, порог {status.predictor.info.threshold}"), status.predictor
    level = WARN if not paths.model.is_file() else FAIL  # файла нет — подсказка; файл испорчен — ошибка
    return Check("Загрузка модели", level, status.message), None


def check_prediction(paths, predictor, rows: int = 2000) -> Check:
    """Модель на первых строках validation должна воспроизвести сохранённые score."""
    from src.evaluation import sources

    name = "Воспроизведение прогнозов"
    if predictor is None:
        return Check(name, WARN, "пропущено: модель не загружена")
    if not paths.transactions.is_file():
        return Check(name, WARN, "пропущено: нет transactions.parquet")
    saved = sources.load_validation_predictions(paths, predictor.metadata)
    if not saved.available:
        return Check(name, WARN, f"пропущено: {saved.message}")
    from src.data.api import load_xy

    X, _ = load_xy("validation", path=paths.transactions)
    X = X.iloc[:rows]
    try:
        predicted = predictor.predict(X)
    except ValueError as exc:
        return Check(name, FAIL, f"прогноз не выполнен: {exc}")
    expected = saved.data.set_index("transaction_id").loc[X.index]
    if not np.array_equal(predicted["score"].to_numpy(), expected["score"].to_numpy()):
        return Check(name, FAIL, "score не совпали с сохранёнными validation-прогнозами")
    if not np.array_equal(predicted["y_pred"].to_numpy(), expected["y_pred"].to_numpy()):
        return Check(name, FAIL, "классы не совпали с сохранёнными validation-прогнозами")
    return Check(name, OK, f"{len(X)} строк validation: score и классы совпали с сохранёнными")


def _render_section(section: str, timeout: int) -> Check:
    from streamlit.testing.v1 import AppTest

    try:
        app = AppTest.from_file(str(APP_PATH), default_timeout=timeout).run()
        app.sidebar.radio(key="section").set_value(section).run()
    except Exception as exc:  # любая ошибка запуска — это результат проверки
        return Check(f"Раздел {section}", FAIL, f"не запустился: {exc}")
    if app.exception:
        return Check(f"Раздел {section}", FAIL, f"исключение: {app.exception[0].value}")
    if app.error:
        return Check(f"Раздел {section}", FAIL, f"ошибка на странице: {app.error[0].value}")
    hints = len(app.warning) + len(app.info)
    return Check(f"Раздел {section}", OK, "отрисован" + (f", подсказок: {hints}" if hints else ""))


def check_app_sections(timeout: int = 120) -> list[Check]:
    """Каждый раздел должен отрисоваться без исключений и без красных ошибок."""
    import logging

    try:
        import streamlit  # noqa: F401
    except ImportError as exc:
        return [Check("Приложение", FAIL, f"streamlit не установлен: {exc}")]
    # Вне «streamlit run» библиотека пишет безвредное «missing ScriptRunContext»; на время проверки
    # предупреждения журнала отключены, результат каждого раздела берётся из самой страницы.
    logging.disable(logging.WARNING)
    try:
        return [_render_section(section, timeout) for section in SECTIONS]
    finally:
        logging.disable(logging.NOTSET)


def run_checks(root=None, with_app: bool = True) -> list[Check]:
    from src.evaluation.sources import project_paths

    paths = project_paths(root)
    checks = [check_python(), check_requirements(), *check_artifacts(paths)]
    model_check, predictor = check_model(paths)
    checks += [model_check, check_prediction(paths, predictor)]
    if with_app:
        checks += check_app_sections()
    return checks


def exit_code(checks: list[Check], strict: bool = False) -> int:
    bad = {FAIL, WARN} if strict else {FAIL}
    return 1 if any(check.status in bad for check in checks) else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Проверка запуска проекта и приложения")
    parser.add_argument("--strict", action="store_true", help="предупреждения считаются ошибками (перед защитой)")
    parser.add_argument("--no-app", action="store_true", help="не запускать разделы Streamlit")
    args = parser.parse_args(argv)
    checks = run_checks(with_app=not args.no_app)
    width = max(len(check.name) for check in checks)
    for check in checks:
        print(f"[{check.status:<4}] {check.name:<{width}}  {check.detail}")
    code = exit_code(checks, strict=args.strict)
    counts = {status: sum(check.status == status for check in checks) for status in (OK, WARN, FAIL)}
    print(f"\nИтог: OK {counts[OK]}, WARN {counts[WARN]}, FAIL {counts[FAIL]}. "
          + ("Готово к запуску: streamlit run app/fraud_app.py" if code == 0 else "Есть проблемы, см. строки выше."))
    return code


if __name__ == "__main__":
    from src.evaluation.cli_utils import run_cli

    raise SystemExit(run_cli(main))
