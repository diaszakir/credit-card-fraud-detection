"""Анализ ошибок: исходы TP/FP/FN/TN, сравнение признаков и SQL-запросы к базе Жандарбека.

SQL выполняется на соединении SQLite **только для чтения**: прогнозы кладутся во временную
таблицу ``eval_predictions`` (живёт в памяти соединения), поэтому ``fraud.db`` не меняется.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS, ID_COLUMN, TIME_COLUMN

SQL_DIR = Path(__file__).resolve().parent / "sql"
OUTCOMES = ("TP", "FP", "FN", "TN")


def label_outcomes(table: pd.DataFrame) -> pd.DataFrame:
    """Добавить колонку outcome по y_true и y_pred."""
    y, p = table["y_true"].astype(int), table["y_pred"].astype(int)
    outcome = np.select([(y == 1) & (p == 1), (y == 0) & (p == 1), (y == 1) & (p == 0)], ["TP", "FP", "FN"], "TN")
    return table.assign(outcome=outcome)


def load_split_features(transactions_path: Path, split: str = "validation") -> pd.DataFrame:
    """Признаки одной части данных через API Жандарбека, индекс — transaction_id."""
    from src.data.api import load_transactions

    if not Path(transactions_path).is_file():
        raise FileNotFoundError(f"Нет {transactions_path}. Подготовьте данные: python -m src.data.cli split.")
    frame = load_transactions(split=split, columns=[ID_COLUMN, TIME_COLUMN, *FEATURE_COLUMNS], path=transactions_path)
    return frame.set_index(ID_COLUMN)


def join_features(predictions: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Прогнозы + признаки; каждая строка прогноза обязана найти свою транзакцию."""
    joined = label_outcomes(predictions).merge(features, left_on=ID_COLUMN, right_index=True, how="left")
    if joined[FEATURE_COLUMNS[0]].isna().any():
        raise ValueError("часть transaction_id из прогнозов не найдена в transactions.parquet")
    return joined


def _smd(a: pd.Series, b: pd.Series) -> float:
    """Стандартизированная разность средних (Cohen's d с общей дисперсией)."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else float("nan")


def feature_profile(joined: pd.DataFrame, features=FEATURE_COLUMNS) -> pd.DataFrame:
    """Средние признаков по исходам и стандартизированные разности.

    * ``smd_fn_vs_tp`` — чем пропущенный fraud отличается от пойманного;
    * ``smd_fn_vs_tn`` — насколько пропущенный fraud похож на обычные транзакции (близко к 0 — похож);
    * ``smd_fp_vs_tn`` — чем ложные тревоги отличаются от обычных транзакций.
    Группы меньше двух строк дают NaN, а не число.
    """
    groups = {name: joined[joined["outcome"] == name] for name in OUTCOMES}
    rows = []
    for feature in features:
        row = {"feature": feature}
        for name in OUTCOMES:
            row[f"mean_{name}"] = float(groups[name][feature].mean()) if len(groups[name]) else float("nan")
        row["smd_fn_vs_tp"] = _smd(groups["FN"][feature], groups["TP"][feature])
        row["smd_fn_vs_tn"] = _smd(groups["FN"][feature], groups["TN"][feature])
        row["smd_fp_vs_tn"] = _smd(groups["FP"][feature], groups["TN"][feature])
        rows.append(row)
    profile = pd.DataFrame(rows)
    order = profile["smd_fn_vs_tp"].abs().sort_values(ascending=False, na_position="last").index
    return profile.loc[order].reset_index(drop=True)


def outcome_sizes(joined: pd.DataFrame) -> dict[str, int]:
    counts = joined["outcome"].value_counts()
    return {name: int(counts.get(name, 0)) for name in OUTCOMES}


# --------------------------------------------------------------------------- #
# SQL
# --------------------------------------------------------------------------- #
def list_error_queries() -> list[str]:
    return sorted(path.stem for path in SQL_DIR.glob("e*.sql"))


def describe_error_query(name: str) -> dict:
    """Заголовок запроса в том же формате, что у запросов Жандарбека."""
    from src.db.queries import parse_query_text

    query = parse_query_text((SQL_DIR / f"{name}.sql").read_text(encoding="utf-8"), name)
    return {"name": query.name, "title": query.title, "question": query.question, "uses": ", ".join(query.uses)}


def run_error_sql(db_path: Path, predictions: pd.DataFrame, names=None) -> dict[str, pd.DataFrame]:
    """Выполнить запросы ошибок на fraud.db, открытой только для чтения."""
    db_path = Path(db_path)
    if not db_path.is_file():
        raise FileNotFoundError(f"Нет {db_path}. Постройте базу: python -m src.data.cli build-db.")
    names = list(names) if names is not None else list_error_queries()
    rows = predictions[[ID_COLUMN, "y_true", "y_pred", "score"]].astype(
        {ID_COLUMN: "int64", "y_true": "int64", "y_pred": "int64", "score": "float64"})
    connection = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        connection.execute("CREATE TEMP TABLE eval_predictions ("
                           "transaction_id INTEGER PRIMARY KEY, y_true INTEGER, y_pred INTEGER, score REAL)")
        connection.executemany("INSERT INTO eval_predictions VALUES (?, ?, ?, ?)",
                               rows.itertuples(index=False, name=None))
        results = {}
        for name in names:
            sql = (SQL_DIR / f"{name}.sql").read_text(encoding="utf-8")
            results[name] = pd.read_sql_query(sql, connection)
        return results
    finally:
        connection.close()
