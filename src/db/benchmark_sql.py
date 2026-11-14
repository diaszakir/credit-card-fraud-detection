"""Benchmark: the same analytical task in Pandas (from Parquet) and in SQL (SQLite).

Requirement 9 asks for a comparison of approaches (e.g. "Pandas and SQL") by time.  Four tasks of
growing weight are solved both ways, and the answers are compared before the times are trusted.

    pandas_end_to_end   read the needed columns from Parquet + compute
    pandas_in_memory    compute only (frame already loaded)
    sql                 run the query on the SQLite file through SQLAlchemy -> DataFrame

Times are the median of ``repeats`` runs inside one process (warm caches), so read them as a
relative comparison on this machine, not as absolute numbers.
"""

from __future__ import annotations

import logging
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import Engine, text

from src.config import (
    AMOUNT_COLUMN,
    BENCHMARK_JSON_PATH,
    ID_COLUMN,
    SECONDS_PER_HOUR,
    SPLIT_COLUMN,
    SPLIT_NAMES,
    TARGET_COLUMN,
    TIME_COLUMN,
    TRANSACTIONS_PARQUET_PATH,
)
from src.data.benchmark import _merge_into_report
from src.data.storage import load_parquet
from src.db.engine import get_engine
from src.db.queries import run_query

logger = logging.getLogger(__name__)

HOURLY_SQL = "SELECT hour_index, COUNT(*) AS n_transactions, SUM(class_id) AS n_fraud FROM transactions GROUP BY hour_index ORDER BY hour_index"


def _timed(function: Callable[[], object], repeats: int) -> tuple[float, object]:
    """Median seconds of ``repeats`` calls and the last result."""
    times, result = [], None
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        times.append(time.perf_counter() - start)
    return statistics.median(times), result


# ---- task implementations ---------------------------------------------------------------------
def _hourly_pandas(df: pd.DataFrame) -> pd.DataFrame:
    hour = df[TIME_COLUMN].astype("int64") // SECONDS_PER_HOUR
    return df.groupby(hour)[TARGET_COLUMN].agg(n_transactions="size", n_fraud="sum").reset_index(names="hour_index")


def _rolling_pandas(df: pd.DataFrame) -> pd.DataFrame:
    hourly = _hourly_pandas(df)
    hourly["rolling_3h_rate_pct"] = 100.0 * hourly["n_fraud"].rolling(3, min_periods=1).sum() / hourly["n_transactions"].rolling(3, min_periods=1).sum()
    return hourly


def _split_pandas(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(df[SPLIT_COLUMN].astype(str))
        .agg(n_transactions=(TARGET_COLUMN, "size"), n_fraud=(TARGET_COLUMN, "sum"), total_amount=(AMOUNT_COLUMN, "sum"))
        .reindex(list(SPLIT_NAMES))
        .reset_index(names="split_name")
    )


def _duplicates_pandas(df: pd.DataFrame) -> int:
    return int(df.drop(columns=[ID_COLUMN, SPLIT_COLUMN]).duplicated().sum())


def _same(a: object, b: object) -> bool:
    if isinstance(a, pd.DataFrame):
        common = [c for c in a.columns if c in b.columns and pd.api.types.is_numeric_dtype(a[c])]
        return all(np.allclose(a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float), atol=0.01, equal_nan=True) for c in common)
    return a == b


def run_sql_vs_pandas(
    engine: Engine | None = None,
    parquet_path: Path = TRANSACTIONS_PARQUET_PATH,
    repeats: int = 5,
) -> dict:
    """Run the four tasks both ways, check the answers agree and record the timings."""
    engine = engine or get_engine()
    in_memory = load_parquet(parquet_path)

    def sql_df(query_name: str) -> Callable[[], pd.DataFrame]:
        return lambda: run_query(engine, query_name)

    def sql_inline() -> pd.DataFrame:
        with engine.connect() as connection:
            return pd.read_sql_query(text(HOURLY_SQL), connection)

    def sql_duplicates() -> int:
        return int(run_query(engine, "q12_duplicate_transactions")["n_extra_rows"].sum())

    tasks = [
        ("hourly counts (GROUP BY)", ["Time", "Class"], _hourly_pandas, sql_inline),
        ("split overview (JOIN + GROUP BY)", [SPLIT_COLUMN, AMOUNT_COLUMN, "Class"], _split_pandas, sql_df("q01_split_overview")),
        ("3-hour rolling rate (CTE + window)", ["Time", "Class"], _rolling_pandas, sql_df("q05_rolling_fraud_rate")),
        ("exact duplicates over 30 columns (GROUP BY ... HAVING)", None, _duplicates_pandas, sql_duplicates),
    ]

    rows = []
    for name, columns, pandas_fn, sql_fn in tasks:
        e2e_seconds, pandas_result = _timed(lambda c=columns, f=pandas_fn: f(load_parquet(parquet_path, columns=c)), repeats)
        mem_seconds, _ = _timed(lambda f=pandas_fn: f(in_memory), repeats)
        sql_seconds, sql_result = _timed(sql_fn, repeats)
        agree = bool(_same(pandas_result, sql_result)) if not isinstance(pandas_result, pd.DataFrame) else bool(
            _same(pandas_result.reset_index(drop=True), sql_result.reset_index(drop=True)))
        rows.append(
            {
                "task": name,
                "pandas_end_to_end_seconds": e2e_seconds,
                "pandas_in_memory_seconds": mem_seconds,
                "sql_seconds": sql_seconds,
                "results_agree": agree,
            }
        )
        logger.info("%-55s pandas e2e %.3fs | in-memory %.3fs | sql %.3fs | agree=%s", name, e2e_seconds, mem_seconds, sql_seconds, agree)

    if not all(r["results_agree"] for r in rows):
        raise AssertionError(f"SQL and Pandas disagree: {[r['task'] for r in rows if not r['results_agree']]}")
    report = {"repeats": repeats, "platform": sys.platform, "tasks": rows}
    _merge_into_report("pandas_vs_sql", report)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_sql_vs_pandas()
