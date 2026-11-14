"""Independent verification: every SQL answer is recomputed in Pandas and compared.

If SQL and Pandas (two independent implementations over the same Parquet data) agree, a wrong
JOIN / window frame / NTILE in a query would be caught here.  Used by the tests and by the CLI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import Engine

from src.config import (
    AMOUNT_BANDS,
    AMOUNT_COLUMN,
    ID_COLUMN,
    SECONDS_PER_HOUR,
    SPLIT_COLUMN,
    SPLIT_NAMES,
    SPLIT_TRAIN,
    TARGET_COLUMN,
    TIME_COLUMN,
    V_COLUMNS,
)
from src.db.queries import run_query


def _hourly(processed: pd.DataFrame) -> pd.DataFrame:
    hour = processed[TIME_COLUMN].astype("int64") // SECONDS_PER_HOUR
    return processed.groupby(hour)[TARGET_COLUMN].agg(n_transactions="size", n_fraud="sum").reset_index(names="hour_index")


def check_split_overview(engine: Engine, processed: pd.DataFrame) -> None:
    sql = run_query(engine, "q01_split_overview").set_index("split_name")
    expected = processed.groupby(processed[SPLIT_COLUMN].astype(str)).agg(
        n_transactions=(TARGET_COLUMN, "size"), n_fraud=(TARGET_COLUMN, "sum"), total_amount=(AMOUNT_COLUMN, "sum")
    )
    for name in SPLIT_NAMES:
        assert sql.loc[name, "n_transactions"] == expected.loc[name, "n_transactions"], f"q01 rows differ for {name}"
        assert sql.loc[name, "n_fraud"] == expected.loc[name, "n_fraud"], f"q01 frauds differ for {name}"
        assert np.isclose(sql.loc[name, "total_amount"], expected.loc[name, "total_amount"], atol=0.01), f"q01 amount differs for {name}"


def check_rolling_rate(engine: Engine, processed: pd.DataFrame) -> None:
    sql = run_query(engine, "q05_rolling_fraud_rate")
    hourly = _hourly(processed)
    assert np.array_equal(sql["n_transactions"], hourly["n_transactions"]), "q05 hourly counts differ"
    assert np.array_equal(sql["n_fraud"], hourly["n_fraud"]), "q05 hourly frauds differ"
    rolling = 100.0 * hourly["n_fraud"].rolling(3, min_periods=1).sum() / hourly["n_transactions"].rolling(3, min_periods=1).sum()
    assert np.allclose(sql["rolling_3h_rate_pct"], rolling, atol=6e-5), "q05 rolling rate differs"


def check_amount_bands(engine: Engine, processed: pd.DataFrame) -> None:
    sql = run_query(engine, "q02_fraud_by_amount_band")
    train = processed[processed[SPLIT_COLUMN].astype(str) == SPLIT_TRAIN]
    edges = [low for _, low, _ in AMOUNT_BANDS] + [np.inf]
    band = pd.cut(train[AMOUNT_COLUMN], bins=edges, right=False, labels=False)
    grouped = train.groupby(band)[TARGET_COLUMN].agg(["size", "sum"])
    for _, row in sql.iterrows():
        position = int(row["band_id"]) - 1
        assert row["n_transactions"] == grouped.loc[position, "size"], f"q02 rows differ for band {position + 1}"
        assert row["n_fraud"] == grouped.loc[position, "sum"], f"q02 frauds differ for band {position + 1}"


def check_amount_deciles(engine: Engine, processed: pd.DataFrame) -> None:
    sql = run_query(engine, "q07_fraud_rate_by_amount_decile")
    train = processed[processed[SPLIT_COLUMN].astype(str) == SPLIT_TRAIN]
    ordered = train.sort_values([AMOUNT_COLUMN, ID_COLUMN])
    # NTILE(10) gives the first (n mod 10) buckets one extra row: exactly numpy.array_split
    buckets = np.array_split(ordered[TARGET_COLUMN].to_numpy(), 10)
    assert [len(b) for b in buckets] == sql["n_transactions"].tolist(), "q07 bucket sizes differ"
    assert [int(b.sum()) for b in buckets] == sql["n_fraud"].tolist(), "q07 fraud per decile differ"


def check_time_between_frauds(engine: Engine, processed: pd.DataFrame) -> None:
    sql = run_query(engine, "q08_time_between_frauds").set_index("split_name")
    fraud = processed[processed[TARGET_COLUMN] == 1]
    for name in SPLIT_NAMES:
        gaps = fraud.loc[fraud[SPLIT_COLUMN].astype(str) == name, TIME_COLUMN].astype("int64").sort_values().diff().dropna()
        assert sql.loc[name, "n_gaps"] == len(gaps), f"q08 gap count differs for {name}"
        assert np.isclose(sql.loc[name, "avg_gap_sec"], gaps.mean(), atol=0.06), f"q08 mean gap differs for {name}"
        assert sql.loc[name, "max_gap_sec"] == gaps.max(), f"q08 max gap differs for {name}"


def check_class_moments(engine: Engine, processed: pd.DataFrame) -> None:
    sql = run_query(engine, "q10_class_feature_moments").set_index("class_label")
    train = processed[processed[SPLIT_COLUMN].astype(str) == SPLIT_TRAIN]
    for cls, label in ((0, "legitimate"), (1, "fraud")):
        part = train[train[TARGET_COLUMN] == cls][V_COLUMNS].astype("float64")
        for column in V_COLUMNS:
            lower = column.lower()
            assert np.isclose(sql.loc[label, f"mean_{lower}"], part[column].mean(), atol=2e-6), f"q10 mean {column} differs"
            assert np.isclose(sql.loc[label, f"var_{lower}"], part[column].var(ddof=0), rtol=1e-6, atol=2e-6), f"q10 var {column} differs"


def check_duplicates(engine: Engine, processed: pd.DataFrame) -> None:
    sql = run_query(engine, "q12_duplicate_transactions")
    raw_columns = processed.drop(columns=[ID_COLUMN, SPLIT_COLUMN])
    repeated = raw_columns.duplicated(keep="first")
    assert sql["n_extra_rows"].sum() == repeated.sum(), "q12 duplicate total differs"
    fraud_extra = sql.loc[sql["class_label"] == "fraud", "n_extra_rows"].sum()
    assert fraud_extra == (repeated & (processed[TARGET_COLUMN] == 1)).sum(), "q12 fraud duplicates differ"


def check_integrity(engine: Engine, processed: pd.DataFrame) -> None:
    sql = run_query(engine, "q11_integrity_anti_joins")
    hard = sql[~sql["check_name"].str.contains("informational")]
    assert (hard["n_violations"] == 0).all(), f"q11 found violations: {hard[hard['n_violations'] > 0].to_dict('records')}"


CHECKS = {
    "q01_split_overview": check_split_overview,
    "q02_fraud_by_amount_band": check_amount_bands,
    "q05_rolling_fraud_rate": check_rolling_rate,
    "q07_fraud_rate_by_amount_decile": check_amount_deciles,
    "q08_time_between_frauds": check_time_between_frauds,
    "q10_class_feature_moments": check_class_moments,
    "q11_integrity_anti_joins": check_integrity,
    "q12_duplicate_transactions": check_duplicates,
}


def crosscheck_with_pandas(engine: Engine, processed: pd.DataFrame) -> dict[str, bool]:
    """Run every SQL-vs-Pandas comparison; raise ``AssertionError`` on the first mismatch."""
    outcome: dict[str, bool] = {}
    for name, check in CHECKS.items():
        check(engine, processed)
        outcome[name] = True
    return outcome
