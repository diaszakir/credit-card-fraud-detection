"""Stable read API for teammates (ML and Evaluation modules).

Everything downstream should obtain data through these functions instead of reading files
directly, so the file layout can change without breaking other modules.

    from src.data.api import load_xy, load_transactions, load_split_manifest
    X_train, y_train = load_xy("train")
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import (
    FEATURE_COLUMNS,
    ID_COLUMN,
    SPLIT_COLUMN,
    SPLIT_MANIFEST_PATH,
    SPLIT_NAMES,
    TARGET_COLUMN,
    TRANSACTIONS_PARQUET_PATH,
)
from src.data.storage import load_parquet


def _check_split(split: str | None) -> None:
    if split is not None and split not in SPLIT_NAMES:
        raise ValueError(f"unknown split {split!r}; choose one of {SPLIT_NAMES}")


def load_transactions(
    split: str | None = None,
    columns: list[str] | None = None,
    path: Path = TRANSACTIONS_PARQUET_PATH,
) -> pd.DataFrame:
    """Processed transactions, optionally only one ``split`` and/or some ``columns``.

    The split filter is pushed down into Parquet (row groups of other splits are skipped).
    """
    _check_split(split)
    filters = [(SPLIT_COLUMN, "==", split)] if split else None
    df = load_parquet(path, columns=columns, filters=filters)
    if SPLIT_COLUMN in df.columns:
        df[SPLIT_COLUMN] = pd.Categorical(df[SPLIT_COLUMN], categories=list(SPLIT_NAMES), ordered=True)
    return df


def load_xy(
    split: str,
    feature_columns: list[str] | None = None,
    path: Path = TRANSACTIONS_PARQUET_PATH,
) -> tuple[pd.DataFrame, pd.Series]:
    """Features ``X`` and target ``y`` of one split, both indexed by ``transaction_id``.

    ``feature_columns`` defaults to ``src.config.FEATURE_COLUMNS`` (V1..V28 + Amount).
    """
    _check_split(split)
    if split is None:
        raise ValueError("split is required for load_xy")
    features = list(FEATURE_COLUMNS if feature_columns is None else feature_columns)
    df = load_transactions(split=split, columns=[ID_COLUMN, *features, TARGET_COLUMN], path=path)
    df = df.set_index(ID_COLUMN)
    return df[features], df[TARGET_COLUMN]


def load_split_manifest(path: Path = SPLIT_MANIFEST_PATH) -> pd.DataFrame:
    """The tiny ``transaction_id -> split`` table shared by the whole team."""
    df = load_parquet(path)
    df[SPLIT_COLUMN] = pd.Categorical(df[SPLIT_COLUMN], categories=list(SPLIT_NAMES), ordered=True)
    return df
