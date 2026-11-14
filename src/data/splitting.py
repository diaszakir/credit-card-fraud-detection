"""Stable transaction ids and the chronological train / validation / test split.

Why chronological?  Fraud patterns drift and the model will be used on *future* transactions.
A random split would let the model see the future (and near-duplicate transactions) during
training and would report optimistic metrics.  The split below is deterministic and needs no
random seed: rows are ordered by ``Time`` and cut by position.

Ties: several transactions can share one ``Time`` value (up to 36 in this dataset).  A cut is
therefore moved forward to the next distinct ``Time`` so a timestamp never straddles two splits.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import (
    ID_COLUMN,
    PROCESSED_COLUMNS,
    SPLIT_COLUMN,
    SPLIT_NAMES,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VALIDATION,
    TARGET_COLUMN,
    TIME_COLUMN,
    TRAIN_FRACTION,
    VALIDATION_FRACTION,
)

logger = logging.getLogger(__name__)


def assign_transaction_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Insert ``transaction_id`` = 0-based position in the file as the first column.

    The raw file must already be ordered by ``Time`` (the audit checks this) so the id is also
    the chronological rank.  Raises ``ValueError`` otherwise - sorting silently would change
    the meaning of the id.
    """
    if ID_COLUMN in df.columns:
        raise ValueError(f"{ID_COLUMN} already present")
    if not df[TIME_COLUMN].is_monotonic_increasing:
        raise ValueError("rows are not ordered by Time; refusing to create positional ids")
    out = df.copy()
    out.insert(0, ID_COLUMN, np.arange(len(out), dtype="int32"))
    return out


def find_cut_positions(
    time: np.ndarray,
    train_fraction: float = TRAIN_FRACTION,
    validation_fraction: float = VALIDATION_FRACTION,
) -> tuple[int, int]:
    """Row positions ``(a, b)``: train = [0, a), validation = [a, b), test = [b, n).

    ``time`` must be sorted.  Nominal cuts are ``floor(n * fraction)``; each is then moved forward
    to the first row whose Time is larger than the Time of the row before the cut.
    """
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1):
        raise ValueError("fractions must be in (0, 1)")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be < 1 (test needs rows)")
    time = np.asarray(time)
    if len(time) and np.any(np.diff(time) < 0):
        raise ValueError("time must be sorted in non-decreasing order")

    n = len(time)
    cuts = []
    for fraction in (train_fraction, train_fraction + validation_fraction):
        position = int(n * fraction)
        if 0 < position < n:
            position = int(np.searchsorted(time, time[position - 1], side="right"))
        cuts.append(position)

    first, second = cuts
    if not (0 < first < second < n):
        raise ValueError(
            f"cannot build three non-empty splits (cuts {first}, {second} of {n} rows); "
            "too few rows or too many ties"
        )
    return first, second


def chronological_split(
    df: pd.DataFrame,
    train_fraction: float = TRAIN_FRACTION,
    validation_fraction: float = VALIDATION_FRACTION,
) -> pd.Series:
    """Categorical split label for every row of a Time-ordered frame."""
    first, second = find_cut_positions(df[TIME_COLUMN].to_numpy(), train_fraction, validation_fraction)
    labels = np.empty(len(df), dtype=object)
    labels[:first] = SPLIT_TRAIN
    labels[first:second] = SPLIT_VALIDATION
    labels[second:] = SPLIT_TEST
    return pd.Series(pd.Categorical(labels, categories=list(SPLIT_NAMES), ordered=True), index=df.index, name=SPLIT_COLUMN)


def add_split_column(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Return ``df`` with the ``split`` column inserted right after ``transaction_id``."""
    out = df.copy()
    position = out.columns.get_loc(ID_COLUMN) + 1 if ID_COLUMN in out.columns else 0
    out.insert(position, SPLIT_COLUMN, chronological_split(out, **kwargs))
    return out


def build_processed_frame(raw_optimized: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """ids + split on top of the optimised frame, columns in contract order."""
    framed = add_split_column(assign_transaction_ids(raw_optimized), **kwargs)
    return framed[PROCESSED_COLUMNS]


def split_summary(df: pd.DataFrame) -> dict:
    """Counts, fraud rate and Time / id ranges per split."""
    summary: dict[str, dict] = {}
    n_total = len(df)
    for name in SPLIT_NAMES:
        part = df[df[SPLIT_COLUMN] == name]
        n_fraud = int(part[TARGET_COLUMN].sum())
        summary[name] = {
            "n_rows": int(len(part)),
            "share_of_rows_percent": 100.0 * len(part) / n_total,
            "n_fraud": n_fraud,
            "n_legitimate": int(len(part) - n_fraud),
            "fraud_rate_percent": 100.0 * n_fraud / len(part) if len(part) else None,
            "time_min": int(part[TIME_COLUMN].min()) if len(part) else None,
            "time_max": int(part[TIME_COLUMN].max()) if len(part) else None,
            "id_min": int(part[ID_COLUMN].min()) if len(part) else None,
            "id_max": int(part[ID_COLUMN].max()) if len(part) else None,
        }
    return summary


def validate_split(df: pd.DataFrame) -> dict:
    """Check the split invariants; raise ``ValueError`` listing every violation.

    1. ids are unique and equal to 0..n-1
    2. every row has one of the allowed split labels
    3. Time ranges are strictly ordered (train < validation < test): no timestamp is shared
    4. every split contains both classes (otherwise PR-AUC / recall are undefined)
    """
    problems: list[str] = []
    ids = df[ID_COLUMN].to_numpy()
    if not np.array_equal(ids, np.arange(len(df))):
        problems.append("transaction_id is not exactly 0..n-1 in row order")

    labels = df[SPLIT_COLUMN]
    if labels.isna().any() or not labels.isin(SPLIT_NAMES).all():
        problems.append("split labels outside the allowed names or missing")

    summary = split_summary(df)
    order = list(SPLIT_NAMES)
    for earlier, later in zip(order[:-1], order[1:]):
        a, b = summary[earlier], summary[later]
        if a["n_rows"] and b["n_rows"] and not a["time_max"] < b["time_min"]:
            problems.append(
                f"Time ranges overlap: {earlier} ends at {a['time_max']}, {later} starts at {b['time_min']}"
            )
    for name, info in summary.items():
        if info["n_rows"] == 0:
            problems.append(f"split {name!r} is empty")
        elif info["n_fraud"] == 0 or info["n_legitimate"] == 0:
            problems.append(f"split {name!r} does not contain both classes")

    if problems:
        raise ValueError("invalid split: " + "; ".join(problems))
    return summary
