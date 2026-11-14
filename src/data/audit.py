"""Data-quality audit of the raw transactions.

Every number in the audit is computed from the file; nothing is assumed.  The result is a
plain dict that is written to ``reports/data_audit.json`` and reused by the report.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    AMOUNT_COLUMN,
    RAW_COLUMNS,
    TARGET_COLUMN,
    TIME_COLUMN,
)
from src.data.memory import memory_mib

logger = logging.getLogger(__name__)

_PLAIN_INTEGER = re.compile(r"^\d+$")


def scan_time_tokens(path: Path, max_examples: int = 5) -> dict:
    """Look at the *text* of the Time column (pandas hides formatting quirks).

    Returns how many tokens are not plain digits (e.g. ``1e+05``) and a few examples.
    """
    tokens = pd.read_csv(path, usecols=[TIME_COLUMN], dtype=str)[TIME_COLUMN]
    odd = tokens[~tokens.str.match(_PLAIN_INTEGER)]
    return {
        "n_tokens": int(len(tokens)),
        "n_not_plain_integer": int(len(odd)),
        "examples": odd.head(max_examples).tolist(),
    }


def iqr_outlier_table(df: pd.DataFrame, columns: list[str] | None = None, k: float = 1.5) -> pd.DataFrame:
    """Tukey fences (Q1 - k*IQR, Q3 + k*IQR) and the number of values outside them."""
    cols = columns or [c for c in df.columns if c != TARGET_COLUMN]
    rows = []
    for col in cols:
        series = df[col].astype("float64")
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - k * iqr, q3 + k * iqr
        n_out = int(((series < lower) | (series > upper)).sum())
        rows.append(
            {
                "column": col,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_fence": lower,
                "upper_fence": upper,
                "n_outliers": n_out,
                "pct_outliers": 100.0 * n_out / len(series),
            }
        )
    return pd.DataFrame(rows)


def _missing_summary(df: pd.DataFrame) -> dict:
    per_column = df.isna().sum()
    return {
        "total_missing": int(per_column.sum()),
        "columns_with_missing": {c: int(n) for c, n in per_column.items() if n > 0},
    }


def _duplicate_summary(df: pd.DataFrame) -> dict:
    dup_later = df.duplicated(keep="first")  # rows equal to an earlier row
    dup_any = df.duplicated(keep=False)  # every member of a duplicate group
    is_fraud = df[TARGET_COLUMN] == 1
    return {
        "n_repeated_rows": int(dup_later.sum()),
        "n_rows_in_duplicate_groups": int(dup_any.sum()),
        "n_duplicate_groups": int(df[dup_any].drop_duplicates().shape[0]),
        "repeated_rows_among_fraud": int((dup_later & is_fraud).sum()),
        "repeated_rows_among_legitimate": int((dup_later & ~is_fraud).sum()),
    }


def _time_summary(df: pd.DataFrame) -> dict:
    time = df[TIME_COLUMN].astype("float64")
    per_timestamp = time.value_counts()
    return {
        "min": float(time.min()),
        "max": float(time.max()),
        "span_hours": float((time.max() - time.min()) / 3600),
        "is_sorted_non_decreasing": bool(time.is_monotonic_increasing),
        "integer_valued": bool(np.array_equal(time.to_numpy(), np.floor(time.to_numpy()))),
        "n_distinct_timestamps": int(per_timestamp.size),
        "max_rows_sharing_one_timestamp": int(per_timestamp.max()),
    }


def _amount_summary(df: pd.DataFrame) -> dict:
    amount = df[AMOUNT_COLUMN].astype("float64")
    by_class = amount.groupby(df[TARGET_COLUMN]).agg(["count", "mean", "median", "max"])
    cents = (amount * 100).round()
    return {
        "min": float(amount.min()),
        "max": float(amount.max()),
        "mean": float(amount.mean()),
        "median": float(amount.median()),
        "n_zero": int((amount == 0).sum()),
        "n_negative": int((amount < 0).sum()),
        "has_at_most_two_decimals": bool(np.abs(cents - amount * 100).max() < 1e-6),
        "by_class": {int(k): {c: float(v) for c, v in row.items()} for k, row in by_class.iterrows()},
    }


def _class_summary(df: pd.DataFrame) -> dict:
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    n_fraud = int(counts.get(1, 0))
    n_legit = int(counts.get(0, 0))
    return {
        "counts": {int(k): int(v) for k, v in counts.items()},
        "n_fraud": n_fraud,
        "n_legitimate": n_legit,
        "fraud_rate_percent": 100.0 * n_fraud / len(df),
        "imbalance_ratio_legit_per_fraud": (n_legit / n_fraud) if n_fraud else None,
    }


def audit_dataframe(df: pd.DataFrame, source_path: Path | None = None) -> dict:
    """Run all checks on the *raw* frame and return the findings as a dict.

    ``source_path`` (optional) enables checks of the file itself: size, SHA-256 and the
    textual form of the Time column.
    """
    missing_columns = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing_columns:
        raise ValueError(f"cannot audit: columns missing {missing_columns}")

    outliers = iqr_outlier_table(df)
    constant = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]

    audit: dict = {
        "overview": {
            "n_rows": int(len(df)),
            "n_columns": int(df.shape[1]),
            "dtypes": df.dtypes.astype(str).value_counts().to_dict(),
            "memory_mib_in_pandas": memory_mib(df),
        },
        "missing_values": _missing_summary(df),
        "duplicates": _duplicate_summary(df),
        "constant_columns": constant,
        "class_balance": _class_summary(df),
        "time": _time_summary(df),
        "amount": _amount_summary(df),
        "outliers_iqr": {
            "rule": "values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]",
            "columns_with_most_outliers": outliers.sort_values("pct_outliers", ascending=False)
            .head(5)[["column", "n_outliers", "pct_outliers"]]
            .to_dict(orient="records"),
            "amount": outliers[outliers["column"] == AMOUNT_COLUMN]
            .iloc[0][["lower_fence", "upper_fence", "n_outliers", "pct_outliers"]]
            .to_dict(),
        },
    }

    if source_path is not None:
        from src.data.download import sha256_of  # local import keeps module import light

        source_path = Path(source_path)
        audit["source_file"] = {
            "path": source_path.name,
            "size_bytes": source_path.stat().st_size,
            "sha256": sha256_of(source_path),
            "time_column_text": scan_time_tokens(source_path),
        }

    audit["findings"] = _findings(audit)
    return audit


def _findings(audit: dict) -> list[str]:
    """Human-readable list of what the numbers above mean for the project."""
    notes: list[str] = []
    if audit["missing_values"]["total_missing"] == 0:
        notes.append("No missing values: imputation is not needed.")
    else:
        notes.append(f"{audit['missing_values']['total_missing']} missing values found.")

    dup = audit["duplicates"]
    if dup["n_repeated_rows"]:
        notes.append(
            f"{dup['n_repeated_rows']} rows repeat an earlier row "
            f"({dup['repeated_rows_among_fraud']} of them fraud); kept, ids stay stable."
        )

    time = audit["time"]
    if time["is_sorted_non_decreasing"]:
        notes.append("Rows are already ordered by Time, so row position is a chronological id.")
    else:
        notes.append("Rows are NOT ordered by Time; sort before assigning transaction_id.")
    if time["max_rows_sharing_one_timestamp"] > 1:
        notes.append(
            f"Up to {time['max_rows_sharing_one_timestamp']} rows share one timestamp: "
            "split cuts must not fall between them."
        )

    cls = audit["class_balance"]
    if cls["imbalance_ratio_legit_per_fraud"] is None:
        notes.append("No fraud rows found: the target is constant.")
    else:
        notes.append(
            f"Fraud is {cls['fraud_rate_percent']:.4f}% of rows "
            f"(about {cls['imbalance_ratio_legit_per_fraud']:.0f} legitimate per fraud): "
            "plain accuracy is meaningless; use PR-AUC / recall at fixed precision."
        )

    token_info = audit.get("source_file", {}).get("time_column_text")
    if token_info and token_info["n_not_plain_integer"]:
        notes.append(
            f"{token_info['n_not_plain_integer']} Time value(s) are written in scientific "
            f"notation {token_info['examples']}, so pandas reads Time as float64; "
            "values are still whole seconds and are cast to int32 after a check."
        )
    return notes
