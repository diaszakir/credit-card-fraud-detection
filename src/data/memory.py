"""Memory optimisation: explicit, verified dtype policy and before / after report.

The raw CSV is read as 30 x float64 + 1 x int64 (~70 MiB).  The policy below shrinks
it without losing information that matters:

* ``Time``   float64 -> int32   (values are whole seconds, max 172 792; checked)
* ``Class``  int64   -> int8    (0 / 1; checked)
* ``V1..V28`` float64 -> float32 (relative error <= ~6e-8; checked against a tolerance)
* ``Amount`` stays float64      (monetary values; SQL aggregates must match the CSV exactly)

Every cast is *verified*: if a column cannot be converted safely the function raises
``ValueError`` instead of silently corrupting data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import AMOUNT_COLUMN, TARGET_COLUMN, TIME_COLUMN, V_COLUMNS

OPTIMIZED_DTYPES: dict[str, str] = {
    TIME_COLUMN: "int32",
    **{col: "float32" for col in V_COLUMNS},
    AMOUNT_COLUMN: "float64",
    TARGET_COLUMN: "int8",
}

# float32 keeps ~7 significant digits (relative error <= 6e-8); 1e-6 is a safe bound.
FLOAT32_RTOL = 1e-6

_MIB = 1024**2


def memory_mib(df: pd.DataFrame) -> float:
    """Deep memory usage of a DataFrame in MiB (index included)."""
    return float(df.memory_usage(index=True, deep=True).sum()) / _MIB


def _check_int_cast(series: pd.Series, dtype: str) -> None:
    """Raise ValueError unless ``series`` is integer-valued and fits into ``dtype``."""
    if series.isna().any():
        raise ValueError(f"column {series.name!r} has missing values; cannot cast to {dtype}")
    values = series.to_numpy(dtype="float64")
    if not np.array_equal(values, np.floor(values)):
        raise ValueError(f"column {series.name!r} has non-integer values; cannot cast to {dtype}")
    info = np.iinfo(dtype)
    if values.min() < info.min or values.max() > info.max:
        raise ValueError(
            f"column {series.name!r} range [{values.min()}, {values.max()}] "
            f"does not fit into {dtype}"
        )


def max_relative_error(original: pd.Series, converted: pd.Series) -> float:
    """Largest |a - b| / |a| over non-zero, finite entries (zeros must stay zero)."""
    a = original.to_numpy(dtype="float64")
    b = converted.to_numpy(dtype="float64")
    finite = np.isfinite(a)
    if not np.array_equal(finite, np.isfinite(b)):
        return float("inf")
    a, b = a[finite], b[finite]
    zero = a == 0
    if np.any(b[zero] != 0):
        return float("inf")
    nonzero = ~zero
    if not nonzero.any():
        return 0.0
    return float((np.abs(a[nonzero] - b[nonzero]) / np.abs(a[nonzero])).max())


def optimize_dtypes(
    df: pd.DataFrame,
    dtypes: dict[str, str] | None = None,
    float_rtol: float = FLOAT32_RTOL,
) -> pd.DataFrame:
    """Return a copy of ``df`` cast to the verified compact dtypes.

    Columns that are not mentioned in ``dtypes`` are left untouched.
    Raises ``ValueError`` if any cast would change the data beyond the tolerance.
    """
    policy = OPTIMIZED_DTYPES if dtypes is None else dtypes
    out = df.copy()
    for column, target in policy.items():
        if column not in out.columns:
            continue
        series = out[column]
        if str(series.dtype) == target:
            continue
        if target.startswith("int"):
            _check_int_cast(series, target)
            out[column] = series.astype(target)
        elif target.startswith("float"):
            converted = series.astype(target)
            if target != "float64":
                error = max_relative_error(series, converted)
                if error > float_rtol:
                    raise ValueError(
                        f"column {column!r}: cast to {target} loses precision "
                        f"(max relative error {error:.3g} > {float_rtol:g})"
                    )
            out[column] = converted
        else:
            out[column] = series.astype(target)
    return out


def cast_report(before: pd.DataFrame, after: pd.DataFrame) -> list[dict]:
    """Per-column summary of what the optimisation did (for the report / audit)."""
    rows: list[dict] = []
    for column in before.columns:
        if column not in after.columns:
            continue
        rel_err = None
        if pd.api.types.is_float_dtype(before[column]) and before[column].dtype != after[column].dtype:
            rel_err = max_relative_error(before[column], after[column])
        rows.append(
            {
                "column": column,
                "dtype_before": str(before[column].dtype),
                "dtype_after": str(after[column].dtype),
                "mib_before": float(before[column].memory_usage(index=False, deep=True)) / _MIB,
                "mib_after": float(after[column].memory_usage(index=False, deep=True)) / _MIB,
                "max_relative_error": rel_err,
                "values_identical": bool(
                    np.array_equal(
                        before[column].to_numpy(dtype="float64"),
                        after[column].to_numpy(dtype="float64"),
                    )
                ),
            }
        )
    return rows


def compare_memory(before: pd.DataFrame, after: pd.DataFrame) -> dict:
    """Total memory before / after optimisation."""
    b, a = memory_mib(before), memory_mib(after)
    return {
        "mib_before": b,
        "mib_after": a,
        "mib_saved": b - a,
        "saved_percent": 100.0 * (b - a) / b if b else 0.0,
    }
