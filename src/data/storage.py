"""Parquet storage helpers."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Small row groups let Parquet skip whole groups when filtering by ``split`` or ``Time``
# (rows are stored in chronological order, so the min/max statistics are tight).
ROW_GROUP_SIZE = 50_000


def save_parquet(
    df: pd.DataFrame,
    path: Path,
    compression: str = "zstd",
    row_group_size: int = ROW_GROUP_SIZE,
) -> Path:
    """Write ``df`` to Parquet (no index) and return the path."""
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(
            path,
            engine="pyarrow",
            compression=compression,
            index=False,
            row_group_size=row_group_size,
        )
    except OSError as exc:
        raise OSError(f"Cannot write Parquet file {path}: {exc}") from exc
    logger.info("wrote %s (%.1f MiB)", path, path.stat().st_size / 1024**2)
    return path


def load_parquet(
    path: Path,
    columns: list[str] | None = None,
    filters: list | None = None,
) -> pd.DataFrame:
    """Read a Parquet file, optionally only some ``columns`` / rows matching ``filters``."""
    path = Path(path)
    try:
        return pd.read_parquet(path, engine="pyarrow", columns=columns, filters=filters)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{path} not found. Build it with `python -m src.data.cli run-all`."
        ) from exc


def file_size_mib(path: Path) -> float:
    """File size in MiB."""
    return Path(path).stat().st_size / 1024**2
