"""Reading the raw CSV: plain and chunked (memory-friendly) loaders."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from src.config import RAW_COLUMNS, RAW_CSV_PATH
from src.data.memory import optimize_dtypes

logger = logging.getLogger(__name__)

DEFAULT_CHUNKSIZE = 50_000


class DataLoadError(RuntimeError):
    """The raw file is missing, empty or does not have the expected columns."""


def _check_columns(columns: list[str], path: Path) -> None:
    missing = [c for c in RAW_COLUMNS if c not in columns]
    if missing:
        raise DataLoadError(f"{path} is missing columns: {missing}")


def read_raw_csv(path: Path = RAW_CSV_PATH) -> pd.DataFrame:
    """Read the whole CSV with pandas defaults (float64) - the *baseline* loader."""
    path = Path(path)
    try:
        df = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise DataLoadError(
            f"{path} not found. Run `python -m src.data.cli download` first."
        ) from exc
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise DataLoadError(f"{path} is not a valid CSV: {exc}") from exc
    _check_columns(list(df.columns), path)
    return df[RAW_COLUMNS]


def iter_csv_chunks(path: Path = RAW_CSV_PATH, chunksize: int = DEFAULT_CHUNKSIZE) -> Iterator[pd.DataFrame]:
    """Yield the CSV as consecutive DataFrames of at most ``chunksize`` rows."""
    path = Path(path)
    try:
        reader = pd.read_csv(path, chunksize=chunksize)
    except FileNotFoundError as exc:
        raise DataLoadError(
            f"{path} not found. Run `python -m src.data.cli download` first."
        ) from exc
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise DataLoadError(f"{path} is not a valid CSV: {exc}") from exc
    with reader:
        for number, chunk in enumerate(reader, start=1):
            if number == 1:
                _check_columns(list(chunk.columns), path)
            yield chunk[RAW_COLUMNS]


def load_csv_chunked(path: Path = RAW_CSV_PATH, chunksize: int = DEFAULT_CHUNKSIZE) -> pd.DataFrame:
    """Read the CSV chunk by chunk, compressing every chunk before keeping it.

    Peak memory is one float64 chunk plus the already compressed rows, instead of the
    whole file as float64.  The result is identical to ``optimize_dtypes(read_raw_csv())``.
    """
    pieces: list[pd.DataFrame] = []
    total = 0
    for chunk in iter_csv_chunks(path, chunksize):
        pieces.append(optimize_dtypes(chunk))
        total += len(chunk)
    if not pieces:
        raise DataLoadError(f"{path} contains no data rows")
    logger.info("loaded %d rows in %d chunks", total, len(pieces))
    return pd.concat(pieces, ignore_index=True)
