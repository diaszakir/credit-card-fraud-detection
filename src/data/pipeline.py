"""Pipeline steps of the data-engineering module.

Each ``step_*`` function does one job, reads / writes well-known files from
``src.config`` and returns what it produced, so the CLI, the tests and a notebook can all
call the same code.  Order:  download -> audit -> optimise (Parquet) -> split -> EDA -> SQLite.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import (
    AUDIT_JSON_PATH,
    OPTIMIZED_PARQUET_PATH,
    RAW_CSV_PATH,
)
from src.data.audit import audit_dataframe
from src.data.download import download_dataset
from src.data.io_utils import write_json
from src.data.loading import load_csv_chunked, read_raw_csv
from src.data.memory import cast_report, compare_memory, optimize_dtypes
from src.data.storage import save_parquet

logger = logging.getLogger(__name__)


def step_download(force: bool = False) -> Path:
    """Fetch and verify ``data/raw/creditcard.csv``."""
    return download_dataset(RAW_CSV_PATH, force=force)


def step_audit(csv_path: Path = RAW_CSV_PATH, out_path: Path = AUDIT_JSON_PATH) -> dict:
    """Audit the raw file, add the memory before / after report, save JSON."""
    raw = read_raw_csv(csv_path)
    audit = audit_dataframe(raw, source_path=csv_path)
    optimized = optimize_dtypes(raw)
    audit["memory_optimization"] = {
        "total": compare_memory(raw, optimized),
        "columns": cast_report(raw, optimized),
    }
    write_json(audit, out_path)
    return audit


def step_optimize(csv_path: Path = RAW_CSV_PATH, out_path: Path = OPTIMIZED_PARQUET_PATH) -> pd.DataFrame:
    """Chunked CSV load with compact dtypes, saved as Parquet (interim file)."""
    df = load_csv_chunked(csv_path)
    save_parquet(df, out_path)
    return df
