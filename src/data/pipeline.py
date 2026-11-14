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
    ID_COLUMN,
    OPTIMIZED_PARQUET_PATH,
    RAW_CSV_PATH,
    SPLIT_COLUMN,
    SPLIT_MANIFEST_PATH,
    SPLIT_SUMMARY_JSON_PATH,
    TRANSACTIONS_PARQUET_PATH,
)
from src.data.audit import audit_dataframe
from src.data.download import download_dataset
from src.data.io_utils import write_json
from src.data.loading import load_csv_chunked, read_raw_csv
from src.data.memory import cast_report, compare_memory, optimize_dtypes
from src.data.splitting import build_processed_frame, validate_split
from src.data.storage import load_parquet, save_parquet

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


def step_split(
    interim_path: Path = OPTIMIZED_PARQUET_PATH,
    out_path: Path = TRANSACTIONS_PARQUET_PATH,
    manifest_path: Path = SPLIT_MANIFEST_PATH,
    summary_path: Path = SPLIT_SUMMARY_JSON_PATH,
) -> pd.DataFrame:
    """Add ``transaction_id`` + chronological ``split``, validate, write the shared files.

    Writes: the processed dataset, the tiny split manifest (committed to git) and a JSON summary.
    """
    optimized = load_parquet(interim_path)
    processed = build_processed_frame(optimized)
    summary = validate_split(processed)  # raises ValueError if an invariant is broken
    save_parquet(processed, out_path)
    save_parquet(processed[[ID_COLUMN, SPLIT_COLUMN]], manifest_path)
    write_json(summary, summary_path)
    return processed


def step_eda(processed_path: Path = TRANSACTIONS_PARQUET_PATH) -> dict:
    """Generate the EDA tables and figures from the processed dataset."""
    from src.data.eda import run_eda  # imported lazily: pulls in matplotlib / plotly

    return run_eda(load_parquet(processed_path))


def step_build_db() -> dict:
    """Build and verify ``data/processed/fraud.db`` from the processed Parquet file."""
    from src.db.builder import build_database  # imported lazily: keeps the data steps light

    return build_database()


def step_er_diagram() -> dict:
    """Regenerate ``docs/er_diagram.png`` and ``docs/er_diagram.md`` from the database."""
    from src.db.er_diagram import generate_er_diagram

    return generate_er_diagram()
