"""Load experiment results written by the ML / Evaluation modules into SQLite.  (PROPOSAL)

Input files in ``reports/experiments/`` (format to be confirmed by Dias and Adam, see
docs/CONTRACTS.md section 6):

    runs.parquet         run_id, model_name, imbalance_strategy, params_json, fit_seconds,
                         model_size_bytes, peak_memory_mb, threshold, created_at
    metrics.parquet      run_id, split, metric, value                (long format)
    predictions.parquet  run_id, transaction_id, y_true, score, y_pred

Nothing is invented here: when a file is absent, it is skipped and reported as missing.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, text

from src.config import EXPERIMENTS_DIR, SPLIT_NAMES
from src.db.builder import SQL_DIR
from src.data.storage import load_parquet

logger = logging.getLogger(__name__)

EXPERIMENTS_SCHEMA_PATH = SQL_DIR / "experiments_schema.sql"
REQUIRED_COLUMNS = {
    "runs": ["run_id", "model_name", "imbalance_strategy"],
    "metrics": ["run_id", "split", "metric", "value"],
    "predictions": ["run_id", "transaction_id", "y_true", "score", "y_pred"],
}
OPTIONAL_RUN_COLUMNS = ["params_json", "fit_seconds", "model_size_bytes", "peak_memory_mb", "threshold", "created_at"]


def create_experiment_tables(engine: Engine, schema_path: Path = EXPERIMENTS_SCHEMA_PATH) -> None:
    """Create the (empty) experiment tables if they do not exist yet."""
    raw = engine.raw_connection()
    try:
        raw.driver_connection.executescript(Path(schema_path).read_text(encoding="utf-8"))
        raw.commit()
    finally:
        raw.close()


def _check_columns(name: str, frame: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS[name] if c not in frame.columns]
    if missing:
        raise ValueError(f"{name}.parquet is missing columns {missing}")


def load_experiments(engine: Engine, experiments_dir: Path = EXPERIMENTS_DIR) -> dict[str, int | str]:
    """Insert runs / metrics / predictions that exist on disk; return row counts per table.

    Existing rows of the same ``run_id`` are replaced, so re-running an experiment is safe.
    Predictions are checked against the labels stored in ``transactions``.
    """
    experiments_dir = Path(experiments_dir)
    create_experiment_tables(engine)
    report: dict[str, int | str] = {}

    files = {name: experiments_dir / f"{name}.parquet" for name in REQUIRED_COLUMNS}
    missing = [name for name, path in files.items() if not path.is_file()]
    if len(missing) == len(files):
        return {"status": "no experiment files found; nothing loaded"}

    frames = {name: load_parquet(path) for name, path in files.items() if path.is_file()}
    if "runs" not in frames:
        raise ValueError("runs.parquet is required: metrics and predictions refer to run_id")
    for name, frame in frames.items():
        _check_columns(name, frame)

    runs = frames["runs"].copy()
    for column in OPTIONAL_RUN_COLUMNS:
        if column not in runs.columns:
            runs[column] = None
    run_ids = runs["run_id"].tolist()

    with engine.begin() as connection:
        for table in ("predictions", "run_metrics", "model_runs"):  # children first
            for run_id in run_ids:
                connection.execute(text(f"DELETE FROM {table} WHERE run_id = :r"), {"r": run_id})
        runs[["run_id", "model_name", "imbalance_strategy", *OPTIONAL_RUN_COLUMNS]].to_sql(
            "model_runs", connection, if_exists="append", index=False)
        report["model_runs"] = len(runs)

        if "metrics" in frames:
            split_ids = dict(connection.execute(text("SELECT split_name, split_id FROM splits")).all())
            metrics = frames["metrics"].copy()
            unknown = set(metrics["split"].astype(str)) - set(SPLIT_NAMES)
            if unknown:
                raise ValueError(f"metrics.parquet has unknown split names {sorted(unknown)}")
            metrics["split_id"] = metrics["split"].astype(str).map(split_ids)
            metrics[["run_id", "split_id", "metric", "value"]].to_sql("run_metrics", connection, if_exists="append", index=False)
            report["run_metrics"] = len(metrics)

        if "predictions" in frames:
            predictions = frames["predictions"]
            labels = pd.read_sql_query(text("SELECT transaction_id, class_id FROM transactions"), connection)
            merged = predictions.merge(labels, on="transaction_id", how="left")
            if merged["class_id"].isna().any():
                raise ValueError("predictions.parquet references transaction_ids that are not in the database")
            if (merged["class_id"] != merged["y_true"]).any():
                raise ValueError("predictions.parquet y_true disagrees with the labels in transactions")
            predictions[["run_id", "transaction_id", "y_true", "score", "y_pred"]].to_sql(
                "predictions", connection, if_exists="append", index=False)
            report["predictions"] = len(predictions)
    if missing:
        report["skipped_missing_files"] = ", ".join(missing)
    return report
