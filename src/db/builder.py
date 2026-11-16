"""Create the SQLite database from the processed Parquet dataset and verify it.

    processed Parquet  --->  splits / classes / time_slots / amount_bands (dimensions)
                       --->  transactions (facts) + transaction_features (V1..V28)

The Parquet file is the single input, so the database contains exactly the values the ML module
trains on (V-features are the float32 values widened to SQLite's 8-byte REAL).
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, text

from src.config import (
    AMOUNT_BANDS,
    AMOUNT_COLUMN,
    ID_COLUMN,
    SECONDS_PER_HOUR,
    SPLIT_COLUMN,
    SPLIT_NAMES,
    SQLITE_PATH,
    TARGET_COLUMN,
    TIME_COLUMN,
    TRAIN_FRACTION,
    TRANSACTIONS_PARQUET_PATH,
    V_COLUMNS,
    VALIDATION_FRACTION,
)
from src.data.storage import load_parquet
from src.db.engine import get_engine

logger = logging.getLogger(__name__)

SQL_DIR = Path(__file__).resolve().parent / "sql"
SCHEMA_PATH = SQL_DIR / "schema.sql"

CLASS_LABELS = {0: "legitimate", 1: "fraud"}
INSERT_CHUNK = 50_000


def split_descriptions() -> dict[str, str]:
    """Human-readable description of every split, derived from the configured fractions."""
    test_fraction = 1 - TRAIN_FRACTION - VALIDATION_FRACTION
    return {
        "train": f"first {TRAIN_FRACTION:.0%} of rows in time order: fit models here",
        "validation": f"next {VALIDATION_FRACTION:.0%}: tune hyper-parameters and the threshold",
        "test": f"last {test_fraction:.0%}: final independent evaluation only",
    }


def create_schema(engine: Engine, schema_path: Path = SCHEMA_PATH) -> None:
    """Execute ``schema.sql`` (several statements, so the raw sqlite3 connection is used)."""
    script = Path(schema_path).read_text(encoding="utf-8")
    raw = engine.raw_connection()
    try:
        raw.driver_connection.executescript(script)
        raw.commit()
    finally:
        raw.close()


def populate_dimensions(engine: Engine, max_hour: int) -> None:
    """Fill the lookup tables (they are tiny and fully determined by the configuration)."""
    descriptions = split_descriptions()
    splits = pd.DataFrame(
        {
            "split_id": range(1, len(SPLIT_NAMES) + 1),
            "split_name": list(SPLIT_NAMES),
            "description": [descriptions[name] for name in SPLIT_NAMES],
        }
    )
    classes = pd.DataFrame({"class_id": list(CLASS_LABELS), "label": list(CLASS_LABELS.values())})
    hours = pd.DataFrame({"hour_index": range(max_hour + 1)})
    hours["day_index"] = hours["hour_index"] // 24
    hours["hour_in_cycle"] = hours["hour_index"] % 24
    bands = pd.DataFrame(
        [(i, label, low, high) for i, (label, low, high) in enumerate(AMOUNT_BANDS, start=1)],
        columns=["band_id", "label", "lower_bound", "upper_bound"],
    )
    with engine.begin() as conn:
        for name, frame in (("splits", splits), ("classes", classes), ("time_slots", hours), ("amount_bands", bands)):
            frame.to_sql(name, conn, if_exists="append", index=False)


def _fact_frame(processed: pd.DataFrame) -> pd.DataFrame:
    split_ids = {name: i for i, name in enumerate(SPLIT_NAMES, start=1)}
    return pd.DataFrame(
        {
            "transaction_id": processed[ID_COLUMN].astype("int64"),
            "split_id": processed[SPLIT_COLUMN].astype(str).map(split_ids).astype("int64"),
            "class_id": processed[TARGET_COLUMN].astype("int64"),
            "hour_index": processed[TIME_COLUMN].astype("int64") // SECONDS_PER_HOUR,
            "time_sec": processed[TIME_COLUMN].astype("int64"),
            "amount": processed[AMOUNT_COLUMN].astype("float64"),
        }
    )


def _feature_frame(processed: pd.DataFrame) -> pd.DataFrame:
    features = processed[V_COLUMNS].astype("float64")
    features.columns = [c.lower() for c in V_COLUMNS]
    features.insert(0, "transaction_id", processed[ID_COLUMN].astype("int64"))
    return features


def load_facts(engine: Engine, processed: pd.DataFrame, chunksize: int = INSERT_CHUNK) -> None:
    """Insert ``transactions`` first (parent), then ``transaction_features`` (child), in chunks."""
    facts, features = _fact_frame(processed), _feature_frame(processed)
    for name, frame in (("transactions", facts), ("transaction_features", features)):
        for start in range(0, len(frame), chunksize):
            with engine.begin() as conn:  # one transaction per chunk keeps memory and journal small
                frame.iloc[start : start + chunksize].to_sql(name, conn, if_exists="append", index=False)
        logger.info("loaded %s: %d rows", name, len(frame))


def verify_database(engine: Engine, processed: pd.DataFrame) -> dict:
    """Compare the database with the Parquet data; raise ``ValueError`` on any mismatch."""
    checks: dict[str, object] = {}
    problems: list[str] = []
    with engine.connect() as conn:
        for table in ("transactions", "transaction_features"):
            n = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one()
            checks[f"rows_{table}"] = n
            if n != len(processed):
                problems.append(f"{table} has {n} rows, expected {len(processed)}")

        db_sum = conn.execute(text("SELECT SUM(amount) FROM transactions")).scalar_one()
        py_sum = float(processed[AMOUNT_COLUMN].astype("float64").sum())
        checks["amount_sum_db"], checks["amount_sum_parquet"] = db_sum, py_sum
        if not math.isclose(db_sum, py_sum, rel_tol=1e-9):
            problems.append(f"amount sums differ: {db_sum} vs {py_sum}")

        db_fraud = dict(
            conn.execute(
                text(
                    "SELECT s.split_name, SUM(t.class_id) FROM transactions t "
                    "JOIN splits s ON s.split_id = t.split_id GROUP BY s.split_name"
                )
            ).all()
        )
        py_fraud = processed.groupby(processed[SPLIT_COLUMN].astype(str))[TARGET_COLUMN].sum().to_dict()
        checks["fraud_per_split_db"] = db_fraud
        if {k: int(v) for k, v in db_fraud.items()} != {k: int(v) for k, v in py_fraud.items()}:
            problems.append(f"fraud counts per split differ: {db_fraud} vs {py_fraud}")

        fk_violations = conn.execute(text("PRAGMA foreign_key_check")).all()
        checks["foreign_key_violations"] = len(fk_violations)
        if fk_violations:
            problems.append(f"{len(fk_violations)} foreign-key violations")

        integrity = conn.execute(text("PRAGMA integrity_check")).scalar_one()
        checks["integrity_check"] = integrity
        if integrity != "ok":
            problems.append(f"integrity_check: {integrity}")

    if problems:
        raise ValueError("database verification failed: " + "; ".join(problems))
    checks["ok"] = True
    return checks


def build_database(
    processed_path: Path = TRANSACTIONS_PARQUET_PATH,
    db_path: Path = SQLITE_PATH,
    overwrite: bool = True,
) -> dict:
    """Build ``db_path`` from the processed Parquet file and return the verification report.

    The database is written to a temporary file and moved into place only after verification,
    so a failed build never leaves a half-filled ``fraud.db`` behind.
    """
    db_path = Path(db_path)
    if db_path.exists() and not overwrite:
        raise FileExistsError(f"{db_path} exists (use overwrite=True)")
    processed = load_parquet(processed_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = db_path.with_name(db_path.name + ".tmp")
    temp_path.unlink(missing_ok=True)

    engine = get_engine(temp_path, must_exist=False)
    try:
        create_schema(engine)
        max_hour = int(processed[TIME_COLUMN].max()) // SECONDS_PER_HOUR
        populate_dimensions(engine, max_hour)
        load_facts(engine, processed)
        report = verify_database(engine, processed)
    except Exception:
        engine.dispose()
        temp_path.unlink(missing_ok=True)
        raise
    engine.dispose()
    os.replace(temp_path, db_path)
    report["size_mib"] = db_path.stat().st_size / 1024**2
    logger.info("database ready: %s (%.1f MiB)", db_path, report["size_mib"])
    return report
