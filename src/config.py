"""Shared project configuration and data-contract constants.

Everything that more than one module (or more than one team member) depends on
lives here: file locations, column names, the split definition and the seed.

Changing a value in the "DATA CONTRACT" section changes what the ML and
Evaluation modules receive -> agree on it with the team first (see
docs/CONTRACTS.md).
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
SQL_RESULTS_DIR = REPORTS_DIR / "sql_results"
EXPERIMENTS_DIR = REPORTS_DIR / "experiments"  # filled by ML / Evaluation modules
DOCS_DIR = ROOT_DIR / "docs"

RAW_CSV_PATH = RAW_DIR / "creditcard.csv"
OPTIMIZED_PARQUET_PATH = INTERIM_DIR / "creditcard_optimized.parquet"
TRANSACTIONS_PARQUET_PATH = PROCESSED_DIR / "transactions.parquet"
SPLIT_MANIFEST_PATH = PROCESSED_DIR / "split_manifest.parquet"
SQLITE_PATH = PROCESSED_DIR / "fraud.db"

AUDIT_JSON_PATH = REPORTS_DIR / "data_audit.json"
SPLIT_SUMMARY_JSON_PATH = REPORTS_DIR / "split_summary.json"
BENCHMARK_JSON_PATH = REPORTS_DIR / "benchmark_storage.json"

# --------------------------------------------------------------------------- #
# Data source
# --------------------------------------------------------------------------- #
# Official source (login required): https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Public mirror of the same file (used for reproducible, login-free download).
DATA_URLS: list[str] = [
    "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv",
]
EXPECTED_ROWS = 284_807
EXPECTED_FRAUD = 492

# --------------------------------------------------------------------------- #
# DATA CONTRACT (shared with ML / Evaluation modules)
# --------------------------------------------------------------------------- #
ID_COLUMN = "transaction_id"  # stable id = 0-based row position in the raw CSV
SPLIT_COLUMN = "split"  # one of SPLIT_NAMES
TIME_COLUMN = "Time"  # seconds since the first transaction (0 .. 172 792)
AMOUNT_COLUMN = "Amount"
TARGET_COLUMN = "Class"  # 1 = fraud, 0 = legitimate

V_COLUMNS: list[str] = [f"V{i}" for i in range(1, 29)]  # anonymised PCA components
FEATURE_COLUMNS: list[str] = V_COLUMNS + [AMOUNT_COLUMN]
RAW_COLUMNS: list[str] = [TIME_COLUMN, *V_COLUMNS, AMOUNT_COLUMN, TARGET_COLUMN]
PROCESSED_COLUMNS: list[str] = [ID_COLUMN, SPLIT_COLUMN, *RAW_COLUMNS]

SPLIT_TRAIN, SPLIT_VALIDATION, SPLIT_TEST = "train", "validation", "test"
SPLIT_NAMES: tuple[str, ...] = (SPLIT_TRAIN, SPLIT_VALIDATION, SPLIT_TEST)

# Chronological split, expressed as fractions of the rows ordered by Time.
# The cut is moved forward to the next distinct Time value so that rows sharing
# one timestamp are never separated.  Test = the remaining rows.
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15

RANDOM_SEED = 42  # only used for sampling in plots; the split itself is deterministic

# --------------------------------------------------------------------------- #
# SQL dimensions
# --------------------------------------------------------------------------- #
SECONDS_PER_HOUR = 3600
# (label, lower bound inclusive, upper bound exclusive) in currency units.
# Upper bound None means "no upper bound".
AMOUNT_BANDS: list[tuple[str, float, float | None]] = [
    ("zero", 0.0, 0.01),
    ("micro (<1)", 0.01, 1.0),
    ("small (1-10)", 1.0, 10.0),
    ("medium (10-50)", 10.0, 50.0),
    ("regular (50-100)", 50.0, 100.0),
    ("large (100-500)", 100.0, 500.0),
    ("very large (500-1000)", 500.0, 1000.0),
    ("huge (>=1000)", 1000.0, None),
]
