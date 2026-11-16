-- =============================================================================
-- PROPOSAL (needs a yes from Dias and Adam, see docs/CONTRACTS.md section 6):
-- tables that hold experiment results next to the transactions, so false positives /
-- false negatives can be joined back to transactions and features in SQL.
--
-- These tables are created EMPTY. They are filled only by src/db/experiments.py from the
-- real Parquet files that the ML / Evaluation modules write to reports/experiments/.
-- =============================================================================
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS model_runs (
    run_id             TEXT PRIMARY KEY,
    model_name         TEXT NOT NULL,          -- e.g. logistic_regression, random_forest, dummy
    imbalance_strategy TEXT NOT NULL,          -- e.g. none, class_weight, smote, undersampling
    params_json        TEXT,
    fit_seconds        REAL,
    model_size_bytes   INTEGER,
    peak_memory_mb     REAL,
    threshold          REAL,                   -- chosen on validation only
    created_at         TEXT
);

CREATE TABLE IF NOT EXISTS run_metrics (
    run_id   TEXT    NOT NULL REFERENCES model_runs (run_id) ON DELETE CASCADE,
    split_id INTEGER NOT NULL REFERENCES splits (split_id),
    metric   TEXT    NOT NULL,                 -- e.g. pr_auc, average_precision, recall_at_precision_90
    value    REAL    NOT NULL,
    PRIMARY KEY (run_id, split_id, metric)
);

CREATE TABLE IF NOT EXISTS predictions (
    run_id         TEXT    NOT NULL REFERENCES model_runs (run_id) ON DELETE CASCADE,
    transaction_id INTEGER NOT NULL REFERENCES transactions (transaction_id),
    y_true         INTEGER NOT NULL CHECK (y_true IN (0, 1)),
    score          REAL    NOT NULL,
    y_pred         INTEGER NOT NULL CHECK (y_pred IN (0, 1)),
    PRIMARY KEY (run_id, transaction_id)
);

CREATE INDEX IF NOT EXISTS idx_predictions_transaction ON predictions (transaction_id);
