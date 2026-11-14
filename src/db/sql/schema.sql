-- =============================================================================
-- Credit Card Fraud Detection: normalised SQLite schema (star-like, 3NF)
--
--   splits, classes, time_slots, amount_bands   small lookup / dimension tables
--   transactions                                one row per transaction (narrow fact table)
--   transaction_features                        the 28 anonymised PCA features, 1:1 with transactions
--
-- Why two tables for one CSV row?
--   * The 28 V-features are only needed for model-related queries. Keeping them in a separate
--     1:1 table makes analytical scans of `transactions` (counts, amounts, time) much narrower.
--   * Repeated text (split name, class label) is stored once and referenced by a small integer key.
--
-- transaction_id is the stable id defined in docs/CONTRACTS.md (0-based row position in the raw CSV).
-- =============================================================================

PRAGMA foreign_keys = ON;

-- train / validation / test (chronological split; see src/data/splitting.py)
CREATE TABLE splits (
    split_id    INTEGER PRIMARY KEY,
    split_name  TEXT    NOT NULL UNIQUE
                CHECK (split_name IN ('train', 'validation', 'test')),
    description TEXT    NOT NULL
);

-- target values: 0 = legitimate, 1 = fraud
CREATE TABLE classes (
    class_id INTEGER PRIMARY KEY CHECK (class_id IN (0, 1)),
    label    TEXT    NOT NULL UNIQUE
);

-- one row per hour since the first transaction (the data has no absolute clock time)
CREATE TABLE time_slots (
    hour_index    INTEGER PRIMARY KEY CHECK (hour_index >= 0),      -- Time / 3600
    day_index     INTEGER NOT NULL CHECK (day_index >= 0),          -- hour_index / 24
    hour_in_cycle INTEGER NOT NULL CHECK (hour_in_cycle BETWEEN 0 AND 23)  -- hour_index % 24
);

-- business bands of the amount; joined to transactions by RANGE (lower_bound <= amount < upper_bound)
CREATE TABLE amount_bands (
    band_id     INTEGER PRIMARY KEY,
    label       TEXT NOT NULL UNIQUE,
    lower_bound REAL NOT NULL CHECK (lower_bound >= 0),   -- inclusive
    upper_bound REAL,                                     -- exclusive; NULL = no upper bound
    CHECK (upper_bound IS NULL OR upper_bound > lower_bound)
);

CREATE TABLE transactions (
    transaction_id INTEGER PRIMARY KEY,
    split_id       INTEGER NOT NULL REFERENCES splits (split_id),
    class_id       INTEGER NOT NULL REFERENCES classes (class_id),
    hour_index     INTEGER NOT NULL REFERENCES time_slots (hour_index),
    time_sec       INTEGER NOT NULL CHECK (time_sec >= 0),   -- seconds since the first transaction
    amount         REAL    NOT NULL CHECK (amount >= 0)
);

CREATE TABLE transaction_features (
    transaction_id INTEGER PRIMARY KEY
                   REFERENCES transactions (transaction_id) ON DELETE CASCADE,
    v1  REAL NOT NULL, v2  REAL NOT NULL, v3  REAL NOT NULL, v4  REAL NOT NULL,
    v5  REAL NOT NULL, v6  REAL NOT NULL, v7  REAL NOT NULL, v8  REAL NOT NULL,
    v9  REAL NOT NULL, v10 REAL NOT NULL, v11 REAL NOT NULL, v12 REAL NOT NULL,
    v13 REAL NOT NULL, v14 REAL NOT NULL, v15 REAL NOT NULL, v16 REAL NOT NULL,
    v17 REAL NOT NULL, v18 REAL NOT NULL, v19 REAL NOT NULL, v20 REAL NOT NULL,
    v21 REAL NOT NULL, v22 REAL NOT NULL, v23 REAL NOT NULL, v24 REAL NOT NULL,
    v25 REAL NOT NULL, v26 REAL NOT NULL, v27 REAL NOT NULL, v28 REAL NOT NULL
);

-- indexes for the access paths used by the analytical queries
CREATE INDEX idx_transactions_split_class ON transactions (split_id, class_id);
CREATE INDEX idx_transactions_hour        ON transactions (hour_index);
CREATE INDEX idx_transactions_time        ON transactions (time_sec);
CREATE INDEX idx_transactions_amount      ON transactions (amount);
