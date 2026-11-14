# Team contracts: data, split and module interfaces

This document is the **single place where the three modules meet**. It is written by the
Data Engineering & SQL owner (Zhandarbek) because the data format and the split come first
in the pipeline. Items marked **PROPOSAL** need a yes/no from the owner of the other side
before they are treated as fixed (convention 10 of the project: *any change of the shared data
format or module API is agreed with the team*).

> Status: this branch introduces the contract and the shared config. The code that implements it
> arrives in the following branches (`feat/de-02-…` to `feat/de-06-…`, in that order).

| Area | Owner | Folder |
|---|---|---|
| Data Engineering & SQL | Zhandarbek | `src/data/`, `src/db/`, `data/`, `reports/figures/`, `reports/sql_results/` |
| Machine Learning | Dias | `src/ml/`, `models/` |
| Evaluation & Application | Adam | `src/evaluation/`, `app/`, root `README.md`, final `requirements.txt` |

Nobody edits another owner's folder without agreeing first. `src/config.py` is shared:
a change in its *DATA CONTRACT* section needs the whole team.

## 1. Repository layout

```
src/config.py            shared constants (paths, column names, split definition)
src/data/                Zhandarbek: download, audit, dtypes, split, Parquet, EDA, benchmarks, CLI
src/db/                  Zhandarbek: SQLite schema, loader, ER diagram, SQL queries
src/ml/                  Dias        (placeholder)
src/evaluation/          Adam        (placeholder)
app/                     Adam        (Streamlit, placeholder)
models/                  Dias        (joblib pipelines + metadata, placeholder)
data/raw|interim|processed   generated, git-ignored (except the split manifest)
reports/figures/         EDA figures
reports/sql_results/     results of the SQL queries (CSV)
reports/experiments/     metrics / predictions written by ML and Evaluation (placeholder)
docs/                    contracts, ER diagram, data-engineering description
tests/data_engineering/  tests of this module (others: tests/ml, tests/evaluation)
```

## 2. Dataset and stable identifier

* Source: ULB *Credit Card Fraud Detection* (Kaggle `mlg-ulb/creditcardfraud`), 284 807 rows,
  492 frauds. The raw CSV is not committed (150 MB); `python -m src.data.cli download`
  fetches it and verifies the row and fraud counts.
* **`transaction_id`** = 0-based position of the row in the raw CSV. The file is already ordered
  by `Time` (verified by the audit), so the id also reflects chronological order. The id never
  changes as long as the same file is used; its SHA-256 is stored in `reports/data_audit.json`.
* The 1 081 exact duplicate rows are **kept** (they have different ids). Dropping them would change
  the dataset size and the ids; the audit reports how many there are.

## 3. Processed dataset (what ML and Evaluation read)

`data/processed/transactions.parquet`

| Column | dtype | Meaning |
|---|---|---|
| `transaction_id` | int32 | stable id (see above) |
| `split` | category/string | `train`, `validation` or `test` |
| `Time` | int32 | seconds since the first transaction |
| `V1` … `V28` | float32 | anonymised PCA components |
| `Amount` | float64 | transaction amount (kept exact) |
| `Class` | int8 | 1 = fraud, 0 = legitimate |

`data/processed/split_manifest.parquet` (committed, tiny): `transaction_id`, `split`.

Loading helpers (stable API, `src/data/api.py`):

```python
from src.data.api import load_transactions, load_xy, load_split_manifest
from src.config import FEATURE_COLUMNS            # V1..V28 + Amount

X_train, y_train = load_xy("train")               # DataFrame, Series (indexed by transaction_id)
X_val,   y_val   = load_xy("validation")
X_test,  y_test  = load_xy("test")                # use only for the final evaluation
df = load_transactions(split="train", columns=["transaction_id", "Time", "Class"])
```

`Time` is delivered but is **not** in `FEATURE_COLUMNS`: the split is chronological, so a raw
"seconds since start" feature only extrapolates on validation/test. Dias decides whether to use it.

## 4. Split definition

* Chronological on `Time`: first 70 % of rows = `train`, next 15 % = `validation`, last 15 % = `test`.
* A cut is never placed between two rows with the same `Time`; it moves forward to the next
  distinct value. (A plain row-count cut would split Time = 151 328 across validation and test.)
* Deterministic, no random seed involved. Fractions live in `src/config.py`.
* Rules that follow from it (project conventions 3-7): fit imputers/scalers/SMOTE/undersampling
  on `train` only; tune the threshold on `validation` only; touch `test` once, at the end.
* Exact row and fraud counts per split are written to `reports/split_summary.json` by the pipeline.
  Current values (computed from the real file, 70/15/15):

  | split | rows | frauds | fraud rate | Time range (s) | `transaction_id` range |
  |---|---|---|---|---|---|
  | train | 199 364 | 384 | 0.193 % | 0 – 132 928 | 0 – 199 363 |
  | validation | 42 722 | 56 | 0.131 % | 132 929 – 151 328 | 199 364 – 242 085 |
  | test | 42 721 | 52 | 0.122 % | 151 329 – 172 792 | 242 086 – 284 806 |

  The nominal validation/test cut (row 242 085) fell on a timestamp shared by two rows
  (Time = 151 328); the rule above moved it one row forward, which is why validation has one row more.

**PROPOSAL (Dias):** confirm 70/15/15. The fraud rate drifts between splits, so validation and test
contain few frauds (see `reports/split_summary.json`); precision/recall on them will be noisy and
the report should say so. If you prefer 60/20/20, change `TRAIN_FRACTION`/`VALIDATION_FRACTION` in
`src/config.py` and rebuild: everything downstream is regenerated.

**PROPOSAL (Dias):** cross-validation inside `train` should respect time (for example
`TimeSeriesSplit` on rows ordered by `transaction_id`) or at least be stratified; your choice, but
please state it in the report.

## 5. SQLite database

`data/processed/fraud.db` is built by `python -m src.data.cli build-db`; schema in
`src/db/schema.sql`, ER diagram in `docs/er_diagram.png` (+ Mermaid in `docs/er_diagram.md`).

```
splits(split_id PK, split_name)            classes(class_id PK, label)
time_slots(hour_index PK, day_index, hour_in_cycle)
transactions(transaction_id PK, split_id FK, class_id FK, hour_index FK, time_sec, amount)
transaction_features(transaction_id PK/FK, v1 … v28)        -- 1:1 with transactions
amount_bands(band_id PK, label, lower_bound, upper_bound)    -- joined by range
```

Analytical queries: `src/db/sql/queries/*.sql`, run through `src/db/queries.py`; every result
is loaded into a Pandas DataFrame and exported to `reports/sql_results/`.

**For Adam (Data Overview page):**

```python
from src.db.queries import get_engine, run_query, list_queries
engine = get_engine()                      # data/processed/fraud.db
df = run_query(engine, "q01_split_overview")
```

## 6. Experiment results (shared SQLite/Parquet storage) — PROPOSAL

Convention 8 says metrics and results are stored in SQLite/Parquet. So that Adam's *Model
Comparison / Error Analysis* pages and SQL joins with `transactions` work without conversions,
Dias and Adam please confirm (or change) these Parquet files in `reports/experiments/`:

| File | Columns |
|---|---|
| `metrics.parquet` (long format) | `run_id`, `model_name`, `imbalance_strategy`, `split`, `metric`, `value` |
| `runs.parquet` | `run_id`, `model_name`, `imbalance_strategy`, `params_json`, `fit_seconds`, `model_size_bytes`, `peak_memory_mb`, `threshold`, `created_at` |
| `predictions.parquet` | `run_id`, `transaction_id`, `split`, `y_true`, `score`, `y_pred` |

Once the format is agreed, `src/db/experiments.py` (Zhandarbek) loads them into the tables
`model_runs`, `run_metrics`, `predictions` of `fraud.db`, so false positives / false negatives can be
joined back to `transactions` and `transaction_features` in SQL. The DDL is in
`src/db/sql/experiments_schema.sql`. Nothing in this repository contains made-up model results:
the tables stay empty until Dias' real runs exist.

Model artefact (Dias → Adam), **PROPOSAL**: `models/<run_id>.joblib` (the whole Pipeline with its
preprocessing) and `models/<run_id>.json` (threshold, validation metrics, feature names,
training split, library versions).

## 7. Change process

1. Open a branch, change `docs/CONTRACTS.md` and `src/config.py` together.
2. Tag the other owners in the pull request; merge only after both agree.
3. Re-run `python -m src.data.cli run-all` and commit the refreshed `split_manifest.parquet`.
