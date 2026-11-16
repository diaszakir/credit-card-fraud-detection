# Team contracts: data, split and module interfaces

This document is the **single place where the three modules meet**. It is written by the
Data Engineering & SQL owner (Zhandarbek) because the data format and the split come first
in the pipeline. Items marked **PROPOSAL** need a yes/no from the owner of the other side
before they are treated as fixed (convention 10 of the project: *any change of the shared data
format or module API is agreed with the team*).

> Approved by the user on 15 November 2026: remove exact raw-row duplicates, then use a
> chronological 70/15/15 split. Keep the original CSV row identifiers for retained transactions.
> The user also fixed fraud precision >= 0.85 and authorized autonomous completion of the ML
> stages with local commits. The working ML protocol is recorded before experiments in
> `src/ml/experiment_config.json`. The implemented interface below is ready for Adam's review;
> no external owner sign-off is claimed.

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
* Provider metadata and the downloaded CSV SHA-256 are recorded in
  `reports/experiments/data_source.json`, including the provider's license designation.
* **`transaction_id`** = 0-based position of the row in the raw CSV. The file is already ordered
  by `Time` (verified by the audit), so the id also reflects chronological order. The id never
  changes as long as the same file is used; its SHA-256 is stored in `reports/data_audit.json`.
* Assign identifiers before removing duplicates. Compare all raw columns (`Time`, `V1`–`V28`,
  `Amount`, `Class`) and keep the first occurrence of each identical row. Perform this check
  before converting V features to `float32`, which can make distinct raw values equal.
* Retained transactions keep their original `transaction_id`; gaps after removal are expected.
  The audit describes the original file, including its duplicate counts. The cleaned dataset
  and its split statistics are generated separately.

## 3. Processed dataset (what ML and Evaluation read)

`data/processed/transactions.parquet`

| Column | dtype | Meaning |
|---|---|---|
| `transaction_id` | int32 | original CSV row id; unique and increasing, with gaps after deduplication |
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

* Chronological on `Time`, **after removing exact raw-row duplicates**: first 70 % of retained
  rows = `train`, next 15 % = `validation`, last 15 % = `test`.
* A cut is never placed between two rows with the same `Time`; it moves forward to the next
  distinct value. Actual percentages may therefore differ slightly from 70/15/15.
* Deterministic, no random seed involved. Fractions live in `src/config.py`.
* Rules that follow from it (project conventions 3-7): fit imputers/scalers/SMOTE/undersampling
  on `train` only; tune the threshold on `validation` only; touch `test` once, at the end.
* Exact row and fraud counts, time ranges and id ranges are written to
  `reports/split_summary.json` after rebuilding the cleaned dataset. Artefacts from before
  this change describe the dataset with duplicates; do not reuse them for ML. The manifest,
  summary, SQLite database and reports have been regenerated using the deduplication rule.

The 70/15/15 proportions are approved. Check fraud counts in the regenerated summary and
describe the uncertainty of precision/recall when validation or test contains few frauds.

ML v1 uses three expanding-window folds within train. Four equal `Time` intervals define
train/evaluation windows 1 -> 2, 1+2 -> 3, 1+2+3 -> 4. Equal timestamps stay together;
each side contains both classes and each training side has enough frauds for SMOTE with k=5.
The exact ordered ids are hashed in `reports/experiments/data_checks.json`.
Hyperparameters maximize mean fold AP. Preprocessing and resampling are fitted only on each
fold's training portion; validation and test retain their natural class balance.

Different-Time legitimate transactions can share all 29 input values. The agreed exact-raw
deduplication does not remove these distinct operations. The ML audit verifies them against
the raw CSV; a supplementary novel-validation report excludes inputs already seen in train
and uses the already selected threshold. Selection always uses the full validation split.

Before test, choose the threshold with maximum validation recall subject to precision >=0.85
and at least one positive prediction. Ties prefer precision, then the larger threshold.
Model ties prefer recall, precision, trapezoidal PR-AUC, then fixed model/strategy/run order.
AP is reported separately. A missing feasible threshold is an explicit unavailable result.

## 5. SQLite database

`data/processed/fraud.db` is built by `python -m src.data.cli build-db`; schema in
`src/db/sql/schema.sql`, ER diagram in `docs/er_diagram.png` (+ Mermaid in `docs/er_diagram.md`).

```
splits(split_id PK, split_name)            classes(class_id PK, label)
time_slots(hour_index PK, day_index, hour_in_cycle)
transactions(transaction_id PK, split_id FK, class_id FK, hour_index FK, time_sec, amount)
transaction_features(transaction_id PK/FK, v1 … v28)        -- 1:1 with transactions
amount_bands(band_id PK, label, lower_bound, upper_bound)    -- joined by range
```

Analytical queries: `src/db/sql/queries/*.sql`, run through `src/db/queries.py`; every result
is loaded into a Pandas DataFrame and exported to `reports/sql_results/`.
Query `q12_duplicate_transactions` checks repeated values in the processed dataset; it does
not measure how many raw copies were removed. Distinct raw values can coincide after
conversion to `float32`, so this result need not equal the raw audit's duplicate count.

**For Adam (Data Overview page):**

```python
from src.db.queries import get_engine, run_query, list_queries
engine = get_engine()                      # data/processed/fraud.db
df = run_query(engine, "q01_split_overview")
```

## 6. Implemented ML experiment and artifact interface

ML exports these Parquet files in `reports/experiments/` for the existing SQLite loader and
Adam's comparison/error-analysis pages. Per-run JSON files retain the complete provenance.

| File | Columns |
|---|---|
| `metrics.parquet` (long format) | `run_id`, `model_name`, `imbalance_strategy`, `split`, `metric`, `value` |
| `runs.parquet` | `run_id`, `model_name`, `imbalance_strategy`, `params_json`, `fit_seconds`, `model_size_bytes`, `peak_memory_mb`, `threshold`, `created_at` |
| `predictions.parquet` | `run_id`, `transaction_id`, `split`, `y_true`, `score`, `y_pred` |

`src/db/experiments.py` loads them into the tables
`model_runs`, `run_metrics`, `predictions` of `fraud.db`, so false positives / false negatives can be
joined back to `transactions` and `transaction_features` in SQL. The DDL is in
`src/db/sql/experiments_schema.sql`. Nothing in this repository contains made-up model results:
the tables stay empty until Dias' real runs exist.

Missing threshold metrics are not written as zeros or NaN. An infeasible run has a nullable
threshold and no row-level class predictions in the export. The loader validates labels only
for ids actually present in prediction files; importing validation results does not read test labels.

Per-run artifacts: `models/runs/<run_id>.joblib` plus JSON. The frozen handoff pair is
`models/fraud_pipeline.joblib` and `models/fraud_pipeline.json`, with `models/selection_lock.json`.
The binary and row-level caches stay out of Git and are reproducible from the committed recipe.
Metadata includes the complete feature schema, positive label, `>=` rule, threshold, min precision,
versions, seed, parameters, commands, CV, validation metrics and data/code/artifact hashes.

Shared function: `src.ml.prediction.predict_transactions(X, pipeline, metadata)` returns a
DataFrame with the same row index/order and `score`, `threshold`, `y_pred`. It checks the exact
29-feature set and finite numeric values, orders/casts inputs to the saved schema, selects the
fraud score using `classes_ == 1`, and applies the saved `>=` comparison. Load only trusted
joblib files through `src.ml.artifacts.load_artifact`.

Full commands and the fresh-clone artifact recipe are in the root `README.md`. A minimal Streamlit
demo in `app/streamlit_app.py` displays saved validation results and uses the shared prediction API;
Docker Compose prepares the frozen recipe once and persists data, models and reports in named volumes.
Final test evaluation and further application integration remain Adam's responsibility: use the
frozen pipeline and threshold without retraining on train+validation or adjusting the threshold
after seeing test results.

## 7. Change process

1. Open a branch, change `docs/CONTRACTS.md` and `src/config.py` together.
2. Tag the other owners in the pull request; merge only after both agree.
3. Re-run `python -m src.data.cli run-all` and commit the refreshed `split_manifest.parquet`.
