# Data Engineering & SQL module

Owner: **Zhandarbek Zhetpissov**. Download and audit of the data, memory optimisation and Parquet, the chronological
split with a stable `transaction_id`, EDA, the SQLite database with an ER diagram, and SQL analysis loaded into Pandas.
The interfaces shared with the ML and Evaluation modules are in [CONTRACTS.md](CONTRACTS.md);
the generated report text is in [report_data_engineering.md](report_data_engineering.md).

## Quick start

```bash
python3.12 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-data.txt
python -m src.data.cli run-all        # download -> audit -> Parquet -> split -> EDA -> SQLite -> SQL -> benchmarks -> report
python -m src.data.cli                # or: the interactive console menu
python -m pytest tests/data_engineering
```

`run-all` downloads `creditcard.csv` (150 MB, a public mirror of the Kaggle file, verified by row and fraud count).
If the download is blocked, put the file from <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud> into `data/raw/creditcard.csv`; every step then works offline.
Nothing large is committed: `data/raw`, `data/interim` and the Parquet/SQLite files in `data/processed` are rebuilt by the command above
(only the tiny `split_manifest.parquet` is in git).

## Commands

| Command | What it does | Main output |
|---|---|---|
| `download` | fetch and verify the CSV | `data/raw/creditcard.csv` |
| `audit` | data-quality audit + memory before/after | `reports/data_audit.json` |
| `optimize` | chunked load with compact dtypes → Parquet | `data/interim/creditcard_optimized.parquet` |
| `split` | add `transaction_id`, chronological `split`, validate | `data/processed/transactions.parquet`, `split_manifest.parquet`, `reports/split_summary.json` |
| `eda` | tables and figures | `reports/eda/`, `reports/figures/` |
| `build-db` | build and verify SQLite | `data/processed/fraud.db` |
| `er-diagram` | draw the ER diagram from the database | `docs/er_diagram.png`, `docs/er_diagram.md` |
| `sql` | run all queries, save results, cross-check with Pandas | `reports/sql_results/*.csv` |
| `query [name] [--param k=v]` | list queries, or run one and print the DataFrame | console |
| `benchmark` | CSV vs Parquet, Pandas vs SQL | `reports/benchmarks.json` |
| `summary` | render report and slide text from the saved results | `docs/report_data_engineering.md`, `docs/presentation_data_engineering.md` |

## Code map

| File | Responsibility (main functions) |
|---|---|
| `src/config.py` | paths, column names, split fractions, amount bands (shared contract) |
| `src/data/download.py` | `download_dataset`, `verify_dataset`, `sha256_of` |
| `src/data/loading.py` | `read_raw_csv` (baseline), `iter_csv_chunks`, `load_csv_chunked` |
| `src/data/memory.py` | `optimize_dtypes` (verified casts), `cast_report`, `compare_memory`, `max_relative_error` |
| `src/data/audit.py` | `audit_dataframe`, `iqr_outlier_table`, `scan_time_tokens` |
| `src/data/storage.py` | `save_parquet`, `load_parquet` |
| `src/data/splitting.py` | `assign_transaction_ids`, `find_cut_positions`, `chronological_split`, `validate_split`, `split_summary` |
| `src/data/api.py` | `load_xy`, `load_transactions`, `load_split_manifest` (read API for teammates) |
| `src/data/eda.py` | `run_eda`, `hourly_profile`, `feature_target_correlation`, `plot_*`, `interactive_*` |
| `src/data/benchmark.py` | `run_storage_benchmark` (each variant in a fresh process) |
| `src/data/pipeline.py` | `step_*` functions used by the CLI, tests and notebooks |
| `src/data/cli.py` | `run_menu`, `main`, the `COMMANDS` dict |
| `src/data/report.py` | `collect_values`, `write_report` (templates in `docs/templates/`) |
| `src/db/sql/schema.sql` | normalised schema with constraints and indexes |
| `src/db/builder.py` | `build_database`, `create_schema`, `populate_dimensions`, `load_facts`, `verify_database` |
| `src/db/er_diagram.py` | `read_model` (introspection), `draw_er_diagram`, `to_mermaid` |
| `src/db/sql/queries/*.sql` | 13 analytical queries with a metadata header |
| `src/db/queries.py` | `run_query`, `run_all`, `save_results`, `construct_coverage`, `feature_separation` |
| `src/db/crosscheck.py` | recomputes 8 queries in Pandas and compares |
| `src/db/benchmark_sql.py` | Pandas vs SQL timings |
| `src/db/experiments.py` | **proposal:** loads ML/Evaluation results into SQLite (see CONTRACTS.md §6) |

## Course requirements → evidence

| Requirement (Final Project, Cluster 7 + technical requirements) | Where |
|---|---|
| ≥ 5 functions besides `main` | every module above; e.g. `optimize_dtypes`, `audit_dataframe`, `chronological_split`, `build_database`, `run_query` |
| list / dict | `COMMANDS` dict, `OPTIMIZED_DTYPES`, query metadata, audit dict, lists of columns and results |
| if / elif / else | `run_menu`, `optimize_dtypes`, `_findings`, `_pandas_vs_sql_values` |
| for / while | `run_menu` (`while True`), `download_dataset` (mirrors), `iter_csv_chunks`, `run_all` |
| File reading and writing | CSV → Parquet → SQLite, JSON reports, CSV query results, Markdown generation |
| try / except | `download_dataset`, `verify_dataset`, `read_raw_csv`, `run_menu` (`EXPECTED_ERRORS`), `write_json` |
| Console menu | `python -m src.data.cli` |
| Modularity | `src/data` and `src/db` packages, `src/config.py`, standard library (`pathlib`, `json`, `argparse`, `hashlib`, `logging`) |
| EDA: statistics, missing values, outliers (IQR / boxplot), class balance, correlations | `src/data/audit.py`, `src/data/eda.py`, `reports/eda/`, `reports/figures/` |
| ≥ 4–5 visualisations, ≥ 1 interactive (Plotly) | 7 static + 2 Plotly figures |
| Split before fitting transformations, no leakage, by time | `src/data/splitting.py`, rules in CONTRACTS.md §4 |
| Parquet / dtype optimisation with memory before and after | `src/data/memory.py`, `audit.py`, report §3 |
| Compare ≥ 2 approaches with a baseline (quality / time / memory) | `src/data/benchmark.py` (CSV baseline vs chunked CSV vs Parquet variants), `src/db/benchmark_sql.py` (Pandas vs SQL) |
| SQLite + ER diagram | `src/db/sql/schema.sql`, `docs/er_diagram.png`, `docs/er_diagram.md` |
| ≥ 8 SQL queries: JOIN, GROUP BY/HAVING, subquery or CTE, window functions | 13 queries; coverage table in the report, tested in `tests/data_engineering/test_queries.py` |
| Query results loaded into Pandas | `run_query` → `pandas.read_sql_query`; `reports/sql_results/` |
| README, `requirements.txt`, GitHub | this file + `requirements-data.txt` (the project-wide `requirements.txt` and root README are consolidated by Adam) |
| Data volume ≥ 500k rows / 200 MB | topic 32 is on the instructor's list of exceptions; memory work and Parquet are done anyway (the CSV is about 144 MiB) |

## Using the data from another module

```python
from src.config import FEATURE_COLUMNS
from src.data.api import load_xy

X_train, y_train = load_xy("train")            # index = transaction_id
X_val,   y_val   = load_xy("validation")       # tune the threshold here only
X_test,  y_test  = load_xy("test")             # final evaluation only
```

```python
from src.db.queries import get_engine, run_query
df = run_query(get_engine(), "q03_hourly_fraud_hotspots", params={"min_transactions": 500})
```

## Design decisions in one place

- **Chronological split, not random:** the model scores future transactions; a random split would let it see the future and near-duplicates.
- **Cuts never separate equal timestamps:** a row-count cut falls inside a group of rows with the same `Time` in this dataset.
- **Duplicates kept:** they have their own ids; removing them changes the dataset the whole team agreed on. They are reported by the audit and by SQL `q12`.
- **`Amount` stays `float64`, V features become `float32`:** money must match exactly in SQL; the V features are PCA scores where a relative error of 6e-8 is irrelevant. Every cast is verified, and the build fails if a cast is unsafe.
- **Two tables for the transactions:** facts are narrow; the 28 features are a 1:1 table. Dimensions are separate to avoid repeating text.
- **SQL results are cross-checked in Pandas:** a wrong JOIN or window frame would otherwise look plausible.
- **Reports are generated:** numbers in the report come from the saved results, so they cannot be mistyped or invented.

## Troubleshooting

- `FileNotFoundError ... run-all`: the processed data has not been built; run `python -m src.data.cli run-all`.
- Download blocked or offline: place `creditcard.csv` manually (see Quick start).
- Windows: all code is cross-platform except the peak-memory column of the storage benchmark, which needs the `resource` module and shows `n/a` there.
- Plotly figures load `plotly.js` from a CDN, so they need internet when opened.
