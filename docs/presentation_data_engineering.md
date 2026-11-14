# Presentation content: Data Engineering & SQL (Zhandarbek Zhetpissov)

Slide text for the author's part of the team presentation, generated from the saved results
(`python -m src.data.cli summary`). Each slide: title, 3–5 bullets, suggested figure, speaker notes.
Merge into the common deck (.pptx or .pdf); the other members add their own slides.

---

## Slide 1. Data and audit

- 284,807 transactions, 31 columns; 492 frauds = 0.1727 % (about 578 : 1)
- contains no missing values; 1,081 rows repeat an earlier row (kept, ids stay stable)
- Rows already ordered by `Time` → row position = stable `transaction_id`
- Figure: `reports/figures/eda_01_class_balance.png`

*Notes:* accuracy is useless at this imbalance, hence PR-AUC, recall at fixed precision and threshold tuning. 1 Time value(s) are written in scientific notation (1e+05); pandas therefore reads the whole column as float64 although every value is a whole number of seconds.

## Slide 2. Memory and Parquet

- DataFrame memory 67.4 MiB → 34.0 MiB (49.6 % smaller): Time `int32`, Class `int8`, V1–V28 `float32`, Amount stays `float64`
- Every cast is verified (integer-valued, in range, relative error < 1e-6)
- CSV 143.8 MiB → Parquet 43.4 MiB (70 % smaller)
- 3-column projection from Parquet: 2.3x the baseline speed

*Notes:* the benchmark runs each variant in its own process; load times are close because pandas' CSV parser is fast, the gains are size, memory and projection. Table: `reports/benchmarks.json`.

## Slide 3. Chronological split without leakage

- Train / validation / test = 70 / 15 / 15 % of rows in time order; cuts never separate rows with the same `Time`
- Frauds: 384 / 56 / 52; fraud rate 0.193 % / 0.131 % / 0.122 %
- Preprocessing and resampling fit on train only; threshold on validation only; test once
- Figure: `reports/figures/eda_07_split_drift.png`

*Notes:* few frauds in validation and test → noisy precision/recall; the committed `split_manifest.parquet` is the shared source of truth.

## Slide 4. EDA

- Largest correlations with Class (train): V17 (-0.384), V14 (-0.326), V12 (-0.277), V10 (-0.256), V3 (-0.244), V7 (-0.238)
- V1–V28 nearly uncorrelated with each other (mean |r| = 0.019)
- Highest hourly fraud rate: 2.05 % in hour 26 (36 of 1,752)
- Figures: `eda_03_time_profile.png`, `eda_04_feature_target_correlation.png`; interactive: `eda_08_interactive_time_profile.html`

*Notes:* feature–target statistics use train only, so EDA does not leak validation/test information.

## Slide 5. SQLite and the ER diagram

- Normalised star-like schema: `transactions` + `transaction_features` (1:1) + `splits`, `classes`, `time_slots`, `amount_bands`
- PK / FK / CHECK / UNIQUE / ON DELETE CASCADE; build verified (counts, sums, `foreign_key_check`, `integrity_check`) before use
- Figure: `docs/er_diagram.png` (generated from the database)

*Notes:* the 28 features live in a separate table so analytical scans stay narrow; `amount_bands` is joined by a range, not by a key.

## Slide 6. 13 SQL queries → Pandas

- JOIN (including a range join), GROUP BY / HAVING, scalar subquery, CTE, window functions (RANK, NTILE, LAG, ROW_NUMBER, running SUM)
- Results loaded into DataFrames with `read_sql_query`; saved to `reports/sql_results/`
- Amounts in the lowest decile of the train split (0.00 - 1.00) hold 138 of the 384 train frauds (35.9 %); the fraud rate there is 0.69 % against 0.19 % for the whole split (`q07`).
- Transactions with Amount = 0 have a fraud rate of 1.31 % (18 of 1,379 in train) (`q02`).
- 12 of 48 hours contain at least half of all frauds (`q06`).
- The highest hourly fraud rate among hours with enough traffic is 2.05 % in hour 26 (36 frauds in 1,752 transactions) (`q03`); 18 hours are above the overall rate (`q04`).
- Mean gap between consecutive frauds: train 345 s, validation 324 s, test 361 s (`q08`).
- 1,081 rows repeat an earlier row (`q12`); this equals the 1,081 found by the Pandas audit on the raw CSV.
- Features with the largest standardised mean difference between fraud and legitimate rows (train): V14 (-2.29), V4 (+2.02), V12 (-1.97), V11 (+1.94), V10 (-1.70), V16 (-1.63) (`q10`).

*Notes:* the full list with questions is in `docs/report_data_engineering.md`, section 7.

## Slide 7. SQL against Pandas, and verification

- Pandas on columnar Parquet was faster than SQLite in all 4 tasks (SQL took 2.0x - 5.0x as long end-to-end).
- 8 SQL queries recomputed independently in Pandas and compared; the pipeline stops on a mismatch
- One command: `python -m src.data.cli run-all`; console menu: `python -m src.data.cli`

*Notes:* SQL's value here is integrity and joins with model predictions, not speed.

## Contribution (for the contribution slide)

Zhandarbek Zhetpissov: data download and audit, Parquet and memory optimisation, split and `transaction_id`, EDA, SQLite schema and ER diagram, SQL queries, benchmarks, data contract.
