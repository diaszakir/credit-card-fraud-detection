# Presentation content: Data Engineering & SQL (Zhandarbek Zhetpissov)

Slide text for the author's part of the team presentation, generated from the saved results
(`python -m src.data.cli summary`). Each slide: title, 3–5 bullets, suggested figure, speaker notes.
Merge into the common deck (.pptx or .pdf); the other members add their own slides.

---

## Slide 1. Data and audit

- Raw file: 284,807 transactions, 31 columns; 492 frauds = 0.1727 % (about 578 : 1)
- contains no missing values; remove 1,081 exact raw-row copies before dtype conversion and splitting
- Keep the first occurrence and its original CSV `transaction_id`; removed copies leave gaps
- Figure: `reports/figures/eda_01_class_balance.png`

*Notes:* deduplication and chronological 70/15/15 were approved by the user on 15 November 2026. Rows remain ordered by `Time`. Accuracy is useless at this imbalance, hence PR-AUC, recall at fixed precision and threshold tuning. 1 Time value(s) are written in scientific notation (1e+05); pandas therefore reads the whole column as float64 although every value is a whole number of seconds.

## Slide 2. Memory and Parquet

- DataFrame memory 67.4 MiB → 34.0 MiB (49.6 % smaller): Time `int32`, Class `int8`, V1–V28 `float32`, Amount stays `float64`
- Every cast is verified (integer-valued, in range, relative error < 1e-6)
- CSV 143.8 MiB → Parquet 43.4 MiB (70 % smaller)
- 3-column projection from Parquet: 34.0x the baseline speed

*Notes:* preparation reads the full raw CSV and removes exact copies before `float32` conversion. The benchmark separately compares loading the original rows, including a chunked CSV alternative, with each variant in its own process. Table: `reports/benchmarks.json`.

## Slide 3. Chronological split without leakage

- After deduplication: train / validation / test target 70 / 15 / 15 % of retained rows in time order; cuts never separate rows with the same `Time`
- Frauds: 366 / 55 / 52; fraud rate 0.184 % / 0.129 % / 0.122 %
- Preprocessing and resampling fit on train only; threshold on validation only; test once
- Figure: `reports/figures/eda_07_split_drift.png`

*Notes:* few frauds in validation and test → noisy precision/recall; rebuild `split_manifest.parquet` and `split_summary.json` after deduplication. Equal-time groups can shift actual percentages slightly. Use the regenerated manifest for all models.

## Slide 4. EDA

- Largest correlations with Class (train): V17 (-0.369), V14 (-0.315), V12 (-0.265), V10 (-0.243), V3 (-0.230), V16 (-0.222)
- V1–V28 nearly uncorrelated with each other (mean |r| = 0.018)
- Highest hourly fraud rate: 1.56 % in hour 26 (27 of 1,735)
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
- Amounts in the lowest decile of the train split (0.00 - 1.00) hold 128 of the 366 train frauds (35.0 %); the fraud rate there is 0.64 % against 0.18 % for the whole split (`q07`).
- Transactions with Amount = 0 have a fraud rate of 1.17 % (16 of 1,371 in train) (`q02`).
- 13 of 48 hours contain at least half of all frauds (`q06`).
- The highest hourly fraud rate among hours with enough traffic is 1.56 % in hour 26 (27 frauds in 1,735 transactions) (`q03`); 17 hours are above the overall rate (`q04`).
- Mean gap between consecutive frauds: train 362 s, validation 330 s, test 361 s (`q08`).
- The raw audit found 1,081 repeated rows, removed before splitting. The stored compact dataset has 0 repeated rows (`q12`); this check uses float32 features, so it can also count rounding coincidences between distinct raw rows.
- Features with the largest standardised mean difference between fraud and legitimate rows (train): V14 (-2.25), V4 (+1.99), V12 (-1.94), V11 (+1.89), V10 (-1.69), V16 (-1.58) (`q10`).

*Notes:* the full list with questions is in `docs/report_data_engineering.md`, section 7.

## Slide 7. SQL against Pandas, and verification

- Neither approach was faster in every task; see the table.
- 8 SQL queries recomputed independently in Pandas and compared; the pipeline stops on a mismatch
- One command: `python -m src.data.cli run-all`; console menu: `python -m src.data.cli`

*Notes:* SQL duplicate counts refer to processed values, while the audit counts original copies before removal; compare SQL with Pandas on the same processed dataset. SQL also provides integrity and joins with model predictions.

## Contribution (for the contribution slide)

Zhandarbek Zhetpissov: data download and audit, Parquet and memory optimisation, split and `transaction_id`, EDA, SQLite schema and ER diagram, SQL queries, benchmarks, data contract.
