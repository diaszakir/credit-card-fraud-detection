"""Generate the report / presentation text of the data-engineering part from saved artefacts.

Every number in the generated documents is read from ``reports/*.json`` and
``reports/sql_results/*.csv`` (produced by the pipeline); the templates in ``docs/templates/``
contain only prose and ``${placeholders}``.  ``string.Template.substitute`` is used on purpose:
a misspelled placeholder raises ``KeyError`` instead of silently leaving a blank.
"""

from __future__ import annotations

import logging
from pathlib import Path
from string import Template

import pandas as pd

from src.config import (
    AUDIT_JSON_PATH,
    BENCHMARK_JSON_PATH,
    DOCS_DIR,
    EDA_DIR,
    SPLIT_NAMES,
    SPLIT_SUMMARY_JSON_PATH,
    SQL_RESULTS_DIR,
)
from src.data.io_utils import read_json

logger = logging.getLogger(__name__)

TEMPLATES_DIR = DOCS_DIR / "templates"
OUTPUTS = {
    "report_data_engineering.md": "data_engineering_report.md.tmpl",
    "presentation_data_engineering.md": "data_engineering_slides.md.tmpl",
}


def md_table(frame: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table."""
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    divider = "|" + "|".join("---" for _ in frame.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, divider, *rows])


def _num(value: float, digits: int = 0) -> str:
    return f"{value:,.{digits}f}"


def _read_csv(name: str) -> pd.DataFrame:
    path = SQL_RESULTS_DIR / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.data.cli sql` first.")
    return pd.read_csv(path)


# ----------------------------------------------------------------------------- value builders
def _audit_values(audit: dict) -> dict[str, str]:
    overview, dup = audit["overview"], audit["duplicates"]
    cls, time, amount = audit["class_balance"], audit["time"], audit["amount"]
    outliers = audit["outliers_iqr"]
    source = audit.get("source_file", {})
    missing = audit["missing_values"]["total_missing"]
    token = source.get("time_column_text", {})

    checks = pd.DataFrame(
        [
            ("Rows x columns", f"{_num(overview['n_rows'])} x {overview['n_columns']}"),
            ("Missing values", str(missing)),
            ("Constant columns", ", ".join(audit["constant_columns"]) or "none"),
            ("Rows repeating an earlier row", f"{_num(dup['n_repeated_rows'])} ({dup['repeated_rows_among_fraud']} fraud, {_num(dup['repeated_rows_among_legitimate'])} legitimate)"),
            ("Time range", f"{_num(time['min'])} - {_num(time['max'])} s ({time['span_hours']:.2f} h)"),
            ("Rows ordered by Time", "yes" if time["is_sorted_non_decreasing"] else "no"),
            ("Most rows sharing one timestamp", str(time["max_rows_sharing_one_timestamp"])),
            ("Amount: zero / negative", f"{_num(amount['n_zero'])} / {amount['n_negative']}"),
            ("Amount: at most 2 decimals", "yes" if amount["has_at_most_two_decimals"] else "no"),
            ("Amount outside IQR fences", f"{outliers['amount']['pct_outliers']:.2f} % (fences {outliers['amount']['lower_fence']:.2f} .. {outliers['amount']['upper_fence']:.2f})"),
            ("Fraud share", f"{cls['fraud_rate_percent']:.4f} % ({cls['n_fraud']} of {_num(overview['n_rows'])})"),
        ],
        columns=["Check", "Result"],
    )
    by_class = amount["by_class"]
    top_outlier_cols = ", ".join(f"{d['column']} ({d['pct_outliers']:.1f} %)" for d in outliers["columns_with_most_outliers"][:3])
    return {
        "n_rows": _num(overview["n_rows"]),
        "n_columns": str(overview["n_columns"]),
        "n_fraud": str(cls["n_fraud"]),
        "n_legit": _num(cls["n_legitimate"]),
        "fraud_rate": f"{cls['fraud_rate_percent']:.4f}",
        "imbalance_ratio": f"{cls['imbalance_ratio_legit_per_fraud']:.0f}",
        "sha256": source.get("sha256", "n/a"),
        "time_span_hours": f"{time['span_hours']:.1f}",
        "max_rows_one_timestamp": str(time["max_rows_sharing_one_timestamp"]),
        "missing_sentence": "contains no missing values" if missing == 0 else f"contains {missing} missing values",
        "n_repeated": _num(dup["n_repeated_rows"]),
        "dup_fraud": str(dup["repeated_rows_among_fraud"]),
        "dup_legit": _num(dup["repeated_rows_among_legitimate"]),
        "dup_groups": _num(dup["n_duplicate_groups"]),
        "time_token_sentence": (
            f"{token['n_not_plain_integer']} Time value(s) are written in scientific notation ({', '.join(token['examples'])}); "
            "pandas therefore reads the whole column as float64 although every value is a whole number of seconds."
            if token.get("n_not_plain_integer") else "All Time values are plain integers."
        ),
        "audit_table": md_table(checks),
        "amount_median_legit": f"{by_class['0']['median']:.2f}",
        "amount_median_fraud": f"{by_class['1']['median']:.2f}",
        "amount_mean_legit": f"{by_class['0']['mean']:.2f}",
        "amount_mean_fraud": f"{by_class['1']['mean']:.2f}",
        "amount_max": f"{amount['max']:.2f}",
        "outlier_columns": top_outlier_cols,
        "amount_outlier_pct": f"{outliers['amount']['pct_outliers']:.1f}",
    }


def _memory_values(audit: dict, benchmarks: dict) -> dict[str, str]:
    columns = pd.DataFrame(audit["memory_optimization"]["columns"])
    columns["group"] = columns["column"].where(~columns["column"].str.match(r"^V\d+$"), "V1..V28")
    grouped = columns.groupby("group", sort=False).agg(
        before=("dtype_before", "first"), after=("dtype_after", "first"),
        mib_before=("mib_before", "sum"), mib_after=("mib_after", "sum"),
        max_rel_error=("max_relative_error", "max"), identical=("values_identical", "all"),
    ).reset_index()
    table = pd.DataFrame({
        "Column(s)": grouped["group"], "dtype before": grouped["before"], "dtype after": grouped["after"],
        "MiB before": grouped["mib_before"].round(2), "MiB after": grouped["mib_after"].round(2),
        "Values identical": grouped["identical"].map({True: "yes", False: "no"}),
        "Max relative error": grouped["max_rel_error"].map(lambda v: "-" if pd.isna(v) else f"{v:.1e}"),
    })
    total = audit["memory_optimization"]["total"]

    storage = benchmarks["storage_formats"]
    variants = pd.DataFrame(storage["variants"])
    storage_table = pd.DataFrame({
        "Variant": variants["variant"], "Load time, s": variants["seconds_median"].round(3),
        "Speed-up vs baseline": variants["speedup_vs_baseline"].round(2),
        "DataFrame, MiB": variants["frame_mib"].round(1),
        "Peak RSS above idle, MiB": variants["peak_above_idle_mib"].map(lambda v: "n/a" if pd.isna(v) else f"{v:.0f}"),
        "Columns": variants["columns"],
    })
    sizes = storage["file_sizes_mib"]
    sizes_table = pd.DataFrame({"File": list(sizes), "Size, MiB": [round(v, 1) for v in sizes.values()]})
    base = variants.set_index("variant")
    return {
        "mem_before": f"{total['mib_before']:.1f}",
        "mem_after": f"{total['mib_after']:.1f}",
        "mem_saved_pct": f"{total['saved_percent']:.1f}",
        "memory_table": md_table(table),
        "storage_table": md_table(storage_table),
        "file_sizes_table": md_table(sizes_table),
        "csv_mib": f"{sizes['csv']:.1f}",
        "parquet_mib": f"{sizes['parquet_optimized_zstd']:.1f}",
        "parquet_saved_pct": f"{100 * (1 - sizes['parquet_optimized_zstd'] / sizes['csv']):.0f}",
        "peak_csv": f"{base.loc['csv_float64', 'peak_above_idle_mib']:.0f}",
        "peak_parquet": f"{base.loc['parquet_optimized', 'peak_above_idle_mib']:.0f}",
        "peak_chunked": f"{base.loc['csv_chunked_optimized', 'peak_above_idle_mib']:.0f}",
        "time_csv": f"{base.loc['csv_float64', 'seconds_median']:.3f}",
        "time_chunked": f"{base.loc['csv_chunked_optimized', 'seconds_median']:.3f}",
        "time_parquet": f"{base.loc['parquet_optimized', 'seconds_median']:.3f}",
        "speedup_projection": f"{base.loc['parquet_optimized_3cols', 'speedup_vs_baseline']:.1f}",
        "bench_platform": storage["platform"],
        "bench_repeats": str(storage["repeats"]),
    }


def _split_values(summary: dict) -> dict[str, str]:
    table = pd.DataFrame({
        "Split": list(SPLIT_NAMES),
        "Rows": [_num(summary[n]["n_rows"]) for n in SPLIT_NAMES],
        "Share of rows": [f"{summary[n]['share_of_rows_percent']:.2f} %" for n in SPLIT_NAMES],
        "Frauds": [summary[n]["n_fraud"] for n in SPLIT_NAMES],
        "Fraud rate": [f"{summary[n]['fraud_rate_percent']:.3f} %" for n in SPLIT_NAMES],
        "Time range, s": [f"{_num(summary[n]['time_min'])} - {_num(summary[n]['time_max'])}" for n in SPLIT_NAMES],
        "transaction_id range": [f"{_num(summary[n]['id_min'])} - {_num(summary[n]['id_max'])}" for n in SPLIT_NAMES],
    })
    return {
        "split_table": md_table(table),
        "train_rows": _num(summary["train"]["n_rows"]),
        "train_fraud": str(summary["train"]["n_fraud"]),
        "val_fraud": str(summary["validation"]["n_fraud"]),
        "test_fraud": str(summary["test"]["n_fraud"]),
        "train_rate": f"{summary['train']['fraud_rate_percent']:.3f}",
        "val_rate": f"{summary['validation']['fraud_rate_percent']:.3f}",
        "test_rate": f"{summary['test']['fraud_rate_percent']:.3f}",
        "val_rows": _num(summary["validation"]["n_rows"]),
        "test_rows": _num(summary["test"]["n_rows"]),
    }


def _eda_values() -> dict[str, str]:
    eda = read_json(EDA_DIR / "eda_summary.json")
    top = eda["top_features_by_abs_correlation_train"]
    dependence = eda["v_feature_dependence_train"]
    peak = eda["hour_with_highest_fraud_rate"]
    return {
        "top_corr_list": ", ".join(f"{d['feature']} ({d['correlation']:+.3f})" for d in top),
        "v_mean_abs_corr": f"{dependence['mean_abs_corr']:.3f}",
        "v_max_abs_corr": f"{dependence['max_abs_corr']:.3f}",
        "peak_hour": str(peak["hour_index"]),
        "peak_rate": f"{peak['fraud_rate_percent']:.2f}",
        "peak_fraud": str(peak["n_fraud"]),
        "peak_tx": _num(peak["n_transactions"]),
    }


def _sql_values(audit: dict) -> dict[str, str]:
    from src.db.crosscheck import CHECKS
    from src.db.queries import construct_coverage, describe_queries

    q02, q03, q07 = _read_csv("q02_fraud_by_amount_band"), _read_csv("q03_hourly_fraud_hotspots"), _read_csv("q07_fraud_rate_by_amount_decile")
    q06, q08, q10 = _read_csv("q06_fraud_pareto_by_hour"), _read_csv("q08_time_between_frauds"), _read_csv("q10_feature_separation")
    q12, q04 = _read_csv("q12_duplicate_transactions"), _read_csv("q04_hours_above_average_fraud_rate")

    train_fraud, train_rows = int(q07["n_fraud"].sum()), int(q07["n_transactions"].sum())
    train_rate = 100 * train_fraud / train_rows
    first = q07.iloc[0]
    zero_band = q02.iloc[0]
    hours_half = int((q06["cumulative_share_of_frauds_pct"] < 50).sum() + 1)
    top_hour = q03.iloc[0]
    gaps = q08.set_index("split_name")

    catalogue = describe_queries()
    queries_table = pd.DataFrame({
        "Query": catalogue["name"].map(lambda n: f"`{n}`"),
        "Question": catalogue["question"],
        "Constructs (detected in the SQL)": catalogue["detected_constructs"],
    })
    coverage = construct_coverage()
    coverage_table = pd.DataFrame({
        "Required construct": list(coverage),
        "Queries using it": [len(v) for v in coverage.values()],
        "Examples": [", ".join(v[:3]) for v in coverage.values()],
    })

    findings = [
        f"Amounts in the lowest decile of the train split ({first['min_amount']:.2f} - {first['max_amount']:.2f}) hold "
        f"{int(first['n_fraud'])} of the {train_fraud} train frauds ({100 * first['n_fraud'] / train_fraud:.1f} %); "
        f"the fraud rate there is {first['fraud_rate_pct']:.2f} % against {train_rate:.2f} % for the whole split (`q07`).",
        f"Transactions with Amount = 0 have a fraud rate of {zero_band['fraud_rate_pct']:.2f} % "
        f"({int(zero_band['n_fraud'])} of {_num(zero_band['n_transactions'])} in train) (`q02`).",
        f"{hours_half} of {len(q06)} hours contain at least half of all frauds (`q06`).",
        f"The highest hourly fraud rate among hours with enough traffic is {top_hour['fraud_rate_pct']:.2f} % in hour "
        f"{int(top_hour['hour_index'])} ({int(top_hour['n_fraud'])} frauds in {_num(top_hour['n_transactions'])} transactions) (`q03`); "
        f"{len(q04)} hours are above the overall rate (`q04`).",
        "Mean gap between consecutive frauds: " + ", ".join(f"{n} {gaps.loc[n, 'avg_gap_sec']:.0f} s" for n in SPLIT_NAMES) + " (`q08`).",
        f"{_num(q12['n_extra_rows'].sum())} rows repeat an earlier row (`q12`); this equals the {_num(audit['duplicates']['n_repeated_rows'])} found by the Pandas audit on the raw CSV.",
        "Features with the largest standardised mean difference between fraud and legitimate rows (train): "
        + ", ".join(f"{r.feature} ({r.standardised_difference:+.2f})" for r in q10.head(6).itertuples()) + " (`q10`).",
    ]
    return {
        "n_queries": str(len(catalogue)),
        "queries_table": md_table(queries_table),
        "coverage_table": md_table(coverage_table),
        "sql_findings": "\n".join(f"- {line}" for line in findings),
        "crosscheck_list": ", ".join(f"`{name.split('_')[0]}`" for name in CHECKS),
        "n_crosschecks": str(len(CHECKS)),
        "decile_share": f"{100 * first['n_fraud'] / train_fraud:.1f}",
        "decile_rate": f"{first['fraud_rate_pct']:.2f}",
        "top_sep_features": ", ".join(q10.head(3)["feature"]),
    }


def _pandas_vs_sql_values(benchmarks: dict) -> dict[str, str]:
    report = benchmarks["pandas_vs_sql"]
    tasks = pd.DataFrame(report["tasks"])
    table = pd.DataFrame({
        "Task": tasks["task"],
        "Pandas from Parquet, s": tasks["pandas_end_to_end_seconds"].round(3),
        "Pandas in memory, s": tasks["pandas_in_memory_seconds"].round(3),
        "SQLite via SQLAlchemy, s": tasks["sql_seconds"].round(3),
        "Same answer": tasks["results_agree"].map({True: "yes", False: "NO"}),
    })
    ratio = tasks["sql_seconds"] / tasks["pandas_end_to_end_seconds"]
    if (ratio > 1).all():
        verdict = (f"Pandas on columnar Parquet was faster than SQLite in all {len(tasks)} tasks "
                   f"(SQL took {ratio.min():.1f}x - {ratio.max():.1f}x as long end-to-end).")
    elif (ratio < 1).all():
        verdict = f"SQLite was faster than Pandas in all {len(tasks)} tasks."
    else:
        verdict = "Neither approach was faster in every task; see the table."
    return {"pandas_sql_table": md_table(table), "pandas_sql_verdict": verdict, "sql_bench_repeats": str(report["repeats"])}


def collect_values() -> dict[str, str]:
    """Read every artefact and return the flat dict of placeholder values."""
    for path in (AUDIT_JSON_PATH, BENCHMARK_JSON_PATH, SPLIT_SUMMARY_JSON_PATH, EDA_DIR / "eda_summary.json"):
        if not Path(path).is_file():
            raise FileNotFoundError(f"{path} not found. Run `python -m src.data.cli run-all` first.")
    audit, benchmarks, summary = read_json(AUDIT_JSON_PATH), read_json(BENCHMARK_JSON_PATH), read_json(SPLIT_SUMMARY_JSON_PATH)
    if "storage_formats" not in benchmarks or "pandas_vs_sql" not in benchmarks:
        raise FileNotFoundError("benchmarks.json is incomplete. Run `python -m src.data.cli benchmark` first.")
    values: dict[str, str] = {}
    for part in (_audit_values(audit), _memory_values(audit, benchmarks), _split_values(summary),
                 _eda_values(), _sql_values(audit), _pandas_vs_sql_values(benchmarks)):
        values.update(part)
    return values


def render(template_name: str, values: dict[str, str]) -> str:
    return Template((TEMPLATES_DIR / template_name).read_text(encoding="utf-8")).substitute(values)


def write_report(out_dir: Path = DOCS_DIR) -> Path:
    """Render every template into ``out_dir``; returns the path of the report section."""
    values = collect_values()
    out_dir = Path(out_dir)
    written = []
    for output_name, template_name in OUTPUTS.items():
        path = out_dir / output_name
        path.write_text(render(template_name, values), encoding="utf-8")
        written.append(path)
        logger.info("wrote %s", path)
    return written[0]
