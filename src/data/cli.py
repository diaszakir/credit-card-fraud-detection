"""Command-line and console-menu interface of the data-engineering module.

    python -m src.data.cli                 interactive console menu
    python -m src.data.cli run-all         whole pipeline, no questions asked
    python -m src.data.cli audit           one step (see --help for the list)
    python -m src.data.cli query q03_hourly_fraud_hotspots --param min_transactions=500

The commands are stored in a dict ``COMMANDS`` (name -> (description, function)); the menu and
the argument parser are both generated from it, so they cannot get out of sync.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable

import pandas as pd

from src.config import AUDIT_JSON_PATH, RAW_CSV_PATH
from src.data import pipeline
from src.data.download import DownloadError
from src.data.loading import DataLoadError

logger = logging.getLogger("src.data.cli")

# Errors that a user can cause or fix (missing file, bad input): reported, not a crash.
EXPECTED_ERRORS = (DownloadError, DataLoadError, FileNotFoundError, FileExistsError, ValueError, KeyError)


# --------------------------------------------------------------------------- command functions
def cmd_download() -> None:
    path = pipeline.step_download()
    print(f"OK: {path} is present and verified (284,807 rows, 492 frauds).")


def cmd_audit() -> None:
    audit = pipeline.step_audit()
    print(f"Rows: {audit['overview']['n_rows']:,} | missing values: {audit['missing_values']['total_missing']} "
          f"| repeated rows: {audit['duplicates']['n_repeated_rows']} | fraud: {audit['class_balance']['fraud_rate_percent']:.4f} %")
    memory = audit["memory_optimization"]["total"]
    print(f"Memory in pandas: {memory['mib_before']:.1f} MiB -> {memory['mib_after']:.1f} MiB ({memory['saved_percent']:.1f} % saved)")
    for note in audit["findings"]:
        print(" -", note)
    print(f"Saved: {AUDIT_JSON_PATH}")


def cmd_optimize() -> None:
    df = pipeline.step_optimize()
    print(f"Chunked load + compact dtypes done: {len(df):,} rows saved as interim Parquet.")


def cmd_split() -> None:
    processed = pipeline.step_split()
    summary = pd.DataFrame(pipeline.validate_split(processed)).T[["n_rows", "n_fraud", "fraud_rate_percent", "time_min", "time_max"]]
    print(summary.round(4).to_string())


def cmd_eda() -> None:
    result = pipeline.step_eda()
    print("Figures and tables written. Files:")
    for name, path in result["files"].items():
        print(f"  {name}: {path}")


def cmd_build_db() -> None:
    report = pipeline.step_build_db()
    print(f"Database built and verified: {report['rows_transactions']:,} transactions, "
          f"{report['size_mib']:.1f} MiB, FK violations: {report['foreign_key_violations']}, integrity: {report['integrity_check']}")


def cmd_er_diagram() -> None:
    outputs = pipeline.step_er_diagram()
    print("ER diagram:", outputs["png"], "and", outputs["markdown"])


def cmd_sql() -> None:
    result = pipeline.step_sql()
    print(f"{result['n_queries']} queries executed; results saved as CSV in reports/sql_results/")
    print("SQL == Pandas cross-checks:", ", ".join(f"{k.split('_')[0]}:{'OK' if v else 'FAIL'}" for k, v in result["crosscheck_with_pandas"].items()))


def cmd_benchmark() -> None:
    result = pipeline.step_benchmark()
    from src.data.benchmark import storage_table

    print(storage_table(result["storage"]).to_string(index=False))
    print(pd.DataFrame(result["pandas_vs_sql"]["tasks"]).round(3).to_string(index=False))


def cmd_summary() -> None:
    from src.data.report import write_report

    path = write_report()
    print("Report section generated from the saved artefacts:", path)


def cmd_run_all() -> None:
    """The full pipeline in dependency order."""
    for name in ("download", "audit", "optimize", "split", "eda", "build-db", "er-diagram", "sql", "benchmark", "summary"):
        print(f"\n=== {name} ===")
        COMMANDS[name][1]()


COMMANDS: dict[str, tuple[str, Callable[[], None]]] = {
    "download": ("Download and verify creditcard.csv", cmd_download),
    "audit": ("Data-quality audit + memory before/after", cmd_audit),
    "optimize": ("Chunked CSV load with compact dtypes -> Parquet", cmd_optimize),
    "split": ("Add transaction_id and the chronological split", cmd_split),
    "eda": ("Exploratory analysis: tables and figures", cmd_eda),
    "build-db": ("Build and verify the SQLite database", cmd_build_db),
    "er-diagram": ("Draw the ER diagram from the database", cmd_er_diagram),
    "sql": ("Run all SQL queries, save results, cross-check with Pandas", cmd_sql),
    "benchmark": ("Benchmarks: CSV vs Parquet, Pandas vs SQL", cmd_benchmark),
    "summary": ("Generate docs/report_data_engineering.md from saved results", cmd_summary),
    "run-all": ("Run everything above in order", cmd_run_all),
}


# --------------------------------------------------------------------------- SQL query browser
def parse_params(pairs: list[str] | None) -> dict[str, object]:
    """['a=1', 'b=x'] -> {'a': 1, 'b': 'x'}; raises ValueError on malformed items."""
    from src.db.queries import _coerce

    params: dict[str, object] = {}
    for item in pairs or []:
        key, separator, value = item.partition("=")
        if not separator or not key:
            raise ValueError(f"parameter must look like name=value, got {item!r}")
        params[key.strip()] = _coerce(value.strip())
    return params


def cmd_query(name: str | None = None, params: dict | None = None, rows: int = 15) -> None:
    from src.db.queries import get_engine, list_queries, load_query, run_query

    if name is None:
        for query_name in list_queries():
            query = load_query(query_name)
            print(f"{query_name:42s} {query.title}")
        return
    query = load_query(name)
    print(f"{query.title}\n  question: {query.question}\n  uses: {', '.join(query.uses)}\n  default params: {query.params}\n")
    result = run_query(get_engine(), name, params)
    with pd.option_context("display.width", 200, "display.max_columns", 12):
        print(result.head(rows).to_string(index=False))
    print(f"\n({len(result)} rows x {result.shape[1]} columns loaded into a pandas DataFrame)")


# --------------------------------------------------------------------------- interactive menu
def run_menu(input_func: Callable[[str], str] = input) -> int:
    """Numbered console menu; loops until the user chooses 0 or closes the input."""
    items = list(COMMANDS.items())
    while True:
        print("\n=== Credit Card Fraud Detection: data engineering & SQL ===")
        for number, (name, (description, _)) in enumerate(items, start=1):
            print(f"  {number:2d} - {description}  [{name}]")
        print(f"  {len(items) + 1:2d} - Browse / run a SQL query")
        print("   0 - Exit")
        try:
            choice = input_func("Choose an option: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        try:
            number = int(choice)
        except ValueError:
            print(f"'{choice}' is not a number. Please enter 0-{len(items) + 1}.")
            continue

        try:
            if number == 0:
                print("Bye.")
                return 0
            elif 1 <= number <= len(items):
                items[number - 1][1][1]()
            elif number == len(items) + 1:
                cmd_query()
                name = input_func("Query name (e.g. q01_split_overview), empty to go back: ").strip()
                if name:
                    cmd_query(name)
            else:
                print(f"Option {number} does not exist.")
        except EXPECTED_ERRORS as exc:
            print(f"Problem: {exc}")


# --------------------------------------------------------------------------- entry point
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.data.cli", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command")
    for name, (description, _) in COMMANDS.items():
        sub.add_parser(name, help=description)
    query = sub.add_parser("query", help="list SQL queries, or run one and print the DataFrame")
    query.add_argument("name", nargs="?", help="query name, e.g. q01_split_overview (omit to list)")
    query.add_argument("--param", action="append", metavar="NAME=VALUE", help="override a query parameter")
    query.add_argument("--rows", type=int, default=15, help="rows to print")
    sub.add_parser("menu", help="interactive console menu (default without arguments)")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    try:
        if args.command in (None, "menu"):
            if args.command is None and not sys.stdin.isatty():
                build_parser().print_help()
                return 0
            return run_menu()
        if args.command == "query":
            cmd_query(args.name, parse_params(args.param), args.rows)
        else:
            COMMANDS[args.command][1]()
    except EXPECTED_ERRORS as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if not RAW_CSV_PATH.exists():
            print("Hint: run `python -m src.data.cli download` first.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
