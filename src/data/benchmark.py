"""Benchmark of storage / loading strategies: time, memory and file size.

Requirement 9 asks to compare at least two approaches with a baseline on quality, time
and memory.  For the data layer the candidates are:

    csv_float64             baseline: ``pd.read_csv`` with default dtypes
    csv_chunked_optimized   CSV read in chunks, every chunk compressed to compact dtypes
    parquet_float64         same dtypes as the baseline, columnar format
    parquet_optimized       columnar format + compact dtypes (what the project uses)
    parquet_optimized_3cols Parquet projection: only Time, Amount, Class

Each variant is executed in a **fresh Python process** so that the peak resident memory
(``ru_maxrss``) belongs to that variant alone.  The memory of an idle interpreter that has
imported pandas + pyarrow is measured too and subtracted ("peak above idle").
Time is the median of ``repeats`` runs; ``frame_mib`` is ``DataFrame.memory_usage(deep=True)``.

Usage:  python -m src.data.benchmark            (runs everything, writes the JSON report)
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet  # noqa: F401  (imported up-front so import cost is not charged to Parquet)

from src.config import (
    AMOUNT_COLUMN,
    BENCHMARK_JSON_PATH,
    INTERIM_DIR,
    RAW_CSV_PATH,
    ROOT_DIR,
    TARGET_COLUMN,
    TIME_COLUMN,
)
from src.data.io_utils import read_json, write_json
from src.data.loading import load_csv_chunked, read_raw_csv
from src.data.memory import memory_mib, optimize_dtypes
from src.data.storage import file_size_mib, load_parquet, save_parquet

logger = logging.getLogger(__name__)

BENCH_DIR = INTERIM_DIR / "benchmark"
PARQUET_FLOAT64 = BENCH_DIR / "creditcard_float64.parquet"
PARQUET_OPTIMIZED = BENCH_DIR / "creditcard_optimized.parquet"

VARIANTS: tuple[str, ...] = (
    "csv_float64",
    "csv_chunked_optimized",
    "parquet_float64",
    "parquet_optimized",
    "parquet_optimized_3cols",
)
BASELINE = "csv_float64"
THREE_COLUMNS = [TIME_COLUMN, AMOUNT_COLUMN, TARGET_COLUMN]


def _peak_rss_mib() -> float | None:
    """Peak resident memory of the current process in MiB (None where unsupported)."""
    try:
        import resource
    except ImportError:  # Windows
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1024**2 if sys.platform == "darwin" else peak / 1024  # bytes vs KiB


def _load_variant(name: str) -> pd.DataFrame:
    """The operation that is being timed for each variant."""
    if name == "csv_float64":
        return read_raw_csv(RAW_CSV_PATH)
    if name == "csv_chunked_optimized":
        return load_csv_chunked(RAW_CSV_PATH)
    if name == "parquet_float64":
        return load_parquet(PARQUET_FLOAT64)
    if name == "parquet_optimized":
        return load_parquet(PARQUET_OPTIMIZED)
    if name == "parquet_optimized_3cols":
        return load_parquet(PARQUET_OPTIMIZED, columns=THREE_COLUMNS)
    raise ValueError(f"unknown benchmark variant: {name}")


def run_child(name: str) -> dict:
    """Executed inside the child process: time one load and report memory."""
    if name == "idle":
        return {"variant": name, "seconds": 0.0, "peak_rss_mib": _peak_rss_mib(), "frame_mib": 0.0, "rows": 0}
    start = time.perf_counter()
    df = _load_variant(name)
    seconds = time.perf_counter() - start
    return {
        "variant": name,
        "seconds": seconds,
        "peak_rss_mib": _peak_rss_mib(),
        "frame_mib": memory_mib(df),
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
    }


def prepare_files(csv_path: Path = RAW_CSV_PATH) -> dict[str, float]:
    """Create the two Parquet files needed by the benchmark; return their sizes in MiB."""
    raw = read_raw_csv(csv_path)
    optimized = optimize_dtypes(raw)
    save_parquet(raw, PARQUET_FLOAT64, compression="zstd")
    save_parquet(optimized, PARQUET_OPTIMIZED, compression="zstd")

    snappy_path = BENCH_DIR / "creditcard_optimized_snappy.parquet"
    save_parquet(optimized, snappy_path, compression="snappy")
    sizes = {
        "csv": file_size_mib(csv_path),
        "parquet_float64_zstd": file_size_mib(PARQUET_FLOAT64),
        "parquet_optimized_zstd": file_size_mib(PARQUET_OPTIMIZED),
        "parquet_optimized_snappy": file_size_mib(snappy_path),
    }
    snappy_path.unlink(missing_ok=True)
    return sizes


def _spawn(variant: str, timeout: float = 600.0) -> dict:
    """Run one variant in a fresh interpreter and parse its JSON line."""
    completed = subprocess.run(
        [sys.executable, "-m", "src.data.benchmark", "--child", variant],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"benchmark child {variant!r} failed:\n{completed.stderr[-2000:]}")
    return json.loads(completed.stdout.strip().splitlines()[-1])


def run_storage_benchmark(repeats: int = 3, csv_path: Path = RAW_CSV_PATH) -> dict:
    """Run every variant ``repeats`` times in separate processes and summarise."""
    sizes = prepare_files(csv_path)
    idle_runs = [_spawn("idle")["peak_rss_mib"] for _ in range(repeats)]
    idle = statistics.median([v for v in idle_runs if v is not None]) if idle_runs[0] is not None else None

    results: list[dict] = []
    for variant in VARIANTS:  # baseline first
        runs = [_spawn(variant) for _ in range(repeats)]
        peaks = [r["peak_rss_mib"] for r in runs if r["peak_rss_mib"] is not None]
        peak = statistics.median(peaks) if peaks else None
        results.append(
            {
                "variant": variant,
                "seconds_median": statistics.median(r["seconds"] for r in runs),
                "seconds_all": [r["seconds"] for r in runs],
                "frame_mib": runs[0]["frame_mib"],
                "peak_rss_mib": peak,
                "peak_above_idle_mib": (peak - idle) if (peak is not None and idle is not None) else None,
                "rows": runs[0]["rows"],
                "columns": runs[0]["columns"],
            }
        )

    base = next(r for r in results if r["variant"] == BASELINE)
    for row in results:
        row["speedup_vs_baseline"] = base["seconds_median"] / row["seconds_median"]
        row["frame_memory_vs_baseline"] = row["frame_mib"] / base["frame_mib"] if base["frame_mib"] else None

    report = {
        "repeats": repeats,
        "platform": sys.platform,
        "idle_peak_rss_mib": idle,
        "file_sizes_mib": sizes,
        "variants": results,
        "notes": [
            "seconds = time of the load call only, median of repeats, each in a fresh process",
            "frame_mib = DataFrame.memory_usage(deep=True)",
            "peak_above_idle_mib = process peak RSS minus an idle interpreter with pandas + pyarrow imported",
        ],
    }
    _merge_into_report("storage_formats", report)
    return report


def _merge_into_report(key: str, value: dict, path: Path = BENCHMARK_JSON_PATH) -> None:
    """Store ``value`` under ``key`` in the shared benchmark JSON (keeps other keys)."""
    try:
        current = read_json(path)
    except (FileNotFoundError, ValueError):
        current = {}
    current[key] = value
    write_json(current, path)


def storage_table(report: dict) -> pd.DataFrame:
    """Compact table of the storage benchmark for printing / the report."""
    rows = [
        {
            "variant": r["variant"],
            "load_seconds": round(r["seconds_median"], 3),
            "speedup_x": round(r["speedup_vs_baseline"], 2),
            "frame_MiB": round(r["frame_mib"], 1),
            "peak_above_idle_MiB": None if r["peak_above_idle_mib"] is None else round(r["peak_above_idle_mib"], 1),
            "columns": r["columns"],
        }
        for r in report["variants"]
    ]
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--child", help="internal: run one variant and print JSON")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)

    if args.child:
        print(json.dumps(run_child(args.child)))
        return 0

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = run_storage_benchmark(repeats=args.repeats)
    print(storage_table(report).to_string(index=False))
    print("file sizes (MiB):", {k: round(v, 1) for k, v in report["file_sizes_mib"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
