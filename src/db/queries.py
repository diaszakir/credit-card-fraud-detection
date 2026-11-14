"""Run the analytical SQL queries and load their results into Pandas.

Every query lives in its own file ``src/db/sql/queries/qNN_name.sql``.  A header of ``-- key: value``
comment lines describes it (title, question, constructs used, scope) and may define default
parameters (``-- param: min_transactions=1000``); the body is plain SQL with named bind parameters.

    from src.db.queries import get_engine, run_query
    df = run_query(get_engine(), "q01_split_overview")          # -> pandas DataFrame
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import Engine, text

from src.config import SQL_RESULTS_DIR
from src.db.builder import SQL_DIR
from src.db.engine import get_engine  # noqa: F401  (re-exported for convenience)

logger = logging.getLogger(__name__)

QUERIES_DIR = SQL_DIR / "queries"
_HEADER_LINE = re.compile(r"^--\s*(\w+):\s*(.*?)\s*$")

# constructs the course requires, detected from the SQL text itself
REQUIRED_CONSTRUCTS: dict[str, re.Pattern] = {
    "JOIN": re.compile(r"\bJOIN\b", re.I),
    "GROUP BY": re.compile(r"\bGROUP\s+BY\b", re.I),
    "HAVING": re.compile(r"\bHAVING\b", re.I),
    "CTE (WITH)": re.compile(r"^\s*WITH\b", re.I | re.M),
    "subquery": re.compile(r"\(\s*SELECT\b", re.I),
    "window function (OVER)": re.compile(r"\bOVER\b", re.I),
}


@dataclass(frozen=True)
class Query:
    name: str
    title: str
    question: str
    uses: tuple[str, ...]
    scope: str
    note: str
    params: dict[str, object] = field(default_factory=dict)
    sql: str = ""


def _coerce(value: str) -> object:
    """'1000' -> 1000, '0.5' -> 0.5, anything else stays a string."""
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def parse_query_text(content: str, fallback_name: str = "unnamed") -> Query:
    """Split a query file into its metadata header and the SQL body."""
    meta: dict[str, str] = {}
    params: dict[str, object] = {}
    body_lines: list[str] = []
    in_header = True
    for line in content.splitlines():
        match = _HEADER_LINE.match(line) if in_header else None
        if match:
            key, value = match.groups()
            if key == "param":
                name, _, raw = value.partition("=")
                params[name.strip()] = _coerce(raw.strip())
            else:
                meta[key] = value
        else:
            if line.strip() and not line.startswith("--"):
                in_header = False
            body_lines.append(line)
    return Query(
        name=meta.get("name", fallback_name),
        title=meta.get("title", ""),
        question=meta.get("question", ""),
        uses=tuple(part.strip() for part in meta.get("uses", "").split(",") if part.strip()),
        scope=meta.get("scope", ""),
        note=meta.get("note", ""),
        params=params,
        sql="\n".join(body_lines).strip() + "\n",
    )


def list_queries(queries_dir: Path = QUERIES_DIR) -> list[str]:
    """Names of all available queries, in file order (q01, q02, ...)."""
    return sorted(p.stem for p in Path(queries_dir).glob("q*.sql"))


def load_query(name: str, queries_dir: Path = QUERIES_DIR) -> Query:
    path = Path(queries_dir) / f"{name}.sql"
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise KeyError(f"unknown query {name!r}; available: {list_queries(queries_dir)}") from exc
    return parse_query_text(content, fallback_name=name)


def run_query(engine: Engine, name: str, params: dict | None = None, queries_dir: Path = QUERIES_DIR) -> pd.DataFrame:
    """Execute one query and return its result as a DataFrame.

    ``params`` override the defaults declared in the file header.
    """
    query = load_query(name, queries_dir)
    statement = text(query.sql)
    wanted = set(statement.compile().params)
    merged = {**query.params, **(params or {})}
    missing = wanted - set(merged)
    if missing:
        raise ValueError(f"query {name!r} needs parameters {sorted(missing)}")
    with engine.connect() as connection:
        return pd.read_sql_query(statement, connection, params={k: merged[k] for k in wanted})


def run_all(engine: Engine, queries_dir: Path = QUERIES_DIR) -> dict[str, pd.DataFrame]:
    """Run every query; returns ``{name: DataFrame}`` in file order."""
    results: dict[str, pd.DataFrame] = {}
    for name in list_queries(queries_dir):
        results[name] = run_query(engine, name, queries_dir=queries_dir)
        logger.info("%s -> %d rows x %d columns", name, *results[name].shape)
    return results


def save_results(results: dict[str, pd.DataFrame], out_dir: Path = SQL_RESULTS_DIR) -> list[Path]:
    """Write every result as CSV (and return the paths)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, frame in results.items():
        path = out_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths.append(path)
    return paths


def describe_queries(queries_dir: Path = QUERIES_DIR) -> pd.DataFrame:
    """Catalogue of the queries: title, question, constructs declared in the header, and detected."""
    rows = []
    for name in list_queries(queries_dir):
        query = load_query(name, queries_dir)
        rows.append(
            {
                "name": name,
                "title": query.title,
                "question": query.question,
                "declared_constructs": ", ".join(query.uses),
                "detected_constructs": ", ".join(detected_constructs(query.sql)),
                "scope": query.scope,
            }
        )
    return pd.DataFrame(rows)


def detected_constructs(sql: str) -> list[str]:
    """Which of the required constructs appear in the SQL (comments stripped first)."""
    code = "\n".join(line for line in sql.splitlines() if not line.lstrip().startswith("--"))
    return [name for name, pattern in REQUIRED_CONSTRUCTS.items() if pattern.search(code)]


def construct_coverage(queries_dir: Path = QUERIES_DIR) -> dict[str, list[str]]:
    """For every required construct, the queries that use it (evidence for the requirements)."""
    coverage: dict[str, list[str]] = {name: [] for name in REQUIRED_CONSTRUCTS}
    for name in list_queries(queries_dir):
        for construct in detected_constructs(load_query(name, queries_dir).sql):
            coverage[construct].append(name)
    return coverage


def feature_separation(moments: pd.DataFrame) -> pd.DataFrame:
    """Rank V features by the standardised mean difference between fraud and legitimate rows.

    ``moments`` is the result of ``q10_class_feature_moments`` (one row per class).  For every
    feature:  d = (mean_fraud - mean_legit) / sqrt((var_fraud + var_legit) / 2).
    """
    by_class = moments.set_index("class_label")
    if not {"legitimate", "fraud"} <= set(by_class.index):
        raise ValueError("moments must contain both 'legitimate' and 'fraud' rows")
    features = sorted(
        (c[len("mean_") :] for c in by_class.columns if c.startswith("mean_")), key=lambda c: int(c[1:])
    )
    rows = []
    for feature in features:
        mean_gap = by_class.loc["fraud", f"mean_{feature}"] - by_class.loc["legitimate", f"mean_{feature}"]
        pooled_sd = np.sqrt((by_class.loc["fraud", f"var_{feature}"] + by_class.loc["legitimate", f"var_{feature}"]) / 2)
        rows.append({"feature": feature.upper(), "mean_difference": mean_gap, "standardised_difference": mean_gap / pooled_sd})
    table = pd.DataFrame(rows)
    table["abs_standardised_difference"] = table["standardised_difference"].abs()
    return table.sort_values("abs_standardised_difference", ascending=False, ignore_index=True)
