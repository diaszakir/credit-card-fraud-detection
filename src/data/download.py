"""Download and verify ``creditcard.csv``.

The official source is Kaggle (``mlg-ulb/creditcardfraud``, needs a login).  For a
login-free, reproducible setup the same file is fetched from a public mirror listed in
``src.config.DATA_URLS`` and then verified: column names, row count (284 807) and number
of frauds (492).  The SHA-256 is recorded in the audit so any copy can be compared.
"""

from __future__ import annotations

import hashlib
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from src.config import (
    DATA_URLS,
    EXPECTED_FRAUD,
    EXPECTED_ROWS,
    RAW_COLUMNS,
    RAW_CSV_PATH,
    TARGET_COLUMN,
)

logger = logging.getLogger(__name__)

_CHUNK = 1024 * 1024  # 1 MiB

MANUAL_HELP = (
    "Download it manually from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
    f"and place creditcard.csv at {RAW_CSV_PATH}"
)


class DownloadError(RuntimeError):
    """The dataset could not be downloaded or failed verification."""


def sha256_of(path: Path, chunk_size: int = _CHUNK) -> str:
    """SHA-256 of a file, computed in chunks (the file is ~150 MB)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_dataset(
    path: Path = RAW_CSV_PATH,
    expected_rows: int | None = EXPECTED_ROWS,
    expected_fraud: int | None = EXPECTED_FRAUD,
) -> dict:
    """Check header, row count and fraud count.  Never raises for data problems.

    Returns ``{"ok": bool, "problems": [...], "rows": int, "fraud": int, "sha256": str, ...}``.
    """
    path = Path(path)
    result: dict = {"path": str(path), "ok": False, "problems": []}
    if not path.is_file():
        result["problems"].append(f"file not found: {path}")
        return result
    try:
        header = list(pd.read_csv(path, nrows=0).columns)
        if header != RAW_COLUMNS:
            result["problems"].append(f"unexpected columns: {header[:5]}...")
        target = pd.read_csv(path, usecols=[TARGET_COLUMN])[TARGET_COLUMN]
    except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError, KeyError) as exc:
        result["problems"].append(f"cannot parse CSV: {exc}")
        return result

    rows, fraud = int(len(target)), int((target == 1).sum())
    result.update(rows=rows, fraud=fraud, size_bytes=path.stat().st_size)
    if expected_rows is not None and rows != expected_rows:
        result["problems"].append(f"expected {expected_rows} rows, found {rows}")
    if expected_fraud is not None and fraud != expected_fraud:
        result["problems"].append(f"expected {expected_fraud} frauds, found {fraud}")
    result["sha256"] = sha256_of(path)
    result["ok"] = not result["problems"]
    return result


def download_dataset(
    dest: Path = RAW_CSV_PATH,
    urls: list[str] | None = None,
    force: bool = False,
    timeout: float = 60.0,
    expected_rows: int | None = EXPECTED_ROWS,
    expected_fraud: int | None = EXPECTED_FRAUD,
) -> Path:
    """Fetch the CSV (if needed) and verify it.  Returns the path of the verified file.

    Raises ``DownloadError`` with manual instructions if every source fails.
    """
    dest = Path(dest)
    sources = list(DATA_URLS if urls is None else urls)

    if dest.is_file() and not force:
        check = verify_dataset(dest, expected_rows, expected_fraud)
        if check["ok"]:
            logger.info("dataset already present and verified: %s", dest)
            return dest
        logger.warning("existing file failed verification (%s); re-downloading", check["problems"])

    dest.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for url in sources:  # try each mirror in turn
        part = dest.with_suffix(dest.suffix + ".part")
        try:
            logger.info("downloading %s", url)
            with urllib.request.urlopen(url, timeout=timeout) as response, part.open("wb") as out:
                for block in iter(lambda: response.read(_CHUNK), b""):
                    out.write(block)
            os.replace(part, dest)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            errors.append(f"{url}: {exc}")
            part.unlink(missing_ok=True)
            continue

        check = verify_dataset(dest, expected_rows, expected_fraud)
        if check["ok"]:
            return dest
        errors.append(f"{url}: verification failed: {check['problems']}")
        dest.unlink(missing_ok=True)

    raise DownloadError("Could not obtain the dataset.\n  " + "\n  ".join(errors) + f"\n{MANUAL_HELP}")
