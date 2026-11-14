"""Small file helpers shared by the data-engineering modules (JSON in / out)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy / pandas / pathlib objects into plain JSON types."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return to_jsonable(obj.item())
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return to_jsonable(obj.to_dict(orient="records"))
    if isinstance(obj, pd.Series):
        return to_jsonable(obj.to_dict())
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None  # JSON has no NaN / Infinity
    return obj


def write_json(obj: Any, path: Path) -> Path:
    """Write ``obj`` as pretty UTF-8 JSON, creating parent folders."""
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(to_jsonable(obj), fh, indent=2, ensure_ascii=False)
            fh.write("\n")
    except OSError as exc:
        raise OSError(f"Cannot write JSON to {path}: {exc}") from exc
    logger.info("wrote %s", path)
    return path


def read_json(path: Path) -> Any:
    """Read a JSON file; raise a clear error if it is missing or malformed."""
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"JSON file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in {path}: {exc}") from exc
