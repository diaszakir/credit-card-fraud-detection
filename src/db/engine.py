"""SQLAlchemy engine for the project's SQLite database."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import Engine, create_engine, event

from src.config import SQLITE_PATH


def get_engine(path: Path = SQLITE_PATH, must_exist: bool = True) -> Engine:
    """Engine for the SQLite file at ``path`` with foreign-key enforcement switched on.

    ``must_exist=True`` raises a clear error instead of silently creating an empty database
    when the file has not been built yet.
    """
    path = Path(path)
    if must_exist and not path.is_file():
        raise FileNotFoundError(f"{path} not found. Build it with `python -m src.data.cli build-db`.")
    engine = create_engine(f"sqlite:///{path}")

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _record) -> None:  # SQLite default is OFF
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.close()

    return engine
