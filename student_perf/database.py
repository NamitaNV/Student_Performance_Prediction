"""SQLite helpers: import the dataset, read tables, run read-only queries.

The project only needs the standard library here: ``sqlite3`` plus Pandas'
``read_sql_query`` / ``to_sql``. Databases can be given as a plain path or as a
``sqlite:///path/to/file.db`` URL.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

from student_perf.config import STUDENTS_TABLE


def resolve_database_path(database: str | Path) -> Path:
    """Accept a filesystem path or a ``sqlite:///`` URL and return the path."""
    text = str(database)
    for prefix in ("sqlite:///", "sqlite://"):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    return Path(text).expanduser()


def _quote_identifier(identifier: str) -> str:
    """Quote a table/column name for SQLite."""
    return '"' + identifier.replace('"', '""') + '"'


def connect(database: str | Path, read_only: bool = False) -> sqlite3.Connection:
    """Open a connection, creating parent directories for writable databases."""
    path = resolve_database_path(database)
    if read_only:
        if not path.exists():
            raise FileNotFoundError(f"SQLite database not found: {path}")
        return sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(path)


def list_tables(database: str | Path) -> list[str]:
    """Return the user tables in a database, sorted by name."""
    with closing(connect(database, read_only=True)) as connection:
        rows = connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        ).fetchall()
    return [name for (name,) in rows]


def write_table(
    frame: pd.DataFrame,
    database: str | Path,
    table: str = STUDENTS_TABLE,
    if_exists: str = "replace",
) -> Path:
    """Write a frame to a SQLite table and return the database path."""
    path = resolve_database_path(database)
    with closing(connect(path)) as connection:
        frame.to_sql(table, connection, if_exists=if_exists, index=False)
        connection.commit()
    return path


def read_table(
    database: str | Path,
    table: str = STUDENTS_TABLE,
    query: str | None = None,
    params: object = None,
) -> pd.DataFrame:
    """Read a table, or the result of a custom read-only SQL query."""
    with closing(connect(database, read_only=True)) as connection:
        if query is None:
            available = [
                name
                for (name,) in connection.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
                    "ORDER BY name"
                ).fetchall()
            ]
            if table not in available:
                raise ValueError(
                    f"Table '{table}' not found in {resolve_database_path(database)}. "
                    f"Available tables: {', '.join(available) or '(none)'}"
                )
            query = f"SELECT * FROM {_quote_identifier(table)}"
        return pd.read_sql_query(query, connection, params=params)
