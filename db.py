"""Import and query the student dataset in SQLite.

Usage:
    python db.py import                          # data/students.csv -> data/students.db
    python db.py import --csv mine.csv --db mine.db
    python db.py tables --db data/students.db
    python db.py query --db data/students.db --sql "SELECT * FROM students LIMIT 5"

The database is plain SQLite, so any client (DB Browser, DBeaver, the sqlite3
shell) can open it as well.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

from student_perf.config import DEFAULT_DATA_PATH, DEFAULT_DB_PATH, STUDENTS_TABLE
from student_perf.data import export_dataset_to_db, load_dataset
from student_perf.database import list_tables, read_table, resolve_database_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import the student dataset into SQLite and run read-only queries."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    importer = subparsers.add_parser("import", help="Load a CSV file into a table.")
    importer.add_argument("--csv", type=Path, default=DEFAULT_DATA_PATH)
    importer.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    importer.add_argument("--table", default=STUDENTS_TABLE)
    importer.add_argument(
        "--if-exists",
        choices=("replace", "append", "fail"),
        default="replace",
        help="What to do when the table already exists (default: replace).",
    )

    tables = subparsers.add_parser("tables", help="List the tables in a database.")
    tables.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)

    query = subparsers.add_parser("query", help="Run a read-only SQL query.")
    query.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    query.add_argument("--sql", required=True, help="SELECT statement to run.")
    query.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Rows to display; 0 shows everything (default: 20).",
    )

    return parser.parse_args(argv)


def run_import(args: argparse.Namespace) -> int:
    if args.csv == DEFAULT_DATA_PATH:
        frame = load_dataset(path=args.csv)
    else:
        frame = pd.read_csv(args.csv)
    path = export_dataset_to_db(
        frame, args.db, table=args.table, if_exists=args.if_exists
    )
    print(
        f"Imported {len(frame):,} rows from {args.csv} into "
        f"'{args.table}' at {path}"
    )
    return 0


def run_tables(args: argparse.Namespace) -> int:
    tables = list_tables(args.db)
    if not tables:
        print(f"No tables in {resolve_database_path(args.db)}")
        return 0
    for name in tables:
        print(name)
    return 0


def _strip_semicolons(sql: str) -> str:
    return sql.strip().rstrip(";").rstrip()


def run_query(args: argparse.Namespace) -> int:
    sql = _strip_semicolons(args.sql)
    if args.limit > 0:
        sql = f"SELECT * FROM (\n{sql}\n) LIMIT {int(args.limit)}"
    frame = read_table(args.db, query=sql)
    if frame.empty:
        print("No rows.")
        return 0
    print(frame.to_string(index=False))
    if args.limit > 0:
        print(f"\n{len(frame)} row(s) shown (limit {args.limit}).")
    else:
        print(f"\n{len(frame)} row(s).")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "import":
            return run_import(args)
        if args.command == "tables":
            return run_tables(args)
        return run_query(args)
    except (
        FileNotFoundError,
        ValueError,
        sqlite3.Error,
        pd.errors.DatabaseError,
    ) as error:
        print(f"Error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
