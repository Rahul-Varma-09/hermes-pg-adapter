"""Copy a Postgres-pinned profile's SQLite file stores (projects.db, cron/executions.db, ...) into Postgres.

Run it with Hermes' own Python, from the Hermes checkout (``source ./activate``), with ``HERMES_PG_DSN`` set::

    python ~/Documents/Projects/hermes-adapter/migrate_file_stores.py --home ~/.hermes projects.db cron/executions.db
    python ~/Documents/Projects/hermes-adapter/migrate_file_stores.py --home ~/.hermes --commit projects.db cron/executions.db

Each named file is opened read-only. Its tables and indexes are created through the provider's SQLite-compatible
connection from the file's own ``sqlite_master`` text (so the store sees the schema it wrote), every row is copied,
and every table is compared row by row; the whole store is one transaction. A dry run copies into a scratch schema and
drops it; ``--commit`` copies into the profile's store, which must hold no tables yet. The SQLite files are never
changed; ``--retire`` (with ``--commit``) moves each migrated file and its ``-wal``/``-shm`` into
``backups/sqlite-retired-<time>/`` afterwards.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any


def _normalize(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, memoryview):
        return bytes(value)
    if isinstance(value, float) and value.is_integer():
        return value
    return value


def _rows(conn: Any, table: str) -> list[tuple]:
    return sorted((tuple(_normalize(v) for v in row) for row in conn.execute(f'SELECT * FROM "{table}"').fetchall()),
                  key=lambda row: tuple((v is None, str(type(v)), v if v is not None else 0) for v in row))


def copy_store(src: sqlite3.Connection, dst: Any) -> dict[str, int]:
    """Create ``src``'s tables and indexes on ``dst`` and copy every row, inside ``dst``'s open transaction."""
    counts: dict[str, int] = {}
    objects = src.execute("SELECT type, name, tbl_name, sql FROM sqlite_master WHERE sql IS NOT NULL "
                          "AND name NOT LIKE 'sqlite_%' ORDER BY CASE type WHEN 'table' THEN 0 ELSE 1 END, rowid").fetchall()
    for kind, name, _table, ddl in objects:
        if kind in ("table", "index"):
            dst.execute(ddl)
        elif kind == "trigger" or kind == "view":
            raise SystemExit(f"{kind} {name}: not copied by this tool yet; refusing rather than dropping it")
    for kind, name, _table, _ddl in objects:
        if kind != "table":
            continue
        columns = [row[1] for row in src.execute(f'PRAGMA table_info("{name}")')]
        names = ", ".join(f'"{c}"' for c in columns)
        marks = ", ".join("?" for _ in columns)
        # In rowid order, so the store's own rowid (assigned as rows arrive) keeps SQLite's order.
        try:
            rows = src.execute(f'SELECT {names} FROM "{name}" ORDER BY rowid').fetchall()
        except sqlite3.OperationalError:  # a WITHOUT ROWID table
            rows = src.execute(f'SELECT {names} FROM "{name}"').fetchall()
        if rows:
            dst.executemany(f'INSERT INTO "{name}" ({names}) VALUES ({marks})', [tuple(r) for r in rows])
        counts[name] = len(rows)
    return counts


def compare(src: sqlite3.Connection, dst: Any, tables: list[str]) -> list[str]:
    problems = []
    for table in tables:
        before, after = _rows(src, table), _rows(dst, table)
        if before != after:
            problems.append(f"{table}: {len(before)} rows in SQLite, {len(after)} in Postgres, contents differ")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("stores", nargs="+", help="store files relative to the home, e.g. projects.db cron/executions.db")
    parser.add_argument("--home", type=Path, default=Path("~/.hermes"))
    parser.add_argument("--commit", action="store_true")
    parser.add_argument("--retire", action="store_true", help="with --commit: move migrated files into backups/")
    args = parser.parse_args()

    import persistence_assembly as pa

    home = args.home.expanduser().resolve()
    scope = pa.profile_scope(home=home)
    provider_name = pa._provider_for_home(scope.profile_dir)
    if provider_name == "sqlite":
        print(f"{home} uses SQLite; pin it to a provider first")
        return 2
    provider = pa._plugin_provider(provider_name)
    settings = pa._persistence_section(scope.profile_dir)
    status = 0
    migrated: list[Path] = []
    for relative in args.stores:
        path = home / relative
        if not path.is_file():
            print(f"{relative}: no file, nothing to migrate")
            continue
        store = pa.compat_store_name(home, path)
        target = store if args.commit else store + "_dryrun"
        src = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        dst = provider.open_compat_store(scope, settings, target)
        try:
            if dst.execute("SELECT count(*) FROM sqlite_master WHERE type = 'table'").fetchone()[0]:
                print(f"{relative}: the Postgres store {target!r} already holds tables; refusing")
                status = 1
                continue
            dst.execute("BEGIN")
            counts = copy_store(src, dst)
            problems = compare(src, dst, list(counts))
            if problems:
                dst.rollback()
                print(f"{relative}: verification failed, nothing kept:\n  " + "\n  ".join(problems))
                status = 1
                continue
            dst.commit()
            print(f"{relative} -> {'store' if args.commit else 'dry-run store'} {target!r}: "
                  + ", ".join(f"{t} {n}" for t, n in counts.items()) + "; every row verified")
            if args.commit:
                migrated.append(path)
        finally:
            schema = dst._schema
            dst.close()
            src.close()
            if not args.commit:
                import psycopg
                from psycopg import sql

                with psycopg.connect(os.environ["HERMES_PG_DSN"], autocommit=True) as conn:
                    conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema)))
    if args.retire and migrated:
        folder = home / "backups" / time.strftime("sqlite-retired-%Y%m%d-%H%M%S")
        for path in migrated:
            destination = folder / path.relative_to(home).parent
            destination.mkdir(parents=True, exist_ok=True)
            for suffix in ("", "-wal", "-shm"):
                part = path.with_name(path.name + suffix)
                if part.exists():
                    shutil.move(str(part), str(destination / part.name))
        print(f"retired into {folder}")
    return status


if __name__ == "__main__":
    sys.exit(main())
