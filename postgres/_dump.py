"""Whole-store export and import for ``hermes backup`` / ``hermes import``, through ``pg_dump`` and ``psql``.

An export is one plain SQL file: a header line naming the schemas it holds, then ``pg_dump``'s output of the
profile's schema, its file stores' schemas and, for the install's root profile, the install's shared group-chat schema. An import loads it in one
transaction and renames each schema to the name this machine derives for the profile (a restore on another machine or
another home path derives other names). It refuses when the profile's store already holds sessions; an existing empty
store (one a Hermes start created) is replaced.
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping

import psycopg
from psycopg import sql

from persistence import ProfileScope, StoreUnavailable

from . import _schema
from ._settings import read_settings

_HEADER = "-- hermes-postgres-store "
_SEARCH = ("/Library/PostgreSQL/*/bin", "/opt/homebrew/opt/postgresql*/bin", "/opt/homebrew/bin", "/usr/local/bin",
           "/Applications/Postgres.app/Contents/Versions/latest/bin", "C:/Program Files/PostgreSQL/*/bin")


def _tool(name: str, settings: Mapping[str, Any]) -> str:
    """The ``pg_dump`` / ``psql`` executable: ``persistence.pg_bin_dir``, then ``PATH``, then the usual install
    folders (newest version first)."""
    configured = settings.get("pg_bin_dir")
    candidates = [Path(str(configured)).expanduser() / name] if configured else []
    found = shutil.which(name)
    if found:
        candidates.append(Path(found))
    for pattern in _SEARCH:
        candidates.extend(Path(match) / name for match in sorted(glob.glob(pattern), reverse=True))
    for candidate in candidates:
        for path in (candidate, candidate.with_suffix(".exe")):
            if path.is_file() and os.access(path, os.X_OK):
                return str(path)
    raise StoreUnavailable(f"postgres persistence: {name} was not found; install the PostgreSQL client tools or set "
                           "persistence.pg_bin_dir to the folder holding them")


def _schemas(scope: ProfileScope, settings: Mapping[str, Any], conn: psycopg.Connection,
             file_stores: tuple[str, ...] = ()) -> dict[str, str]:
    """``{role: schema}`` this profile's export holds: its own store, each of its file stores (``file:<store>``:
    projects, cron, kanban boards, ...), and the install's shared group-chat store for the root profile. File stores
    are found in the database; ``file_stores`` adds the names an export brought, so an import maps them too."""
    from ._provider import compat_schema
    from .gateway._hosted_store import shared_state_path
    import hashlib

    config = read_settings(settings)
    profile = _schema.schema_name(config.schema_prefix, scope.hermes_home_key)
    names = {"profile": profile}
    if Path(scope.profile_dir).parent.name != "profiles":
        path = str(shared_state_path(Path(scope.profile_dir)))
        names["shared"] = f"{config.schema_prefix}shared_{hashlib.sha256(path.encode('utf-8')).hexdigest()[:10]}"
    prefix = profile + "__"
    found = [row[0][len(prefix):] for row in conn.execute(
        "SELECT nspname FROM pg_namespace WHERE starts_with(nspname, %s)", (prefix,)).fetchall()]
    for store in found:  # as named (a long store name is already cut to fit, with its hash)
        names["file:" + store] = prefix + store
    for store in file_stores:
        names.setdefault("file:" + store, compat_schema(profile, store) if len(prefix + store) > 63 else prefix + store)
    return names


def _run(command: list[str], dsn: str) -> None:
    done = subprocess.run(command, capture_output=True, text=True, env={**os.environ, "PGCONNECT_TIMEOUT": "10"})
    if done.returncode != 0:
        raise StoreUnavailable(f"postgres persistence: {Path(command[0]).name} failed: "
                               f"{(done.stderr or done.stdout).strip()[-2000:]}")


def export_store(scope: ProfileScope, settings: Mapping[str, Any], destination: Path) -> bool:
    config = read_settings(settings)
    with psycopg.connect(config.dsn, connect_timeout=10, autocommit=True) as conn:
        present = {role: name for role, name in _schemas(scope, settings, conn).items()
                   if _schema.schema_exists(conn, name)}
    if not present:
        return False
    body = destination.with_name(destination.name + ".body")
    try:
        _run([_tool("pg_dump", settings), "--dbname", config.dsn, "--no-owner", "--no-privileges", "--file", str(body),
              *(arg for name in present.values() for arg in ("--schema", name))], config.dsn)
        with open(destination, "w", encoding="utf-8") as out, open(body, encoding="utf-8") as dump:
            out.write(_HEADER + json.dumps(present, sort_keys=True) + "\n")
            shutil.copyfileobj(dump, out)
    finally:
        body.unlink(missing_ok=True)
    return True


def _header(source: Path) -> dict[str, str]:
    with open(source, encoding="utf-8") as fh:
        first = fh.readline()
    if not first.startswith(_HEADER):
        raise StoreUnavailable(f"{source} is not a Hermes Postgres store export")
    return json.loads(first[len(_HEADER):])


def _retarget(source: Path, destination: Path, renames: Mapping[str, str]) -> None:
    """Copy the export with each schema name replaced in its SQL statements and ``COPY`` headers; the rows of each
    ``COPY ... FROM stdin`` block (up to its ``\\.`` line) are copied untouched, so stored text never changes."""
    pattern = re.compile(r"\b(" + "|".join(re.escape(old) for old in renames) + r")\b")
    in_rows = False
    with open(source, encoding="utf-8") as src, open(destination, "w", encoding="utf-8") as out:
        src.readline()  # the header line
        for line in src:
            if in_rows:
                in_rows = line != "\\.\n"
                out.write(line)
                continue
            if renames:
                line = pattern.sub(lambda match: renames[match.group(1)], line)
            in_rows = line.startswith("COPY ") and line.rstrip().endswith("FROM stdin;")
            out.write(line)


def import_store(scope: ProfileScope, settings: Mapping[str, Any], source: Path) -> None:
    config = read_settings(settings)
    exported = _header(source)
    with psycopg.connect(config.dsn, connect_timeout=10, autocommit=True) as conn:
        wanted = _schemas(scope, settings, conn, tuple(role[5:] for role in exported if role.startswith("file:")))
        renames = {old: wanted[role] for role, old in exported.items() if role in wanted and old != wanted[role]}
        targets = {role: wanted[role] for role in exported if role in wanted}
        for role, target in targets.items():
            if _schema.schema_exists(conn, target) and _holds_data(conn, target, role):
                raise StoreUnavailable(f"postgres persistence: schema {target} already holds data; import refused so "
                                       "a restore never overwrites a live store")
    loadable = source.with_name(source.name + ".load")
    try:
        _retarget(source, loadable, renames)
        # An empty store a Hermes start created is replaced by the export, in the load's own transaction.
        drops = "".join(f'DROP SCHEMA IF EXISTS "{target}" CASCADE;\n' for target in targets.values())
        _run([_tool("psql", settings), "--dbname", config.dsn, "--quiet", "--no-psqlrc", "-v", "ON_ERROR_STOP=1",
              "--single-transaction", "--command", drops, "--file", str(loadable)], config.dsn)
    finally:
        loadable.unlink(missing_ok=True)


def _holds_data(conn: psycopg.Connection, schema: str, role: str) -> bool:
    if role.startswith("file:"):  # any row in any of the store's own tables
        tables = [row[0] for row in conn.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = %s AND tablename NOT IN ('sqlite_master', 'hermes_compat_meta')",
            (schema,)).fetchall()]
        return any(conn.execute(sql.SQL("SELECT 1 FROM {}.{} LIMIT 1").format(sql.Identifier(schema), sql.Identifier(t)))
                   .fetchone() is not None for t in tables)
    table = "sessions" if role == "profile" else "hosted_room_rows"
    found = conn.execute("SELECT 1 FROM information_schema.tables WHERE table_schema = %s AND table_name = %s",
                         (schema, table)).fetchone()
    if found is None:
        return False
    return conn.execute(sql.SQL("SELECT 1 FROM {}.{} LIMIT 1").format(sql.Identifier(schema), sql.Identifier(table))).fetchone() is not None
