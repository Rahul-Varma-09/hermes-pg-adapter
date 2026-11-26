"""A ``sqlite3.Connection`` look-alike over Postgres, for Hermes' file stores (projects, cron history, kanban, ...).

Each store is one Postgres schema. The connection runs in autocommit and keeps SQLite's transaction rules itself:
``BEGIN`` (any kind) opens a transaction that takes the store's advisory write lock, as SQLite's single writer would;
the ``sqlite3`` module's implicit transaction opens the same way before the first INSERT/UPDATE/DELETE/REPLACE;
``commit()``, ``rollback()`` and ``with conn:`` close it. Inside a transaction every statement runs in a savepoint, so
a failed statement leaves the transaction usable, as in SQLite. Parameters go to the server untyped (SQLite compares
loosely), results come back as SQLite returns them (integers, floats, text, bytes), and errors are ``sqlite3``'s
exception classes with SQLite's wording where callers match on it.

``sqlite_master`` is a real table of the schema: each CREATE through the connection records its original SQLite text
there, so a store that reads its own DDL back (to decide an upgrade) sees what it wrote.
"""

from __future__ import annotations

import decimal
import hashlib
import re
import sqlite3
import threading
import weakref
import zlib
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Sequence

import psycopg
from psycopg import errors as pg_errors, sql as pg_sql
from psycopg.adapt import Dumper

from .translate import Statement, Token, split_statements, translate

_IMPLICIT_BEGIN = {"INSERT", "UPDATE", "DELETE", "REPLACE"}

_HELPERS = r"""
CREATE OR REPLACE FUNCTION hermes_gc_step(state text, value anyelement) RETURNS text LANGUAGE sql IMMUTABLE AS $$
  SELECT CASE WHEN value IS NULL THEN state WHEN state IS NULL THEN value::text ELSE state || ',' || value::text END $$;
CREATE TABLE IF NOT EXISTS sqlite_master (type text, name text, tbl_name text, rootpage bigint DEFAULT 0, sql text);
CREATE TABLE IF NOT EXISTS hermes_compat_meta (key text PRIMARY KEY, value text);

CREATE OR REPLACE FUNCTION hermes_ts(value text) RETURNS timestamp LANGUAGE plpgsql VOLATILE AS $$
BEGIN
  IF value IS NULL THEN RETURN NULL; END IF;
  IF value ~ '^\s*-?\d+(\.\d+)?\s*$' THEN RETURN to_timestamp(value::double precision) AT TIME ZONE 'UTC'; END IF;
  IF lower(value) = 'now' THEN RETURN clock_timestamp() AT TIME ZONE 'UTC'; END IF;
  RETURN (value::timestamptz) AT TIME ZONE 'UTC';
EXCEPTION WHEN others THEN RETURN NULL;
END $$;

CREATE OR REPLACE FUNCTION hermes_ts_mod(t timestamp, modifiers text[]) RETURNS timestamp LANGUAGE plpgsql STABLE AS $$
DECLARE m text; n double precision; unit text;
BEGIN
  IF t IS NULL THEN RETURN NULL; END IF;
  FOREACH m IN ARRAY coalesce(modifiers, ARRAY[]::text[]) LOOP
    m := lower(btrim(m));
    IF m IN ('utc', 'localtime', 'unixepoch', 'auto', 'subsec', 'subsecond') THEN CONTINUE; END IF;
    IF m = 'start of day' THEN t := date_trunc('day', t);
    ELSIF m = 'start of month' THEN t := date_trunc('month', t);
    ELSIF m = 'start of year' THEN t := date_trunc('year', t);
    ELSE
      n := split_part(m, ' ', 1)::double precision;
      unit := rtrim(split_part(m, ' ', 2), 's');
      t := t + make_interval(secs => CASE unit WHEN 'second' THEN n WHEN 'minute' THEN n * 60 WHEN 'hour' THEN n * 3600
                                                WHEN 'day' THEN n * 86400 ELSE 0 END)
             + make_interval(months => CASE unit WHEN 'month' THEN n::int WHEN 'year' THEN n::int * 12 ELSE 0 END);
    END IF;
  END LOOP;
  RETURN t;
EXCEPTION WHEN others THEN RETURN NULL;
END $$;

CREATE OR REPLACE FUNCTION hermes_julianday(value text, VARIADIC modifiers text[] DEFAULT ARRAY[]::text[])
RETURNS double precision LANGUAGE sql VOLATILE AS $$
  SELECT extract(epoch FROM hermes_ts_mod(hermes_ts(value), modifiers))::double precision / 86400.0 + 2440587.5 $$;
CREATE OR REPLACE FUNCTION hermes_unixepoch(value text DEFAULT 'now', VARIADIC modifiers text[] DEFAULT ARRAY[]::text[])
RETURNS bigint LANGUAGE sql VOLATILE AS $$
  SELECT floor(extract(epoch FROM hermes_ts_mod(hermes_ts(value), modifiers)))::bigint $$;
CREATE OR REPLACE FUNCTION hermes_datetime(value text DEFAULT 'now', VARIADIC modifiers text[] DEFAULT ARRAY[]::text[])
RETURNS text LANGUAGE sql VOLATILE AS $$
  SELECT to_char(hermes_ts_mod(hermes_ts(value), modifiers), 'YYYY-MM-DD HH24:MI:SS') $$;
CREATE OR REPLACE FUNCTION hermes_date(value text DEFAULT 'now', VARIADIC modifiers text[] DEFAULT ARRAY[]::text[])
RETURNS text LANGUAGE sql VOLATILE AS $$
  SELECT to_char(hermes_ts_mod(hermes_ts(value), modifiers), 'YYYY-MM-DD') $$;
CREATE OR REPLACE FUNCTION hermes_time(value text DEFAULT 'now', VARIADIC modifiers text[] DEFAULT ARRAY[]::text[])
RETURNS text LANGUAGE sql VOLATILE AS $$
  SELECT to_char(hermes_ts_mod(hermes_ts(value), modifiers), 'HH24:MI:SS') $$;
CREATE OR REPLACE FUNCTION hermes_julianday(value double precision, VARIADIC modifiers text[])
RETURNS double precision LANGUAGE sql VOLATILE AS $$ SELECT hermes_julianday(value::text, VARIADIC modifiers) $$;
CREATE OR REPLACE FUNCTION hermes_unixepoch(value double precision, VARIADIC modifiers text[])
RETURNS bigint LANGUAGE sql VOLATILE AS $$ SELECT hermes_unixepoch(value::text, VARIADIC modifiers) $$;
CREATE OR REPLACE FUNCTION hermes_datetime(value double precision, VARIADIC modifiers text[])
RETURNS text LANGUAGE sql VOLATILE AS $$ SELECT hermes_datetime(value::text, VARIADIC modifiers) $$;
CREATE OR REPLACE FUNCTION hermes_date(value double precision, VARIADIC modifiers text[])
RETURNS text LANGUAGE sql VOLATILE AS $$ SELECT hermes_date(value::text, VARIADIC modifiers) $$;
CREATE OR REPLACE FUNCTION hermes_time(value double precision, VARIADIC modifiers text[])
RETURNS text LANGUAGE sql VOLATILE AS $$ SELECT hermes_time(value::text, VARIADIC modifiers) $$;
CREATE OR REPLACE FUNCTION hermes_julianday(value double precision)
RETURNS double precision LANGUAGE sql VOLATILE AS $$ SELECT hermes_julianday(value::text) $$;
CREATE OR REPLACE FUNCTION hermes_unixepoch(value double precision)
RETURNS bigint LANGUAGE sql VOLATILE AS $$ SELECT hermes_unixepoch(value::text) $$;
CREATE OR REPLACE FUNCTION hermes_datetime(value double precision)
RETURNS text LANGUAGE sql VOLATILE AS $$ SELECT hermes_datetime(value::text) $$;
CREATE OR REPLACE FUNCTION hermes_strftime(fmt text, value text DEFAULT 'now', VARIADIC modifiers text[] DEFAULT ARRAY[]::text[])
RETURNS text LANGUAGE plpgsql VOLATILE AS $$
DECLARE t timestamp := hermes_ts_mod(hermes_ts(value), modifiers); out text := ''; i int := 1; c text;
BEGIN
  IF t IS NULL OR fmt IS NULL THEN RETURN NULL; END IF;
  WHILE i <= length(fmt) LOOP
    c := substr(fmt, i, 1);
    IF c = '%' AND i < length(fmt) THEN
      i := i + 1; c := substr(fmt, i, 1);
      out := out || CASE c
        WHEN 'Y' THEN to_char(t, 'YYYY') WHEN 'm' THEN to_char(t, 'MM') WHEN 'd' THEN to_char(t, 'DD')
        WHEN 'H' THEN to_char(t, 'HH24') WHEN 'M' THEN to_char(t, 'MI') WHEN 'S' THEN to_char(t, 'SS')
        WHEN 'f' THEN to_char(t, 'SS.MS') WHEN 'j' THEN to_char(t, 'DDD') WHEN 'w' THEN extract(dow FROM t)::int::text
        WHEN 's' THEN floor(extract(epoch FROM t))::bigint::text WHEN 'J' THEN (extract(epoch FROM t) / 86400.0 + 2440587.5)::text
        WHEN '%' THEN '%' ELSE '%' || c END;
    ELSE out := out || c; END IF;
    i := i + 1;
  END LOOP;
  RETURN out;
END $$;

CREATE OR REPLACE FUNCTION hermes_strftime(fmt text, value double precision, VARIADIC modifiers text[])
RETURNS text LANGUAGE sql VOLATILE AS $$ SELECT hermes_strftime(fmt, value::text, VARIADIC modifiers) $$;
CREATE OR REPLACE FUNCTION hermes_strftime(fmt text, value double precision)
RETURNS text LANGUAGE sql VOLATILE AS $$ SELECT hermes_strftime(fmt, value::text) $$;
CREATE OR REPLACE FUNCTION hermes_truthy(value boolean) RETURNS boolean LANGUAGE sql IMMUTABLE AS $$ SELECT value $$;
CREATE OR REPLACE FUNCTION hermes_truthy(value bigint) RETURNS boolean LANGUAGE sql IMMUTABLE AS $$ SELECT value <> 0 $$;
CREATE OR REPLACE FUNCTION hermes_truthy(value double precision) RETURNS boolean LANGUAGE sql IMMUTABLE AS $$
  SELECT value <> 0 $$;
CREATE OR REPLACE FUNCTION hermes_truthy(value text) RETURNS boolean LANGUAGE sql IMMUTABLE AS $$
  SELECT coalesce(substring(value from '^\s*[-+]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][-+]?[0-9]+)?')::double precision, 0) <> 0 $$;
CREATE OR REPLACE FUNCTION hermes_blob(value text) RETURNS bytea LANGUAGE sql IMMUTABLE AS $$ SELECT convert_to(value, 'UTF8') $$;
CREATE OR REPLACE FUNCTION hermes_blob(value bytea) RETURNS bytea LANGUAGE sql IMMUTABLE AS $$ SELECT value $$;
CREATE OR REPLACE FUNCTION hermes_blob(value double precision) RETURNS bytea LANGUAGE sql IMMUTABLE AS $$
  SELECT convert_to(value::text, 'UTF8') $$;
CREATE OR REPLACE FUNCTION hermes_instr(haystack text, needle text) RETURNS bigint LANGUAGE sql IMMUTABLE AS $$
  SELECT CASE WHEN haystack IS NULL OR needle IS NULL THEN NULL ELSE strpos(haystack, needle) END $$;
CREATE OR REPLACE FUNCTION hermes_typeof(value anyelement) RETURNS text LANGUAGE sql IMMUTABLE AS $$
  SELECT CASE WHEN value IS NULL THEN 'null' WHEN pg_typeof(value)::text IN ('bigint', 'integer', 'smallint') THEN 'integer'
              WHEN pg_typeof(value)::text IN ('double precision', 'real', 'numeric') THEN 'real'
              WHEN pg_typeof(value)::text = 'bytea' THEN 'blob' ELSE 'text' END $$;
CREATE OR REPLACE FUNCTION hermes_json_valid(value text) RETURNS bigint LANGUAGE plpgsql IMMUTABLE AS $$
BEGIN
  IF value IS NULL THEN RETURN NULL; END IF;
  PERFORM value::json; RETURN 1;
EXCEPTION WHEN others THEN RETURN 0;
END $$;
CREATE OR REPLACE FUNCTION hermes_json_extract(doc text, path text) RETURNS text LANGUAGE plpgsql IMMUTABLE AS $$
DECLARE parts text[]; v jsonb;
BEGIN
  IF doc IS NULL OR path IS NULL THEN RETURN NULL; END IF;
  parts := regexp_split_to_array(regexp_replace(regexp_replace(path, '^\$\.?', ''), '\[(\d+)\]', '.\1', 'g'), '\.');
  parts := array_remove(parts, '');
  v := doc::jsonb #> parts;
  IF v IS NULL OR jsonb_typeof(v) = 'null' THEN RETURN NULL; END IF;
  IF jsonb_typeof(v) = 'string' THEN RETURN v #>> '{}'; END IF;
  IF jsonb_typeof(v) = 'boolean' THEN RETURN CASE WHEN v::text = 'true' THEN '1' ELSE '0' END; END IF;
  RETURN v::text;
EXCEPTION WHEN others THEN RETURN NULL;
END $$;
CREATE OR REPLACE FUNCTION hermes_glob(value text, pattern text) RETURNS boolean LANGUAGE sql IMMUTABLE AS $$
  SELECT value ~ ('^' || replace(replace(replace(regexp_replace(pattern, '([.+^$(){}|\\])', '\\\1', 'g'), '*', '.*'), '?', '.'), '\\.*', '\\*') || '$') $$;
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_operator WHERE oprname = '#~#' AND oprnamespace = current_schema()::regnamespace) THEN
    EXECUTE 'CREATE OPERATOR #~# (LEFTARG = text, RIGHTARG = text, FUNCTION = hermes_glob)';
  END IF;
END $$;
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'hermes_group_concat' AND pronamespace = current_schema()::regnamespace) THEN
    EXECUTE 'CREATE AGGREGATE hermes_group_concat(anyelement) (SFUNC = hermes_gc_step, STYPE = text)';
  END IF;
END $$;

DO $$
DECLARE t record; pk text;
BEGIN
  -- Tables made before the implicit rowid column get it: the INTEGER PRIMARY KEY's value, else a counter.
  FOR t IN SELECT c.relname FROM pg_class c WHERE c.relnamespace = current_schema()::regnamespace AND c.relkind = 'r'
             AND c.relname NOT IN ('sqlite_master', 'hermes_compat_meta')
             AND NOT EXISTS (SELECT 1 FROM pg_attribute a WHERE a.attrelid = c.oid AND a.attname = 'rowid' AND NOT a.attisdropped)
  LOOP
    SELECT a.attname INTO pk FROM pg_attribute a WHERE a.attrelid = (quote_ident(current_schema()) || '.' || quote_ident(t.relname))::regclass
      AND a.attidentity <> '' AND NOT a.attisdropped LIMIT 1;
    IF pk IS NOT NULL THEN
      EXECUTE format('ALTER TABLE %I ADD COLUMN rowid bigint GENERATED ALWAYS AS (%I) STORED', t.relname, pk);
    ELSE
      EXECUTE format('ALTER TABLE %I ADD COLUMN rowid bigint GENERATED BY DEFAULT AS IDENTITY', t.relname);
    END IF;
  END LOOP;
END $$;
"""
_WRITES = {"INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP", "REPLACE"}
_NAMES_ROWID = re.compile(r'\browid\b', re.IGNORECASE)
_HELPERS_MARK = "hermes-compat " + hashlib.sha256(_HELPERS.encode("utf-8")).hexdigest()[:16]


class _UntypedNumberDumper(Dumper):
    """An int sent as an untyped literal: Postgres takes its type from the comparison, as SQLite does."""
    oid = 0

    def dump(self, obj: Any) -> bytes:
        return str(int(obj)).encode()


class _FloatDumper(Dumper):
    """A float as ``double precision``: untyped, ``COALESCE(?, an_integer_column)`` would read 1.5 as a bigint, where
    SQLite keeps the real value."""
    oid = 701

    def dump(self, obj: Any) -> bytes:
        return repr(float(obj)).encode()


class _UntypedBoolDumper(Dumper):
    """SQLite stores a bool as the integer 0 or 1."""
    oid = 0

    def dump(self, obj: Any) -> bytes:
        return b"1" if obj else b"0"


class Row(tuple):
    """``sqlite3.Row``: positional and by-name access, ``keys()``, and ``dict(row)``."""

    def __new__(cls, names: Sequence[str], values: Sequence[Any]) -> "Row":
        row = super().__new__(cls, values)
        row._names = tuple(names)  # type: ignore[attr-defined]
        return row

    def keys(self) -> list[str]:
        return list(self._names)  # type: ignore[attr-defined]

    def __getitem__(self, key: Any) -> Any:  # type: ignore[override]
        if isinstance(key, str):
            names = self._names  # type: ignore[attr-defined]
            try:
                return tuple.__getitem__(self, names.index(key))
            except ValueError:
                lowered = [n.lower() for n in names]
                if key.lower() in lowered:
                    return tuple.__getitem__(self, lowered.index(key.lower()))
                raise IndexError("No item with that key") from None
        return tuple.__getitem__(self, key)


def _value(value: Any) -> Any:
    if isinstance(value, decimal.Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, memoryview):
        return bytes(value)
    if isinstance(value, bool):
        return int(value)
    return value


def _sqlite_error(exc: Exception, sql: str) -> Exception:
    """The ``sqlite3`` exception a caller of the SQLite store expects for a Postgres error."""
    diag = getattr(exc, "diag", None)
    table, column = getattr(diag, "table_name", None), getattr(diag, "column_name", None)
    if isinstance(exc, pg_errors.UniqueViolation):
        detail = getattr(diag, "message_detail", "") or ""
        cols = re.search(r"Key \(([^)]*)\)", detail)
        names = ", ".join(f"{table}.{c.strip()}" for c in cols.group(1).split(",")) if cols and table else table or ""
        return sqlite3.IntegrityError(f"UNIQUE constraint failed: {names}")
    if isinstance(exc, pg_errors.NotNullViolation):
        return sqlite3.IntegrityError(f"NOT NULL constraint failed: {table}.{column}")
    if isinstance(exc, pg_errors.CheckViolation):
        return sqlite3.IntegrityError(f"CHECK constraint failed: {getattr(diag, 'constraint_name', '')}")
    if isinstance(exc, pg_errors.ForeignKeyViolation):
        return sqlite3.IntegrityError("FOREIGN KEY constraint failed")
    if isinstance(exc, pg_errors.DuplicateColumn):
        match = re.search(r'column "([^"]+)"', str(exc))
        return sqlite3.OperationalError(f"duplicate column name: {match.group(1) if match else ''}")
    if isinstance(exc, (pg_errors.UndefinedTable,)):
        match = re.search(r'relation "([^"]+)"', str(exc))
        return sqlite3.OperationalError(f"no such table: {match.group(1) if match else ''}")
    if isinstance(exc, pg_errors.UndefinedColumn):
        match = re.search(r'column "?([^" ]+)"? does not exist', str(exc))
        return sqlite3.OperationalError(f"no such column: {match.group(1) if match else ''}")
    if isinstance(exc, pg_errors.DuplicateTable):
        match = re.search(r'relation "([^"]+)"', str(exc))
        return sqlite3.OperationalError(f"table {match.group(1) if match else ''} already exists")
    if isinstance(exc, (pg_errors.LockNotAvailable, pg_errors.DeadlockDetected, pg_errors.QueryCanceled)):
        return sqlite3.OperationalError("database is locked")
    if isinstance(exc, (psycopg.OperationalError, psycopg.InterfaceError)):
        return sqlite3.OperationalError(f"unable to open database file: {exc}")
    if isinstance(exc, (pg_errors.SyntaxError, pg_errors.UndefinedFunction, pg_errors.DatatypeMismatch,
                        pg_errors.InvalidTextRepresentation)):
        return sqlite3.OperationalError(f"{exc} [compat SQL: {sql[:300]}]")
    return sqlite3.DatabaseError(str(exc))


class Cursor:
    """``sqlite3.Cursor`` over a :class:`Connection`."""

    arraysize = 1

    def __init__(self, connection: "Connection") -> None:
        self.connection = connection
        self._rows: list[Any] = []
        self._position = 0
        self.description: Optional[tuple] = None
        self.rowcount = -1
        self.lastrowid: Optional[int] = None
        self.row_factory = None

    def execute(self, sql: str, parameters: Any = ()) -> "Cursor":
        self.connection._run(self, sql, parameters)
        return self

    def executemany(self, sql: str, seq_of_parameters: Iterable[Any]) -> "Cursor":
        total = 0
        for parameters in seq_of_parameters:
            self.connection._run(self, sql, parameters)
            total += max(self.rowcount, 0)
        self.rowcount = total
        return self

    def executescript(self, script: str) -> "Cursor":
        self.connection.executescript(script)
        return self

    def _set(self, names: Optional[list[str]], rows: list[tuple], rowcount: int) -> None:
        factory = self.row_factory or self.connection.row_factory
        if names is None:
            self.description, self._rows = None, []
        else:
            self.description = tuple((name, None, None, None, None, None, None) for name in names)
            values = [tuple(_value(v) for v in row) for row in rows]
            if factory is None:
                self._rows = values
            elif factory is sqlite3.Row or factory is Row:
                self._rows = [Row(names, v) for v in values]
            else:
                self._rows = [factory(self, v) for v in values]
        self._position = 0
        self.rowcount = rowcount

    def fetchone(self) -> Any:
        if self._position >= len(self._rows):
            return None
        row = self._rows[self._position]
        self._position += 1
        return row

    def fetchmany(self, size: Optional[int] = None) -> list[Any]:
        size = self.arraysize if size is None else size
        rows = self._rows[self._position:self._position + size]
        self._position += len(rows)
        return rows

    def fetchall(self) -> list[Any]:
        rows = self._rows[self._position:]
        self._position = len(self._rows)
        return rows

    def __iter__(self) -> Iterator[Any]:
        while (row := self.fetchone()) is not None:
            yield row

    def close(self) -> None:
        self._rows = []


# Closed connections wait here for the next open on the same database, a few per DSN: stores that connect per
# operation (shared metrics, Discord recovery) then neither reconnect nor hold more server slots than they use.
_IDLE: dict[str, list[psycopg.Connection]] = {}
_IDLE_LOCK = threading.Lock()
_IDLE_PER_DSN = 8
# The schema each pooled connection's search_path names, so a reuse for the same store skips setting it.
_SEARCH_PATH: "weakref.WeakKeyDictionary[psycopg.Connection, str]" = weakref.WeakKeyDictionary()


def _take_idle(dsn: str) -> Optional[psycopg.Connection]:
    with _IDLE_LOCK:
        pool = _IDLE.get(dsn, [])
        while pool:
            conn = pool.pop()
            if not conn.closed and conn.info.transaction_status == psycopg.pq.TransactionStatus.IDLE:
                return conn
            conn.close()
    return None


def _borrow(dsn: str, schema: str, connect_timeout: int) -> psycopg.Connection:
    conn = _take_idle(dsn)
    if conn is None:
        conn = psycopg.connect(dsn, autocommit=True, connect_timeout=connect_timeout)
        for kind, dumper in ((int, _UntypedNumberDumper), (float, _FloatDumper), (bool, _UntypedBoolDumper)):
            conn.adapters.register_dumper(kind, dumper)
        conn.execute("SET TIME ZONE 'UTC'")
    if _SEARCH_PATH.get(conn) != schema:
        conn.execute(pg_sql.SQL("SET search_path TO {}").format(pg_sql.Identifier(schema)))
        _SEARCH_PATH[conn] = schema
    return conn


def _give_idle(dsn: str, conn: psycopg.Connection) -> None:
    if conn.info.transaction_status == psycopg.pq.TransactionStatus.IDLE:
        with _IDLE_LOCK:
            pool = _IDLE.setdefault(dsn, [])
            if len(pool) < _IDLE_PER_DSN:
                pool.append(conn)
                return
    conn.close()


class Connection:
    """``sqlite3.Connection`` for one store schema (module docstring)."""

    def __init__(self, dsn: str, schema: str, *, row_factory: Any = None, connect_timeout: int = 10) -> None:
        self._schema = schema
        self._lock_key = zlib.crc32(("hermes.compat." + schema).encode()) - 2**31
        self._dsn, self._connect_timeout = dsn, connect_timeout
        # A server connection is held only during a statement or an open transaction, borrowed from the process's
        # idle ones otherwise: like an idle sqlite3 connection, an open store costs the server nothing.
        self._pg: Optional[psycopg.Connection] = None
        self._closed = False
        # The SQLite file this store stands in for, which ``PRAGMA database_list`` reports (callers key on it).
        self.filename = ""
        # ``mode=ro``: every write is refused as SQLite refuses it.
        self.read_only = False
        self._in_transaction = False
        self._savepoints = 0
        self._last_rowid = 0
        self._changes = 0
        self._keys: dict[str, tuple[list[str], list[str], Optional[str]]] = {}
        self.row_factory = row_factory
        self.isolation_level: Optional[str] = ""
        self.text_factory = str
        self.total_changes = 0
        current = self._conn.execute("SELECT obj_description(oid, 'pg_namespace') FROM pg_namespace WHERE nspname = %s",
                                     (schema,)).fetchone()
        if current is not None and current[0] == _HELPERS_MARK:
            self._release()
            return
        with self._conn.transaction():
            self._conn.execute(pg_sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(pg_sql.Identifier(schema)))
            self._conn.execute(pg_sql.SQL("SET search_path TO {}").format(pg_sql.Identifier(schema)))
            self._conn.execute("SET TIME ZONE 'UTC'")
            self._conn.execute("SELECT pg_advisory_xact_lock(%s)", (self._lock_key,))
            # The schema's comment names the helper set installed in it, so a newer set reaches existing stores.
            installed = self._conn.execute("SELECT obj_description(current_schema()::regnamespace, 'pg_namespace')").fetchone()[0]
            if installed != _HELPERS_MARK:
                for statement in _plpgsql_split(_HELPERS):
                    self._conn.execute(statement)
                self._reconcile_master()
                self._conn.execute(pg_sql.SQL("COMMENT ON SCHEMA {} IS {}").format(pg_sql.Identifier(schema),
                                                                                  pg_sql.Literal(_HELPERS_MARK)))
        self._release()

    # ── the sqlite3.Connection surface ──

    @property
    def in_transaction(self) -> bool:
        return self._in_transaction

    def cursor(self, factory: Any = None) -> Cursor:
        return Cursor(self)

    def execute(self, sql: str, parameters: Any = ()) -> Cursor:
        return self.cursor().execute(sql, parameters)

    def executemany(self, sql: str, seq_of_parameters: Iterable[Any]) -> Cursor:
        return self.cursor().executemany(sql, seq_of_parameters)

    def executescript(self, script: str) -> Cursor:
        self.commit()
        cursor = self.cursor()
        for statement in split_statements(script):
            self._run(cursor, statement, ())
        self.commit()
        return cursor

    def commit(self) -> None:
        if self._in_transaction:
            self._finish("COMMIT")

    def rollback(self) -> None:
        if self._in_transaction:
            self._finish("ROLLBACK")

    @property
    def _conn(self) -> psycopg.Connection:
        if self._closed:
            raise sqlite3.ProgrammingError("Cannot operate on a closed database.")
        if self._pg is None:
            self._pg = _borrow(self._dsn, self._schema, self._connect_timeout)
        return self._pg

    def _release(self) -> None:
        """Give the server connection back between statements outside a transaction."""
        if self._pg is not None and not self._in_transaction:
            conn, self._pg = self._pg, None
            _give_idle(self._dsn, conn)

    def close(self) -> None:
        self._closed = True
        conn, self._pg = getattr(self, "_pg", None), None
        if conn is None or conn.closed:
            return
        if self._in_transaction:
            try:
                conn.execute("ROLLBACK")
            except psycopg.Error:
                conn.close()
                return
            self._in_transaction = False
        _give_idle(self._dsn, conn)

    def __del__(self) -> None:
        # A store dropped unclosed gives its server slot back now, as a dropped sqlite3 connection frees its file.
        conn = getattr(self, "_pg", None)
        if conn is not None and not conn.closed:
            try:
                conn.close()
            except Exception:
                pass

    def __enter__(self) -> "Connection":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if exc_type is None:
            self.commit()
        else:
            self.rollback()
        return False

    def create_function(self, *args: Any, **kwargs: Any) -> None:
        raise sqlite3.NotSupportedError("user functions are not available on a Postgres-backed store")

    def set_trace_callback(self, callback: Any) -> None:
        pass

    def interrupt(self) -> None:
        self._conn.cancel()

    # ── execution ──

    def _begin(self) -> None:
        try:
            self._conn.execute("BEGIN")
            self._conn.execute("SELECT pg_advisory_xact_lock(%s)", (self._lock_key,))
        except psycopg.Error as exc:
            raise _sqlite_error(exc, "BEGIN") from exc
        self._in_transaction = True

    def _finish(self, verb: str) -> None:
        try:
            self._conn.execute(verb)
        except psycopg.Error as exc:
            raise _sqlite_error(exc, verb) from exc
        finally:
            self._in_transaction = False
            self._savepoints = 0
            self._release()

    def _params(self, parameters: Any) -> tuple[Callable[[Token], str], Any]:
        if isinstance(parameters, Mapping):
            return (lambda token: f"%({token.text[1:]})s"), dict(parameters)
        values = list(parameters or ())
        numbered: list[Any] = []
        counter = iter(range(len(values)))

        def positional(token: Token) -> str:
            if len(token.text) > 1:  # ?NNN
                numbered.append(values[int(token.text[1:]) - 1])
            else:
                numbered.append(values[next(counter)])
            return "%s"
        return positional, numbered

    def _run(self, cursor: Cursor, sql: str, parameters: Any) -> None:
        try:
            self._run_statement(cursor, sql, parameters)
        finally:
            self._release()

    def _run_statement(self, cursor: Cursor, sql: str, parameters: Any) -> None:
        style, values = self._params(parameters)
        try:
            stmt = translate(sql, param_style=style)
        except ValueError as exc:
            raise sqlite3.OperationalError(str(exc)) from exc
        kind = stmt.kind
        if self.read_only and kind.split()[0] in _WRITES:
            raise sqlite3.OperationalError("attempt to write a readonly database")
        cursor.lastrowid = None
        if kind in ("EMPTY", "NOOP"):
            cursor._set(None, [], -1)
            return
        if kind == "BEGIN":
            if self._in_transaction:
                raise sqlite3.OperationalError("cannot start a transaction within a transaction")
            self._begin()
            cursor._set(None, [], -1)
            return
        if kind in ("COMMIT", "ROLLBACK") and not stmt.sql.upper().startswith("ROLLBACK TO"):
            if not self._in_transaction:
                raise sqlite3.OperationalError(f"cannot {kind.lower()} - no transaction is active")
            self._finish(kind)
            cursor._set(None, [], -1)
            return
        if kind == "SAVEPOINT" and not self._in_transaction:
            self._begin()
        if kind == "PRAGMA":
            self._pragma(cursor, *stmt.pragma)  # type: ignore[misc]
            return
        if kind in _IMPLICIT_BEGIN and not self._in_transaction and self.isolation_level is not None:
            self._begin()
        if kind == "INSERT" and stmt.insert_or_replace:
            stmt.sql = self._replace_target(stmt)
        if "{last_insert_rowid}" in stmt.sql or "{changes}" in stmt.sql:
            stmt.sql = (stmt.sql.replace("{last_insert_rowid}", str(int(self._last_rowid)))
                        .replace("{changes}", str(int(self._changes))))
        if kind == "INSERT" and "{columns}" in stmt.sql:
            columns, _keys, _identity = self._table_keys(stmt.table or "")
            stmt.sql = stmt.sql.replace("{columns}", "(" + ", ".join(f'"{c}"' for c in columns) + ")" if columns else "")
        if kind == "INSERT":
            stmt.sql = self._with_rowid(stmt)
        if stmt.update_or == "IGNORE":
            self._update_or_ignore(cursor, stmt, values)
            return
        self._execute(cursor, stmt, values)

    def _execute(self, cursor: Cursor, stmt: Statement, values: Any) -> None:
        guarded = self._in_transaction and stmt.kind not in ("SAVEPOINT", "RELEASE", "ROLLBACK")
        try:
            if guarded:
                self._conn.execute("SAVEPOINT hermes_compat_stmt")
            result = self._conn.execute(stmt.sql, values)
            names = [column.name for column in result.description] if result.description else None
            rows = result.fetchall() if names is not None else []
            if guarded:
                self._conn.execute("RELEASE SAVEPOINT hermes_compat_stmt")
        except psycopg.Error as exc:
            if guarded:
                try:
                    self._conn.execute("ROLLBACK TO SAVEPOINT hermes_compat_stmt")
                    self._conn.execute("RELEASE SAVEPOINT hermes_compat_stmt")
                except psycopg.Error:
                    pass
            raise _sqlite_error(exc, stmt.sql) from exc
        rowcount = result.rowcount if stmt.kind in ("INSERT", "UPDATE", "DELETE") else -1
        if stmt.kind == "INSERT" and stmt.table and rowcount > 0:
            self._advance_identity(stmt)
        if stmt.kind == "INSERT" and names == ["__hermes_rowid"]:
            cursor.lastrowid = int(rows[-1][0]) if rows else None
            if cursor.lastrowid is not None:
                self._last_rowid = cursor.lastrowid
            names, rows = None, []
        if stmt.kind in ("INSERT", "UPDATE", "DELETE"):
            self._changes = max(rowcount, 0)
        if names is not None and "rowid" in names and not _NAMES_ROWID.search(stmt.sql):
            # SQLite's rowid is hidden: ``SELECT *`` never returns it.
            keep = [n for n, name in enumerate(names) if name != "rowid"]
            names, rows = [names[n] for n in keep], [tuple(row[n] for n in keep) for row in rows]
        if rowcount > 0:
            self.total_changes += rowcount
        cursor._set(names, rows, rowcount)
        self._after_ddl(stmt)

    def _update_or_ignore(self, cursor: Cursor, stmt: Statement, values: Any) -> None:
        """SQLite's ``UPDATE OR IGNORE``: each matching row is updated on its own, and one whose update would break a
        UNIQUE or PRIMARY KEY constraint is left as it was. Rows are picked by the hidden rowid."""
        head, _, where = stmt.sql.partition("/*hermes_where*/")
        if isinstance(values, Mapping):
            set_values, where_values = values, values
        else:
            split = len(re.findall(r"(?<!%)%s", head))
            set_values, where_values = list(values)[:split], list(values)[split:]
        began = not self._in_transaction
        if began:
            self._begin()
        try:
            rowids = [row[0] for row in self._conn.execute(
                f'SELECT rowid FROM "{stmt.table}" {where}', where_values).fetchall()]
            changed = 0
            for rowid in rowids:
                self._conn.execute("SAVEPOINT hermes_compat_or_ignore")
                try:
                    result = self._conn.execute(f'{head} WHERE rowid = %s' if isinstance(set_values, list)
                                                else f'{head} WHERE rowid = %(__hermes_rowid)s',
                                                [*set_values, rowid] if isinstance(set_values, list)
                                                else {**set_values, "__hermes_rowid": rowid})
                    changed += result.rowcount
                    self._conn.execute("RELEASE SAVEPOINT hermes_compat_or_ignore")
                except (pg_errors.UniqueViolation, pg_errors.ExclusionViolation):
                    self._conn.execute("ROLLBACK TO SAVEPOINT hermes_compat_or_ignore")
                    self._conn.execute("RELEASE SAVEPOINT hermes_compat_or_ignore")
        except psycopg.Error as exc:
            if began:
                self._finish("ROLLBACK")
            raise _sqlite_error(exc, stmt.sql) from exc
        if began and self.isolation_level is None:
            self._finish("COMMIT")
        self._changes = changed
        self.total_changes += changed
        cursor._set(None, [], changed)

    # ── DDL bookkeeping ──

    def _after_ddl(self, stmt: Statement) -> None:
        if stmt.kind.startswith("CREATE ") and stmt.ddl:
            kind = stmt.kind.split()[1].lower()
            self._conn.execute("INSERT INTO sqlite_master (type, name, tbl_name, sql) SELECT %s, %s, %s, %s "
                               "WHERE NOT EXISTS (SELECT 1 FROM sqlite_master WHERE type = %s AND name = %s)",
                               (kind, stmt.table, stmt.tbl_name, stmt.ddl, kind, stmt.table))
        elif stmt.kind == "DROP" and stmt.table:
            self._conn.execute("DELETE FROM sqlite_master WHERE name = %s OR (type <> 'table' AND tbl_name = %s)",
                               (stmt.table, stmt.table))
            self._keys.pop(stmt.table, None)
        elif stmt.kind == "ALTER RENAME" and stmt.rename:
            old, new = stmt.rename
            self._conn.execute("UPDATE sqlite_master SET name = CASE WHEN name = %s THEN %s ELSE name END, tbl_name = %s "
                               "WHERE tbl_name = %s", (old, new, new, old))
            self._keys.pop(old, None)
        elif stmt.kind == "ALTER ADD":
            self._keys.pop(stmt.table or "", None)
            if stmt.ddl:
                self._append_column_text(stmt.table or "", stmt.ddl)

    def _append_column_text(self, table: str, column_ddl: str) -> None:
        """Add a column's definition to its table's CREATE TABLE in sqlite_master, before the column list's closing
        parenthesis (any WITHOUT ROWID / STRICT after it stays), as SQLite rewrites it on ADD COLUMN."""
        row = self._conn.execute("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = %s", (table,)).fetchone()
        if row is None or not row[0] or ")" not in row[0]:
            return
        text = row[0]
        close = text.rindex(")")
        self._conn.execute("UPDATE sqlite_master SET sql = %s WHERE type = 'table' AND name = %s",
                           (f"{text[:close].rstrip()}, {column_ddl}{text[close:]}", table))

    def _reconcile_master(self) -> None:
        """Columns added before sqlite_master followed ADD COLUMN: each one its table's text lacks is appended by
        name and type, so a copy of the store (export, backup) recreates every column."""
        recorded = self._conn.execute("SELECT name, sql FROM sqlite_master WHERE type = 'table' AND sql IS NOT NULL").fetchall()
        for table, text in recorded:
            names = {(quoted or bare).lower() for quoted, bare in re.findall(r'"([^"]+)"|\b([A-Za-z_][A-Za-z0-9_]*)\b', text)}
            for column, dtype in self._conn.execute(
                    "SELECT column_name, upper(data_type) FROM information_schema.columns WHERE table_schema = current_schema() "
                    "AND table_name = %s AND column_name <> 'rowid' ORDER BY ordinal_position", (table,)).fetchall():
                if column.lower() not in names:
                    self._append_column_text(table, f"{column} {_sqlite_type(dtype)}")

    def _table_keys(self, table: str) -> tuple[list[str], list[str], Optional[str]]:
        """``(columns, primary key columns, identity column)`` of a table of this schema."""
        if table not in self._keys:
            columns = [r[0] for r in self._conn.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_schema = current_schema() "
                "AND table_name = %s AND column_name <> 'rowid' ORDER BY ordinal_position", (table,)).fetchall()]
            keys = [r[0] for r in self._conn.execute(
                "SELECT a.attname FROM pg_index i JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey) "
                "WHERE i.indrelid = to_regclass(quote_ident(current_schema()) || '.' || quote_ident(%s)) AND i.indisprimary "
                "ORDER BY array_position(i.indkey::int2[], a.attnum)", (table,)).fetchall()]
            identity = self._conn.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_schema = current_schema() "
                "AND table_name = %s AND is_identity = 'YES'", (table,)).fetchone()
            self._keys[table] = (columns, keys, identity[0] if identity else None)
        return self._keys[table]

    def _replace_target(self, stmt: Statement) -> str:
        columns, keys, _identity = self._table_keys(stmt.table or "")
        if not keys:
            unique = self._conn.execute(
                "SELECT array_agg(a.attname ORDER BY a.attnum) FROM pg_index i JOIN pg_attribute a "
                "ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey) WHERE i.indrelid = "
                "to_regclass(quote_ident(current_schema()) || '.' || quote_ident(%s)) AND i.indisunique "
                "GROUP BY i.indexrelid LIMIT 1", (stmt.table,)).fetchone()
            keys = list(unique[0]) if unique else []
        if not keys:
            return stmt.sql.replace(" ON CONFLICT {target} DO UPDATE SET {set} ", " ")
        written = stmt.replace_columns or columns
        updates = ", ".join(f'"{c}" = excluded."{c}"' for c in written if c not in keys) or f'"{keys[0]}" = excluded."{keys[0]}"'
        target = "(" + ", ".join(f'"{k}"' for k in keys) + ")"
        return stmt.sql.replace("{target}", target).replace("{set}", updates)

    def _advance_identity(self, stmt: Statement) -> None:
        """An INSERT that set the identity key itself moves the key's sequence past it: SQLite's next rowid is always
        above the largest stored one."""
        _columns, _keys, identity = self._table_keys(stmt.table or "")
        if identity is None or stmt.replace_columns is None or identity not in stmt.replace_columns:
            return
        self._conn.execute(
            pg_sql.SQL("SELECT setval(pg_get_serial_sequence({}, {}), greatest(coalesce(max({}), 0), 1), max({}) IS NOT NULL) "
                       "FROM {}").format(pg_sql.Literal(stmt.table), pg_sql.Literal(identity), pg_sql.Identifier(identity),
                                         pg_sql.Identifier(identity), pg_sql.Identifier(stmt.table)))

    def _with_rowid(self, stmt: Statement) -> str:
        """An INSERT into a table with an identity key returns it, for ``cursor.lastrowid``."""
        if " RETURNING " in stmt.sql.upper() or not stmt.table:
            return stmt.sql
        _columns, _keys, identity = self._table_keys(stmt.table)
        if identity is None:
            return stmt.sql
        return f'{stmt.sql} RETURNING "{identity}" AS __hermes_rowid'

    # ── PRAGMA ──

    def _pragma(self, cursor: Cursor, name: str, argument: Optional[str]) -> None:
        if name == "table_info" and argument:
            columns, keys, _identity = self._table_keys(argument)
            info = self._conn.execute(
                "SELECT column_name, upper(data_type), is_nullable, column_default FROM information_schema.columns "
                "WHERE table_schema = current_schema() AND table_name = %s AND column_name <> 'rowid' "
                "ORDER BY ordinal_position", (argument,)).fetchall()
            rows = [(cid, name_, _sqlite_type(dtype), 0 if nullable == "YES" else 1, _sqlite_default(default),
                     (keys.index(name_) + 1) if name_ in keys else 0) for cid, (name_, dtype, nullable, default) in enumerate(info)]
            cursor._set(["cid", "name", "type", "notnull", "dflt_value", "pk"], rows, -1)
        elif name == "index_list" and argument:
            rows = self._conn.execute(
                "SELECT row_number() OVER () - 1, c.relname, CASE WHEN i.indisunique THEN 1 ELSE 0 END, 'c', 0 "
                "FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid WHERE i.indrelid = "
                "to_regclass(quote_ident(current_schema()) || '.' || quote_ident(%s))", (argument,)).fetchall()
            cursor._set(["seq", "name", "unique", "origin", "partial"], rows, -1)
        elif name in ("user_version", "application_id"):
            key = "pragma." + name
            if argument is not None:
                self._conn.execute("INSERT INTO hermes_compat_meta (key, value) VALUES (%s, %s) "
                                   "ON CONFLICT (key) DO UPDATE SET value = excluded.value", (key, str(int(argument))))
                cursor._set(None, [], -1)
            else:
                found = self._conn.execute("SELECT value FROM hermes_compat_meta WHERE key = %s", (key,)).fetchone()
                cursor._set([name], [(int(found[0]) if found else 0,)], -1)
        elif name == "journal_mode":
            cursor._set(["journal_mode"], [("wal",)], -1)
        elif name in ("integrity_check", "quick_check"):
            cursor._set([name], [("ok",)], -1)
        elif name == "foreign_keys" and argument is None:
            cursor._set(["foreign_keys"], [(1,)], -1)
        elif name == "database_list":
            cursor._set(["seq", "name", "file"], [(0, "main", self.filename)], -1)
        elif name == "wal_checkpoint":
            cursor._set(["busy", "log", "checkpointed"], [(0, 0, 0)], -1)
        else:
            cursor._set(None, [], -1)


def _sqlite_default(default: Optional[str]) -> Optional[str]:
    """A column default as SQLite reports it: ``'scratch'`` for Postgres' ``'scratch'::text``; none for a
    sequence's ``nextval(...)``."""
    if default is None or default.startswith("nextval("):
        return None
    text = default
    while (match := re.fullmatch(r"(.*)::[a-z ]+(\[\])?", text, re.DOTALL)) and not match.group(1).endswith("'::"):
        text = match.group(1)
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1]
    return text


def _sqlite_type(pg_type: str) -> str:
    return {"BIGINT": "INTEGER", "INTEGER": "INTEGER", "DOUBLE PRECISION": "REAL", "BYTEA": "BLOB"}.get(pg_type, "TEXT")


def _plpgsql_split(script: str) -> list[str]:
    """Split the helper script at semicolons outside ``$$`` bodies."""
    out, current, inside = [], [], False
    for part in re.split(r"(\$\$|;)", script):
        if part == "$$":
            inside = not inside
            current.append(part)
        elif part == ";" and not inside:
            text = "".join(current).strip()
            if text:
                out.append(text)
            current = []
        else:
            current.append(part)
    text = "".join(current).strip()
    if text:
        out.append(text)
    return out
