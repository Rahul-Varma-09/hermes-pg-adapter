"""The hosted-room table layer on Postgres, shared by every hosted-room repository.

The hosted-room logic is written against ``HostedTables``: each logical table a mapping, a per-room table a mapping
of room id to that room's rows. Here those mappings are lazy views over ``hosted_room_rows`` of one placement's schema:
a key's row loads on first touch, iterating loads the rest, and a write block saves back only the rows it changed,
inserted or removed. The ``'profile'`` placement is the bound profile's schema; ``'shared'`` is the install's shared
schema, one per root ``shared-state.db`` path, created on first write.

Every public verb of a hosted-room repository runs in one transaction under one database-wide advisory lock
(:func:`hosted_verb`), so hosted-room verbs serialize as SQLite's single writer does and every read sees one state.
"""

from __future__ import annotations

import contextvars
import copy
import functools
import hashlib
import json
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Optional

from psycopg import Connection, sql
from psycopg.types.json import Jsonb

from .._schema import _MIGRATIONS_DIR

#: Tables whose rows are grouped per room (``table[room_id][inner key]``); every other table is keyed once.
PER_ROOM = frozenset({
    "hosted_room_attachments", "hosted_room_driver_tasks", "hosted_room_events", "hosted_room_links",
    "hosted_room_peer_reservations", "hosted_room_policy_events", "hosted_room_policy_publications",
    "hosted_room_policy_threads", "hosted_room_policy_transcript", "hosted_room_policy_watermarks",
    "hosted_room_remote_runs", "hosted_room_replica_events", "hosted_room_router_claims"})
_LOCK = "SELECT pg_advisory_xact_lock(hashtext('hermes.hosted_rooms'))"
_MISSING = object()
_ACTIVE: contextvars.ContextVar[dict[int, Connection]] = contextvars.ContextVar("hermes_pg_hosted", default={})


def _encode(key: Any) -> str:
    return json.dumps(list(key) if isinstance(key, tuple) else key, separators=(",", ":"))


def _decode(text: str) -> Any:
    value = json.loads(text)
    return tuple(value) if isinstance(value, list) else value


def _same(a: Any, b: Any) -> bool:
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def _copy(value: Any) -> Any:
    return json.loads(json.dumps(value))


# ── the lazy table views ──

class _Rows(MutableMapping):
    """One keyed set of rows (a table keyed once, or one room's rows of a per-room table), loaded lazily."""

    def __init__(self, store: "_Placement", tbl: str, k1: Optional[str]) -> None:
        self._store, self._tbl, self._k1 = store, tbl, k1
        self._cache: dict[Any, Any] = {}
        self._original: dict[Any, Any] = {}
        self._full = False

    def _where(self) -> tuple[str, tuple[Any, ...]]:
        if self._k1 is None:
            return "tbl = %s AND k2 = ''", (self._tbl,)
        return "tbl = %s AND k1 = %s", (self._tbl, self._k1)

    def _key_column(self) -> str:
        return "k1" if self._k1 is None else "k2"

    def _load(self, key: Any) -> Any:
        if key in self._cache:
            return self._cache[key]
        if self._full or key in self._original:
            return _MISSING
        where, params = self._where()
        found = self._store.scalar(f"SELECT row FROM {{table}} WHERE {where} AND {self._key_column()} = %s",
                                   (*params, _encode(key)))
        self._original[key] = _MISSING if found is None else _copy(found)
        if found is not None:
            self._cache[key] = found
        return self._cache.get(key, _MISSING)

    def _load_all(self) -> None:
        if self._full:
            return
        where, params = self._where()
        for item in self._store.rows(f"SELECT {self._key_column()} AS key, row FROM {{table}} WHERE {where}", params):
            key = _decode(item["key"])
            if key not in self._original:
                self._original[key] = _copy(item["row"])
                self._cache[key] = item["row"]
        self._full = True

    def __getitem__(self, key: Any) -> Any:
        value = self._load(key)
        if value is _MISSING:
            raise KeyError(key)
        return value

    def __contains__(self, key: object) -> bool:
        return self._load(key) is not _MISSING

    def __setitem__(self, key: Any, value: Any) -> None:
        if key not in self._original and not self._full:
            self._load(key)
        self._original.setdefault(key, _MISSING)
        self._cache[key] = value

    def __delitem__(self, key: Any) -> None:
        if self._load(key) is _MISSING:
            raise KeyError(key)
        del self._cache[key]

    def __iter__(self) -> Iterator[Any]:
        self._load_all()
        return iter(list(self._cache))

    def __len__(self) -> int:
        self._load_all()
        return len(self._cache)

    def __deepcopy__(self, memo: dict[int, Any]) -> dict[Any, Any]:
        """A detached plain copy of every row, as copying a dict of rows gives."""
        return {key: copy.deepcopy(value, memo) for key, value in self.items()}

    def changes(self) -> Iterator[tuple[Any, Any]]:
        """``(key, value or _MISSING)`` of every row this view changed."""
        for key, before in self._original.items():
            after = self._cache.get(key, _MISSING)
            if before is _MISSING and after is _MISSING:
                continue
            if before is _MISSING or after is _MISSING or not _same(before, after):
                yield key, after
        for key in self._cache.keys() - self._original.keys():
            yield key, self._cache[key]


class _RoomTable(MutableMapping):
    """A per-room table: room id to that room's rows, each a lazy :class:`_Rows`."""

    def __init__(self, store: "_Placement", tbl: str) -> None:
        self._store, self._tbl = store, tbl
        self._rooms: dict[str, _Rows] = {}
        self._cleared: set[str] = set()
        self._created: set[str] = set()

    def _view(self, room_id: str) -> _Rows:
        if room_id not in self._rooms:
            self._rooms[room_id] = _Rows(self._store, self._tbl, room_id)
        return self._rooms[room_id]

    def __contains__(self, room_id: object) -> bool:
        if room_id in self._created:
            return True
        if room_id in self._cleared:
            return False
        return self._store.scalar("SELECT 1 FROM {table} WHERE tbl = %s AND k1 = %s LIMIT 1", (self._tbl, room_id)) is not None

    def __getitem__(self, room_id: Any) -> _Rows:
        if room_id not in self:
            raise KeyError(room_id)
        return self._view(room_id)

    def get(self, room_id: Any, default: Any = None) -> Any:
        return self._view(room_id) if room_id in self else default

    def setdefault(self, room_id: Any, default: Any = None) -> _Rows:
        self._created.add(room_id)
        return self._view(room_id)

    def __setitem__(self, room_id: Any, rows: Any) -> None:
        view = self.setdefault(room_id)
        for key in list(view):
            del view[key]
        for key, value in dict(rows).items():
            view[key] = value

    def __delitem__(self, room_id: Any) -> None:
        if room_id not in self:
            raise KeyError(room_id)
        self.pop(room_id)

    def pop(self, room_id: Any, default: Any = _MISSING) -> Any:
        if room_id not in self:
            if default is _MISSING:
                raise KeyError(room_id)
            return default
        view = self._view(room_id)
        rows = dict(view.items())
        self._cleared.add(room_id)
        self._created.discard(room_id)
        self._rooms.pop(room_id, None)
        return rows

    def __iter__(self) -> Iterator[str]:
        stored = {item["k1"] for item in self._store.rows("SELECT DISTINCT k1 FROM {table} WHERE tbl = %s", (self._tbl,))}
        return iter(sorted((stored - self._cleared) | self._created))

    def __len__(self) -> int:
        return len(list(iter(self)))

    def __deepcopy__(self, memo: dict[int, Any]) -> dict[Any, Any]:
        return {room_id: copy.deepcopy(self._view(room_id), memo) for room_id in self}

    def flush(self) -> None:
        for room_id in self._cleared:
            self._store.execute("DELETE FROM {table} WHERE tbl = %s AND k1 = %s", (self._tbl, room_id))
        for room_id, view in self._rooms.items():
            for key, value in view.changes():
                self._store.put(self._tbl, room_id, _encode(key), value)


class HostedTables:
    """One placement's hosted-room tables (module docstring); ``tables[name]`` is the named table."""

    def __init__(self, placement: "_Placement") -> None:
        self._placement = placement
        self._tables: dict[str, Any] = {}

    def __getitem__(self, name: str) -> Any:
        from .hosted_room_tables import TABLES

        if name not in TABLES:
            raise KeyError(name)
        if name not in self._tables:
            self._tables[name] = (_RoomTable(self._placement, name) if name in PER_ROOM
                                  else _Rows(self._placement, name, None))
        return self._tables[name]

    def rows(self, name: str, room_id: str) -> Any:
        """The rows of one room in a per-room table, created on first touch."""
        return self[name].setdefault(room_id, {})

    def find_event(self, room_id: str, event_id: str) -> Optional[dict[str, Any]]:
        """The event row of ``(room_id, event_id)`` through the event-id index, or None."""
        events = self["hosted_room_events"]
        found = self._placement.rows("SELECT k2 FROM {table} WHERE tbl = 'hosted_room_events' AND k1 = %s "
                                     "AND row ->> 'event_id' = %s", (room_id, event_id))
        for item in found:
            row = events.get(room_id, {}).get(_decode(item["k2"]))
            if row is not None and row["event_id"] == event_id:
                return row
        created = events.get(room_id, {})
        return next((row for key, row in getattr(created, "_cache", {}).items() if row["event_id"] == event_id), None)

    def flush(self) -> None:
        for name, table in self._tables.items():
            if isinstance(table, _RoomTable):
                table.flush()
            else:
                for key, value in table.changes():
                    self._placement.put(name, _encode(key), "", value)


class _Placement:
    """One placement's schema on the verb's connection."""

    def __init__(self, conn: Connection, schema: str) -> None:
        self.conn, self.schema = conn, schema
        self._table = sql.Identifier(schema, "hosted_room_rows").as_string(conn)

    def _sql(self, query: str) -> str:
        return query.replace("{table}", self._table)

    def rows(self, query: str, params: Any = ()) -> list[dict[str, Any]]:
        from .._sql import rows

        return rows(self.conn, self._sql(query), params)

    def scalar(self, query: str, params: Any = ()) -> Any:
        from .._sql import scalar

        return scalar(self.conn, self._sql(query), params)

    def execute(self, query: str, params: Any = ()) -> int:
        from .._sql import execute

        return execute(self.conn, self._sql(query), params)

    def put(self, tbl: str, k1: str, k2: str, value: Any) -> None:
        if value is _MISSING:
            self.execute("DELETE FROM {table} WHERE tbl = %s AND k1 = %s AND k2 = %s", (tbl, k1, k2))
        else:
            self.execute("INSERT INTO {table} (tbl, k1, k2, row) VALUES (%s, %s, %s, %s) "
                         "ON CONFLICT (tbl, k1, k2) DO UPDATE SET row = excluded.row", (tbl, k1, k2, Jsonb(value)))


# ── placements, verbs and the table blocks ──

def shared_state_path(profile_dir: Path) -> Path:
    """The install's root ``shared-state.db`` derived from a profile folder."""
    return (profile_dir.parent.parent if profile_dir.parent.name == "profiles" else profile_dir) / "shared-state.db"


def _shared_schema(store: Any) -> str:
    path = str(shared_state_path(Path(store.scope.profile_dir)))
    return f"{store._backend.settings.schema_prefix}shared_{hashlib.sha256(path.encode('utf-8')).hexdigest()[:10]}"


def _schema_exists(conn: Connection, schema: str) -> bool:
    return conn.execute("SELECT 1 FROM pg_namespace WHERE nspname = %s", (schema,)).fetchone() is not None


def _ensure_shared(conn: Connection, schema: str) -> None:
    """Create the install's shared placement schema with the hosted-room tables, on first use."""
    if _schema_exists(conn, schema):
        return
    ddl = (_MIGRATIONS_DIR / "0008_hosted_rooms.sql").read_text(encoding="utf-8")
    conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema)))
    conn.execute(sql.SQL("SET LOCAL search_path TO {}").format(sql.Identifier(schema)))
    try:
        conn.execute(ddl)
    finally:
        conn.execute(sql.SQL("SET LOCAL search_path TO {}").format(sql.Identifier(_current_schema(conn))))


def _current_schema(conn: Connection) -> str:
    return _ACTIVE_SCHEMA.get()


_ACTIVE_SCHEMA: contextvars.ContextVar[str] = contextvars.ContextVar("hermes_pg_hosted_schema", default="public")


def current_conn(store: Any) -> Connection:
    conn = _ACTIVE.get().get(id(store))
    if conn is None:
        raise RuntimeError("hosted-room tables are read inside a hosted verb")
    return conn


def _placement(store: Any, placement: str, *, create: bool) -> Optional[_Placement]:
    conn = current_conn(store)
    if placement == "profile":
        return _Placement(conn, store.schema)
    if placement == "shared":
        schema = _shared_schema(store)
        if create:
            _ensure_shared(conn, schema)
        elif not _schema_exists(conn, schema):
            return None
        return _Placement(conn, schema)
    raise ValueError(f"unknown hosted-room placement {placement!r}")


def read_tables(store: Any, placement: str) -> HostedTables:
    """The tables of ``placement`` for a read; opening the shared placement creates it."""
    return HostedTables(_placement(store, placement, create=True))  # type: ignore[arg-type]


def peek_tables(store: Any, placement: str) -> Optional[HostedTables]:
    """The tables of ``placement``, or None when its schema was never created (the probes never create it)."""
    found = _placement(store, placement, create=False)
    return None if found is None else HostedTables(found)


@contextmanager
def write_tables(store: Any, placement: str) -> Iterator[HostedTables]:
    """One write block: the rows it changed are saved when the block returns (as one savepoint), nothing when it
    raises."""
    tables = HostedTables(_placement(store, placement, create=True))  # type: ignore[arg-type]
    yield tables
    with current_conn(store).transaction():
        tables.flush()


def hosted_verb(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Run one hosted-room verb in one transaction under the hosted-room lock.

    Every completed write block of the verb is kept even when the verb then raises, as each block publishes on its own
    when it returns: the transaction commits what the verb wrote, then the refusal is raised."""
    @functools.wraps(fn)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        from .._backend import translated

        store = self._store
        if id(store) in _ACTIVE.get():
            return fn(self, *args, **kwargs)
        failure: Optional[BaseException] = None
        with store.transaction() as conn:
            conn.execute(_LOCK)
            token = _ACTIVE.set({**_ACTIVE.get(), id(store): conn})
            schema_token = _ACTIVE_SCHEMA.set(store.schema)
            try:
                return fn(self, *args, **kwargs)
            except Exception as exc:  # committed below, then raised
                failure = exc
            finally:
                _ACTIVE_SCHEMA.reset(schema_token)
                _ACTIVE.reset(token)
        with translated():
            raise failure  # type: ignore[misc]
    return wrapper


def hosted_verbs(cls: type) -> type:
    """Wrap every public verb of ``cls`` (its own and its mixins') in :func:`hosted_verb`."""
    for klass in reversed(cls.__mro__[:-1]):
        for name, value in list(vars(klass).items()):
            if not name.startswith("_") and callable(value) and not isinstance(value, type):
                setattr(cls, name, hosted_verb(getattr(cls, name)))
    return cls


# ── the profile's state_meta, as the hosted-room logic reads it ──

class StateMeta(MutableMapping):
    """The bound profile's ``state_meta`` rows on the verb's connection, in insertion order."""

    def __init__(self, store: Any) -> None:
        self._conn = current_conn(store)

    def __getitem__(self, key: str) -> str:
        found = self._conn.execute("SELECT value FROM state_meta WHERE key = %s", (key,)).fetchone()
        if found is None:
            raise KeyError(key)
        return found[0]

    def __contains__(self, key: object) -> bool:
        return self._conn.execute("SELECT 1 FROM state_meta WHERE key = %s", (key,)).fetchone() is not None

    def __setitem__(self, key: str, value: str) -> None:
        self._conn.execute("INSERT INTO state_meta (key, value) VALUES (%s, %s) "
                           "ON CONFLICT (key) DO UPDATE SET value = excluded.value", (key, value))

    def __delitem__(self, key: str) -> None:
        if self._conn.execute("DELETE FROM state_meta WHERE key = %s", (key,)).rowcount == 0:
            raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter([item[0] for item in self._conn.execute("SELECT key FROM state_meta ORDER BY seq").fetchall()])

    def __len__(self) -> int:
        return int(self._conn.execute("SELECT count(*) FROM state_meta").fetchone()[0])

    def setdefault(self, key: str, default: Any = None) -> Any:
        self._conn.execute("INSERT INTO state_meta (key, value) VALUES (%s, %s) ON CONFLICT (key) DO NOTHING", (key, default))
        return self[key]


# ── attachment blob bytes ──

class BlobFiles(MutableMapping):
    """One placement's blob bytes: file name to bytes, in ``hosted_room_blob_files`` of its schema."""

    before_read: Optional[Callable[[], None]] = None

    def __init__(self, placement: _Placement) -> None:
        self._p = placement
        self._table = sql.Identifier(placement.schema, "hosted_room_blob_files").as_string(placement.conn)

    def __getitem__(self, name: str) -> bytes:
        found = self._p.conn.execute(f"SELECT data FROM {self._table} WHERE name = %s", (name,)).fetchone()
        if found is None:
            raise KeyError(name)
        return bytes(found[0])

    def __setitem__(self, name: str, data: bytes) -> None:
        self._p.conn.execute(f"INSERT INTO {self._table} (name, data) VALUES (%s, %s) "
                             "ON CONFLICT (name) DO UPDATE SET data = excluded.data", (name, bytes(data)))

    def __delitem__(self, name: str) -> None:
        if self._p.conn.execute(f"DELETE FROM {self._table} WHERE name = %s", (name,)).rowcount == 0:
            raise KeyError(name)

    def __iter__(self) -> Iterator[str]:
        return iter([item[0] for item in self._p.conn.execute(f"SELECT name FROM {self._table} ORDER BY name").fetchall()])

    def __len__(self) -> int:
        return int(self._p.conn.execute(f"SELECT count(*) FROM {self._table}").fetchone()[0])


def blob_files(store: Any, placement: str) -> BlobFiles:
    return BlobFiles(_placement(store, placement, create=True))  # type: ignore[arg-type]
