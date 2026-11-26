"""The assembled Postgres backend.

One ``PostgresBackend`` serves the stores one assembly was built for, each profile's ``hermes_home_key`` mapped to one
schema (``_schema.schema_name``). ``open(scope)`` returns a ``PostgresStore`` bound to that profile: every repository
of the handle runs its SQL through ``store.transaction()``, which checks a connection out of the process-wide pool for
the DSN, sets ``search_path`` to the profile's schema for that transaction only, and translates driver failures into
the contract's error family. A writable store creates and migrates its schema on first touch; a read-only one never
does and refuses a schema that is not there.

Repository modules call ``register_repository`` at import time with a factory that receives the bound handle, as the
SQLite backend's do. An attribute with no registered repository is absent from the handle and its capability flags
stay ``False``; ``missing_repositories()`` names them, and the provider refuses to serve Hermes while any is missing.
"""

from __future__ import annotations

import contextvars
import dataclasses
import os
import threading
import typing
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional

import psycopg
from psycopg import errors as pg_errors, sql
from psycopg_pool import ConnectionPool, PoolTimeout

from persistence import (
    AwaitableStore, DiskFull, PermanentStoreError, PersistenceCapabilities, PersistenceError, ProfileScope,
    StoreContention, StoreReplaced, StoreUnavailable, TransientStoreError,
)
from persistence.core.admin import PHYSICAL_ADMIN, AdminUnsupported
from persistence.core.backend import PersistenceStore

from . import _schema
from ._settings import PostgresSettings

__all__ = [
    "CAPABILITY_ATTRIBUTES", "PostgresBackend", "PostgresStore", "close_all_pools", "missing_repositories",
    "register_repository", "translated",
]

PROVIDER_NAME = "postgres"

# The repository attributes a store handle can carry: every ``PersistenceStore`` attribute but the handle's own facts.
CAPABILITY_ATTRIBUTES: tuple[str, ...] = tuple(
    name for name in typing.get_type_hints(PersistenceStore) if name not in {"scope", "read_only", "capabilities", "aio"})

# Facts of a shared, networked store without file-level admin; every flag a repository earns by its behaviour starts
# False and is set by that repository's registration.
_BASE_CAPABILITIES = PersistenceCapabilities(
    full_text_search=False,
    substring_search=False,
    transactional_claims=False,
    cross_process=True,
    remote=True,
    reopen_durability=True,
    authority_fences=False,
    durable_admissions=False,
    receipted_writes=False,
    physical_admin=False,
)

RepositoryFactory = Callable[["PostgresStore"], Any]

_REPOSITORY_FACTORIES: dict[str, RepositoryFactory] = {}
_CAPABILITY_OVERRIDES: dict[str, bool] = {}


def register_repository(
    attribute: str, factory: RepositoryFactory, capabilities: Optional[Mapping[str, bool]] = None,
) -> None:
    """Register the repository assembled as ``attribute`` on every new store handle, with the capability flags its
    behaviour earns. Registering an attribute twice fails."""
    if attribute not in CAPABILITY_ATTRIBUTES:
        raise ValueError(f"unknown capability attribute {attribute!r}")
    if attribute in _REPOSITORY_FACTORIES:
        raise ValueError(f"postgres repository {attribute!r} already registered")
    _REPOSITORY_FACTORIES[attribute] = factory
    for name, value in (capabilities or {}).items():
        if not hasattr(_BASE_CAPABILITIES, name):
            raise ValueError(f"unknown capability field {name!r}")
        _CAPABILITY_OVERRIDES[name] = value


def missing_repositories() -> tuple[str, ...]:
    """The repository attributes no module has registered yet."""
    return tuple(name for name in CAPABILITY_ATTRIBUTES if name not in _REPOSITORY_FACTORIES)


# SQLSTATEs a retry may get past: serialization failure, deadlock, lock or statement timeout.
_CONTENTION = {"40001", "40P01", "55P03", "57014"}
# The server is going away or not accepting connections yet.
_TRANSIENT = {"57P01", "57P02", "57P03", "53300"}


def _store_error(exc: psycopg.Error) -> PersistenceError:
    """The contract error for one driver error."""
    message = str(exc)
    state = getattr(exc, "sqlstate", None) or ""
    if state in _CONTENTION:
        return StoreContention(message)
    if state == "53100":
        return DiskFull(message)
    if state in _TRANSIENT:
        return TransientStoreError(message)
    if isinstance(exc, psycopg.OperationalError) and not state:
        return StoreUnavailable(message)
    return PermanentStoreError(message)


@contextmanager
def translated() -> Iterator[None]:
    """Raise the store's failures as the contract's error family, chaining the original."""
    try:
        yield
    except PoolTimeout as exc:
        raise StoreContention(str(exc)) from exc
    except psycopg.Error as exc:
        raise _store_error(exc) from exc


_EXECUTOR_WORKERS = min(32, (os.cpu_count() or 1) + 4)
_CONNECT_TIMEOUT_S = 5.0
# How long a verb waits for a free pooled connection before it raises ``StoreContention``.
_CHECKOUT_TIMEOUT_S = 15.0
_executor: Optional[ThreadPoolExecutor] = None
_pools: dict[str, list[Any]] = {}  # dsn -> [pool, references]
_pools_lock = threading.Lock()
_open_lock = threading.Lock()


def _store_executor() -> ThreadPoolExecutor:
    global _executor
    with _pools_lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(max_workers=_EXECUTOR_WORKERS, thread_name_prefix="hermes-postgres-store")
        return _executor


def _acquire_pool(dsn: str) -> ConnectionPool:
    """The DSN's shared pool with one more reference; the first acquisition connects, and a database it cannot
    reach raises ``StoreUnavailable`` within ``_CONNECT_TIMEOUT_S``."""
    with _pools_lock:
        entry = _pools.get(dsn)
        if entry is None:
            pool = ConnectionPool(dsn, min_size=1, max_size=10, open=False, timeout=_CHECKOUT_TIMEOUT_S,
                                  name="hermes-persistence",
                                  kwargs={"autocommit": True, "application_name": "hermes-agent",
                                          "connect_timeout": int(_CONNECT_TIMEOUT_S)})
            entry = _pools[dsn] = [pool, 0]
        entry[1] += 1
        pool = entry[0]
    try:
        # One opener at a time: a pool another backend is still opening is not opened twice.
        with _open_lock:
            pool.open(wait=True, timeout=_CONNECT_TIMEOUT_S)
    except PoolTimeout as exc:
        _release_pool(dsn)
        raise StoreUnavailable(f"postgres persistence: cannot connect within {_CONNECT_TIMEOUT_S:.0f}s "
                               f"(check the server, HERMES_PG_DSN and ~/.pgpass)") from exc
    return pool


def _release_pool(dsn: str) -> None:
    with _pools_lock:
        entry = _pools.get(dsn)
        if entry is None:
            return
        entry[1] -= 1
        if entry[1] > 0:
            return
        del _pools[dsn]
    entry[0].close()


def close_all_pools() -> int:
    """Close every pool in the process regardless of references; returns how many closed."""
    with _pools_lock:
        entries = list(_pools.values())
        _pools.clear()
    for pool, _references in entries:
        pool.close()
    return len(entries)


# Every live backend of this process, for the provider's ``close_stores_under``.
_LIVE_BACKENDS: "weakref.WeakSet[PostgresBackend]" = weakref.WeakSet()


def close_stores_under(directory: Path) -> int:
    """Close, under their open handles, the stores of every live backend whose home lies under ``directory``."""
    return sum(backend.close_stores_under(directory) for backend in list(_LIVE_BACKENDS))


class PostgresBackend:
    """Every store one assembly was built for; ``open(scope)`` binds a handle to one of them."""

    def __init__(self, stores: Mapping[str, Path], settings: PostgresSettings, *, read_only: bool = False,
                 read_patience_s: Optional[float] = None, check_drift: bool = False,
                 owner_root: Optional[Path] = None) -> None:
        """``stores`` maps each served ``hermes_home_key`` to the ``state.db`` path the assembly derived; only the
        key is used. ``read_patience_s`` becomes each read transaction's ``lock_timeout``. ``owner_root`` is the
        Hermes root a store's owning profile is read against (the default root when None)."""
        self._schemas = {key: _schema.schema_name(settings.schema_prefix, key) for key in stores}
        self.settings = settings
        self.owner_root = owner_root
        self.read_only = read_only
        self._read_patience_s = read_patience_s
        self._check_drift = check_drift
        self.executor = _store_executor()
        self._lock = threading.Lock()
        self._acquire_lock = threading.Lock()
        self._pool: Optional[ConnectionPool] = None
        self._ready: set[str] = set()
        self._closed = False
        self.capabilities = dataclasses.replace(_BASE_CAPABILITIES, **_CAPABILITY_OVERRIDES)
        self._closed_under: set[str] = set()
        _LIVE_BACKENDS.add(self)

    def open(self, scope: ProfileScope) -> PostgresStore:
        """A handle bound to ``scope``'s schema; opening touches no connection. A profile this backend holds no
        store for is refused with ``ValueError``."""
        schema = self._schemas.get(scope.hermes_home_key)
        if schema is None:
            raise ValueError(f"this Postgres backend holds no store for {scope.profile_dir}")
        self.require_open()
        return PostgresStore(self, scope, schema)

    def schema_for(self, scope: ProfileScope) -> str:
        return self._schemas[scope.hermes_home_key]

    def close_stores_under(self, directory: Path) -> int:
        """Close the stores of homes under ``directory`` under their open handles (a profile being deleted): every
        later transaction of their handles refuses. Returns how many were closed now."""
        root = Path(directory).resolve()
        closed = 0
        with self._lock:
            for key in self._schemas:
                home = Path(key).resolve()
                if key not in self._closed_under and (home == root or root in home.parents):
                    self._closed_under.add(key)
                    closed += 1
        return closed

    def require_held(self, home_key: str) -> None:
        """``PermanentStoreError`` once the store of ``home_key`` was closed under its handles."""
        with self._lock:
            if home_key in self._closed_under:
                raise PermanentStoreError("this profile's store was closed under its handle")

    def require_open(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("Postgres backend used after close")

    def pool(self) -> ConnectionPool:
        """The DSN's shared pool, acquired on first touch and again after ``reopen``; concurrent first touches
        acquire it once."""
        with self._acquire_lock:
            with self._lock:
                if self._closed:
                    raise RuntimeError("Postgres backend used after close")
                if self._pool is not None:
                    return self._pool
            pool = _acquire_pool(self.settings.dsn)
            with self._lock:
                if not self._closed:
                    self._pool = pool
                    return pool
            _release_pool(self.settings.dsn)
            raise RuntimeError("Postgres backend used after close")
        return self._pool

    def ensure_schema(self, home_key: str, schema: str) -> None:
        """Migrate a writable store's schema once per generation; check a read-only store's schema exists."""
        with self._lock:
            if schema in self._ready:
                return
        with translated(), self.pool().connection() as conn:
            if self.read_only:
                if not _schema.schema_exists(conn, schema):
                    raise StoreUnavailable(f"no Hermes store in schema {schema} (it is created on first write)")
                if self._check_drift and _pending_migrations(conn, schema):
                    from persistence import DeclarativeDrift
                    raise DeclarativeDrift(f"schema {schema} lags this provider's migrations")
            else:
                _schema.migrate(conn, schema, home_key)
        with self._lock:
            self._ready.add(schema)

    def open_store_admin(self, target: ProfileScope | Path, *, read_only: bool = True) -> AdminUnsupported:
        return AdminUnsupported(PHYSICAL_ADMIN, PROVIDER_NAME)

    def open_profile_repair(self, scope: ProfileScope, *, read_only: bool) -> AdminUnsupported:
        return AdminUnsupported(PHYSICAL_ADMIN, PROVIDER_NAME)

    def reopen(self) -> None:
        """Drop the pool reference and the migrated marks; the next touch acquires and checks again."""
        with self._lock:
            pool, self._pool = self._pool, None
            self._ready.clear()
        if pool is not None:
            _release_pool(self.settings.dsn)

    def close(self) -> None:
        """Release the pool reference and refuse further use."""
        with self._lock:
            self._closed = True
            pool, self._pool = self._pool, None
        if pool is not None:
            _release_pool(self.settings.dsn)


def _pending_migrations(conn: psycopg.Connection, schema: str) -> bool:
    row = conn.execute(sql.SQL("SELECT coalesce(max(version), 0) FROM {}.schema_migrations").format(
        sql.Identifier(schema))).fetchone()
    latest = max((version for version, _text in _schema.migrations()), default=0)
    return row[0] < latest


def _owner_profile(profile_dir: Path, root: Optional[Path]) -> Optional[str]:
    """The profile owning a store at ``profile_dir``: ``default`` at the root, the folder name directly under
    ``<root>/profiles`` when it is a valid profile name, else None (an underivable owner is never fabricated)."""
    if root is None:
        from persistence_assembly import store_owner_name

        return store_owner_name(profile_dir)
    from hermes_constants import PROFILE_ID_RE

    if profile_dir == root:
        return "default"
    if profile_dir.parent == root / "profiles" and PROFILE_ID_RE.fullmatch(profile_dir.name):
        return profile_dir.name
    return None


# The open transaction of each store handle in this context, by ``id(handle)``.
_ACTIVE: contextvars.ContextVar[dict[int, psycopg.Connection]] = contextvars.ContextVar("hermes_pg_transaction", default={})


class PostgresStore:
    """One store handle bound to one profile: its repositories, capability facts and awaitable view ``aio``."""

    def __init__(self, backend: PostgresBackend, scope: ProfileScope, schema: str) -> None:
        self.scope = scope
        self.schema = schema
        self.read_only = backend.read_only
        self.capabilities = backend.capabilities
        self.executor = backend.executor
        self._backend = backend
        self.owner_profile = _owner_profile(scope.profile_dir, backend.owner_root)
        self._incarnation: Optional[str] = None
        self.aio = AwaitableStore(self)
        for attribute in CAPABILITY_ATTRIBUTES:
            factory = _REPOSITORY_FACTORIES.get(attribute)
            if factory is not None:
                setattr(self, attribute, factory(self))

    def ready(self) -> None:
        """Connect and migrate now, so an unreachable store raises its typed error here instead of on the first verb."""
        self._backend.ensure_schema(self.scope.hermes_home_key, self.schema)

    def exists(self) -> bool:
        """Whether this profile's schema exists, answered without creating or migrating it."""
        with translated(), self._backend.pool().connection() as conn:
            return _schema.schema_exists(conn, self.schema)

    @contextmanager
    def transaction(self) -> Iterator[psycopg.Connection]:
        """One transaction on this profile's schema: read-only on a reader's handle, rolled back on any exception.

        Re-entrant: a call inside an open transaction of this handle (one verb composing another's steps) runs as a
        savepoint on the same connection, so the composed write commits or rolls back with its caller."""
        active = _ACTIVE.get().get(id(self))
        if active is not None:
            with translated(), active.transaction():
                yield active
            return
        self._backend.require_held(self.scope.hermes_home_key)
        self.ready()
        with translated(), self._backend.pool().connection() as conn, conn.transaction():
            if self.read_only:
                conn.execute("SET TRANSACTION READ ONLY")
                patience = self._backend._read_patience_s
                if patience is not None:
                    conn.execute(sql.SQL("SET LOCAL lock_timeout = {}").format(
                        sql.Literal(f"{max(1, int(patience * 1000))}ms")))
            conn.execute(sql.SQL("SET LOCAL search_path TO {}").format(sql.Identifier(self.schema)))
            if self._incarnation is None:
                self.require_unreplaced(conn)
            token = _ACTIVE.set({**_ACTIVE.get(), id(self): conn})
            try:
                yield conn
            finally:
                _ACTIVE.reset(token)

    def require_unreplaced(self, conn: psycopg.Connection) -> None:
        """``StoreReplaced`` when the store at this handle's schema is no longer the one it first saw."""
        current = conn.execute(sql.SQL("SELECT incarnation FROM {}.store_identity").format(
            sql.Identifier(self.schema))).fetchone()
        token = None if current is None else current[0]
        if self._incarnation is None:
            self._incarnation = token
        elif token != self._incarnation:
            raise StoreReplaced(f"the store of {self.scope.name!r} was replaced under its handle")

    def serves_same_store(self, other: object) -> bool:
        """Whether ``other`` reads and writes the same schema of the same database."""
        return (isinstance(other, PostgresStore) and self.schema == other.schema
                and self._backend.settings.dsn == other._backend.settings.dsn)

    def require_open(self) -> None:
        """Raise ``RuntimeError`` once the backend has closed; touches no connection."""
        self._backend.require_open()
