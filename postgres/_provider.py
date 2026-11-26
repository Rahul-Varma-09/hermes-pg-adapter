"""The ``PersistenceProvider`` Hermes loads for ``persistence.provider: postgres``."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import psycopg
import psycopg.sql

from persistence import ProfileScope, StoreUnavailable
from persistence.core.provider import PersistenceProvider

from . import _backend, _schema
from ._settings import read_settings
from .services import PostgresStorageHealth


def compat_schema(profile_schema: str, store: str) -> str:
    """The schema of a profile's file store, within Postgres' 63-byte names (a long name keeps a hash of the rest)."""
    import hashlib

    name = f"{profile_schema}__{store}"
    if len(name) <= 63:
        return name
    return name[:52] + "_" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:10]


# One idle autocommit connection per DSN for the change token, which watchers poll every few seconds.
_TOKEN_CONNECTIONS: dict[str, psycopg.Connection] = {}
_TOKEN_LOCK = threading.Lock()


def _cached_query(dsn: str, query: Callable[[psycopg.Connection], Any]) -> Any:
    """``query`` run on this process's idle autocommit connection to ``dsn``, reconnecting once if it dropped."""
    with _backend.translated(), _TOKEN_LOCK:
        for attempt in (1, 2):
            conn = _TOKEN_CONNECTIONS.get(dsn)
            try:
                if conn is None or conn.closed:
                    conn = _TOKEN_CONNECTIONS[dsn] = psycopg.connect(dsn, connect_timeout=5, autocommit=True)
                return query(conn)
            except psycopg.OperationalError:
                _TOKEN_CONNECTIONS.pop(dsn, None)
                if attempt == 2:
                    raise
    return None


class PostgresProvider(PersistenceProvider):
    name = _backend.PROVIDER_NAME

    def assemble(self, stores: Mapping[str, Path], settings: Mapping[str, Mapping[str, Any]], *, read_only: bool = False,
                 read_patience_s: Optional[float] = None, check_drift: bool = False) -> _backend.PostgresBackend:
        """A backend over ``stores``. Every store of one assembly must name the same database; until every
        repository is ported the provider refuses to serve unless ``allow_incomplete`` is set."""
        resolved = {read_settings(section) for section in settings.values()}
        if len(resolved) != 1:
            raise StoreUnavailable("postgres persistence: one assembly's profiles name different databases")
        (config,) = resolved
        missing = _backend.missing_repositories()
        if missing and not config.allow_incomplete:
            raise StoreUnavailable(
                "postgres persistence is not complete yet; missing repositories: " + ", ".join(missing)
                + " (set persistence.allow_incomplete: true only for development)")
        return _backend.PostgresBackend(stores, config, read_only=read_only, read_patience_s=read_patience_s,
                                        check_drift=check_drift)

    def storage_health(self, scope: ProfileScope, settings: Mapping[str, Any]) -> PostgresStorageHealth:
        config = read_settings(settings)
        return PostgresStorageHealth(config.dsn, _schema.schema_name(config.schema_prefix, scope.hermes_home_key))

    def export_store(self, scope: ProfileScope, settings: Mapping[str, Any], destination: Path) -> bool:
        from ._dump import export_store

        return export_store(scope, settings, destination)

    def import_store(self, scope: ProfileScope, settings: Mapping[str, Any], source: Path) -> None:
        from ._dump import import_store

        import_store(scope, settings, source)

    def open_compat_store(self, scope: Optional[ProfileScope], settings: Mapping[str, Any], store: str, *,
                          row_factory: Any = None) -> Any:
        """The profile's file store ``store`` as a SQLite-compatible connection on schema ``<profile schema>__<store>``
        (``scope`` None: ``store`` is the schema itself, for the test suites)."""
        from .compat import Connection

        config = read_settings(settings)
        schema = store if scope is None else compat_schema(_schema.schema_name(config.schema_prefix, scope.hermes_home_key),
                                                           store)
        return Connection(config.dsn, schema, row_factory=row_factory)

    def compat_store_exists(self, scope: Optional[ProfileScope], settings: Mapping[str, Any], store: str) -> bool:
        """Whether the file store ``store`` holds any table yet: the provider's answer to a SQLite file's existing."""
        config = read_settings(settings)
        schema = self._compat_schema(scope, config, store)
        return bool(_cached_query(config.dsn, lambda conn: conn.execute(
            "SELECT EXISTS (SELECT 1 FROM pg_tables WHERE schemaname = %s "
            "AND tablename NOT IN ('sqlite_master', 'hermes_compat_meta'))", (schema,)).fetchone()[0]))

    def move_compat_store(self, scope: Optional[ProfileScope], settings: Mapping[str, Any], store: str,
                          new_store: str) -> bool:
        """Rename file store ``store`` to ``new_store`` (a board archived to another folder); False when there was
        nothing to move. Refuses when ``new_store`` already exists."""
        config = read_settings(settings)
        old, new = (self._compat_schema(scope, config, name) for name in (store, new_store))

        def rename(conn: psycopg.Connection) -> bool:
            if not _schema.schema_exists(conn, old):
                return False
            if _schema.schema_exists(conn, new):
                raise FileExistsError(f"file store {new_store!r} already exists")
            conn.execute(psycopg.sql.SQL("ALTER SCHEMA {} RENAME TO {}").format(psycopg.sql.Identifier(old),
                                                                                psycopg.sql.Identifier(new)))
            return True

        return bool(_cached_query(config.dsn, rename))

    def drop_compat_store(self, scope: Optional[ProfileScope], settings: Mapping[str, Any], store: str) -> bool:
        """Delete file store ``store`` and everything in it (a board deleted for good); False when absent."""
        config = read_settings(settings)
        schema = self._compat_schema(scope, config, store)

        def drop(conn: psycopg.Connection) -> bool:
            if not _schema.schema_exists(conn, schema):
                return False
            conn.execute(psycopg.sql.SQL("DROP SCHEMA {} CASCADE").format(psycopg.sql.Identifier(schema)))
            return True

        return bool(_cached_query(config.dsn, drop))

    @staticmethod
    def _compat_schema(scope: Optional[ProfileScope], config: Any, store: str) -> str:
        return store if scope is None else compat_schema(_schema.schema_name(config.schema_prefix, scope.hermes_home_key),
                                                         store)

    def change_token(self, scope: ProfileScope, settings: Mapping[str, Any]) -> Optional[str]:
        """The database's next transaction id: every committed write anywhere in it moves the token, and reads take
        none, so a cache keyed on it never misses a change of this profile's store (it may refresh for another's)."""
        config = read_settings(settings)
        schema = _schema.schema_name(config.schema_prefix, scope.hermes_home_key)
        return _cached_query(config.dsn, lambda conn: str(conn.execute(
            "SELECT pg_snapshot_xmax(pg_current_snapshot())").fetchone()[0]) if _schema.schema_exists(conn, schema) else None)

    def close_stores_under(self, directory: Path) -> int:
        return _backend.close_stores_under(directory)

    def close_all_stores(self) -> int:
        with _TOKEN_LOCK:
            for conn in _TOKEN_CONNECTIONS.values():
                conn.close()
            _TOKEN_CONNECTIONS.clear()
        return _backend.close_all_pools()
