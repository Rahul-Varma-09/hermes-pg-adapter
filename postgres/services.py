"""Store services that answer without opening a store: the readiness probe.

The probe uses its own short-lived connection, never the pool, and only reads the catalog: it creates, migrates and
repairs nothing. Postgres guards its own pages, so there is no corruption latch to hold; ``storage_state`` is ``OK``.
"""

from __future__ import annotations

import psycopg

from persistence import StorageProbe, StorageState

from . import _schema
from ._backend import register_repository

__all__ = ["PostgresStorageHealth"]

_PROBE_TIMEOUT_S = 2


class PostgresStorageHealth:
    """``StorageHealth`` for one profile's schema."""

    def __init__(self, dsn: str, schema: str) -> None:
        self._dsn = dsn
        self._schema = schema

    def storage_state(self) -> StorageState:
        return StorageState.OK

    def probe(self) -> StorageProbe:
        try:
            with psycopg.connect(self._dsn, connect_timeout=_PROBE_TIMEOUT_S, autocommit=True) as conn:
                if not _schema.schema_exists(conn, self._schema):
                    return StorageProbe(status="ok", detail="not initialized")
            return StorageProbe(status="ok", detail=None)
        except Exception as exc:  # health: allow BLE001 -- any probe failure is reported, never raised
            return StorageProbe(status="degraded", detail=type(exc).__name__)


register_repository("health", lambda store: PostgresStorageHealth(store._backend.settings.dsn, store.schema))
