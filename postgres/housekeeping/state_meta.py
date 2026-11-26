"""``StateMetaRepository`` on Postgres: the core's bookkeeping keys in ``state_meta``.

``list_prefix`` matches like SQLite's ``LIKE``: ASCII letters fold, every other character is compared as is, and rows
read in insertion order (``seq``; a ``set`` on a stored key keeps its place).
"""

from __future__ import annotations

from typing import Optional

from persistence.housekeeping.state_meta import StateMetaEntry

from .._backend import PostgresStore, register_repository
from .._sql import execute, rows as _rows, scalar

_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_LOWER = "abcdefghijklmnopqrstuvwxyz"


def _require(key: str, what: str = "key") -> None:
    if not key:
        raise ValueError(f"state_meta {what} must be non-empty")


class PostgresStateMetaRepository:
    """Core-owned bookkeeping keys of the bound profile's schema."""

    def __init__(self, store: PostgresStore) -> None:
        self._store = store

    def get(self, key: str) -> Optional[str]:
        _require(key)
        with self._store.transaction() as conn:
            return scalar(conn, "SELECT value FROM state_meta WHERE key = %s", (key,))

    def set(self, key: str, value: str) -> None:
        _require(key)
        with self._store.transaction() as conn:
            execute(conn, "INSERT INTO state_meta (key, value) VALUES (%s, %s) "
                          "ON CONFLICT (key) DO UPDATE SET value = excluded.value", (key, value))

    def list_prefix(self, prefix: str) -> tuple[StateMetaEntry, ...]:
        _require(prefix, "prefix")
        with self._store.transaction() as conn:
            found = _rows(conn, "SELECT key, value FROM state_meta "
                                "WHERE translate(left(key, %s), %s, %s) = translate(%s, %s, %s) ORDER BY seq",
                          (len(prefix), _UPPER, _LOWER, prefix, _UPPER, _LOWER))
        return tuple(StateMetaEntry(item["key"], item["value"]) for item in found)


register_repository("state_meta", PostgresStateMetaRepository)
