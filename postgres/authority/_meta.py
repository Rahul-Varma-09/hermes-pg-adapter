"""The authority ``meta`` key/value space over ``authority_meta``: string keys as written, tuple keys as their JSON
array text, values as jsonb (a string value round-trips as a string)."""

from __future__ import annotations

import json
from typing import Any, Optional, Union

from psycopg import Connection
from psycopg.types.json import Jsonb

from .._sql import execute, rows as _rows, scalar

Key = Union[str, tuple]


def _key(key: Key) -> str:
    return key if isinstance(key, str) else json.dumps(list(key))


def get(conn: Connection, key: Key, *, lock: bool = False) -> Any:
    """The stored value, None when absent."""
    return scalar(conn, "SELECT value FROM authority_meta WHERE key = %s" + (" FOR UPDATE" if lock else ""), (_key(key),))


def has(conn: Connection, key: Key) -> bool:
    return scalar(conn, "SELECT 1 FROM authority_meta WHERE key = %s", (_key(key),)) is not None


def put(conn: Connection, key: Key, value: Any) -> None:
    execute(conn, "INSERT INTO authority_meta (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value = excluded.value",
            (_key(key), Jsonb(value)))


def put_new(conn: Connection, key: Key, value: Any) -> bool:
    """Insert unless the key exists; returns whether it was written (the first write is kept)."""
    return bool(execute(conn, "INSERT INTO authority_meta (key, value) VALUES (%s, %s) ON CONFLICT (key) DO NOTHING",
                        (_key(key), Jsonb(value))))


def delete(conn: Connection, key: Key, *, only_if: Optional[Any] = None) -> bool:
    """Remove the key; with ``only_if``, only while it still holds that value. Returns whether it was removed."""
    if only_if is None:
        return bool(execute(conn, "DELETE FROM authority_meta WHERE key = %s", (_key(key),)))
    return bool(execute(conn, "DELETE FROM authority_meta WHERE key = %s AND value = %s", (_key(key), Jsonb(only_if))))


def scan(conn: Connection, prefix: str) -> list[tuple[str, Any]]:
    """Every ``(key, value)`` whose string key starts with ``prefix``, in key order."""
    found = _rows(conn, "SELECT key, value FROM authority_meta WHERE starts_with(key, %s) ORDER BY key COLLATE \"C\"",
                  (prefix,))
    return [(item["key"], item["value"]) for item in found]


def local_receipt(conn: Connection, session_id: str) -> Optional[dict[str, Any]]:
    """The local creation receipt the bindings capability stores for ``session_id`` (its ``binding:`` key)."""
    from ..conversation.sessions_delete import LOCAL_RECEIPT_PREFIX

    value = get(conn, LOCAL_RECEIPT_PREFIX + session_id)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return None
    return value if isinstance(value, dict) else None
