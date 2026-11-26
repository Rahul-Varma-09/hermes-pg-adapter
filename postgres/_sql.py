"""Small query helpers over one ``store.transaction()`` connection: rows come back as dicts."""

from __future__ import annotations

import hashlib
from typing import Any, Optional, Sequence

from psycopg import Connection
from psycopg.rows import dict_row


def rows(conn: Connection, query: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, params)
        return cur.fetchall()


def row(conn: Connection, query: str, params: Sequence[Any] = ()) -> Optional[dict[str, Any]]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, params)
        return cur.fetchone()


def scalar(conn: Connection, query: str, params: Sequence[Any] = ()) -> Any:
    found = conn.execute(query, params).fetchone()
    return None if found is None else found[0]


def execute(conn: Connection, query: str, params: Sequence[Any] = ()) -> int:
    """Run one statement; returns the affected row count."""
    return conn.execute(query, params).rowcount


def has_marker(alias: str, key: str) -> str:
    """SQL true when ``alias.model_config`` holds ``key`` with a non-null value (an absent and a JSON-null key read
    alike, as ``dict.get`` does)."""
    return f"coalesce({alias}.model_config -> '{key}', 'null'::jsonb) <> 'null'::jsonb"


def store_text(conn: Connection, text: Optional[str]) -> Optional[str]:
    """Store ``text`` content-addressed in ``system_prompts``; returns its sha256 (None for None)."""
    if text is None:
        return None
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    conn.execute("INSERT INTO system_prompts (hash, prompt) VALUES (%s, %s) ON CONFLICT (hash) DO NOTHING",
                 (digest, text))
    return digest
