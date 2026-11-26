"""``ProfileIdentityRepository`` on Postgres: renaming and purging a profile's identity inside this store.

Both verbs are one transaction. A rename checks every collision before its first write, then moves the owner tags
(sessions, heartbeats, delivery obligations, topic tables), the ``agent:<old>:`` key namespace (sessions, delivery
obligations, topic bindings, routing entries), the ``profile`` of stored origins (sessions, routing payloads, receipt
entries) and the API binding and local creation receipts. Every facet's table exists here, so every count is present.
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional

from psycopg import Connection

from persistence.housekeeping.profile_identity import (
    ProfileIdentityCollision, ProfilePurgeCounts, ProfileRekeyCounts, profile_namespace,
)

from .._backend import PostgresStore, register_repository
from .._sql import execute, rows as _rows
from ..authority import _meta
from ..authority._bindings_lineage import RECEIPT_PREFIX, decode
from ..authority.bindings import API_BINDING_PREFIX


def _rekeyed(key: Any, old_ns: str, new_ns: str) -> Any:
    return new_ns + key[len(old_ns):] if isinstance(key, str) and key.startswith(old_ns) else key


def _parse(text: Any) -> Any:
    try:
        return json.loads(text)
    except (TypeError, ValueError, RecursionError):
        return None


def _rekeyed_receipt(raw: Any, old: str, new: str, old_ns: str, new_ns: str) -> Optional[dict[str, Any]]:
    """``raw`` with its ``route`` and ``entry['session_key']`` moved out of ``old_ns`` and its entry origin's
    ``profile`` renamed; None when nothing changed or it does not decode as an object."""
    try:
        receipt = decode(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(receipt, dict):
        return None
    receipt, changed = json.loads(json.dumps(receipt)), False
    if isinstance(receipt.get("route"), str) and receipt["route"].startswith(old_ns):
        receipt["route"], changed = _rekeyed(receipt["route"], old_ns, new_ns), True
    entry = receipt.get("entry")
    if isinstance(entry, dict):
        if isinstance(entry.get("session_key"), str) and entry["session_key"].startswith(old_ns):
            entry["session_key"], changed = _rekeyed(entry["session_key"], old_ns, new_ns), True
        origin = entry.get("origin")
        if isinstance(origin, dict) and origin.get("profile") == old:
            origin["profile"], changed = new, True
    return receipt if changed else None


def _rekeyed_entry(text: str, old: str, new: str, old_ns: str, new_ns: str) -> str:
    """A routing entry's text with its stored ``session_key`` moved and its origin's ``profile`` renamed."""
    parsed = _parse(text)
    if not isinstance(parsed, dict):
        return text
    changed = {}
    if isinstance(parsed.get("session_key"), str) and parsed["session_key"].startswith(old_ns):
        changed["session_key"] = _rekeyed(parsed["session_key"], old_ns, new_ns)
    origin = parsed.get("origin")
    if isinstance(origin, dict) and origin.get("profile") == old:
        changed["origin"] = {**origin, "profile": new}
    return json.dumps({**parsed, **changed}) if changed else text


class PostgresProfileIdentityRepository:
    """Profile rename and purge over the bound profile's schema."""

    def __init__(self, store: PostgresStore) -> None:
        self._store = store

    @staticmethod
    def _check_collisions(conn: Connection, old: str, new: str, old_ns: str, new_ns: str) -> None:
        clash = _rows(conn, "SELECT a.scope, %(new)s || substr(a.session_key, %(cut)s) AS target FROM gateway_routing a "
                            "JOIN gateway_routing b ON b.scope = a.scope AND b.session_key = %(new)s || substr(a.session_key, %(cut)s) "
                            "WHERE starts_with(a.session_key, %(old)s) ORDER BY a.scope, a.session_key LIMIT 1",
                      {"new": new_ns, "old": old_ns, "cut": len(old_ns) + 1})
        if clash:
            raise ProfileIdentityCollision(new, f"profile routing collision in scope {clash[0]['scope']!r}: {clash[0]['target']!r}")
        if _rows(conn, "SELECT 1 FROM telegram_dm_topic_mode a JOIN telegram_dm_topic_mode b "
                       "ON b.profile_name = %s AND b.chat_id = a.chat_id WHERE a.profile_name = %s LIMIT 1", (new, old)):
            raise ProfileIdentityCollision(new, "profile identity collision in telegram_dm_topic_mode")
        if _rows(conn, "SELECT 1 FROM telegram_dm_topic_bindings a JOIN telegram_dm_topic_bindings b "
                       "ON b.profile_name = %s AND b.chat_id = a.chat_id AND b.thread_id = a.thread_id "
                       "WHERE a.profile_name = %s LIMIT 1", (new, old)):
            raise ProfileIdentityCollision(new, "profile identity collision in telegram_dm_topic_bindings")

    @staticmethod
    def _rekey_column(conn: Connection, table: str, column: str, old_ns: str, new_ns: str) -> int:
        return execute(conn, f"UPDATE {table} SET {column} = %s || substr({column}, %s) WHERE starts_with({column}, %s)",
                       (new_ns, len(old_ns) + 1, old_ns))

    @staticmethod
    def _rekey_origins(conn: Connection, old: str, new: str) -> int:
        moved = 0
        for item in _rows(conn, "SELECT id, origin_json FROM sessions WHERE origin_json IS NOT NULL "
                                "AND strpos(origin_json, %s) > 0", (old,)):
            origin = _parse(item["origin_json"])
            if isinstance(origin, dict) and origin.get("profile") == old:
                execute(conn, "UPDATE sessions SET origin_json = %s WHERE id = %s",
                        (json.dumps({**origin, "profile": new}), item["id"]))
                moved += 1
        return moved

    @staticmethod
    def _rekey_routing(conn: Connection, old: str, new: str, old_ns: str, new_ns: str) -> int:
        moved = _rows(conn, "DELETE FROM gateway_routing WHERE starts_with(session_key, %s) "
                            "RETURNING scope, session_key, entry_json, session_id, updated_at", (old_ns,))
        for item in moved:
            execute(conn, "INSERT INTO gateway_routing (scope, session_key, entry_json, session_id, updated_at) "
                          "VALUES (%s, %s, %s, %s, %s)",
                    (item["scope"], _rekeyed(item["session_key"], old_ns, new_ns),
                     _rekeyed_entry(item["entry_json"], old, new, old_ns, new_ns), item["session_id"], item["updated_at"]))
        return len(moved)

    @staticmethod
    def _rekey_receipts(conn: Connection, prefix: str, *names: str) -> int:
        moved = 0
        for key, raw in _meta.scan(conn, prefix):
            if (receipt := _rekeyed_receipt(raw, *names)) is not None:
                _meta.put(conn, key, receipt)
                moved += 1
        return moved

    def rekey_profile(self, old_name: str, new_name: str) -> ProfileRekeyCounts:
        old, new = (old_name or "").strip(), (new_name or "").strip()
        if not old or not new or old == new:
            return ProfileRekeyCounts()
        old_ns, new_ns = profile_namespace(old), profile_namespace(new)
        names = (old, new, old_ns, new_ns)
        with self._store.transaction() as conn:
            self._check_collisions(conn, *names)
            owners = execute(conn, "UPDATE sessions SET profile_name = %s WHERE profile_name = %s", (new, old))
            keys = self._rekey_column(conn, "sessions", "session_key", old_ns, new_ns)
            origins = self._rekey_origins(conn, old, new)
            heartbeats = execute(conn, "UPDATE gateway_heartbeats SET profile = %s WHERE profile = %s", (new, old))
            obligation_owners = execute(conn, "UPDATE delivery_obligations SET adapter_profile = %s WHERE adapter_profile = %s",
                                        (new, old))
            obligation_keys = self._rekey_column(conn, "delivery_obligations", "session_key", old_ns, new_ns)
            modes = execute(conn, "UPDATE telegram_dm_topic_mode SET profile_name = %s WHERE profile_name = %s", (new, old))
            binding_keys = self._rekey_column(conn, "telegram_dm_topic_bindings", "session_key", old_ns, new_ns)
            binding_owners = execute(conn, "UPDATE telegram_dm_topic_bindings SET profile_name = %s WHERE profile_name = %s",
                                     (new, old))
            routing = self._rekey_routing(conn, *names)
            api = self._rekey_receipts(conn, API_BINDING_PREFIX, *names)
            local = self._rekey_receipts(conn, RECEIPT_PREFIX, *names)
        return ProfileRekeyCounts(
            sessions_profile_name=owners, gateway_heartbeats_profile=heartbeats, sessions_session_key=keys,
            sessions_origin_json=origins, delivery_obligations_adapter_profile=obligation_owners,
            delivery_obligations_session_key=obligation_keys, telegram_dm_topic_mode_profile_name=modes,
            telegram_dm_topic_bindings_profile_name=binding_owners, telegram_dm_topic_bindings_session_key=binding_keys,
            gateway_routing=routing, api_binding_receipts=api, local_policy_receipts=local)

    def purge_profile(self, profile: str) -> ProfilePurgeCounts:
        name = (profile or "").strip()
        if not name:
            return ProfilePurgeCounts()
        ns = profile_namespace(name)
        with self._store.transaction() as conn:
            routing = execute(conn, "DELETE FROM gateway_routing WHERE starts_with(session_key, %s)", (ns,))
            heartbeats = execute(conn, "DELETE FROM gateway_heartbeats WHERE profile = %s", (name,))
            obligations = execute(conn, "UPDATE delivery_obligations SET state = 'abandoned', updated_at = %s "
                                        "WHERE state NOT IN ('delivered', 'abandoned') "
                                        "AND (adapter_profile = %s OR starts_with(session_key, %s))", (time.time(), name, ns))
            modes = execute(conn, "DELETE FROM telegram_dm_topic_mode WHERE profile_name = %s", (name,))
            bindings = execute(conn, "DELETE FROM telegram_dm_topic_bindings WHERE profile_name = %s "
                                     "OR starts_with(session_key, %s)", (name, ns))
        return ProfilePurgeCounts(gateway_routing=routing, gateway_heartbeats=heartbeats, delivery_obligations=obligations,
                                  telegram_dm_topic_mode=modes, telegram_dm_topic_bindings=bindings)


register_repository("profile_identity", PostgresProfileIdentityRepository)
