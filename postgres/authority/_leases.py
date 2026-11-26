"""Lease reads shared by the delete and prune guards: a compaction lease on a session, a turn lease on its
conversation.

A lease is live until it expires, or until its holder's local process is provably gone. Only a holder stamped by this
host's PID namespace is probed (``pidns=<id>`` on Linux; every holder elsewhere), so a remote holder always reads live.
"""

from __future__ import annotations

import os
import re
import sys
import time
from typing import Optional

from psycopg import Connection

from persistence.authority.coordination import turn_lease_holder_pid
from persistence.conversation.sessions import END_REASON_COMPRESSION

from .._sql import row as _row
from ..conversation._lineage import config_of

_PIDNS_TOKEN = re.compile(r"(?:^|:)pidns=(\d+)(?::|$)")


def _local_pid_namespace() -> tuple[Optional[str], bool]:
    """``(id, supported)`` of this process's PID namespace."""
    if sys.platform != "linux":
        return None, False
    try:
        link = os.readlink("/proc/self/ns/pid")
    except OSError:
        return None, True
    match = re.search(r"\[(\d+)\]", link)
    return (match.group(1) if match else None), True


def _holder_pid_checkable(holder: str) -> bool:
    ns, supported = _local_pid_namespace()
    if not supported:
        return True
    match = _PIDNS_TOKEN.search(holder or "")
    return match is not None and ns is not None and match.group(1) == ns


def holder_process_is_dead(holder: str) -> bool:
    """True only when the holder's ``pid=<n>`` names a local process that is provably gone."""
    pid = turn_lease_holder_pid(holder)
    if pid is None or pid == os.getpid() or not _holder_pid_checkable(holder):
        return False
    if os.name == "nt":
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except (OSError, OverflowError):
        return False
    return False


def _is_explicit_fork_child(row) -> bool:
    if row.get("source") == "tool":
        return True
    parent = row.get("parent_session_id")
    config = config_of(row)
    return bool(parent) and parent in (config.get("_branched_from"), config.get("_delegate_from"),
                                       config.get("_reset_from"))


def conversation_key(conn: Connection, session_id: str) -> str:
    """The turn-lease key: walk to the parent while the row is a plain continuation and the parent ended by
    compression; an unknown id is its own key."""
    query = "SELECT id, parent_session_id, source, model_config, end_reason FROM sessions WHERE id = %s"
    current = _row(conn, query, (session_id,))
    key, seen = session_id, {session_id}
    while current is not None:
        parent_id = current.get("parent_session_id")
        if not parent_id or parent_id in seen or _is_explicit_fork_child(current):
            break
        parent = _row(conn, query, (parent_id,))
        if parent is None or parent.get("end_reason") != END_REASON_COMPRESSION:
            break
        seen.add(parent_id)
        key, current = parent_id, parent
    return key


def lease_rows(conn: Connection, session_id: str) -> list[tuple[str, str, str, str, bool]]:
    """``(kind, table, key column, key, live)`` of the compaction lease on the session and the turn lease on its
    conversation, in that check order; absent rows are omitted. Stale = expired or a provably dead local holder."""
    now = time.time()
    found = []
    for kind, table, column, key in (("compaction", "compression_locks", "session_id", session_id),
                                     ("turn", "session_turn_leases", "conversation_id", conversation_key(conn, session_id))):
        lease = _row(conn, f"SELECT holder, expires_at FROM {table} WHERE {column} = %s FOR UPDATE", (key,))
        if lease is not None:
            found.append((kind, table, column, key,
                          not (lease["expires_at"] <= now or holder_process_is_dead(lease["holder"]))))
    return found


def lease_guarded(conn: Connection, session_id: str) -> bool:
    """A live compaction lease on the session or a live turn lease on its conversation, holder-blind."""
    return any(live for *_rest, live in lease_rows(conn, session_id))


def verify_turn_lease(conn: Connection, session_id: str, holder: Optional[str], ttl_seconds: float, *,
                      renew: bool = True) -> None:
    """Raise ``TurnLeaseLost`` unless ``holder`` is empty or the stored turn lease on the session's conversation still
    names it, even expired; with ``renew``, extend an expired row that still names the holder."""
    from persistence import TurnLeaseLost

    if not holder:
        return
    key = conversation_key(conn, session_id)
    lease = _row(conn, "SELECT holder, expires_at FROM session_turn_leases WHERE conversation_id = %s FOR UPDATE", (key,))
    if lease is None or lease["holder"] != holder:
        raise TurnLeaseLost(session_id)
    now = time.time()
    if renew and lease["expires_at"] <= now:
        conn.execute("UPDATE session_turn_leases SET expires_at = %s WHERE conversation_id = %s",
                     (now + max(0.1, float(ttl_seconds)), key))


def verify_compression_lease(conn: Connection, session_id: str, holder: Optional[str]) -> None:
    """Raise ``CompressionInProgress`` unless ``holder`` is None or names the live compression lock on the session;
    an expired lock is lost, never renewed."""
    from persistence import CompressionInProgress

    if holder is None:
        return
    lease = _row(conn, "SELECT holder, expires_at FROM compression_locks WHERE session_id = %s FOR UPDATE", (session_id,))
    if lease is None or lease["expires_at"] <= time.time() or lease["holder"] != holder:
        raise CompressionInProgress(session_id)
