"""``CoordinationRepository`` on Postgres: compression locks, turn leases, hygiene streaks, handoffs, and the
delegation, obligation and subagent ledgers.

Locks and leases are ``compression_locks`` / ``session_turn_leases`` rows (a turn lease keyed by the conversation's
compression root); a claim deletes a stale row, inserts its own if none is left and reads back whether the row names
the caller, under the row lock. Handoffs are the session row's ``handoff_*`` columns. The ledgers are row-locked table
transactions (``_rowtables``): every public verb is one transaction.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Optional

from psycopg import Connection

from persistence import PersistenceError, SessionId, StoreReplaced
from persistence.authority.coordination import (
    TURN_LEASE_SWEEP_GRACE_SECONDS, HandoffState, HandoffStatus, PendingHandoff,
)
from persistence.core.errors import HandoffUnsettled

from .._backend import PostgresStore, register_repository
from .._rowtables import LedgerTable, atomic, current
from .._sql import execute, row as _row, rows as _rows, scalar
from ..conversation import _lineage
from . import _leases
from ._bindings_lineage import runtime_lineage
from .coordination_delegations import DelegationLedger
from .coordination_obligations import ObligationLedger
from .coordination_subagents import SubagentRecords

logger = logging.getLogger(__name__)


def require_idle(conn: Connection, session_ids: Any) -> Optional[str]:
    """The refusal reason of the first id with live ledger work (``unknown_execution`` wins over ``session_busy``)
    or a live transcript lease, else None."""
    for sid in session_ids:
        states = {item["status"] for item in _rows(
            conn, "SELECT status FROM session_admissions WHERE target_session_id = %s AND status IN ('queued', 'started', 'unknown') "
                  "UNION SELECT status FROM worker_executions WHERE session_id = %s AND status <> 'terminal'", (sid, sid))}
        if "unknown" in states:
            return "unknown_execution"
        if states or _leases.lease_guarded(conn, sid):
            return "session_busy"
    return None


def _all_public(cls: type) -> type:
    """Wrap every public verb of ``cls`` (its own and its mixins') in one ledger transaction."""
    for klass in reversed(cls.__mro__[:-1]):
        for name, value in list(vars(klass).items()):
            if not name.startswith("_") and callable(value):
                setattr(cls, name, atomic(getattr(cls, name)))
    return cls


@_all_public
class PostgresCoordinationRepository(SubagentRecords, DelegationLedger, ObligationLedger):
    """Coordination of the bound profile's schema."""

    def __init__(self, store_handle: PostgresStore) -> None:
        self._store = store_handle

    def _conn(self) -> Connection:
        return current(self._store).conn

    def _table(self, name: str) -> LedgerTable:
        return current(self._store).table(name)

    # ── compression lock ──

    def try_acquire_compression_lock(self, *, session_id: SessionId, holder: str, ttl_seconds: float = 300.0) -> bool:
        if not session_id:
            return False
        conn, now = self._conn(), time.time()
        self._store.require_unreplaced(conn)
        row = _row(conn, "SELECT holder, expires_at FROM compression_locks WHERE session_id = %s FOR UPDATE", (session_id,))
        if row is not None and (row["expires_at"] < now or _leases.holder_process_is_dead(row["holder"])):
            execute(conn, "DELETE FROM compression_locks WHERE session_id = %s", (session_id,))
            logger.warning("Reclaimed stale compression lock for session=%s (holder=%s)", session_id, row["holder"])
        execute(conn, "INSERT INTO compression_locks (session_id, holder, acquired_at, expires_at) VALUES (%s, %s, %s, %s) "
                      "ON CONFLICT (session_id) DO NOTHING", (session_id, holder, now, now + ttl_seconds))
        return scalar(conn, "SELECT holder FROM compression_locks WHERE session_id = %s", (session_id,)) == holder

    def refresh_compression_lock(self, *, session_id: SessionId, holder: str, ttl_seconds: float = 300.0) -> bool:
        if not session_id or not holder:
            return False
        self._store.require_unreplaced(self._conn())
        return bool(execute(self._conn(), "UPDATE compression_locks SET expires_at = %s WHERE session_id = %s AND holder = %s",
                            (time.time() + ttl_seconds, session_id, holder)))

    def release_compression_lock(self, *, session_id: SessionId, holder: str) -> None:
        if session_id:
            self._store.require_unreplaced(self._conn())
            execute(self._conn(), "DELETE FROM compression_locks WHERE session_id = %s AND holder = %s", (session_id, holder))

    def get_compression_lock_holder(self, *, session_id: SessionId) -> Optional[str]:
        if not session_id:
            return None
        row = _row(self._conn(), "SELECT holder, expires_at FROM compression_locks WHERE session_id = %s", (session_id,))
        return row["holder"] if row is not None and row["expires_at"] >= time.time() else None

    # ── turn lease ──

    def try_acquire_turn_lease(self, *, session_id: SessionId, holder: str, ttl_seconds: float = 300.0,
                               patience_s: Optional[float] = None) -> bool:
        if not session_id or not holder:
            return False
        conn, now = self._conn(), time.time()
        execute(conn, "DELETE FROM session_turn_leases WHERE expires_at < %s", (now - TURN_LEASE_SWEEP_GRACE_SECONDS,))
        key = _leases.conversation_key(conn, str(session_id))
        row = _row(conn, "SELECT holder, expires_at FROM session_turn_leases WHERE conversation_id = %s FOR UPDATE", (key,))
        if row is not None and (float(row["expires_at"]) <= now or _leases.holder_process_is_dead(row["holder"])):
            execute(conn, "DELETE FROM session_turn_leases WHERE conversation_id = %s", (key,))
        execute(conn, "INSERT INTO session_turn_leases (conversation_id, holder, acquired_at, expires_at) VALUES (%s, %s, %s, %s) "
                      "ON CONFLICT (conversation_id) DO NOTHING", (key, holder, now, now + max(0.1, float(ttl_seconds))))
        return scalar(conn, "SELECT holder FROM session_turn_leases WHERE conversation_id = %s", (key,)) == holder

    def refresh_turn_lease(self, *, session_id: SessionId, holder: str, ttl_seconds: float = 300.0,
                           patience_s: Optional[float] = None) -> bool:
        if not session_id or not holder:
            return False
        conn = self._conn()
        return bool(execute(conn, "UPDATE session_turn_leases SET expires_at = %s WHERE conversation_id = %s AND holder = %s",
                            (time.time() + max(0.1, float(ttl_seconds)), _leases.conversation_key(conn, str(session_id)), holder)))

    def release_turn_lease(self, *, session_id: SessionId, holder: str) -> None:
        if session_id and holder:
            conn = self._conn()
            execute(conn, "DELETE FROM session_turn_leases WHERE conversation_id = %s AND holder = %s",
                    (_leases.conversation_key(conn, str(session_id)), holder))

    def turn_lease_expires_at(self, *, session_id: SessionId, holder: str) -> Optional[float]:
        if not session_id or not holder:
            return None
        conn = self._conn()
        value = scalar(conn, "SELECT expires_at FROM session_turn_leases WHERE conversation_id = %s AND holder = %s",
                       (_leases.conversation_key(conn, str(session_id)), holder))
        return None if value is None else float(value)

    # ── hygiene streak ──

    def increment_hygiene_failure_streak(self, *, session_key: str) -> int:
        if not session_key:
            return 1
        return int(scalar(self._conn(), "INSERT INTO gateway_hygiene_state (session_key, failure_streak) VALUES (%s, 1) "
                                        "ON CONFLICT (session_key) DO UPDATE SET failure_streak = gateway_hygiene_state.failure_streak + 1 "
                                        "RETURNING failure_streak", (session_key,)))

    def reset_hygiene_failure_streak(self, *, session_key: str) -> None:
        if session_key:
            execute(self._conn(), "DELETE FROM gateway_hygiene_state WHERE session_key = %s", (session_key,))

    # ── handoffs ──

    def _handoff_row(self, session_id: str) -> Optional[dict[str, Any]]:
        return _row(self._conn(), "SELECT handoff_state AS status, handoff_platform AS platform, handoff_error AS error "
                                  "FROM sessions WHERE id = %s FOR UPDATE", (session_id,))

    def _set_handoff(self, session_id: str, **values: Any) -> None:
        assignments = ", ".join(f"handoff_{'state' if name == 'status' else name} = %s" for name in values)
        execute(self._conn(), f"UPDATE sessions SET {assignments} WHERE id = %s", (*values.values(), session_id))

    def _require_settled(self, session_id: str) -> None:
        conn = self._conn()
        reason = require_idle(conn, sorted(runtime_lineage(conn, session_id)))
        if reason is not None:
            raise HandoffUnsettled(reason)

    def check_handoff_settled(self, *, session_id: SessionId) -> None:
        self._require_settled(str(session_id))

    def request_handoff(self, *, session_id: SessionId, platform: str, require_settled: bool = False) -> bool:
        if require_settled:
            self._require_settled(str(session_id))
        row = self._handoff_row(str(session_id))
        if row is None or row["status"] not in (None, HandoffStatus.COMPLETED, HandoffStatus.FAILED):
            return False
        self._set_handoff(str(session_id), status=HandoffStatus.PENDING.value, platform=platform, error=None)
        return True

    def claim_handoff(self, *, session_id: SessionId) -> bool:
        row = self._handoff_row(str(session_id))
        if row is None or row["status"] != HandoffStatus.PENDING:
            return False
        self._set_handoff(str(session_id), status=HandoffStatus.RUNNING.value)
        return True

    def complete_handoff(self, *, session_id: SessionId) -> None:
        if self._handoff_row(str(session_id)) is not None:
            self._set_handoff(str(session_id), status=HandoffStatus.COMPLETED.value, error=None)

    def fail_handoff(self, *, session_id: SessionId, error: str,
                     only_states: Optional[tuple[HandoffStatus, ...]] = None) -> bool:
        row = self._handoff_row(str(session_id))
        if row is None or (only_states and row["status"] not in {str(state) for state in only_states}):
            return False
        self._set_handoff(str(session_id), status=HandoffStatus.FAILED.value, error=error[:500])
        return True

    def reclaim_stale_running_handoffs(self, *, error: str) -> tuple[SessionId, ...]:
        found = _rows(self._conn(), "UPDATE sessions SET handoff_state = %s, handoff_error = %s WHERE handoff_state = %s "
                                    "RETURNING id", (HandoffStatus.FAILED.value, error[:500], HandoffStatus.RUNNING.value))
        return tuple(SessionId(item["id"]) for item in found)

    def get_handoff(self, *, session_id: SessionId) -> Optional[HandoffState]:
        row = _row(self._conn(), "SELECT handoff_state, handoff_platform, handoff_error FROM sessions WHERE id = %s",
                   (session_id,))
        if row is None or row["handoff_state"] is None:
            return None
        return HandoffState(state=HandoffStatus(row["handoff_state"]), platform=row["handoff_platform"],
                            error=row["handoff_error"])

    def list_pending_handoffs(self) -> tuple[PendingHandoff, ...]:
        found = _rows(self._conn(), "SELECT id, handoff_platform, title, started_at FROM sessions WHERE handoff_state = %s "
                                    "ORDER BY started_at", (HandoffStatus.PENDING.value,))
        return tuple(PendingHandoff(session_id=SessionId(item["id"]), platform=item["handoff_platform"], title=item["title"],
                                    started_at=item["started_at"]) for item in found)

    def has_pending_handoffs(self) -> bool:
        return scalar(self._conn(), "SELECT 1 FROM sessions WHERE handoff_state = %s LIMIT 1",
                      (HandoffStatus.PENDING.value,)) is not None


def _tolerant(method: Any, default: Any) -> Any:
    """A verb that answers ``default`` instead of raising when the store is closed or failing; a replaced store still
    raises."""
    @functools.wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return method(self, *args, **kwargs)
        except StoreReplaced:
            raise
        except (RuntimeError, PersistenceError) as exc:
            logger.warning("%s failed: %s", method.__name__, exc)
            return default
    return wrapper


def _peeking(method: Any, default: Any) -> Any:
    """A read that answers ``default`` for a store that does not exist yet, never creating it."""
    @functools.wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        return method(self, *args, **kwargs) if self._store.exists() else default
    return wrapper


for _name, _default in (("try_acquire_compression_lock", False), ("refresh_compression_lock", False),
                        ("release_compression_lock", None), ("reclaim_stale_running_handoffs", ()),
                        ("get_handoff", None), ("list_pending_handoffs", ()), ("has_pending_handoffs", False)):
    setattr(PostgresCoordinationRepository, _name, _tolerant(getattr(PostgresCoordinationRepository, _name), _default))
for _name, _default in (("has_owed_subagents", False), ("list_pinned_subagents", ())):
    setattr(PostgresCoordinationRepository, _name, _peeking(getattr(PostgresCoordinationRepository, _name), _default))

register_repository("coordination", PostgresCoordinationRepository, {"transactional_claims": True})
