"""Deletion, empty sweeps, backfills and the ledger seeding seam of the Postgres session repository, mixed into
``PostgresSessionRepository``.

A delete's targets are the session, the rest of its local conversation and every delegate child under them; the
compression-chain variant takes the whole chain both ways. Each delete checks the ledger and retires its targets in one
transaction (``_retirement.retire``).
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional, Sequence

from psycopg import Connection

from persistence import JsonValue, SessionId, SessionWriteGuarded
from persistence.conversation.sessions import RETAG_GATE_PREFIX, RETAG_SOURCE_FROM, RETAG_SOURCE_TO, DeleteManyOutcome

from .._sql import execute, row as _row, rows as _rows, scalar
from ..authority._leases import lease_guarded
from . import _lineage, _retirement

# A local conversation's creation receipt, keyed by its owner (creation) id, names its reset lineage.
LOCAL_RECEIPT_PREFIX = "binding:"
_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_LOWER = "abcdefghijklmnopqrstuvwxyz"


class SessionDeleteMixin:
    """Delete, retirement and backfill verbs; the repository supplies ``_store``."""

    _store: Any

    # ── the ledger seeding seam ──

    def seed_ledger_admission(self, session_id: SessionId, *, status: str = "queued") -> None:
        """Suite seeding seam standing in for the admissions capability: one admission row on the session."""
        with self._store.transaction() as conn:
            count = scalar(conn, "SELECT count(*) FROM session_admissions")
            started = status == "started"
            execute(conn, "INSERT INTO session_admissions (admission_id, request_id, principal_id, target_session_id, "
                          "lineage, payload, digest, intent, status, owner_epoch, generation) "
                          "VALUES (%s, %s, 'seed', %s, '[]', '{}', '', 'queue', %s, %s, %s)",
                    (f"seed-adm-{count}", f"seed-req-{count}", session_id, status,
                     1 if started else None, 1 if started else None))

    # ── delete targets ──

    def _local_conversation_ids(self, conn: Connection, session_id: str) -> list[str]:
        """Every row of the local conversation ``session_id`` belongs to: the creation id, then each segment of its
        receipt lineage; ``[session_id]`` outside one."""
        found = _row(conn, "SELECT chat_id FROM sessions WHERE id = %s", (session_id,))
        owner = found["chat_id"] if found is not None else None
        if not owner or owner == session_id:
            owner = session_id
        from ..authority._meta import local_receipt

        receipt = local_receipt(conn, owner)
        lineage = receipt.get("lineage") if receipt is not None else None
        if not isinstance(lineage, list) or (owner != session_id and session_id not in lineage):
            return [session_id]
        return list(dict.fromkeys([owner, *(sid for sid in lineage if isinstance(sid, str))]))

    def _owned_lineage_ids(self, conn: Connection, session_id: str) -> list[str]:
        return list(dict.fromkeys([session_id, *self._local_conversation_ids(conn, session_id)]))

    def _delegate_children(self, conn: Connection, parent_ids: Sequence[str]) -> list[str]:
        """Delegate children of ``parent_ids`` walked recursively (a ``_delegate_from`` naming a frontier row, or any
        delegate marker on a child of one); the parents themselves never return."""
        seeds = [sid for sid in parent_ids if sid]
        found, frontier = set(seeds), list(seeds)
        while frontier:
            step = [item["id"] for item in _rows(conn, """
                SELECT id FROM sessions
                WHERE NOT id = ANY(%s) AND coalesce(model_config -> '_delegate_from', 'null'::jsonb) <> 'null'::jsonb
                  AND ((jsonb_typeof(model_config -> '_delegate_from') = 'string'
                        AND model_config ->> '_delegate_from' = ANY(%s))
                       OR parent_session_id = ANY(%s))""", (list(found), frontier, frontier))]
            found.update(step)
            frontier = step
        return [sid for sid in found if sid not in seeds]

    def _with_delegates(self, conn: Connection, session_ids: Sequence[str]) -> list[str]:
        return [*session_ids, *self._delegate_children(conn, session_ids)]

    def _expand_compression_lineage(self, conn: Connection, session_ids: Sequence[str]) -> list[str]:
        """Every member of the ids' compression chains, both directions, plus every segment of a local conversation
        they belong to."""
        seen: dict[str, None] = {}
        for sid in session_ids:
            seen.update(dict.fromkeys([sid, *_lineage.lineage_ids(conn, sid)]))
        for sid in list(seen):
            seen.update(dict.fromkeys(self._local_conversation_ids(conn, sid)))
        return list(seen)

    def _targets(self, conn: Connection, session_id: str) -> tuple[str, ...]:
        if scalar(conn, "SELECT 1 FROM sessions WHERE id = %s", (session_id,)) is None:
            return ()
        lineage = self._owned_lineage_ids(conn, session_id)
        delegates = self._delegate_children(conn, lineage)
        return (session_id, *sorted(set(lineage) - {session_id}), *sorted(delegates))

    def delete_targets(self, session_id: SessionId) -> tuple[SessionId, ...]:
        with self._store.transaction() as conn:
            return tuple(SessionId(sid) for sid in self._targets(conn, session_id))

    def _guarded(self, conn: Connection, session_ids: Sequence[str]) -> bool:
        return any(lease_guarded(conn, sid) for sid in session_ids)

    def _display_unchanged(self, session_id: SessionId, rows: Sequence[Mapping[str, JsonValue]]) -> bool:
        """``expected_display``'s per-session check: the session exists and its DISPLAY view reads exactly *rows*."""
        from persistence.conversation.messages import TranscriptRead, TranscriptView

        if self.get(session_id) is None:  # type: ignore[attr-defined]
            return False
        page = TranscriptRead(view=TranscriptView.DISPLAY)
        current = [row.as_wire_row() for row in self._store.messages.read_transcript(session_id, page)]
        return current == [dict(row) for row in rows]

    # ── delete ──

    def delete(self, session_id: SessionId, *, expected_ids: Optional[Sequence[SessionId]] = None,
               expected_display: Optional[Mapping[SessionId, Sequence[Mapping[str, JsonValue]]]] = None,
               refuse_guarded: bool = False, include_compression_chain: bool = False,
               media_root: str) -> tuple[SessionId, ...]:
        if include_compression_chain:
            # The chain variant takes the bulk path, where a guarded root is kept rather than refused.
            outcome = self.delete_many([session_id], skip_guarded=refuse_guarded, include_compression_chain=True,
                                       media_root=media_root)
            return outcome.removed_ids if outcome.deleted else ()
        if expected_display is not None and not all(self._display_unchanged(sid, rows)
                                                    for sid, rows in expected_display.items()):
            return ()
        with self._store.transaction() as conn:
            targets = self._targets(conn, session_id)
            if not targets:
                return ()
            if refuse_guarded and self._guarded(conn, targets):
                raise SessionWriteGuarded(session_id)
            if expected_ids is not None and set(expected_ids) != set(targets):
                return ()
            _retirement.require_ledger_idle(conn, targets)
            _retirement.retire(conn, targets, media_root=media_root)
        return tuple(SessionId(sid) for sid in targets)

    def delete_many(self, session_ids: Sequence[SessionId], *, skip_guarded: bool = False,
                    include_compression_chain: bool = False, media_root: str) -> DeleteManyOutcome:
        with self._store.transaction() as conn:
            requested = list(dict.fromkeys(str(sid) for sid in session_ids if sid))
            # Locked: a concurrent delete of the same rows waits, then sees them gone and counts none of them.
            present = {item["id"] for item in _rows(conn, "SELECT id FROM sessions WHERE id = ANY(%s) ORDER BY id FOR UPDATE",
                                                    (requested,))}
            existing = [sid for sid in requested if sid in present]

            def expand(ids: Sequence[str]) -> list[str]:
                if include_compression_chain:
                    return self._expand_compression_lineage(conn, ids)
                return list(dict.fromkeys(member for sid in ids for member in self._owned_lineage_ids(conn, sid)))

            # A root is kept when it, its lineage or any delegate child the cascade would take is guarded.
            skipped = sorted(sid for sid in existing
                             if skip_guarded and self._guarded(conn, self._with_delegates(conn, expand([sid]))))
            existing = [sid for sid in existing if sid not in skipped]
            doomed = list(dict.fromkeys(self._with_delegates(conn, expand(existing)))) if existing else []
            if doomed:
                _retirement.require_ledger_idle(conn, doomed)
                _retirement.retire(conn, doomed, media_root=media_root)
        return DeleteManyOutcome(deleted=len(existing), skipped_guarded=tuple(SessionId(sid) for sid in skipped),
                                 removed_ids=tuple(SessionId(sid) for sid in doomed))

    # ── empty rows ──

    def _is_empty_candidate(self, conn: Connection, session_id: str) -> bool:
        return scalar(conn, """
            SELECT 1 FROM sessions s WHERE s.id = %s AND s.title IS NULL
              AND NOT EXISTS (SELECT 1 FROM messages m WHERE m.session_id = s.id)
              AND NOT EXISTS (SELECT 1 FROM sessions c WHERE c.parent_session_id = s.id)""", (session_id,)) is not None

    def is_empty_candidate(self, session_id: SessionId) -> bool:
        with self._store.transaction() as conn:
            return self._is_empty_candidate(conn, session_id)

    def delete_if_empty(self, session_id: SessionId, *, media_root: str) -> tuple[SessionId, ...]:
        with self._store.transaction() as conn:
            # A guarded row, or an empty reset tip that is still its conversation's current target, is no candidate.
            if self._guarded(conn, [session_id]) or self._owned_lineage_ids(conn, str(session_id)) != [session_id]:
                return ()
            if not self._is_empty_candidate(conn, session_id):
                return ()
            _retirement.require_ledger_idle(conn, [str(session_id)])
            _retirement.retire(conn, [str(session_id)], media_root=media_root)
        return (session_id,)

    _EMPTY_ENDED_SQL = """
        SELECT s.id FROM sessions s WHERE s.ended_at IS NOT NULL AND NOT s.archived
          AND NOT EXISTS (SELECT 1 FROM messages m WHERE m.session_id = s.id)"""

    def count_empty(self) -> int:
        with self._store.transaction() as conn:
            return int(scalar(conn, f"SELECT count(*) FROM ({self._EMPTY_ENDED_SQL}) empty"))

    def delete_empty(self, *, media_root: str) -> tuple[SessionId, ...]:
        with self._store.transaction() as conn:
            # Skip-busy polarity: one busy row must not abort the sweep; it survives untouched.
            doomed = sorted(item["id"] for item in _rows(conn, self._EMPTY_ENDED_SQL + " FOR UPDATE OF s")
                            if not _retirement.live_ledger_states(conn, item["id"]))
            _retirement.retire(conn, doomed, media_root=media_root)
        return tuple(SessionId(sid) for sid in doomed)

    def discard_unadmitted(self, session_id: SessionId, *, media_root: str) -> tuple[SessionId, ...]:
        with self._store.transaction() as conn:
            ledgered = scalar(conn, "SELECT 1 FROM session_admissions WHERE target_session_id = %s UNION ALL "
                                    "SELECT 1 FROM worker_executions WHERE session_id = %s LIMIT 1",
                              (session_id, session_id)) is not None
            if not ledgered:
                execute(conn, "DELETE FROM messages WHERE session_id = %s", (session_id,))
                execute(conn, "UPDATE sessions SET parent_session_id = NULL WHERE parent_session_id = %s", (session_id,))
                existed = execute(conn, "DELETE FROM sessions WHERE id = %s", (session_id,))
                return (SessionId(str(session_id)),) if existed else ()
        # A real session: the fenced delete path, reporting what it removed.
        return self.delete(session_id, media_root=media_root)

    # ── backfills ──

    def backfill_repo_roots(self, cwd_to_root: Mapping[str, str]) -> None:
        pairs = [(cwd, root) for cwd, root in cwd_to_root.items() if cwd and root]
        with self._store.transaction() as conn:
            for cwd, root in pairs:
                execute(conn, "UPDATE sessions SET git_repo_root = %s WHERE cwd = %s AND coalesce(git_repo_root, '') = ''",
                        (root, cwd))

    def backfill_session_owners(self, *, stamp: Optional[str] = None) -> int:
        stamp = ((self._store.owner_profile if stamp is None else stamp) or "").strip()
        if not stamp:
            return 0
        with self._store.transaction() as conn:
            return execute(conn, "UPDATE sessions SET profile_name = %s WHERE coalesce(trim(profile_name), '') = ''",
                           (stamp,))

    def backfill_acp_session_cwds(self) -> int:
        with self._store.transaction() as conn:
            return execute(conn, "UPDATE sessions SET cwd = model_config ->> 'cwd' WHERE source = 'acp' "
                                 "AND coalesce(cwd, '') = '' AND coalesce(model_config ->> 'cwd', '') <> ''")

    def retag_worker_sessions(self, workspaces_root: str) -> int:
        prefix = str(workspaces_root).rstrip("/\\")
        if not prefix:
            return 0
        gate_key = RETAG_GATE_PREFIX + prefix
        with self._store.transaction() as conn:
            if scalar(conn, "SELECT value FROM state_meta WHERE key = %s", (gate_key,)) == "1":
                return 0
            # ASCII-folded prefix match on either separator, as the SQLite store's LIKE does.
            retagged = execute(conn, """
                UPDATE sessions SET source = %s WHERE source = %s AND cwd IS NOT NULL
                  AND (cwd = %s OR starts_with(translate(cwd, %s, %s), translate(%s, %s, %s))
                       OR starts_with(translate(cwd, %s, %s), translate(%s, %s, %s)))""",
                (RETAG_SOURCE_TO, RETAG_SOURCE_FROM, prefix,
                 _UPPER, _LOWER, prefix + "/", _UPPER, _LOWER, _UPPER, _LOWER, prefix + "\\", _UPPER, _LOWER))
            execute(conn, "INSERT INTO state_meta (key, value) VALUES (%s, '1') ON CONFLICT (key) DO UPDATE SET value = '1'",
                    (gate_key,))
        return retagged
