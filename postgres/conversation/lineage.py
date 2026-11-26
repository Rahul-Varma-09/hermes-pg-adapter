"""``LineageRepository`` on Postgres: compression chains over the ``sessions`` table and the compression publish.

It owns no table: every walk reads the session rows, and continuation edges are evaluated from the parent's
``end_reason`` and the child's fork markers when read (``_lineage``). The publish writes the continuation's session
row, its handoff rows and the parent's tail clones, closes the parent and refreshes the compression lease, in one
transaction under the parent's row lock.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from typing import Any, Mapping, Optional, Sequence

from psycopg import Connection
from psycopg.types.json import Jsonb

from persistence import CompressionBusy, CompressionPublishRefused, Conflict, RowVisibility, SessionId
from persistence.conversation.lineage import CompressionChild
from persistence.conversation.messages import ConversationMessage, NewMessage, RowStamp
from persistence.conversation.sessions import (
    AUTOMATIC_END_REASONS, END_REASON_COMPRESSION, END_REASON_ORPHANED_COMPRESSION, SessionRecord,
)
from persistence.core.identity import MessageId

from .._backend import PostgresStore, register_repository
from .._sql import execute, row as _row, rows as _rows, scalar, store_text
from . import _lineage, _transcript as store
from ..authority._meta import local_receipt

_ORPHAN_FINALIZE_AGE_SECONDS = 7 * 86400.0
_RESUME_DESCENT_DEPTH = 32
# model_config keys that belong to the whole conversation: a rotation carries them onto the child.
_CONVERSATION_MODEL_CONFIG_KEYS = ("onboarding_run", "prompt_refresh_epoch")
# Columns a continuation copies from its parent.
_COPIED = ("git_branch", "git_repo_root", "user_id", "session_key", "chat_id", "chat_type", "thread_id", "display_name",
           "origin_json", "tool_names")
_COUNTERS = ("compression_fallback_streak", "compression_ineffective_count", "compression_overload_streak")


def _non_continuation(row: Mapping[str, Any], parent_id: str) -> bool:
    """A child that is no compression continuation of ``parent_id``: a marker naming it, or a tool row."""
    config = _lineage.config_of(row)
    return row.get("source") == "tool" or any(config.get(marker) == parent_id for marker in _lineage.FORK_MARKERS)


def _carry_parent_timestamps(parent_rows: Sequence[store.StoredRow], handoff: Sequence[NewMessage]) -> list[NewMessage]:
    """Handoff rows without a timestamp take the matching parent row's, so the child copy keeps the original's display
    identity: donors are the parent's active rows by role, content, tool call id, tool calls and tool name."""
    donors: dict[tuple[Any, ...], list[float]] = {}
    for row in parent_rows:
        if row.visibility is not RowVisibility.ACTIVE:
            continue
        m = row.message
        key = (m.role, store.hashable(m.content), m.tool_call_id, store.hashable(list(m.tool_calls)), m.tool_name)
        donors.setdefault(key, []).append(row.epoch)
    carried: list[NewMessage] = []
    for message in handoff:
        stored = store.payload(message)
        queue = donors.get((stored.role, store.hashable(stored.content), stored.tool_call_id,
                            store.hashable(list(stored.tool_calls)), stored.tool_name))
        carried.append(message if message.timestamp is not None or not queue else replace(message, timestamp=queue.pop(0)))
    return carried


class PostgresLineageRepository:
    """Compression lineages of the bound profile's schema."""

    def __init__(self, store_handle: PostgresStore) -> None:
        self._store = store_handle

    # ── walks ──

    def chain(self, session_id: SessionId) -> tuple[SessionId, ...]:
        if not session_id:
            return ()
        with self._store.transaction() as conn:
            return tuple(SessionId(sid) for sid in self._chain(conn, str(session_id)))

    @staticmethod
    def _chain(conn: Connection, session_id: str) -> list[str]:
        walked = [session_id]
        for _ in range(_lineage.CHAIN_CAP):
            child = _lineage.chain_step(conn, walked[-1])
            if child is None or child in walked:
                break
            walked.append(child)
        return walked

    def tip(self, session_id: SessionId) -> SessionId:
        walked = self.chain(session_id)
        return walked[-1] if walked else session_id

    def lineage(self, session_id: SessionId) -> tuple[SessionId, ...]:
        with self._store.transaction() as conn:
            session = _lineage.lineage_row(conn, session_id)
            if session is None:
                return ()
            if _lineage.is_fork_child(session):
                return (session_id,)
            root, ancestors = session, {session["id"]}
            while _lineage.is_compression_child(conn, root):
                parent = _lineage.lineage_row(conn, root["parent_session_id"])
                if parent is None or parent["id"] in ancestors:
                    break
                root = parent
                ancestors.add(root["id"])
            walked, current = [root["id"]], root
            while current["end_reason"] == END_REASON_COMPRESSION:
                children = _rows(conn, "SELECT id, parent_session_id, source, model_config, end_reason, ended_at, "
                                       "started_at FROM sessions WHERE parent_session_id = %s ORDER BY started_at",
                                 (current["id"],))
                following = next((child for child in children if _lineage.is_compression_child(conn, child)), None)
                if following is None or following["id"] in walked:
                    break
                walked.append(following["id"])
                current = following
        return tuple(SessionId(sid) for sid in walked) if session_id in walked else (session_id,)

    def conversation_root(self, session_id: SessionId) -> SessionId:
        walked: list[str] = []
        current: Optional[str] = str(session_id)
        with self._store.transaction() as conn:
            while current and current not in walked and len(walked) < _lineage.WALK_NODES:
                walked.append(current)
                row = _row(conn, "SELECT parent_session_id FROM sessions WHERE id = %s", (current,))
                if row is None:
                    break
                current = row["parent_session_id"]
        return SessionId(walked[-1]) if walked else session_id

    def tool_rotation_root(self, session_id: SessionId) -> SessionId:
        with self._store.transaction() as conn:
            row, seen = _lineage.lineage_row(conn, session_id), {str(session_id)}
            while row is not None and row["source"] == "tool" and (parent_id := row["parent_session_id"]) not in seen:
                parent = _lineage.lineage_row(conn, parent_id) if parent_id else None
                if (parent is None or parent["source"] != "tool"
                        or not _lineage.is_compression_child(conn, {**row, "source": None})):
                    break
                seen.add(parent_id)
                row = parent
        return SessionId(row["id"]) if row is not None else session_id

    def is_fork_child(self, session_id: SessionId) -> bool:
        with self._store.transaction() as conn:
            row = _lineage.lineage_row(conn, session_id)
        return row is not None and _lineage.is_fork_child(row)

    def is_compression_child(self, child: SessionRecord) -> bool:
        edge = {"source": child.source, "model_config": dict(child.model_config),
                "parent_session_id": child.parent_session_id}
        with self._store.transaction() as conn:
            return _lineage.is_compression_child(conn, edge)

    def resolve_resume_target(self, session_id: SessionId) -> SessionId:
        if not session_id:
            return session_id
        start = str(self.tip(session_id))
        with self._store.transaction() as conn:
            current, seen, best = start, {start}, None
            for _ in range(_RESUME_DESCENT_DEPTH):
                if scalar(conn, "SELECT 1 FROM messages WHERE session_id = %s LIMIT 1", (current,)) is not None:
                    best = current
                parent = _lineage.lineage_row(conn, current)
                children = [child for child in _rows(
                    conn, "SELECT id, parent_session_id, source, model_config, end_reason, ended_at, started_at, "
                          "last_activity_at, session_key FROM sessions WHERE parent_session_id = %s "
                          "ORDER BY started_at DESC, id DESC", (current,))
                    if parent is not None and _lineage._descent_child(parent, child)]
                if not children or children[0]["id"] in seen:
                    break
                current = children[0]["id"]
                seen.add(current)
        return SessionId(best if best is not None else start)

    def resume_lineage_ids(self, session_id: SessionId) -> tuple[SessionId, ...]:
        with self._store.transaction() as conn:
            return tuple(SessionId(sid) for sid in reversed(_lineage.resume_lineage_ids(conn, str(session_id))))

    def delegated_child_lineage_ids(self, session_id: SessionId) -> tuple[SessionId, ...]:
        with self._store.transaction() as conn:
            row = _lineage.lineage_row(conn, session_id)
        if row is None or not _lineage.config_of(row).get("_delegate_from"):
            return (session_id,)
        return self.resume_lineage_ids(session_id)

    def delegate_descendant_ids(self, session_id: SessionId) -> tuple[SessionId, ...]:
        with self._store.transaction() as conn:
            seeds: list[str] = []
            current: Optional[str] = str(session_id)
            while current and current not in seeds and len(seeds) < _lineage.WALK_NODES:
                row = _lineage.lineage_row(conn, current)
                config = _lineage.config_of(row) if row is not None else {}
                if config.get("_delegate_from"):
                    break
                seeds.append(current)
                if row is None or config.get("_branched_from"):
                    break
                current = row["parent_session_id"]
            found: list[str] = []
            frontier = seeds
            repository = self._store.sessions
            while frontier:
                children = [child for child in repository._delegate_children(conn, frontier) if child not in found]
                segments = [segment for child in children for segment in self._chain(conn, child)]
                frontier = [segment for segment in dict.fromkeys(segments) if segment not in found]
                found.extend(frontier)
        return tuple(SessionId(sid) for sid in found)

    def current_segments(self, session_ids: Sequence[SessionId]) -> Mapping[SessionId, SessionId]:
        out: dict[SessionId, SessionId] = {}
        with self._store.transaction() as conn:
            for sid in dict.fromkeys(str(s) for s in session_ids):
                row = _row(conn, "SELECT end_reason FROM sessions WHERE id = %s", (sid,))
                if row is None:
                    continue
                receipt = local_receipt(conn, sid)
                try:
                    entry = receipt["entry"]["session_id"] if receipt is not None else None
                except (KeyError, TypeError):
                    entry = None
                if isinstance(entry, str) and entry:
                    out[SessionId(sid)] = SessionId(entry)
                elif row["end_reason"] == END_REASON_COMPRESSION:
                    out[SessionId(sid)] = SessionId(self._chain(conn, sid)[-1])
                else:
                    out[SessionId(sid)] = SessionId(sid)
        return out

    def ancestor_display_prefix(self, session_id: SessionId) -> tuple[ConversationMessage, ...]:
        with self._store.transaction() as conn:
            ids = list(reversed(_lineage.resume_lineage_ids(conn, str(session_id))))
            if len(ids) <= 1:
                return ()
            rows = store.dedupe_display(store.load(conn, ids))
        ancestor_ids = {row.id for row in rows if row.session_id != str(session_id)}
        if not ancestor_ids:
            return ()
        dicts = store.conversation_dicts(rows, include_lineage=True, repair_alternation=False, include_row_ids=True,
                                         include_summary_markers=False)
        return tuple(ConversationMessage.from_openai_message({k: v for k, v in message.items() if k != "_row_id"})
                     for message in dicts if message.get("_row_id") in ancestor_ids)

    # ── the compression publish ──

    def publish_compression_child(self, parent_session_id: SessionId, child: CompressionChild,
                                  handoff: Sequence[NewMessage], *, compression_lock_holder: Optional[str] = None,
                                  require_compression_lease: bool = True, require_lease_refresh: bool = False,
                                  lease_ttl_seconds: float = 300.0, watermark: Optional[MessageId] = None,
                                  watermark_ceiling: Optional[MessageId] = None) -> tuple[RowStamp, ...]:
        parent_id, child_id = str(parent_session_id), str(child.session_id)
        with self._store.transaction() as conn:
            lock = _row(conn, "SELECT holder, expires_at FROM compression_locks WHERE session_id = %s FOR UPDATE",
                        (parent_id,))
            now = time.time()
            # A refresh by the holder lands first, so a lease that just expired still admits it.
            refresh = (require_lease_refresh and bool(compression_lock_holder) and lock is not None
                       and lock["holder"] == compression_lock_holder)
            expires = now + lease_ttl_seconds if refresh else (lock["expires_at"] if lock is not None else 0.0)
            if require_compression_lease and (lock is None or not compression_lock_holder
                                              or lock["holder"] != compression_lock_holder or expires <= now):
                raise CompressionBusy(parent_session_id)
            parent = _row(conn, "SELECT * FROM sessions WHERE id = %s FOR UPDATE", (parent_id,))
            if parent is None:
                raise CompressionPublishRefused(parent_session_id, f"Compression parent not found: {parent_id}")
            if parent["ended_at"] is not None and parent["end_reason"] not in AUTOMATIC_END_REASONS:
                raise CompressionPublishRefused(parent_session_id, f"Compression parent already ended: {parent_id}")
            if not handoff:
                raise CompressionPublishRefused(parent_session_id, "Compression child handoff must not be empty")
            if scalar(conn, "SELECT 1 FROM sessions WHERE id = %s", (child_id,)) is not None:
                raise Conflict("session", child_id)
            self._insert_child(conn, parent, child)
            parent_rows = store.load(conn, [parent_id])
            published = store.insert_rows(conn, child_id, _carry_parent_timestamps(parent_rows, handoff))
            if watermark is not None:
                # The parent's concurrent tail is cloned after the handoff; the originals stay in the closed parent.
                store.clone_rows(conn, child_id, [row for row in parent_rows if row.visibility is RowVisibility.ACTIVE
                                                  and row.id > int(watermark)
                                                  and (watermark_ceiling is None or row.id <= int(watermark_ceiling))])
                store.recount(conn, child_id)
            if refresh:
                execute(conn, "UPDATE compression_locks SET expires_at = %s WHERE session_id = %s", (expires, parent_id))
            execute(conn, "UPDATE sessions SET ended_at = %s, end_reason = %s WHERE id = %s",
                    (time.time(), END_REASON_COMPRESSION, parent_id))
            _lineage.stamp_ancestors(conn, child_id)
            if parent["archived"]:
                self._store.sessions._unarchive_auto_archived_lineage(conn, child_id)
        return tuple(store.row_stamp(row) for row in published)

    def _insert_child(self, conn: Connection, parent: Mapping[str, Any], child: CompressionChild) -> None:
        """The continuation's session row copied from the parent: routing, user, git metadata, display name, origin and
        the tools pin; the conversation-wide config keys, pin and archive state too unless the child is a branch or
        reset copy."""
        continuation = not any(key in (child.model_config or {}) for key in ("_branched_from", "_reset_from"))
        parent_config = _lineage.config_of(parent)
        config = {**{key: value for key, value in parent_config.items()
                     if continuation and key in _CONVERSATION_MODEL_CONFIG_KEYS}, **(child.model_config or {})}
        created = next((label for label in (parent["created_source"], child.source)
                        if label not in _lineage.UNSET_CREATED), None)
        values: dict[str, Any] = {
            "id": str(child.session_id), "source": child.source, "created_source": created, "model": child.model,
            "system_prompt_hash": store_text(conn, child.system_prompt), "parent_session_id": parent["id"],
            "cwd": child.cwd or parent["cwd"],
            "profile_name": child.profile_name or parent["profile_name"] or self._store.owner_profile,
            "started_at": time.time(), "model_config": Jsonb(config) if config else None,
            "pinned": bool(parent["pinned"]) and continuation, "archived": bool(parent["archived"]) and continuation,
            "auto_archived": bool(parent["auto_archived"]) and continuation,
            **{name: parent[name] for name in _COPIED},
        }
        execute(conn, f"INSERT INTO sessions ({', '.join(values)}) VALUES ({', '.join(['%s'] * len(values))})",
                tuple(values.values()))

    def stamp_legacy_compression_chains(self) -> int:
        with self._store.transaction() as conn:
            edges = [item for item in _rows(conn, """
                SELECT c.id, c.source, c.model_config, p.id AS parent_id FROM sessions c
                JOIN sessions p ON p.id = c.parent_session_id
                WHERE p.end_reason = 'compression' AND coalesce(p.created_source, '') IN ('', 'unknown', 'local')
                  AND coalesce(c.created_source, '') NOT IN ('', 'unknown', 'local')""")
                     if not _non_continuation(item, item["parent_id"])]
            return sum(_lineage.stamp_ancestors(conn, item["id"]) for item in edges)

    def reopen_orphaned_compression(self, session_id: SessionId) -> bool:
        if not session_id:
            return False
        with self._store.transaction() as conn:
            row = _row(conn, "SELECT ended_at, end_reason FROM sessions WHERE id = %s FOR UPDATE", (session_id,))
            if row is None or row["ended_at"] is None or row["end_reason"] != END_REASON_COMPRESSION:
                return False
            children = _rows(conn, "SELECT source, model_config FROM sessions WHERE parent_session_id = %s", (session_id,))
            if any(not _non_continuation(child, str(session_id)) for child in children):
                return False
            lock = _row(conn, "SELECT expires_at FROM compression_locks WHERE session_id = %s FOR UPDATE", (session_id,))
            if lock is not None:
                if lock["expires_at"] >= time.time():
                    return False
                execute(conn, "DELETE FROM compression_locks WHERE session_id = %s", (session_id,))
            execute(conn, "UPDATE sessions SET ended_at = NULL, end_reason = NULL WHERE id = %s", (session_id,))
            return True

    def finalize_orphaned_compression_children(self) -> int:
        now = time.time()
        with self._store.transaction() as conn:
            return execute(conn, """
                UPDATE sessions c SET ended_at = %s, end_reason = %s
                FROM sessions p
                WHERE p.id = c.parent_session_id AND p.end_reason = 'compression' AND p.ended_at IS NOT NULL
                  AND c.ended_at IS NULL AND c.end_reason IS NULL AND c.started_at < %s AND c.api_call_count = 0
                  AND EXISTS (SELECT 1 FROM messages m WHERE m.session_id = c.id)""",
                           (now, END_REASON_ORPHANED_COMPRESSION, now - _ORPHAN_FINALIZE_AGE_SECONDS))

    # ── durable compressor counters ──

    def _counter(self, session_id: SessionId, column: str) -> int:
        if not session_id:
            return 0
        with self._store.transaction() as conn:
            value = scalar(conn, f"SELECT {column} FROM sessions WHERE id = %s", (session_id,))
        return max(0, int(value or 0))

    def _set_counter(self, session_id: SessionId, column: str, value: int) -> None:
        if not session_id:
            return
        with self._store.transaction() as conn:
            execute(conn, f"UPDATE sessions SET {column} = %s WHERE id = %s", (max(0, int(value)), session_id))

    def compression_fallback_streak(self, session_id: SessionId) -> int:
        return self._counter(session_id, "compression_fallback_streak")

    def set_compression_fallback_streak(self, session_id: SessionId, value: int) -> None:
        self._set_counter(session_id, "compression_fallback_streak", value)

    def compression_ineffective_count(self, session_id: SessionId) -> int:
        return self._counter(session_id, "compression_ineffective_count")

    def set_compression_ineffective_count(self, session_id: SessionId, value: int) -> None:
        self._set_counter(session_id, "compression_ineffective_count", value)

    def compression_overload_streak(self, session_id: SessionId) -> int:
        return self._counter(session_id, "compression_overload_streak")

    def set_compression_overload_streak(self, session_id: SessionId, value: int) -> None:
        self._set_counter(session_id, "compression_overload_streak", value)

    def increment_compression_overload_streak(self, session_id: SessionId) -> Optional[int]:
        if not session_id:
            return None
        with self._store.transaction() as conn:
            value = scalar(conn, "UPDATE sessions SET compression_overload_streak = compression_overload_streak + 1 "
                                 "WHERE id = %s RETURNING compression_overload_streak", (session_id,))
        return None if value is None else int(value)

    def compression_recovery_deadline(self, session_id: SessionId) -> float:
        if not session_id:
            return 0.0
        with self._store.transaction() as conn:
            value = scalar(conn, "SELECT compression_recovery_deadline FROM sessions WHERE id = %s", (session_id,))
        return max(0.0, float(value or 0.0))

    def set_compression_recovery_deadline(self, session_id: SessionId, deadline: Optional[float]) -> None:
        if not session_id:
            return
        try:
            normalized = max(0.0, float(deadline or 0.0))
        except (TypeError, ValueError):
            normalized = 0.0
        with self._store.transaction() as conn:
            execute(conn, "UPDATE sessions SET compression_recovery_deadline = %s WHERE id = %s",
                    (normalized or None, session_id))


register_repository("lineage", PostgresLineageRepository)
