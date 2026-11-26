"""``MessageRepository`` on Postgres: a session's transcript.

Each verb runs in one ``store.transaction()``. A write first takes the session row's lock (the per-session write
order SQLite gets from its single writer), runs the transcript write guards, then loads the rows it decides on, applies
the shared transcript policy (``_transcript``, ``_policy``) and writes back only what changed. The session's counters
(``message_count``, ``tool_call_count``), ``transcript_revision`` and the compression-failure cooldown are its own
columns.
"""

from __future__ import annotations

import copy
import json
import time
from contextlib import contextmanager
from dataclasses import replace
from typing import Any, Callable, ContextManager, Iterator, Mapping, Optional, Sequence

from psycopg import Connection

from persistence import (
    CompressionClosed, CompressionInProgress, CooldownRestoreRefused, RowVisibility, SessionRowMissing, TurnLeaseLost,
)
from persistence.conversation.messages import (
    AnchoredView, AnchoredWindow, BatchAppendProgress, BatchAppendResult, CompressionCooldown, ConversationMessage,
    ConversationPeer, ConversationRequest, ConversationRevision, CooldownSnapshot, DelegationDelivery,
    MarkerPurgeReport, MessageRecord, MessageStorageState, NewMessage, PagedMessage, PromptPage, PrUrlHit, Reaction,
    ReadCheck, ReplaceMode, ReplaceResult, ResumeConversations, ResumeTooLarge, RowDisposition, RowOutcome, RowUids,
    SegmentPage, SessionTimeline, TimelineEntry, TranscriptRead, TranscriptRow, TranscriptView, UnseenReaction,
    Watermark,
)
from persistence.conversation.sessions import END_REASON_COMPRESSION
from persistence.core.identity import JsonValue, MessageId, SessionId
from session_rules.media import ARCHIVE_POINTER_MARKERS
from session_rules.turns import TYPED_ASK

from .._backend import PostgresStore, register_repository
from .._sql import execute, row as _row, rows as _rows, scalar
from ..authority import _leases
from . import _lineage, _policy as policy, _transcript as store
from ._transcript import StoredRow
from .messages_compact import MessageCompactMixin, row_identity
from .messages_turns import MessageTurnsMixin

_DELIVERY_KINDS = ("async_delegation_complete", "hidden")
# The text markers a stored row that can hold a compaction handoff carries.
_HANDOFF_MARKERS = ("[CONTEXT COMPACTION", "[CONTEXT SUMMARY]")
_IN_VIEW: dict[TranscriptView, Callable[[StoredRow], bool]] = {
    TranscriptView.ACTIVE: lambda row: row.visibility is RowVisibility.ACTIVE,
    TranscriptView.DISPLAY: lambda row: row.visibility is not RowVisibility.REWOUND,
    TranscriptView.ALL: lambda row: True,
}
# What a restore may leave off a copy of a stored row: the row's value stands in for a falsy one.
_RESTORE_MAY_OMIT = ("token_count", "compressed_summary", "tool_call_uids", "absorbed_message_uids")
# The columns a restored copy must replay exactly.
_REPLAY_FIELDS = (*store.OWNED_FIELDS, "tool_call_uids", "absorbed_message_uids", "api_content", "codex_reasoning_items")


def _page(items: list[Any], *, limit: Optional[int], offset: int, latest: bool) -> list[Any]:
    """``limit`` items after skipping ``offset``, from the newest end when ``latest``, always in chronological order."""
    if latest:
        items = items[::-1][offset:]
        return (items if limit is None else items[:limit])[::-1]
    items = items[offset:]
    return items if limit is None else items[:limit]


def _messages(dicts: list[dict[str, Any]]) -> tuple[ConversationMessage, ...]:
    return tuple(ConversationMessage.from_openai_message(message) for message in dicts)


def _merge_patch(target: Mapping[str, JsonValue], patch: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """``patch`` merged into ``target`` as a JSON merge patch (RFC 7396)."""
    merged = dict(target)
    for key, value in patch.items():
        if value is None:
            merged.pop(key, None)
        elif isinstance(value, Mapping):
            merged[key] = _merge_patch(merged[key] if isinstance(merged.get(key), Mapping) else {}, value)
        else:
            merged[key] = value
    return merged


def _same_as_stored(message: NewMessage, stored: NewMessage) -> bool:
    """``message`` stored now would replay exactly as ``stored``."""
    candidate = store.payload(message, keep_reasoning=message.role == "assistant")
    standing_in = {name: getattr(stored, name) for name in _RESTORE_MAY_OMIT if not getattr(message, name)}
    if candidate.content == store.stored_content(policy.loaded_view(stored.role, stored.content)):
        standing_in["content"] = stored.content
    candidate = replace(candidate, **standing_in)
    return all(getattr(candidate, name) == getattr(stored, name) for name in _REPLAY_FIELDS)


def _resolve_target(target: StoredRow, message: NewMessage) -> RowOutcome:
    """One batch row resolved onto the stored row it names."""
    message = copy.deepcopy(message)
    if message.db_row_snapshot is not None:
        if store.row_snapshot(target.message) != message.db_row_snapshot:
            # Another writer changed the payload after the caller's flush: adopt its durable version, write nothing.
            return store.row_outcome(target, RowDisposition.ADOPTED, store.decoded_row(target.message))
        # Still what the caller committed: the caller's owned payload wins; the stored timestamp, other writers'
        # metadata and the identity columns the caller does not carry stay.
        kept = {name: getattr(target.message, name) for name in store.METADATA_FIELDS}
        kept.update({name: getattr(message, name) or getattr(target.message, name) for name in store.IDENTITY_FIELDS})
        stamped = kept["display_metadata"] or {}
        if (stamped.get(TYPED_ASK) is not None and store.payload(message).content != target.message.content
                and (message.display_metadata or {}).get(TYPED_ASK) is None):
            kept["display_metadata"] = {k: v for k, v in stamped.items() if k != TYPED_ASK} or None
        target.message = replace(store.payload(message, keep_reasoning=message.role == "assistant"),
                                 role=target.message.role, timestamp=target.message.timestamp, **kept)
        lent: dict[str, JsonValue] = {name: getattr(target.message, name) for name in store.LIVE_MISSING_METADATA
                                      if getattr(message, name) is None and getattr(target.message, name) is not None}
        return store.row_outcome(target, RowDisposition.REPAIRED, {**lent, "_metadata_only": True} if lent else None)
    if message.role == "assistant" and policy.is_content_blank(target.message.content):
        # The provider streamed nothing before the first flush: fill the blank row's content in place.
        target.message = replace(target.message, content=message.content)
        return store.row_outcome(target, RowDisposition.REPAIRED)
    if message.role == "assistant":
        # A concurrent winner already holds content the model saw: adopt its content only, never overwrite.
        return store.row_outcome(target, RowDisposition.ADOPTED, {"content": target.message.content, "_content_only": True})
    return store.row_outcome(target, RowDisposition.REPAIRED)


def _repair_target(rows: Sequence[StoredRow], message: NewMessage) -> Optional[StoredRow]:
    """The same-role row ``message.repair_target`` names: itself while active, else the one active clone a compaction
    made of it, else the addressed row; ``None`` inserts fresh."""
    if message.repair_target is None:
        return None
    original = next((row for row in rows if row.id == int(message.repair_target)), None)
    if original is None or original.message.role != message.role:
        return None
    if original.visibility is RowVisibility.ACTIVE:
        return original
    clones = [r for r in rows if r.visibility is RowVisibility.ACTIVE and r.id != original.id and r.message == original.message]
    return clones[0] if len(clones) == 1 else original


def _restored_row(rows: Sequence[StoredRow], message: NewMessage) -> Optional[StoredRow]:
    """The active row a restored batch row is a copy of; ``None`` inserts fresh."""
    if (not message.restored_from_store or message.repair_target is not None or message.db_row_snapshot is not None
            or not message.message_uid):
        return None
    active = [row for row in rows if row.visibility is RowVisibility.ACTIVE and row.message.message_uid == message.message_uid]
    if not any(_same_as_stored(message, row.message) for row in active):
        return None
    same_role = [row for row in active if row.message.role == message.role]
    return same_role[-1] if same_role else None


def _holds_handoff(row: StoredRow) -> bool:
    m = row.message
    text = m.content if isinstance(m.content, str) else json.dumps(m.content, ensure_ascii=False)
    return m.role in ("user", "assistant") and (m.compressed_summary or any(marker in text for marker in _HANDOFF_MARKERS))


def _paged(message: dict[str, Any], position: int, durable: int) -> PagedMessage:
    return PagedMessage(message=ConversationMessage.from_openai_message(message), position=position,
                        durable_id=MessageId(durable))


class PostgresMessageRepository(MessageCompactMixin, MessageTurnsMixin):
    """The transcripts of the bound profile's schema."""

    def __init__(self, store_handle: PostgresStore) -> None:
        self._store = store_handle

    # ── guards and plumbing ──

    def _lock_session(self, conn: Connection, session_id: str) -> Optional[dict[str, Any]]:
        return _row(conn, "SELECT id, end_reason FROM sessions WHERE id = %s FOR UPDATE", (session_id,))

    def _write_guards(self, conn: Connection, session_id: str, *, turn_lease_holder: Optional[str] = None,
                      ttl: float = 300.0, reject_live: tuple[str, ...] = (), insert: bool = True) -> None:
        """The transcript-write admission, refusals before any change: live leases of ``reject_live`` kinds, the
        caller's own turn lease, a session closed by compression, a missing session row for an insert; then the
        stale-lease reclaim and the caller's lease renewal."""
        session = self._lock_session(conn, session_id)
        stale = self._refuse_live_leases(conn, session_id, reject_live) if reject_live else []
        _leases.verify_turn_lease(conn, session_id, turn_lease_holder, ttl, renew=False)
        self._refuse_compression_closed(conn, session_id, session)
        if insert and session is None:
            raise SessionRowMissing(session_id)
        self._reclaim(conn, stale)
        _leases.verify_turn_lease(conn, session_id, turn_lease_holder, ttl)

    def _refuse_live_leases(self, conn: Connection, session_id: str,
                            kinds: tuple[str, ...]) -> list[tuple[str, str, str]]:
        """Refuse on a live lease of ``kinds`` held by anyone (compaction first, then turn) and return the stale rows
        to reclaim once every refusal has passed."""
        stale = []
        for kind, table, column, key, live in _leases.lease_rows(conn, session_id):
            if kind not in kinds:
                continue
            if live and kind == "compaction":
                raise CompressionInProgress(session_id)
            if live:
                raise TurnLeaseLost(session_id, f"Session has an active turn lease; refusing transcript mutation for "
                                                f"{str(session_id)!r}")
            stale.append((table, column, key))
        return stale

    @staticmethod
    def _reclaim(conn: Connection, stale: Sequence[tuple[str, str, str]]) -> None:
        for table, column, key in stale:
            execute(conn, f"DELETE FROM {table} WHERE {column} = %s", (key,))

    def _refuse_compression_closed(self, conn: Connection, session_id: str,
                                   session: Optional[Mapping[str, Any]] = None) -> None:
        """A session ended by compression takes no new rows; the refusal names its live continuation."""
        if session is None:
            session = _row(conn, "SELECT end_reason FROM sessions WHERE id = %s", (session_id,))
        if session is not None and session["end_reason"] == END_REASON_COMPRESSION:
            raise CompressionClosed(session_id, SessionId(_lineage.compression_tip(conn, session_id)))

    def _merge_model_config(self, conn: Connection, session_id: str, patch: Mapping[str, JsonValue]) -> None:
        """Merge ``patch`` into the session config (``None`` removes a key); a missing session row is a
        ``ValueError``, so nothing commits against it."""
        from psycopg.types.json import Jsonb

        found = _row(conn, "SELECT model_config FROM sessions WHERE id = %s FOR UPDATE", (session_id,))
        if found is None:
            raise ValueError(f"Session not found: {session_id}")
        config = dict(_lineage.config_of(found))
        for key, value in patch.items():
            if value is None:
                config.pop(key, None)
            else:
                config[key] = copy.deepcopy(value)
        execute(conn, "UPDATE sessions SET model_config = %s WHERE id = %s", (Jsonb(config), session_id))

    def _session_exists(self, conn: Connection, session_id: str) -> bool:
        return scalar(conn, "SELECT 1 FROM sessions WHERE id = %s", (session_id,)) is not None

    def _resume_lineage(self, conn: Connection, session_id: str) -> list[str]:
        """The sessions a display resume reads, root first."""
        return list(reversed(_lineage.resume_lineage_ids(conn, session_id)))

    def _ids(self, conn: Connection, session_id: str, include_lineage: bool) -> list[str]:
        return self._resume_lineage(conn, session_id) if include_lineage else [str(session_id)]

    def _compression_ancestors(self, conn: Connection, session_id: str) -> list[str]:
        """The session plus its parents followed only across parents ended by compression, cycle-safe."""
        chain = [session_id]
        current = session_id
        while True:
            found = _row(conn, "SELECT p.id, p.end_reason FROM sessions c JOIN sessions p ON p.id = c.parent_session_id "
                               "WHERE c.id = %s", (current,))
            if found is None or found["end_reason"] != END_REASON_COMPRESSION or found["id"] in chain:
                return chain
            chain.append(found["id"])
            current = found["id"]

    # ── appends ──

    def append(self, session_id: SessionId, message: NewMessage, *, turn_lease_holder: Optional[str] = None,
               turn_lease_ttl_seconds: float = 300.0) -> MessageId:
        # Only the fields an append takes are stored: given tool-call uids, absorbed uids and data_use are dropped.
        message = replace(message, tool_call_uids=None, tool_call_uid=None, absorbed_message_uids=(), data_use=())
        with self._store.transaction() as conn:
            self._write_guards(conn, session_id, turn_lease_holder=turn_lease_holder, ttl=turn_lease_ttl_seconds)
            return MessageId(store.insert_rows(conn, session_id, [message], keep_reasoning=True)[0].id)

    def append_batch(self, session_id: SessionId, messages: Sequence[NewMessage], *,
                     turn_lease_holder: Optional[str] = None, chunk_rows: Optional[int] = None,
                     turn_lease_ttl_seconds: float = 300.0) -> BatchAppendResult:
        if not messages:
            return BatchAppendResult(outcomes=())
        stamped, chunks = list(messages), [messages]
        if chunk_rows is not None and len(messages) > chunk_rows:
            # Tool-call uids stamped over the whole batch first, so a result pairs with its call across a chunk boundary.
            index: dict[str, str] = {}
            stamped = [store.stamp_tool_call_uids(message, index) for message in messages]
            chunks = [stamped[start:start + chunk_rows] for start in range(0, len(stamped), chunk_rows)]
        outcomes: list[RowOutcome] = []
        try:
            for chunk in chunks:
                with self._store.transaction() as conn:
                    outcomes.extend(self._append_chunk(conn, session_id, chunk, turn_lease_holder, turn_lease_ttl_seconds))
        except BaseException as exc:
            committed = [RowUids(o.message_uid, o.tool_call_uids, o.tool_call_uid) for o in outcomes]
            pending = [RowUids(m.message_uid, m.tool_call_uids, m.tool_call_uid) for m in stamped[len(outcomes):]]
            exc.batch_progress = BatchAppendProgress(outcomes=tuple(outcomes), uids=(*committed, *pending))  # type: ignore[attr-defined]
            raise
        return BatchAppendResult(outcomes=tuple(outcomes))

    def _append_chunk(self, conn: Connection, session_id: str, messages: Sequence[NewMessage],
                      turn_lease_holder: Optional[str], ttl: float) -> list[RowOutcome]:
        """One batch transaction: the guards, every row resolved onto the row it names, then the fresh rows inserted."""
        self._write_guards(conn, session_id, turn_lease_holder=turn_lease_holder, ttl=ttl)
        rows = store.load(conn, [session_id], lock=True)
        resolved: list[Optional[RowOutcome]] = []
        fresh: list[NewMessage] = []
        for message in messages:
            target = _repair_target(rows, message)
            if target is None and (restored := _restored_row(rows, message)) is not None:
                # The writer's own durable row handed back: nothing is written.
                resolved.append(store.row_outcome(restored, RowDisposition.REPAIRED))
                continue
            if target is None:
                resolved.append(None)
                fresh.append(message)
                continue
            payload_before = target.message
            resolved.append(_resolve_target(target, message))
            # A rewrite in place is not an insert: the stored counters keep the calls they counted.
            if target.message is not payload_before:
                store.save(conn, target)
                store.bump_transcript_revision(conn, session_id)
        inserted = iter(store.insert_rows(conn, session_id, fresh))
        return [outcome if outcome is not None else store.row_outcome(next(inserted), RowDisposition.APPENDED)
                for outcome in resolved]

    def _append_detached(self, session_id: str, content: str, display_kind: str, metadata: Mapping[str, JsonValue],
                         match: Callable[[NewMessage], bool]) -> MessageId:
        """A result that arrived between turns recorded once as a user row: a row ``match`` finds in the session or
        its compression ancestors, any visibility, is returned instead; a new insert refuses under a live turn lease
        or on a session closed by compression."""
        with self._store.transaction() as conn:
            for row in store.load(conn, self._compression_ancestors(conn, session_id)):
                if match(row.message):
                    return MessageId(row.id)
        message = NewMessage(role="user", content=content, timestamp=time.time(), display_kind=display_kind,
                             display_metadata=dict(metadata))
        with self._store.transaction() as conn:
            self._write_guards(conn, session_id, reject_live=("turn",))
            for row in store.load(conn, self._compression_ancestors(conn, session_id)):
                if match(row.message):
                    return MessageId(row.id)
            return MessageId(store.insert_rows(conn, session_id, [message], keep_reasoning=True)[0].id)

    def append_delegation_delivery(self, session_id: SessionId, delivery: DelegationDelivery) -> MessageId:
        delegation_id, notice = delivery.identity
        kind = "hidden" if delivery.metadata.get("presentation_suppressed") else "async_delegation_complete"

        def match(message: NewMessage) -> bool:
            meta = message.display_metadata or {}
            stored_notice = meta.get("delivery_notice")
            return (message.display_kind in _DELIVERY_KINDS and meta.get("delegation_id") == delegation_id
                    and ("" if stored_notice is None else stored_notice) == notice)

        return self._append_detached(session_id, delivery.content, kind, delivery.metadata, match)

    def append_completion_note(self, session_id: SessionId, admission_id: str, content: str, display_kind: str,
                               metadata: Mapping[str, JsonValue]) -> MessageId:
        return self._append_detached(
            session_id, content, display_kind, {**metadata, "completion_admission": admission_id},
            lambda message: (message.display_metadata or {}).get("completion_admission") == admission_id)

    # ── replace and clear ──

    def replace_transcript(self, session_id: SessionId, messages: Sequence[NewMessage], mode: ReplaceMode, *,
                           reject_active_turn_lease: bool = False) -> ReplaceResult:
        with self._store.transaction() as conn:
            return self._replace_on(conn, session_id, messages, mode, reject_active_turn_lease=reject_active_turn_lease)

    def _replace_on(self, conn: Connection, session_id: str, messages: Sequence[NewMessage], mode: ReplaceMode, *,
                    reject_active_turn_lease: bool = False) -> ReplaceResult:
        """``replace_transcript`` on the caller's transaction (an admission's ``/retry`` rewrite commits with it)."""
        self._write_guards(conn, session_id, reject_live=("compaction", "turn") if reject_active_turn_lease else (),
                           insert=bool(messages))
        rows = store.load(conn, [session_id], lock=True)
        kept: list[StoredRow] = []
        if mode is ReplaceMode.DESTRUCTIVE:
            execute(conn, "DELETE FROM messages WHERE session_id = %s", (session_id,))
        elif mode is ReplaceMode.PRESERVE_EXISTING_ARCHIVE:
            execute(conn, "DELETE FROM messages WHERE session_id = %s AND active", (session_id,))
        else:
            live = [row for row in rows if row.visibility is RowVisibility.ACTIVE]
            for row, message in zip(live, messages):
                m = row.message
                if (row_identity(m.role, m.content, m.tool_call_id, m.tool_calls)
                        != row_identity(message.role, message.content, message.tool_call_id, message.tool_calls)):
                    break
                kept.append(row)
            rewound = live[len(kept):]
            for row in rewound:
                row.visibility = RowVisibility.REWOUND
            store.save_visibility(conn, rewound)
        inserted = store.insert_rows(conn, session_id, messages[len(kept):])
        # Every mode ends with the absolute counter SET over the rows left active.
        store.recount(conn, session_id)
        # A kept row is stamped with only its stored id and uid: its stamp has no snapshot.
        return ReplaceResult(rows=(*(replace(store.row_stamp(row), db_row_snapshot=None) for row in kept),
                                   *(store.row_stamp(row) for row in inserted)))

    def clear_transcript(self, session_id: SessionId) -> None:
        with self._store.transaction() as conn:
            execute(conn, "DELETE FROM messages WHERE session_id = %s", (session_id,))
            store.recount(conn, session_id)

    # ── row edits ──

    def _active_row(self, conn: Connection, session_id: str, message_id: MessageId,
                    role: Optional[str] = None) -> Optional[StoredRow]:
        found = _row(conn, "SELECT * FROM messages WHERE id = %s AND session_id = %s AND active FOR UPDATE",
                     (int(message_id), session_id))
        if found is None:
            return None
        row = store.decode(found)
        return None if role is not None and row.message.role != role else row

    def set_user_message_content(self, session_id: SessionId, message_id: MessageId, content: JsonValue,
                                 display_metadata: Optional[Mapping[str, JsonValue]] = None) -> bool:
        with self._store.transaction() as conn:
            row = self._active_row(conn, session_id, message_id, "user")
            if row is None:
                return False
            if display_metadata is None:
                row.message = replace(row.message, content=store.payload(replace(row.message, content=content)).content)
            else:
                merged = _merge_patch(row.message.display_metadata or {}, display_metadata)
                stored = store.payload(replace(row.message, content=content, display_metadata=merged))
                row.message = replace(row.message, content=stored.content, display_metadata=stored.display_metadata,
                                      api_content=None)
            store.save(conn, row)
            store.bump_transcript_revision(conn, session_id)
            return True

    def deactivate_message(self, session_id: SessionId, message_id: MessageId) -> bool:
        with self._store.transaction() as conn:
            # The UPDATE touches no counter: the counters still include the row and its calls.
            if scalar(conn, "SELECT 1 FROM messages WHERE id = %s AND session_id = %s", (int(message_id), session_id)) is None:
                return False
            execute(conn, "UPDATE messages SET active = false, compacted = false WHERE id = %s AND active",
                    (int(message_id),))
            return True

    def deactivate_messages_by_display_kind(self, session_id: SessionId, display_kind: str) -> int:
        if not display_kind:
            return 0
        with self._store.transaction() as conn:
            return execute(conn, "UPDATE messages SET active = false, compacted = false "
                                 "WHERE session_id = %s AND active AND display_kind = %s", (session_id, display_kind))

    def set_latest_matching_message_display_kind(self, session_id: SessionId, *, role: str, content: str,
                                                 display_kind: str,
                                                 display_metadata: Optional[Mapping[str, JsonValue]] = None) -> bool:
        if not session_id or not content or not display_kind:
            return False
        stored = store.payload(NewMessage(role=role, content=content)).content
        with self._store.transaction() as conn:
            for row in reversed([r for r in store.load(conn, [session_id], lock=True)
                                 if r.visibility is RowVisibility.ACTIVE]):
                if row.message.role == role and row.message.content == stored:
                    metadata = copy.deepcopy(dict(display_metadata)) if display_metadata else None
                    row.message = replace(row.message, display_kind=display_kind, display_metadata=metadata)
                    store.save(conn, row)
                    store.bump_transcript_revision(conn, session_id)
                    return True
        return False

    def set_latest_user_api_content(self, session_id: SessionId, content: JsonValue, api_content: str) -> bool:
        with self._store.transaction() as conn:
            found = _row(conn, "SELECT * FROM messages WHERE session_id = %s AND active AND role = 'user' "
                               "ORDER BY id DESC LIMIT 1 FOR UPDATE", (session_id,))
            return found is not None and self._stamp_api_content(conn, session_id, store.decode(found), content, api_content)

    def set_message_api_content(self, session_id: SessionId, message_id: MessageId, content: JsonValue,
                                api_content: str) -> bool:
        with self._store.transaction() as conn:
            row = self._active_row(conn, session_id, message_id, "user")
            return row is not None and self._stamp_api_content(conn, session_id, row, content, api_content)

    def _stamp_api_content(self, conn: Connection, session_id: str, row: StoredRow, content: JsonValue,
                           api_content: str) -> bool:
        """Stamp the sidecar when the row still holds ``content`` as the store compares it."""
        if row.message.content != store.payload(NewMessage(role="user", content=content)).content:
            return False
        row.message = replace(row.message, api_content=api_content)
        store.save(conn, row)
        store.bump_transcript_revision(conn, session_id)
        return True

    def purge_stale_tool_call_markers(self, *, dry_run: bool = False, backup: bool = True) -> MarkerPurgeReport:
        with self._store.transaction() as conn:
            candidates = [store.decode(item) for item in _rows(
                conn, "SELECT * FROM messages WHERE role = 'assistant' AND tool_calls IS NOT NULL "
                      "AND content IS NOT NULL ORDER BY id FOR UPDATE")]
            matches = [row for row in candidates if row.message.tool_calls and isinstance(row.message.content, str)
                       and policy.STALE_TOOL_CALL_MARKER_RE.fullmatch(row.message.content.strip())]
            backup_path = None
            if not dry_run and matches:
                if backup:
                    backup_path = self._backup_rows(conn, [row.id for row in matches])
                for row in matches:
                    row.message = replace(row.message, content="")
                    store.save(conn, row)
                    store.bump_transcript_revision(conn, row.session_id)
        return MarkerPurgeReport(dry_run=dry_run, rows_affected=len(matches),
                                 row_ids=tuple(MessageId(r.id) for r in matches), backup_path=backup_path)

    def _backup_rows(self, conn: Connection, row_ids: Sequence[int]) -> str:
        """Copy the rows a destructive rewrite is about to change, every column as stored, to a JSON-lines file under
        the Hermes home's ``backups/`` before the write; returns its path."""
        from hermes_constants import get_hermes_home

        folder = get_hermes_home() / "backups"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"messages-{self._store.schema}-{time.strftime('%Y%m%dT%H%M%S')}-{time.time_ns() % 10**9}.jsonl"
        found = _rows(conn, "SELECT row_to_json(m)::text AS row FROM messages m WHERE id = ANY(%s) ORDER BY id",
                      (list(row_ids),))
        path.write_text("".join(item["row"] + "\n" for item in found), encoding="utf-8")
        return str(path)

    # ── reactions ──

    def set_reaction(self, session_id: SessionId, message_id: MessageId, emoji: Optional[str], *,
                     author: str) -> Optional[tuple[Reaction, ...]]:
        if not session_id:
            return None
        with self._store.transaction() as conn:
            lineage = self._resume_lineage(conn, session_id)
            found = _row(conn, "SELECT * FROM messages WHERE id = %s AND session_id = ANY(%s) FOR UPDATE",
                         (int(message_id), lineage))
            if found is None:
                return None
            row = store.decode(found)
            if row.visibility is RowVisibility.REWOUND:
                return None
            previous = next((r for r in row.reactions if r.get("author") == author), None)
            kept = [r for r in row.reactions if r.get("author") != author]
            if emoji and (previous is None or previous.get("emoji") != emoji):
                kept.append({"emoji": emoji, "author": author, "at": time.time()})
            row.reactions = kept
            store.save(conn, row)
            # A display_metadata rewrite moves the owning segment's transcript revision.
            store.bump_transcript_revision(conn, row.session_id)
        return tuple(Reaction(emoji=str(r.get("emoji") or ""), author=str(r.get("author") or ""),
                              at=float(r.get("at") or 0.0), seen=bool(r.get("seen"))) for r in kept)

    def take_unseen_reactions(self, session_id: SessionId, *, author: str = "user") -> tuple[UnseenReaction, ...]:
        if not session_id:
            return ()
        taken: list[UnseenReaction] = []
        with self._store.transaction() as conn:
            for row in store.load(conn, self._resume_lineage(conn, session_id), lock=True):
                if row.visibility is RowVisibility.REWOUND or not row.reactions:
                    continue
                marked: list[dict[str, Any]] = []
                for reaction in row.reactions:
                    if reaction.get("author") == author and not reaction.get("seen"):
                        content = row.message.content
                        taken.append(UnseenReaction(row_id=MessageId(row.id), role=row.message.role,
                                                    emoji=str(reaction.get("emoji") or ""),
                                                    text=content if isinstance(content, str) else ""))
                        reaction = {**reaction, "seen": True}
                    marked.append(reaction)
                if marked != row.reactions:
                    row.reactions = marked
                    store.save(conn, row)
                    store.bump_transcript_revision(conn, row.session_id)
        return tuple(taken)

    # ── the compression-failure cooldown ──

    def record_compression_cooldown(self, session_id: SessionId, until: float, error: Optional[str] = None) -> None:
        if not session_id:
            return
        with self._store.transaction() as conn:
            execute(conn, "UPDATE sessions SET compression_failure_cooldown_until = CASE "
                          "WHEN compression_failure_cooldown_until IS NULL OR NOT compression_failure_cooldown_until > %s "
                          "THEN %s ELSE compression_failure_cooldown_until END, compression_failure_error = %s WHERE id = %s",
                    (until, until, error, session_id))

    def clear_compression_cooldown(self, session_id: SessionId) -> None:
        with self._store.transaction() as conn:
            execute(conn, "UPDATE sessions SET compression_failure_cooldown_until = NULL, compression_failure_error = NULL "
                          "WHERE id = %s", (session_id,))

    def restore_compression_cooldown(self, session_id: SessionId, snapshot: CooldownSnapshot) -> None:
        with self._store.transaction() as conn:
            present = self._session_exists(conn, session_id)
            if not snapshot.present:
                if present:
                    raise CooldownRestoreRefused(session_id, "cannot restore absent compression cooldown row: session now exists")
                return
            if present:
                execute(conn, "UPDATE sessions SET compression_failure_cooldown_until = %s, compression_failure_error = %s "
                              "WHERE id = %s", (snapshot.until, snapshot.error, session_id))

    def read_compression_cooldown(self, session_id: SessionId, at: float) -> Optional[CompressionCooldown]:
        with self._store.transaction() as conn:
            found = _row(conn, "SELECT compression_failure_cooldown_until AS until, compression_failure_error AS error "
                               "FROM sessions WHERE id = %s", (session_id,))
        if found is None or found["until"] is None or found["until"] <= at:
            return None
        return CompressionCooldown(until=found["until"], error=found["error"])

    def read_compression_cooldown_snapshot(self, session_id: SessionId) -> CooldownSnapshot:
        if not session_id:
            return CooldownSnapshot(present=False)
        with self._store.transaction() as conn:
            found = _row(conn, "SELECT compression_failure_cooldown_until AS until, compression_failure_error AS error "
                               "FROM sessions WHERE id = %s", (session_id,))
        if found is None:
            return CooldownSnapshot(present=False)
        return CooldownSnapshot(present=True, until=found["until"], error=found["error"])

    # ── reads ──

    def _load(self, session_ids: Sequence[str]) -> list[StoredRow]:
        with self._store.transaction() as conn:
            return store.load(conn, session_ids)

    def read_transcript(self, session_id: SessionId, request: TranscriptRead = TranscriptRead()) -> tuple[TranscriptRow, ...]:
        with self._store.transaction() as conn:
            ids = self._ids(conn, session_id, request.include_lineage)
            rows = store.load(conn, ids)
        keep = _IN_VIEW[request.view]
        if len(ids) > 1:
            # A merged lineage dedupes generations, then pages.
            display = request.view is TranscriptView.DISPLAY
            chosen = store.dedupe_display(rows if display else [r for r in rows if keep(r)], drop_removed=display)
        elif request.view is TranscriptView.DISPLAY:
            chosen = [row for _slot, row in store.display_slots(rows)]
        else:
            chosen = [r for r in rows if keep(r) and (request.after_id is None or r.id > int(request.after_id))]
        page = _page(chosen, limit=request.limit, offset=request.offset, latest=request.latest)
        return tuple(row.as_transcript_row() for row in page)

    def active_message_ids(self, session_id: SessionId) -> tuple[MessageId, ...]:
        with self._store.transaction() as conn:
            found = _rows(conn, "SELECT id FROM messages WHERE session_id = %s AND active ORDER BY id", (session_id,))
        return tuple(MessageId(item["id"]) for item in found)

    def active_watermark(self, session_id: SessionId) -> Watermark:
        with self._store.transaction() as conn:
            newest = scalar(conn, "SELECT max(id) FROM messages WHERE session_id = %s AND active", (session_id,))
        return Watermark(last_active=None if newest is None else MessageId(newest))

    def count_all(self) -> int:
        with self._store.transaction() as conn:
            return int(scalar(conn, "SELECT count(*) FROM messages"))

    def display_count(self, session_id: SessionId) -> int:
        return len(store.display_slots(self._load([str(session_id)])))

    def conversation_revision(self, session_id: SessionId, *, include_lineage: bool = False) -> ConversationRevision:
        with self._store.transaction() as conn:
            ids = self._ids(conn, session_id, include_lineage)
            found = _row(conn, "SELECT count(*) AS n, coalesce(max(id), 0) AS newest FROM messages "
                               "WHERE session_id = ANY(%s) AND active", (ids,))
            revision = scalar(conn, "SELECT coalesce(sum(transcript_revision), 0) FROM sessions WHERE id = ANY(%s)",
                              (list(dict.fromkeys(ids)),))
        return ConversationRevision(lineage_ids=tuple(SessionId(sid) for sid in ids), active_count=int(found["n"]),
                                    max_active_id=int(found["newest"]), transcript_revision=int(revision))

    def _conversation_rows(self, session_id: str, view: TranscriptView, include_lineage: bool) -> list[StoredRow]:
        with self._store.transaction() as conn:
            rows = store.load(conn, self._ids(conn, session_id, include_lineage))
        if view is TranscriptView.DISPLAY:
            return store.dedupe_display(rows)
        return [row for row in rows if _IN_VIEW[view](row)]

    def read_conversation(self, session_id: SessionId,
                          request: ConversationRequest = ConversationRequest()) -> tuple[ConversationMessage, ...]:
        rows = self._conversation_rows(session_id, request.view, request.include_lineage)
        return _messages(store.conversation_dicts(
            rows, include_lineage=request.include_lineage, repair_alternation=request.repair_alternation,
            include_row_ids=request.include_row_ids, include_summary_markers=request.repair_alternation))

    def read_resume_conversations(self, session_id: SessionId) -> ResumeConversations:
        with self._store.transaction() as conn:
            rows = store.load(conn, self._resume_lineage(conn, session_id))
        # The model projection is the tip's working context: active rows, without the session_meta bookkeeping row.
        model = [row for row in rows if row.session_id == str(session_id) and row.visibility is RowVisibility.ACTIVE
                 and row.message.role != "session_meta"]
        display = store.dedupe_display(rows)
        return ResumeConversations(
            model=_messages(store.conversation_dicts(model, include_lineage=False, repair_alternation=True,
                                                     include_row_ids=True, include_summary_markers=True)),
            display=_messages(store.conversation_dicts(display, include_lineage=True, repair_alternation=False,
                                                       include_row_ids=True, include_summary_markers=False)))

    def assert_resume_safe(self, session_id: SessionId, *, max_messages: int, tip_only: bool = False) -> int:
        if max_messages < 0:
            raise ValueError("max_messages must be non-negative")
        if max_messages == 0:
            return 0
        with self._store.transaction() as conn:
            if tip_only:
                count = int(scalar(conn, "SELECT count(*) FROM (SELECT 1 FROM messages WHERE session_id = %s AND active "
                                         "LIMIT %s) found", (session_id, max_messages + 1)))
            else:
                count = int(scalar(conn, "SELECT count(*) FROM (SELECT 1 FROM messages WHERE session_id = ANY(%s) "
                                         "AND (active OR compacted) LIMIT %s) found",
                                   (self._resume_lineage(conn, session_id), max_messages + 1)))
        if count > max_messages:
            raise ResumeTooLarge(count, max_messages, scope="in its tip segment" if tip_only else "across its lineage")
        return count

    def read_around(self, session_id: SessionId, anchor: MessageId, *, window: int = 5) -> AnchoredWindow:
        window = max(window, 0)
        rows = self._load([str(session_id)])
        if not any(row.id == int(anchor) for row in rows):
            return AnchoredWindow(window=(), before=0, after=0)
        before_rows = [r for r in rows if r.id <= int(anchor)][-(window + 1):]
        after_rows = [r for r in rows if r.id > int(anchor)][:window]
        return AnchoredWindow(window=tuple(r.as_record() for r in (*before_rows, *after_rows)),
                              before=len(before_rows) - 1, after=len(after_rows))

    def read_anchored_view(self, session_id: SessionId, anchor: MessageId, *, window: int = 5, bookend: int = 3,
                           keep_roles: Optional[tuple[str, ...]] = ("user", "assistant")) -> AnchoredView:
        bookend = max(bookend, 0)
        primitive = self.read_around(session_id, anchor, window=window)
        if not primitive.window:
            return AnchoredView(window=(), before=0, after=0, bookend_start=(), bookend_end=())
        kept = (primitive.window if keep_roles is None
                else tuple(m for m in primitive.window if int(m.id) == int(anchor) or m.role in keep_roles))
        start: tuple[MessageRecord, ...] = ()
        end: tuple[MessageRecord, ...] = ()
        if bookend > 0:
            rows = self._load([str(session_id)])
            eligible = [r for r in rows if (keep_roles is None or r.message.role in keep_roles)
                        and store.text_length(r.message.content) > 0]
            first_id, last_id = int(primitive.window[0].id), int(primitive.window[-1].id)
            start = tuple(r.as_record() for r in [r for r in eligible if r.id < first_id][:bookend])
            end = tuple(r.as_record() for r in [r for r in eligible if r.id > last_id][-bookend:])
        return AnchoredView(window=kept, before=primitive.before, after=primitive.after, bookend_start=start,
                            bookend_end=end)

    def find_pr_url_hits(self, session_ids: Sequence[SessionId]) -> tuple[PrUrlHit, ...]:
        ids = [str(s) for s in session_ids if s]
        with self._store.transaction() as conn:
            found = _rows(conn, "SELECT session_id, content FROM messages WHERE session_id = ANY(%s) AND role = 'tool' "
                                "AND strpos(lower(content), '/pull/') > 0 ORDER BY id", (ids,))
        return tuple(PrUrlHit(session_id=SessionId(item["session_id"]), content=item["content"]) for item in found
                     if store.pr_url_content(item["content"]))

    def latest_todo_result(self, session_id: SessionId, *, max_chars: int) -> Optional[str]:
        if not session_id:
            return None
        with self._store.transaction() as conn:
            lineage = self._resume_lineage(conn, session_id)
            rows = store.load(conn, lineage)
        removed = _removed_todo_copies(rows)
        for sid in reversed(lineage):
            shown = [row for row in rows if row.session_id == sid and row.visibility is not RowVisibility.REWOUND]
            for index in range(len(shown) - 1, -1, -1):
                if any(removed.get(copy_key, 0) > shown[index].id for copy_key in store.copy_identities(shown[index])):
                    continue
                content = _todo_content(shown, index, max_chars)
                if content is not None:
                    return content
        return None

    def _timeline_slots(self, session_id: str) -> list[tuple[int, StoredRow, str]]:
        """The session's displayed human prompts: ``(slot, representative row, preview)`` in display order."""
        out = []
        for slot, row in store.display_slots(self._load([session_id])):
            if row.message.role != "user":
                continue
            preview = policy.prompt_preview(row.message.content, row.message.display_kind, row.message.compressed_summary)
            if preview:
                out.append((slot, row, preview))
        return out

    def read_session_timeline(self, session_id: SessionId, *, limit: int = 500, after_row_id: int = 0) -> SessionTimeline:
        prompts = self._timeline_slots(session_id)
        following = [entry for entry in prompts if entry[0] > after_row_id][:limit + 1]
        has_more = len(following) > limit
        page = following[:limit]
        return SessionTimeline(
            entries=tuple(TimelineEntry(row_id=MessageId(row.id), preview=preview, timestamp=row.epoch)
                          for _slot, row, preview in page),
            limit=limit, after_row_id=after_row_id, returned=len(page), total=len(prompts), has_more=has_more,
            next_cursor=page[-1][0] if has_more else None)

    def read_prompt_page(self, session_id: SessionId, row_id: MessageId, *, limit: int = 120) -> Optional[PromptPage]:
        rows = self._load([str(session_id)])
        anchor = next((row for row in rows if row.id == int(row_id)), None)
        if anchor is None or anchor.message.role != "user" or anchor.visibility is RowVisibility.REWOUND:
            return None
        if not policy.prompt_preview(anchor.message.content, anchor.message.display_kind, anchor.message.compressed_summary):
            return None
        slots = store.display_slots(rows)
        start = next((slot for slot, row in slots if row.id == int(row_id)), None)
        if start is None:
            return None
        offset = sum(1 for slot, _row in slots if slot < start)
        page = [row for slot, row in slots if slot >= start][:limit]
        return PromptPage(messages=tuple(row.as_transcript_row() for row in page), row_id=row_id, limit=limit,
                          returned=len(page), offset=offset, total=len(slots), has_older=offset > 0,
                          has_newer=offset + len(page) < len(slots))

    def latest_conversation_role(self, session_id: SessionId) -> Optional[str]:
        with self._store.transaction() as conn:
            return scalar(conn, "SELECT role FROM messages WHERE session_id = %s AND active "
                                "AND role NOT IN ('session_meta', 'system') ORDER BY id DESC LIMIT 1", (session_id,))

    def latest_dialogue_message(self, session_id: SessionId) -> Optional[str]:
        for row in reversed([r for r in self._load([str(session_id)]) if r.visibility is RowVisibility.ACTIVE]):
            if row.message.role in ("user", "assistant") and store.has_text(row.message.content):
                return store.content_text(row.message.content)
        return None

    def message_storage_state(self, message_id: MessageId) -> Optional[MessageStorageState]:
        with self._store.transaction() as conn:
            found = _row(conn, "SELECT session_id, active, compacted FROM messages WHERE id = %s", (int(message_id),))
        if found is None:
            return None
        return MessageStorageState(session_id=SessionId(found["session_id"]),
                                   visibility=store._visibility(found["active"], found["compacted"]))

    def latest_row_id(self, session_id: SessionId, *, role: str = "user", offset: int = 0,
                      require_text: bool = True) -> Optional[MessageId]:
        if not session_id or role not in ("user", "assistant") or offset < 0:
            return None
        candidates = [row for row in reversed(self._load([str(session_id)])) if row.visibility is RowVisibility.ACTIVE
                      and row.message.role == role and (not require_text or store.has_text(row.message.content))]
        return MessageId(candidates[offset].id) if offset < len(candidates) else None

    def message_role(self, session_id: SessionId, message_id: MessageId) -> Optional[str]:
        if not session_id:
            return None
        with self._store.transaction() as conn:
            return scalar(conn, "SELECT role FROM messages WHERE id = %s AND session_id = %s AND active",
                          (int(message_id), session_id))

    def resolve_active_row_id(self, session_id: SessionId, message_id: MessageId) -> Optional[MessageId]:
        if not session_id:
            return None
        rows = self._load([str(session_id)])
        origin = next((row for row in rows if row.id == int(message_id)), None)
        if origin is None:
            return None
        if origin.visibility is RowVisibility.ACTIVE:
            return message_id
        o = origin.message
        clones = [row.id for row in rows if row.visibility is RowVisibility.ACTIVE and row.id > origin.id
                  and row.message.role == o.role and row.message.content == o.content
                  and row.message.timestamp == o.timestamp]
        return MessageId(clones[0]) if len(clones) == 1 else None

    def has_platform_message_id(self, session_id: SessionId, platform_message_id: str) -> bool:
        with self._store.transaction() as conn:
            return scalar(conn, "SELECT 1 FROM messages WHERE session_id = %s AND platform_message_id = %s LIMIT 1",
                          (session_id, platform_message_id)) is not None

    def has_gateway_input_owner(self, session_id: SessionId, owner: str) -> bool:
        with self._store.transaction() as conn:
            return scalar(conn, "SELECT 1 FROM messages WHERE session_id = %s AND role = 'user' AND NOT observed "
                                "AND (active OR compacted) AND display_metadata ->> 'gateway_input_owner' = %s LIMIT 1",
                          (session_id, owner)) is not None

    def latest_conversation_boundary(self, peer: ConversationPeer) -> Optional[int]:
        if not peer.session_key or not peer.source:
            return None
        with self._store.transaction() as conn:
            generation = scalar(conn, "SELECT generation FROM conversation_generations WHERE source = %s "
                                      "AND session_key = %s", (peer.source, peer.session_key))
        return int(generation) if generation else None

    def terminal_tool_call_payloads(self, *, since_ts: float) -> Iterator[str]:
        with self._store.transaction() as conn:
            found = _rows(conn, "SELECT tool_calls::text AS calls FROM messages WHERE role = 'assistant' "
                                "AND tool_calls IS NOT NULL AND json_typeof(tool_calls) = 'array' "
                                "AND json_array_length(tool_calls) > 0 AND timestamp >= %s ORDER BY id", (since_ts,))
        payloads = [json.dumps(json.loads(item["calls"])) for item in found]
        # ASCII-case-insensitive, over the whole stored payload.
        return iter([payload for payload in payloads if "terminal" in payload.lower()])

    def blocked_tool_result_payloads(self, *, since_ts: float) -> Iterator[tuple[str, Optional[str]]]:
        with self._store.transaction() as conn:
            found = _rows(conn, "SELECT tool_call_id, content FROM messages WHERE role = 'tool' AND tool_call_id IS NOT NULL "
                                "AND timestamp >= %s AND content IS NOT NULL "
                                "AND (strpos(lower(content), 'blocked') > 0 OR strpos(lower(content), 'approval') > 0) "
                                "ORDER BY id", (since_ts,))
        return iter([(item["tool_call_id"], item["content"]) for item in found])

    def has_any_message(self) -> bool:
        with self._store.transaction() as conn:
            return scalar(conn, "SELECT 1 FROM messages LIMIT 1") is not None

    def spillover_reference_rows(self) -> Iterator[tuple[Optional[str], Optional[str]]]:
        with self._store.transaction() as conn:
            rows = [store.decode(item) for item in _rows(conn, "SELECT * FROM messages ORDER BY id")]
        hits = [(store.content_text(row.message.content), row.message.api_content) for row in rows
                if any(marker in (store.content_text(row.message.content) or "") or marker in (row.message.api_content or "")
                       for marker in ARCHIVE_POINTER_MARKERS)]
        return iter(hits)

    # ── stored-session reads ──

    def read_snapshot(self) -> ContextManager[ReadCheck]:
        return self._snapshot()

    def _fingerprint(self) -> tuple[Any, ...]:
        with self._store.transaction() as conn:
            found = _row(conn, "SELECT (SELECT count(*) FROM messages) AS rows, (SELECT max(id) FROM messages) AS newest, "
                               "(SELECT coalesce(sum(transcript_revision), 0) FROM sessions) AS revision, "
                               "(SELECT coalesce(sum(message_count + tool_call_count), 0) FROM sessions) AS counted, "
                               "(SELECT count(*) FROM messages WHERE active) AS active, "
                               # Any committed write anywhere advances the next transaction id (reads take none), as
                               # SQLite's data_version moves on any commit: an unpinned check must see every write.
                               "pg_snapshot_xmax(pg_current_snapshot())::text AS next_xid")
        return tuple(found.values())

    @contextmanager
    def _snapshot(self) -> Iterator[ReadCheck]:
        # Each read is its own transaction: the check compares the transcript state at the block's start and end.
        start = self._fingerprint()
        yield ReadCheck(lambda: self._fingerprint() == start, pinned=False)

    def display_index_complete(self, session_id: SessionId) -> bool:
        return True

    def ensure_display_index(self, session_id: SessionId) -> bool:
        return True

    def display_generations(self, session_id: SessionId, lineage_ids: Sequence[SessionId]) -> tuple[PagedMessage, ...]:
        first_ids: dict[int, int] = {}
        rows = store.dedupe_display(self._load([str(s) for s in lineage_ids]), first_ids=first_ids)
        messages = store.conversation_dicts(rows, include_lineage=True, repair_alternation=False, include_row_ids=True,
                                            include_summary_markers=False)
        return tuple(_paged(m, first_ids.get(m["_row_id"], m["_row_id"]), first_ids.get(m["_row_id"], m["_row_id"]))
                     for m in messages)

    @staticmethod
    def _durable_ids(rows: Sequence[StoredRow], slots: Sequence[tuple[int, StoredRow]]) -> dict[int, int]:
        """``{row id: first durable id}`` of the shown rows."""
        by_key: dict[Any, int] = {}
        for row in rows:
            if row.visibility is RowVisibility.REWOUND:
                by_key.setdefault(("key", store.display_key(row)), row.id)
                if row.message.message_uid:
                    by_key.setdefault(("uid", row.message.message_uid), row.id)
        out = {}
        for slot, row in slots:
            uid = row.message.message_uid
            carrier = by_key.get(("uid", uid), row.id) if row.message.compressed_summary and uid else row.id
            out[row.id] = min(slot, by_key.get(("key", store.display_key(row)), row.id), carrier)
        return out

    @staticmethod
    def _cursor_slot(rows: Sequence[StoredRow], durable: int) -> Optional[int]:
        row = next((row for row in rows if row.id == durable), None)
        if row is None:
            return None
        shown = {r.id: slot for slot, r in store._placed(rows)}
        key = store.display_key(row)
        copies = [r for r in rows if r.id in shown and store.display_key(r) == key]
        if copies:
            return shown[max(copies, key=lambda r: (r.visibility is RowVisibility.ACTIVE, r.id)).id]
        uid = row.message.message_uid
        carriers = [r for r in rows if uid and r.id in shown and r.visibility is RowVisibility.ACTIVE
                    and r.message.compressed_summary and r.message.message_uid == uid]
        if carriers:
            return shown[max(r.id for r in carriers)]
        return shown.get(row.id)

    def transcript_segment_page(self, session_id: SessionId, *, latest: bool, after: Optional[tuple[int, int]],
                                count: int, inclusive: bool = False) -> SegmentPage:
        rows = self._load([str(session_id)])
        slots = store.display_slots(rows)
        position = None
        if after is not None:
            found = self._cursor_slot(rows, int(after[1]))
            position = int(after[0]) if found is None else found
            if inclusive:
                position += 1 if latest else -1
        past = [item for item in slots if position is None or (item[0] < position if latest else item[0] > position)]
        raw = past[::-1][:count][::-1] if latest else past[:count]
        durable = self._durable_ids(rows, slots)
        before = [item for item in slots if raw and item[0] < raw[0][0]][-1:]
        keys = {row.id: (slot, durable[row.id]) for slot, row in raw}
        context = {row.id for _slot, row in before}
        messages = [m for m in store.conversation_dicts([row for _slot, row in (*before, *raw)], include_lineage=False,
                                                        repair_alternation=False, include_row_ids=True,
                                                        include_summary_markers=False)
                    if m["_row_id"] not in context]
        edge = None if not raw else keys[raw[0][1].id if latest else raw[-1][1].id]
        return SegmentPage(messages=tuple(_paged(m, *keys[m["_row_id"]]) for m in messages), full=len(raw) == count,
                           edge=edge, start=raw[0][0] if raw else None)

    def transcript_handoff_rows(self, shown_ids: Sequence[MessageId],
                                segment_ids: Sequence[SessionId]) -> Mapping[MessageId, tuple[TranscriptRow, ...]]:
        if not shown_ids:
            return {}
        segments = {str(s) for s in segment_ids}
        with self._store.transaction() as conn:
            shown_rows = {item["id"]: store.decode(item) for item in _rows(
                conn, "SELECT * FROM messages WHERE id = ANY(%s)", (sorted({int(i) for i in shown_ids}),))}
            uids = [row.message.message_uid for row in shown_rows.values() if row.message.message_uid]
            peers = [store.decode(item) for item in _rows(
                conn, "SELECT * FROM messages WHERE session_id = ANY(%s) AND message_uid = ANY(%s) ORDER BY id",
                (list(segments), uids))]
        found: dict[MessageId, tuple[TranscriptRow, ...]] = {}
        for shown in sorted(shown_rows):
            row = shown_rows[shown]
            uid = row.message.message_uid
            held = [r for r in peers if uid and r.message.message_uid == uid and r.id < shown and _holds_handoff(r)]
            if row.session_id in segments and _holds_handoff(row):
                held.append(row)
            if held:
                found[MessageId(shown)] = tuple(r.as_transcript_row() for r in held)
        return found

    def transcript_handoffs_shown_before(self, session_id: SessionId, before: int,
                                         handoffs: Sequence[tuple[Optional[str], JsonValue]]) -> tuple[TranscriptRow, ...]:
        uids = {uid for uid, _ in handoffs if uid}
        contents = [content for uid, content in handoffs if not uid and isinstance(content, (str, list, dict))]
        rows = self._load([str(session_id)])
        chosen = {row.id: slot for slot, row in store.display_slots(rows)}
        out = []
        for g in rows:
            m = g.message
            if not _holds_handoff(g) or not ((m.message_uid in uids) or (not m.message_uid and m.content in contents)):
                continue
            if any(chosen[w.id] < before for w in rows if w.id in chosen
                   and (w.id == g.id or (m.message_uid and w.message.message_uid == m.message_uid and w.id > g.id))):
                out.append(g.as_transcript_row())
        return tuple(out)

    def transcript_tool_calls_before(self, segment_ids: Sequence[SessionId], before_id: MessageId, *,
                                     limit: int = 16) -> tuple[tuple[Optional[str], Optional[str]], ...]:
        found = [row.as_transcript_row() for row in self._load([str(s) for s in segment_ids])
                 if row.message.role == "assistant" and row.message.tool_calls and row.id < int(before_id)]
        return tuple((json.dumps(row.tool_calls), row.tool_call_uids) for row in found[::-1][:limit])

    def transcript_cursor_state(self, session_id: SessionId, through_id: MessageId,
                                lineage_ids: Sequence[SessionId]) -> tuple[int, int, int]:
        ids = [str(s) for s in lineage_ids] or [str(session_id)]
        with self._store.transaction() as conn:
            found = _row(conn, """
                SELECT count(*) FILTER (WHERE active AND (session_id <> %s OR id <= %s)) AS live,
                       count(*) FILTER (WHERE NOT active) AS inactive
                FROM messages WHERE session_id = ANY(%s)""", (str(session_id), int(through_id), ids))
            revision = scalar(conn, "SELECT coalesce(sum(transcript_revision), 0) FROM sessions WHERE id = ANY(%s)",
                              (list(dict.fromkeys(ids)),))
        return int(found["live"]), int(found["inactive"]), int(revision)


# ── todo results ──

def _is_todo_call(call: Any, call_id: str) -> bool:
    if not isinstance(call, dict) or call.get("id") != call_id:
        return False
    function = call.get("function")
    if not isinstance(function, dict):
        return False
    if function.get("name") in ("todo_list", "todo"):
        return True
    if function.get("name") != "tool_call":
        return False
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    inner = arguments.get("calls") if isinstance(arguments, dict) else None
    return isinstance(inner, list) and any(isinstance(item, dict) and item.get("name") in ("todo_list", "todo") for item in inner)


def _removed_todo_copies(rows: Sequence[StoredRow]) -> dict[Any, int]:
    """Copy identity to the newest rewound Todo result row id: a result with a newer removed copy is hidden from
    display, so the resume must not restore it."""
    removed: dict[Any, int] = {}
    for row in rows:
        if row.visibility is RowVisibility.REWOUND and row.message.role == "tool" and row.message.tool_name in ("todo_list", "todo"):
            for copy_key in store.copy_identities(row):
                removed[copy_key] = max(removed.get(copy_key, 0), row.id)
    return removed


def _todo_content(rows: Sequence[StoredRow], index: int, max_chars: int) -> Optional[str]:
    """The row's content when it is a valid todo result paired with the assistant call right before it."""
    message = rows[index].message
    content = message.content
    if message.role != "tool" or message.tool_name not in ("todo_list", "todo") or not message.tool_call_id:
        return None
    if store.text_length(content) > max_chars or not isinstance(content, str):
        return None
    prior = next((row.message for row in reversed(rows[:index]) if row.message.role in ("assistant", "user", "system")), None)
    if prior is None or prior.role != "assistant":
        return None
    try:
        if not any(_is_todo_call(call, message.tool_call_id) for call in prior.tool_calls):
            return None
        data = json.loads(content)
        if not isinstance(data, dict) or not isinstance(data.get("todos"), list):
            return None
        int(data.get("revision") or 0)
    except (ValueError, TypeError):
        return None
    return content


register_repository("messages", PostgresMessageRepository)
