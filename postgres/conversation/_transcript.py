"""The ``messages`` table as the message repository reads and writes it: the stored row, its column codec, and the
projections every transcript read shares (display slots, generation dedupe, the replay dicts).

A row's payload is decoded to a ``NewMessage`` whose ``timestamp`` and ``message_uid`` are always set. Content is in at
most one of ``content`` (text), ``content_json`` (structured) and ``content_bytes`` (text holding a NUL byte, which a
Postgres text value cannot). Ids are the table's identity column, so id order is the append sequence.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import time
from dataclasses import dataclass, fields, replace
from typing import Any, Iterable, Optional, Sequence

from psycopg import Connection
from psycopg.types.json import Json

from persistence import RowVisibility
from persistence.conversation.messages import (
    CONTENT_JSON_PREFIX, DB_PERSISTED_MARKER_KEY, MessageRecord, NewMessage, RowDisposition, RowOutcome, RowStamp,
    TranscriptRow,
)
from persistence.core.identity import JsonValue, MessageId, SessionId

from .._sql import execute, rows as _rows
from . import _policy as policy

SIDECARS = ("reasoning_details", "codex_reasoning_items", "codex_message_items")
REASONING_FIELDS = ("reasoning", "reasoning_content", *SIDECARS)
# Columns other same-process writers stamp after a flush, and the identity columns minted at insert: neither is part
# of the ownership version, and a snapshot-matched rewrite keeps the stored values.
METADATA_FIELDS = ("display_kind", "display_metadata", "api_content", "codex_reasoning_items", "platform_message_id")
IDENTITY_FIELDS = ("message_uid", "absorbed_message_uids", "tool_call_uids", "tool_call_uid")
LIVE_MISSING_METADATA = ("display_kind", "display_metadata")
_UNOWNED_FIELDS = frozenset({"role", "timestamp", "repair_target", "db_row_snapshot", "data_use", "restored_from_store",
                             *METADATA_FIELDS, *IDENTITY_FIELDS})
OWNED_FIELDS = tuple(f.name for f in fields(NewMessage) if f.name not in _UNOWNED_FIELDS)
_PR_URL = re.compile("/pull/", re.IGNORECASE | re.ASCII)
_NUL = "\x00"

# The payload columns a row is written with, in the order ``_columns`` yields them.
PAYLOAD_COLUMNS = (
    "role", "content", "content_json", "content_bytes", "timestamp", "tool_call_id", "tool_calls", "tool_name",
    "effect_disposition", "token_count", "finish_reason", "reasoning", "reasoning_content", "reasoning_details",
    "codex_reasoning_items", "codex_message_items", "platform_message_id", "observed", "compressed_summary",
    "api_content", "display_kind", "display_metadata", "message_uid", "absorbed_message_uids", "tool_call_uids",
    "tool_call_uid",
)


@dataclass
class StoredRow:
    id: int
    session_id: str
    message: NewMessage
    visibility: RowVisibility = RowVisibility.ACTIVE
    # The display slot an in-place prune kept for a rewritten row (its group's first row id), else None.
    display_slot: Optional[int] = None

    @property
    def reactions(self) -> list[dict[str, Any]]:
        stored = (self.message.display_metadata or {}).get("reactions")
        return [item for item in stored if isinstance(item, dict)] if isinstance(stored, list) else []

    @reactions.setter
    def reactions(self, values: list[dict[str, Any]]) -> None:
        metadata = dict(self.message.display_metadata or {})
        if values:
            metadata["reactions"] = values
        else:
            metadata.pop("reactions", None)
        self.message = replace(self.message, display_metadata=metadata or None)

    @property
    def epoch(self) -> float:
        return float(self.message.timestamp or 0.0)

    def as_record(self) -> MessageRecord:
        m = copy.deepcopy(self.message)
        return MessageRecord(
            id=MessageId(self.id), role=m.role, content=m.content, timestamp=self.epoch,
            tool_calls=m.tool_calls, tool_name=m.tool_name, tool_call_id=m.tool_call_id,
            display_kind=m.display_kind, compressed_summary=m.compressed_summary, visibility=self.visibility,
            api_content=m.api_content,
            display_metadata={key: value for key, value in (m.display_metadata or {}).items() if key != "reactions"} or None,
            platform_message_id=m.platform_message_id, observed=m.observed,
            effect_disposition=m.effect_disposition, finish_reason=m.finish_reason,
            reasoning=m.reasoning, reasoning_content=m.reasoning_content,
            reasoning_details=_record_sidecar(m.reasoning_details),
            codex_reasoning_items=_record_sidecar(m.codex_reasoning_items),
            codex_message_items=_record_sidecar(m.codex_message_items),
            message_uid=m.message_uid,
        )

    def as_transcript_row(self) -> TranscriptRow:
        m = copy.deepcopy(self.message)
        return TranscriptRow(
            id=MessageId(self.id), session_id=SessionId(self.session_id), role=m.role, content=m.content,
            timestamp=self.epoch, visibility=self.visibility,
            tool_call_id=m.tool_call_id, tool_calls=list(m.tool_calls) if m.tool_calls else None,
            tool_name=m.tool_name, effect_disposition=m.effect_disposition, token_count=m.token_count,
            finish_reason=m.finish_reason, reasoning=m.reasoning, reasoning_content=m.reasoning_content,
            reasoning_details=_sidecar_text(m.reasoning_details),
            codex_reasoning_items=_sidecar_text(m.codex_reasoning_items),
            codex_message_items=_sidecar_text(m.codex_message_items), platform_message_id=m.platform_message_id,
            observed=m.observed, compressed_summary=m.compressed_summary, api_content=m.api_content,
            display_kind=m.display_kind, display_metadata=dict(m.display_metadata) if m.display_metadata else None,
            message_uid=m.message_uid,
            absorbed_message_uids=json.dumps(list(m.absorbed_message_uids)) if m.absorbed_message_uids else None,
            tool_call_uids=json.dumps(dict(m.tool_call_uids), sort_keys=True) if m.tool_call_uids else None,
            tool_call_uid=m.tool_call_uid,
        )


# ── the column codec ──

def _visibility(active: bool, compacted: bool) -> RowVisibility:
    if active:
        return RowVisibility.ACTIVE
    return RowVisibility.COMPACTION_ARCHIVED if compacted else RowVisibility.REWOUND


def visibility_flags(visibility: RowVisibility) -> tuple[bool, bool]:
    """``(active, compacted)`` of a visibility."""
    return visibility is RowVisibility.ACTIVE, visibility is RowVisibility.COMPACTION_ARCHIVED


def decode(row: dict[str, Any]) -> StoredRow:
    """A selected ``messages`` row as a stored row."""
    if row["content_bytes"] is not None:
        content: Any = bytes(row["content_bytes"]).decode("utf-8")
    elif row["content_json"] is not None:
        content = row["content_json"]
    else:
        content = row["content"]
    message = NewMessage(
        role=row["role"], content=content, timestamp=row["timestamp"], tool_calls=tuple(row["tool_calls"] or ()),
        tool_name=row["tool_name"], tool_call_id=row["tool_call_id"], display_kind=row["display_kind"],
        compressed_summary=bool(row["compressed_summary"]), api_content=row["api_content"],
        display_metadata=row["display_metadata"], platform_message_id=row["platform_message_id"],
        observed=bool(row["observed"]), effect_disposition=row["effect_disposition"],
        finish_reason=row["finish_reason"], reasoning=row["reasoning"], reasoning_content=row["reasoning_content"],
        reasoning_details=row["reasoning_details"], codex_reasoning_items=row["codex_reasoning_items"],
        codex_message_items=row["codex_message_items"], message_uid=row["message_uid"],
        absorbed_message_uids=tuple(row["absorbed_message_uids"] or ()), tool_call_uids=row["tool_call_uids"],
        tool_call_uid=row["tool_call_uid"], token_count=row["token_count"],
    )
    return StoredRow(id=int(row["id"]), session_id=row["session_id"], message=message,
                     visibility=_visibility(row["active"], row["compacted"]), display_slot=row["display_slot"])


def _json(value: Any) -> Optional[Json]:
    return None if value is None else Json(value)


def columns(message: NewMessage) -> tuple[Any, ...]:
    """The stored payload of ``message`` (already normalized by ``payload``) as ``PAYLOAD_COLUMNS`` values."""
    content = message.content
    text = structured = raw = None
    if isinstance(content, (list, dict)):
        structured = Json(content)
    elif isinstance(content, str) and _NUL in content:
        raw = content.encode("utf-8")
    else:
        text = content
    return (
        message.role, text, structured, raw, message.timestamp, message.tool_call_id,
        _json(list(message.tool_calls)) if message.tool_calls else None, message.tool_name,
        message.effect_disposition, message.token_count, message.finish_reason, message.reasoning,
        message.reasoning_content, _json(message.reasoning_details or None), _json(message.codex_reasoning_items or None),
        _json(message.codex_message_items or None), message.platform_message_id, bool(message.observed),
        bool(message.compressed_summary), message.api_content, message.display_kind,
        _json(dict(message.display_metadata)) if message.display_metadata else None, message.message_uid,
        _json(list(message.absorbed_message_uids)) if message.absorbed_message_uids else None,
        _json(dict(message.tool_call_uids)) if message.tool_call_uids else None, message.tool_call_uid,
    )


def load(conn: Connection, session_ids: Sequence[str], *, lock: bool = False) -> list[StoredRow]:
    """Every stored row of ``session_ids`` in id order; ``lock`` takes their row locks."""
    found = _rows(conn, "SELECT * FROM messages WHERE session_id = ANY(%s) ORDER BY id" + (" FOR UPDATE" if lock else ""),
                  (list(dict.fromkeys(session_ids)),))
    return [decode(item) for item in found]


def insert(conn: Connection, session_id: str, message: NewMessage, *,
           visibility: RowVisibility = RowVisibility.ACTIVE, display_slot: Optional[int] = None) -> StoredRow:
    """Insert one normalized payload; returns the stored row with its minted id."""
    active, compacted = visibility_flags(visibility)
    new_id = conn.execute(
        f"INSERT INTO messages (session_id, {', '.join(PAYLOAD_COLUMNS)}, active, compacted, display_slot) "
        f"VALUES (%s, {', '.join(['%s'] * len(PAYLOAD_COLUMNS))}, %s, %s, %s) RETURNING id",
        (session_id, *columns(message), active, compacted, display_slot)).fetchone()[0]
    return StoredRow(id=int(new_id), session_id=session_id, message=message, visibility=visibility,
                     display_slot=display_slot)


def save(conn: Connection, row: StoredRow) -> None:
    """Write a stored row's payload, visibility and display slot back."""
    active, compacted = visibility_flags(row.visibility)
    assignments = ", ".join(f"{column} = %s" for column in PAYLOAD_COLUMNS)
    execute(conn, f"UPDATE messages SET {assignments}, active = %s, compacted = %s, display_slot = %s WHERE id = %s",
            (*columns(row.message), active, compacted, row.display_slot, row.id))


def save_visibility(conn: Connection, rows: Iterable[StoredRow]) -> None:
    for row in rows:
        active, compacted = visibility_flags(row.visibility)
        execute(conn, "UPDATE messages SET active = %s, compacted = %s WHERE id = %s", (active, compacted, row.id))


# ── payload normalization ──

def stored_content(content: Any) -> Any:
    """Content as the store keeps it: lone surrogates replaced, structured values copied, other scalars as their TEXT
    spelling. A string carrying the reserved ``\\x00json:`` prefix reads back verbatim."""
    if isinstance(content, str):
        return re.sub(r"[\ud800-\udfff]", "�", content)
    if isinstance(content, (dict, list, tuple)):
        return json.loads(json.dumps(content))
    if isinstance(content, bool):
        return str(int(content))
    if isinstance(content, int):
        return str(content)
    if isinstance(content, float):
        return policy.real_as_text(content)
    return content


def payload(message: NewMessage, *, keep_reasoning: bool = True) -> NewMessage:
    """The stored form of a written row, a copy that shares no value with the caller's: content normalized,
    ``data_use`` folded into ``display_metadata``, empty sidecars absent, reasoning dropped unless kept, and the
    caller's resolution bookkeeping never stored."""
    message = copy.deepcopy(message)
    metadata = (dict(message.display_metadata)
                if isinstance(message.display_metadata, dict) and message.display_metadata else None)
    if message.data_use:
        metadata = {**(metadata or {}), "data_use": list(message.data_use)}
    reasoning: dict[str, Any] = ({name: None for name in REASONING_FIELDS} if not keep_reasoning
                                 else {name: getattr(message, name) or None for name in SIDECARS})
    return replace(message, content=stored_content(message.content), display_metadata=metadata, data_use=(),
                   repair_target=None, db_row_snapshot=None, restored_from_store=False, **reasoning)


def _sidecar_text(value: Any) -> Optional[str]:
    if not value:
        return None
    return value if isinstance(value, str) else json.dumps(value)


def replayed_sidecar(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return None
    return value


def _record_sidecar(value: Any) -> Any:
    if not value:
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def text_length(content: Any) -> int:
    """SQL ``length()`` of the stored content in the SQLite store: characters before the first NUL for text, 0 for
    NULL and for structured content."""
    if content is None or isinstance(content, (list, dict)):
        return 0
    text = content if isinstance(content, str) else str(content)
    return len(text.split(_NUL, 1)[0])


def content_text(content: Any) -> Optional[str]:
    """The content as one text cell: text as is, structured content as its sentinel-prefixed JSON."""
    if content is None or isinstance(content, str):
        return content
    if isinstance(content, (list, dict)):
        return CONTENT_JSON_PREFIX + json.dumps(content)
    return str(content)


def has_text(content: Any) -> bool:
    """``content IS NOT NULL AND TRIM(content) != ''`` (spaces only)."""
    return content is not None and (content_text(content) or "").strip(" ") != ""


def hashable(value: Any) -> Any:
    return value if isinstance(value, (str, int, float, bytes, type(None))) else json.dumps(value)


def row_snapshot(message: NewMessage) -> str:
    """The opaque ownership version: a digest of the owned payload only."""
    owned = [[name, getattr(message, name)] for name in OWNED_FIELDS]
    encoded = json.dumps(owned, sort_keys=True, default=repr, ensure_ascii=False).encode("utf-8", "surrogatepass")
    return hashlib.blake2b(encoded, digest_size=16).hexdigest()


def identity_keys(message: NewMessage) -> dict[str, JsonValue]:
    out: dict[str, JsonValue] = {}
    if message.message_uid:
        out["message_uid"] = message.message_uid
    if message.tool_call_uids:
        out["_tool_call_uids"] = dict(message.tool_call_uids)
    if message.tool_call_uid:
        out["_tool_call_uid"] = message.tool_call_uid
    if message.absorbed_message_uids:
        out["_absorbed_message_uids"] = list(message.absorbed_message_uids)
    return out


def decoded_row(message: NewMessage) -> dict[str, JsonValue]:
    """A stored row decoded for a live dict: present keys only, reasoning on assistant rows."""
    message = copy.deepcopy(message)
    row: dict[str, JsonValue] = {"role": message.role, "content": message.content}
    for key in ("tool_call_id", "tool_name", "effect_disposition", "token_count", "finish_reason"):
        if getattr(message, key) is not None:
            row[key] = getattr(message, key)
    if message.tool_calls:
        row["tool_calls"] = list(message.tool_calls)
    if message.platform_message_id is not None:
        row["message_id"] = row["platform_message_id"] = message.platform_message_id
    if message.observed:
        row["observed"] = True
    if message.compressed_summary:
        row["_compressed_summary"] = True
    row.update(identity_keys(message))
    for key in ("api_content", "display_kind", "display_metadata"):
        if getattr(message, key) is not None:
            row[key] = getattr(message, key)
    if message.role == "assistant":
        for key in ("reasoning", "reasoning_content"):
            if getattr(message, key) is not None:
                row[key] = getattr(message, key)
        for key in SIDECARS:
            if getattr(message, key):
                row[key] = replayed_sidecar(getattr(message, key))
    return row


def row_stamp(row: StoredRow) -> RowStamp:
    m = row.message
    return RowStamp(id=MessageId(row.id), timestamp=row.epoch, db_row_snapshot=row_snapshot(m),
                    message_uid=str(m.message_uid),
                    tool_call_uids=copy.deepcopy(dict(m.tool_call_uids)) if m.tool_call_uids else None,
                    tool_call_uid=m.tool_call_uid)


def row_outcome(row: StoredRow, disposition: RowDisposition,
                canonical: Optional[dict[str, JsonValue]] = None) -> RowOutcome:
    stamp = row_stamp(row)
    return RowOutcome(id=stamp.id, disposition=disposition, timestamp=stamp.timestamp,
                      db_row_snapshot=stamp.db_row_snapshot, canonical_row=copy.deepcopy(canonical),
                      message_uid=stamp.message_uid, tool_call_uids=stamp.tool_call_uids,
                      tool_call_uid=stamp.tool_call_uid)


def stamp_tool_call_uids(message: NewMessage, index: dict[str, str]) -> NewMessage:
    """Per-occurrence tool-call identity at insert: an assistant row mints a uid for every call without one; a result
    row without one takes the uid its batch's assistant row named."""
    if message.role == "assistant" and message.tool_calls:
        uids = dict(message.tool_call_uids or {})
        for call in message.tool_calls:
            call_id = policy.coalesce_tool_call_id(call)
            if call_id and call_id not in uids:
                uids[call_id] = policy.mint_uid()
        policy.index_tool_call_uids(index, {"tool_calls": list(message.tool_calls), "_tool_call_uids": uids})
        return replace(message, tool_call_uids=uids or None)
    if message.role == "user":
        index.clear()
    elif message.role == "tool" and not message.tool_call_uid:
        uid = policy.resolve_tool_call_uid(index, message.tool_call_id)
        if uid:
            return replace(message, tool_call_uid=uid)
    return message


def insert_rows(conn: Connection, session_id: str, messages: Sequence[NewMessage], *,
                keep_reasoning: Optional[bool] = None, now: Optional[float] = None) -> list[StoredRow]:
    """Insert ``messages`` as fresh active rows: a missing or invalid timestamp takes the clock, kept strictly
    increasing across the batch; a missing ``message_uid`` is minted; reasoning is kept on assistant rows unless
    ``keep_reasoning`` says otherwise. The session's counters count every inserted row and call."""
    now_ts = time.time() if now is None else now
    index: dict[str, str] = {}
    inserted: list[StoredRow] = []
    for message in messages:
        stamp = policy.coerce_epoch(message.timestamp) if message.timestamp is not None else None
        stamp = now_ts if stamp is None else (0.0 if stamp == 0.0 else stamp)
        keep = message.role == "assistant" if keep_reasoning is None else keep_reasoning
        stored = replace(payload(message, keep_reasoning=keep), timestamp=stamp,
                         message_uid=message.message_uid or policy.mint_uid())
        inserted.append(insert(conn, session_id, stamp_tool_call_uids(stored, index)))
        now_ts = max(now_ts, stamp) + 1e-6
    if inserted:
        bump_counters(conn, session_id, len(inserted), sum(len(row.message.tool_calls) for row in inserted))
    return inserted


def clone_rows(conn: Connection, session_id: str, originals: Iterable[StoredRow]) -> list[StoredRow]:
    """Column clones of ``originals`` as fresh active rows of ``session_id``: the payload exact, a new id."""
    return [insert(conn, session_id, row.message) for row in originals]


def bump_counters(conn: Connection, session_id: str, messages: int, tool_calls: int) -> None:
    execute(conn, "UPDATE sessions SET message_count = message_count + %s, tool_call_count = tool_call_count + %s "
                  "WHERE id = %s", (messages, tool_calls, session_id))


def recount(conn: Connection, session_id: str) -> None:
    """The absolute counter SET over the session's active rows that replace, rewind and compaction end with."""
    execute(conn, """
        UPDATE sessions s SET message_count = c.messages, tool_call_count = c.calls
        FROM (SELECT count(*) AS messages,
                     coalesce(sum(CASE WHEN json_typeof(tool_calls) = 'array' THEN json_array_length(tool_calls)
                                       ELSE 0 END), 0) AS calls
              FROM messages WHERE session_id = %s AND active) c
        WHERE s.id = %s""", (session_id, session_id))


def bump_transcript_revision(conn: Connection, session_id: str) -> None:
    """An in-place rewrite of a row's payload columns moves the owning session's ``transcript_revision``."""
    execute(conn, "UPDATE sessions SET transcript_revision = transcript_revision + 1 WHERE id = %s", (session_id,))


# ── display identity and dedupe ──

def display_key(row: StoredRow) -> tuple[Any, ...]:
    """A row's display identity: a tool-calling assistant row by its call ids, a user carrier by its live view,
    every other row by role, content, timestamp and tool columns."""
    m = row.message
    call_ids = tuple(policy.coalesce_tool_call_id(call) for call in m.tool_calls)
    if m.role == "assistant" and call_ids and all(call_ids):
        return ("assistant", None, m.timestamp, m.tool_call_id, call_ids, m.tool_name)
    content: Any = m.content
    if m.role == "user":
        handoff, live = policy.split_user_originated_turn(
            {"role": "user", "content": m.content, "display_kind": m.display_kind, "display_metadata": m.display_metadata})
        if handoff is not None and live is not None:
            content = live.get("content")
    return (m.role, hashable(content), m.timestamp, m.tool_call_id,
            hashable(list(m.tool_calls)) if m.tool_calls else None, m.tool_name)


def _fold_summary_carrier(row: StoredRow, key: tuple[Any, ...], first_by_uid: dict[str, tuple[Any, ...]]) -> tuple[Any, ...]:
    """A compaction carrier keeps its message's uid but not its content, so it shows as the first row with that uid."""
    uid = row.message.message_uid
    if not uid:
        return key
    if row.message.compressed_summary:
        return first_by_uid.get(uid, key)
    first_by_uid.setdefault(uid, key)
    return key


def model_only(row: StoredRow) -> bool:
    return bool((row.message.display_metadata or {}).get(policy.MODEL_ONLY_KEY))


def _rank(row: StoredRow) -> tuple[bool, int]:
    return (row.visibility is RowVisibility.ACTIVE, row.id)


def _placed(rows: Sequence[StoredRow]) -> list[tuple[int, StoredRow]]:
    """``(slot, row)`` for each row a display read can show; the slot is its identity's first row id in its session,
    or the slot an in-place prune kept."""
    slots: dict[tuple[str, tuple[Any, ...]], int] = {}
    first_by_uid: dict[str, dict[str, tuple[Any, ...]]] = {}
    placed = []
    for row in rows:
        if row.visibility is RowVisibility.REWOUND or model_only(row):
            continue
        key = _fold_summary_carrier(row, display_key(row), first_by_uid.setdefault(row.session_id, {}))
        slot = slots.setdefault((row.session_id, key), row.display_slot if row.display_slot is not None else row.id)
        placed.append((slot, row))
    return placed


def _later_tool_copies(placed: Sequence[tuple[int, StoredRow]]) -> set[int]:
    """Ids of the tool rows a display read skips so a compacted tool result paints once."""
    def owner(row: StoredRow) -> Optional[tuple[str, str, float]]:
        m = row.message
        return ((row.session_id, m.message_uid, m.timestamp)
                if m.role == "tool" and m.message_uid and m.timestamp is not None else None)

    first: dict[tuple[str, str, float], int] = {}
    for slot, row in placed:
        if (key := owner(row)) and row.visibility is RowVisibility.COMPACTION_ARCHIVED:
            first[key] = min(first.get(key, slot), slot)
    return {row.id for slot, row in placed if (key := owner(row)) and first.get(key, slot) < slot}


def display_slots(rows: Sequence[StoredRow]) -> list[tuple[int, StoredRow]]:
    """One session's display set: each logical message once in its slot, the live row preferred, then the newest;
    ``(slot, row)`` in slot order."""
    chosen: dict[int, StoredRow] = {}
    placed = _placed(rows)
    later = _later_tool_copies(placed)
    for slot, row in placed:
        if row.id in later:
            continue
        current = chosen.get(slot)
        if current is None or _rank(row) > _rank(current):
            chosen[slot] = row
    return [(slot, chosen[slot]) for slot in sorted(chosen)]


def copy_identities(row: StoredRow) -> tuple[Any, ...]:
    m = row.message
    return (("key", display_key(row)), *(("uid", uid) for uid in (m.message_uid, *m.absorbed_message_uids) if uid))


def dedupe_display(rows: Sequence[StoredRow], *, drop_removed: bool = True,
                   first_ids: Optional[dict[int, int]] = None) -> list[StoredRow]:
    """Generations collapsed by display identity: rows in id order in, each logical message once out (live row
    preferred, then newest, ordered by its first row). With ``drop_removed`` the input holds every visibility, a
    message whose newest copy was removed by a rewind or a replacement is left out and a later copy of a compacted
    tool result is not shown. ``first_ids`` gets each kept row's id -> its message's first durable row id."""
    seen: dict[tuple[Any, ...], StoredRow] = {}
    first_id: dict[tuple[Any, ...], int] = {}
    first_by_uid: dict[str, tuple[Any, ...]] = {}
    removed: dict[Any, bool] = {}
    removed_first: dict[Any, int] = {}
    identities: dict[tuple[Any, ...], tuple[Any, ...]] = {}
    later = _later_tool_copies(_placed(rows)) if drop_removed else set()
    for row in rows:
        gone = row.visibility is RowVisibility.REWOUND
        copies = copy_identities(row)
        if drop_removed:
            removed.update(dict.fromkeys(copies, gone))
            if gone:
                for copy_key in copies:
                    removed_first.setdefault(copy_key, row.id)
                continue
        if model_only(row) or row.id in later:
            continue
        key = _fold_summary_carrier(row, display_key(row), first_by_uid)
        current = seen.get(key)
        if current is None or _rank(row) > _rank(current):
            seen[key] = row
            identities[key] = copies
        first_id[key] = min(first_id.get(key, row.id), row.id)
    keys = [key for key in sorted(seen, key=first_id.__getitem__)
            if not (drop_removed and any(removed[copy_key] for copy_key in identities[key]))]
    if first_ids is not None:
        first_ids.update((seen[key].id, min([first_id[key], *(removed_first[copy_key] for copy_key in identities[key]
                                                             if copy_key in removed_first)])) for key in keys)
    return [seen[key] for key in keys]


# ── the replay projection ──

def _replay_dict(row: StoredRow, *, include_row_ids: bool, include_summary_markers: bool,
                 stamp_snapshot: bool) -> dict[str, Any]:
    m = copy.deepcopy(row.message)
    msg: dict[str, Any] = {"role": m.role, "content": policy.loaded_view(m.role, m.content), DB_PERSISTED_MARKER_KEY: True}
    if stamp_snapshot:
        msg["_db_row_snapshot"] = row_snapshot(m)
    if include_row_ids:
        msg["_row_id"] = row.id
    msg.update(identity_keys(m))
    msg.update((key, getattr(m, key)) for key in ("api_content", "display_kind") if getattr(m, key))
    if m.display_metadata:
        msg["display_metadata"] = dict(m.display_metadata)
        if m.display_metadata.get("data_use"):
            msg["data_use"] = list(m.display_metadata["data_use"])
    if include_summary_markers and m.compressed_summary:
        msg["_compressed_summary"] = True
    msg.update((key, getattr(m, key)) for key in ("timestamp", "tool_call_id", "tool_name", "effect_disposition")
               if getattr(m, key))
    if m.tool_calls:
        msg["tool_calls"] = list(m.tool_calls)
    if m.platform_message_id:
        msg["message_id"] = m.platform_message_id
    if m.observed:
        msg["observed"] = True
    if m.role == "assistant":
        msg.update((key, getattr(m, key)) for key in ("finish_reason", "reasoning") if getattr(m, key))
        if m.reasoning_content is not None:
            msg["reasoning_content"] = m.reasoning_content
        msg.update((key, replayed_sidecar(getattr(m, key))) for key in SIDECARS if getattr(m, key))
    return msg


def conversation_dicts(rows: Sequence[StoredRow], *, include_lineage: bool, repair_alternation: bool,
                       include_row_ids: bool, include_summary_markers: bool) -> list[dict[str, Any]]:
    """Rows (id order, already filtered) as the replay projection, after the shared projection policy."""
    stamp_snapshot = repair_alternation and not include_row_ids
    messages: list[dict[str, Any]] = []
    clones: dict[Any, dict[str, Any]] = {}
    index: dict[str, str] = {}
    owners: list[dict[str, Any]] = []
    for row in rows:
        msg = _replay_dict(row, include_row_ids=include_row_ids, include_summary_markers=include_summary_markers,
                           stamp_snapshot=stamp_snapshot)
        role = row.message.role
        if role == "assistant" and msg.get("tool_calls"):
            owners.append(msg)
        elif role == "user":
            index.clear()
            owners.clear()
        elif role == "tool" and not row.message.tool_call_uid and row.message.tool_call_id:
            for owner in owners:
                policy.index_tool_call_uids(index, owner)
            owners.clear()
            uid = policy.resolve_tool_call_uid(index, row.message.tool_call_id)
            if uid:
                msg["_tool_call_uid"] = uid
        if include_lineage:
            skip, clone_key = policy.dedupe_replayed_user(messages, msg, clones)
            if skip:
                continue
            if clone_key is not None:
                clones[clone_key] = msg
        messages.append(msg)
    messages = policy.strip_stale_tool_call_markers(policy.strip_background_review_harness(messages))
    if repair_alternation and messages:
        policy.repair_message_sequence(messages)
    return messages


def pr_url_content(content: Any) -> bool:
    return isinstance(content, str) and bool(_PR_URL.search(content))
