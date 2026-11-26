"""``SessionMaintenance`` on Postgres: prune, ghost and orphan sweeps, idle archive, export, import, foreign import,
backend heartbeats and the read-only probes.

The prune, sweep and archive selections read the session rows with their recency (the freshest of the activity stamp
and the newest stored message, else ``started_at``) in one query and apply the filter rules to the ``SessionRecord``
of each row. Deletes go through the session family's ledger check and retirement, in one transaction. Imports insert
the rows and their transcripts through the message family's row writer, then count the live rows.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from psycopg import Connection

from persistence import Conflict, ExportTooLarge, RowVisibility, SessionId
from persistence.conversation.messages import NewMessage, TranscriptRead, TranscriptRow, TranscriptView
from persistence.conversation.sessions import CANONICAL_BOT_CHAT_TITLE, END_REASON_COMPRESSION, SessionRecord
from persistence.core.errors import AuthorityRefusal
from persistence.housekeeping.maintenance import (
    GHOST_SESSION_MIN_AGE, GHOST_SESSION_SOURCE, IMPORT_MAX_SESSIONS, ORPHAN_REAP_END_REASON,
    SESSION_PROJECTION_FIELDS, ForeignImportResult, ForeignOrigin, ForeignTurn, ImportResult, PruneCandidate,
    PruneCriteria, PruneReclaimEstimate, PruneResult, SessionExport, SessionExportRow, SessionLineageExport,
    TreeServeFacts,
)
from session_rules.imports import IMPORT_MESSAGE_JSON_FIELDS, validate_import_payload
from session_rules.legacy_local import bindings_agree
from session_rules.session_ids import new_session_id

from .._backend import PostgresStore, register_repository
from .._sql import execute, rows as _rows, scalar, store_text
from ..authority import _leases, _meta
from ..authority._bindings_lineage import HISTORY_PREFIX
from ..authority.admissions import bound_id
from ..authority.bindings import binding_text
from ..conversation import _lineage, _policy as policy, _retirement, _transcript as transcript_store
from ..conversation.sessions import _SELECT_RESOLVED, session_record

_IMPORT_PASSTHROUGH = ("user_id", "model", "end_reason", "cwd", "git_branch", "git_repo_root", "billing_provider",
                       "billing_base_url", "billing_mode", "cost_status", "cost_source", "pricing_version", "title")
_IMPORT_INTS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens",
                "api_call_count")
_IMPORT_FLOATS = ("ended_at", "estimated_cost_usd", "actual_cost_usd")
_IMPORT_FLAGS = ("archived", "auto_archived", "pinned", "hidden")
_MESSAGE_PASSTHROUGH = (
    "tool_call_id", "tool_name", "effect_disposition", "finish_reason", "reasoning", "reasoning_content",
    "platform_message_id", "token_count", "display_kind", "api_content", "message_uid", "tool_call_uid",
    "reasoning_details", "codex_reasoning_items", "codex_message_items",
)
_INT_RANGE = (-(2**63), 2**63 - 1)
_DAY = 86400.0
_RECENCY = ("coalesce(greatest(s.last_activity_at, (SELECT max(m.timestamp) FROM messages m WHERE m.session_id = s.id)), "
            "s.started_at)")
_WITH_RECENCY = f"{_SELECT_RESOLVED.replace('SELECT s.*,', f'SELECT s.*, {_RECENCY} AS recency,', 1)}"


# ── matching rules ──

def _ascii_lower(text: str) -> str:
    return "".join(ch.lower() if ch.isascii() else ch for ch in text)


def _like(haystack: Optional[str], needle: str) -> bool:
    """``LOWER(COALESCE(col, '')) LIKE '%' || needle.lower() || '%'``: ASCII folds, the needle lowers in full."""
    return _ascii_lower(needle.lower()) in _ascii_lower(haystack or "")


def _under(path: Optional[str], prefix: str) -> bool:
    """The exact cwd (case-sensitive) or a path below it (ASCII-case-insensitive)."""
    if not path:
        return False
    root = prefix.rstrip("/\\") or prefix
    if path == root:
        return True
    low_path, low_root = _ascii_lower(path), _ascii_lower(root)
    return low_path.startswith(low_root + "/") or low_path.startswith(low_root + "\\")


def _inside_tree(worktree_path: str) -> Callable[[Optional[str]], bool]:
    """``inside(cwd)``: the tree or below it, over the tree's own spelling and its resolved one."""
    def key(value: object) -> str:
        return os.path.normcase(str(value)).replace("\\", "/").rstrip("/")

    roots = {key(Path(worktree_path)), key(Path(worktree_path).resolve())}
    return lambda cwd: bool(cwd) and any(key(cwd) == r or key(cwd).startswith(r + "/") for r in roots)


def _cost(record: SessionRecord) -> float:
    if record.actual_cost_usd is not None:
        return record.actual_cost_usd
    return record.estimated_cost_usd if record.estimated_cost_usd is not None else 0.0


def _bounded(value: float, low: Optional[float], high: Optional[float]) -> bool:
    return (low is None or value >= low) and (high is None or value <= high)


# ── import helpers ──

def _coerce_or(value: Any, cast: Callable[[Any], Any], default: Any) -> Any:
    try:
        return default if value is None else cast(value)
    except (TypeError, ValueError):
        return default


def _finite_or_none(value: Any) -> Optional[float]:
    try:
        number = None if value is None else float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number is not None and math.isfinite(number) else None


def _int_or_zero(value: Any) -> int:
    number = _coerce_or(value, int, 0)
    return number if not isinstance(number, bool) and _INT_RANGE[0] <= number <= _INT_RANGE[1] else 0


def _decoded(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except ValueError:
        return value


def _imported_message(message: Mapping[str, Any]) -> NewMessage:
    """One normalized payload message as the stored row payload."""
    tool_calls = _decoded(message.get("tool_calls"))
    metadata = _decoded(message.get("display_metadata"))
    absorbed = _decoded(message.get("absorbed_message_uids"))
    uids = _decoded(message.get("tool_call_uids"))
    return NewMessage(
        role=message["role"], content=message.get("content"),
        timestamp=policy.coerce_epoch(message["timestamp"]) if message.get("timestamp") is not None else None,
        tool_calls=tuple(tool_calls) if isinstance(tool_calls, list) else (),
        display_metadata=metadata if isinstance(metadata, dict) else None,
        observed=bool(_coerce_or(message.get("observed"), int, 0)), compressed_summary=bool(message.get("_compressed_summary")),
        absorbed_message_uids=tuple(absorbed) if isinstance(absorbed, list) else (),
        tool_call_uids=uids if isinstance(uids, dict) else None,
        **{field: _decoded(message.get(field)) if field in IMPORT_MESSAGE_JSON_FIELDS else message.get(field)
           for field in _MESSAGE_PASSTHROUGH})


def _stored_bytes(row: TranscriptRow) -> int:
    """The summed stored length of a message row's text and binary fields."""
    size = 0
    for value in (getattr(row, f.name) for f in dataclasses.fields(row)):
        if isinstance(value, str):
            size += len(value.encode("utf-8", "surrogatepass"))
        elif isinstance(value, (bytes, bytearray)):
            size += len(value)
        elif isinstance(value, (dict, list, tuple)):
            size += len(json.dumps(value).encode("utf-8"))
    return size


def _export_row(record: SessionRecord, *, last_active: Optional[float] = None) -> SessionExportRow:
    """The stored row shape of a record: parsed columns back to their stored text, flags to 0/1."""
    columns = {f.name: getattr(record, f.name) for f in dataclasses.fields(SessionExportRow) if hasattr(record, f.name)}
    columns.update(
        id=record.session_id,
        origin_json=None if record.origin is None else json.dumps(dict(record.origin), ensure_ascii=False),
        model_config=json.dumps(dict(record.model_config)) if record.model_config else None,
        title_source=getattr(record.title_source, "value", record.title_source),
        tool_names=None if record.tool_pin is None else json.dumps(record.tool_pin),
        last_active=last_active,
        **{flag: int(getattr(record, flag)) for flag in ("expiry_finalized", "archived", "auto_archived", "pinned", "hidden")})
    return SessionExportRow(**columns)


def _active_state(value: Optional[str]) -> bool:
    if value is None:
        return False
    try:
        parsed = json.loads(value)
    except ValueError:
        return True
    status = parsed.get("status") if isinstance(parsed, dict) else None
    return (status if status is not None else "active") == "active"


class PostgresMaintenance:
    """Session maintenance of the bound profile's schema."""

    def __init__(self, store: PostgresStore) -> None:
        self._store = store

    # ── selection plumbing ──

    @staticmethod
    def _ranked(conn: Connection, where: str = "true", params: Any = ()) -> list[tuple[float, SessionRecord, dict[str, Any]]]:
        """``(recency, record, row)`` of every row matching ``where`` (over alias ``s``)."""
        return [(item["recency"], session_record(item), item) for item in _rows(conn, f"{_WITH_RECENCY} WHERE {where}", params)]

    @staticmethod
    def _pinned_tail(conn: Connection) -> set[str]:
        """Unpinned continuations a pinned segment covers."""
        edge = _lineage.CONTINUATION_EDGE_SQL
        return {item["id"] for item in _rows(conn, f"""
            WITH RECURSIVE tail(id) AS (
                SELECT c.id FROM sessions p JOIN sessions c ON c.parent_session_id = p.id WHERE p.pinned AND {edge}
                UNION
                SELECT c.id FROM tail t JOIN sessions p ON p.id = t.id JOIN sessions c ON c.parent_session_id = p.id
                WHERE {edge})
            SELECT id FROM tail""")}

    @staticmethod
    def _kept_ancestors(conn: Connection, chosen: Sequence[str]) -> set[str]:
        """Every compression ancestor of a row outside ``chosen``."""
        edge = _lineage.CONTINUATION_EDGE_SQL
        return {item["id"] for item in _rows(conn, f"""
            WITH RECURSIVE up(id) AS (
                SELECT p.id FROM sessions c JOIN sessions p ON p.id = c.parent_session_id
                WHERE NOT c.id = ANY(%s) AND {edge}
                UNION
                SELECT p.id FROM up JOIN sessions c ON c.id = up.id JOIN sessions p ON p.id = c.parent_session_id
                WHERE {edge})
            SELECT id FROM up""", (list(chosen),))}

    @staticmethod
    def _resolved(older_than_days: Optional[float], criteria: PruneCriteria) -> PruneCriteria:
        if older_than_days is None or criteria.last_active_before is not None or criteria.started_before is not None:
            return criteria
        if older_than_days < 0:
            raise ValueError(f"older_than_days must be >= 0, got {older_than_days!r}: a negative retention builds a "
                             "future cutoff that matches every ended session.")
        return dataclasses.replace(criteria, last_active_before=time.time() - older_than_days * _DAY)

    @staticmethod
    def _times_match(record: SessionRecord, last_active: float, c: PruneCriteria) -> bool:
        if c.last_active_before is not None and not (
            last_active < c.last_active_before
            and (record.end_reason != ORPHAN_REAP_END_REASON
                 or (record.ended_at is not None and record.ended_at < c.last_active_before))):
            return False
        return not ((c.last_active_after is not None and last_active < c.last_active_after)
                    or (c.started_before is not None and record.started_at >= c.started_before)
                    or (c.started_after is not None and record.started_at < c.started_after))

    @staticmethod
    def _text_match(record: SessionRecord, c: PruneCriteria, source: Optional[str]) -> bool:
        return not (
            (source and record.source != source)
            or (c.title_like and not _like(record.title, c.title_like))
            or (c.end_reason and record.end_reason != c.end_reason)
            or (c.cwd_prefix and not _under(record.cwd, c.cwd_prefix))
            or (c.model_like and not _like(record.model, c.model_like))
            or (c.provider and _ascii_lower(record.billing_provider or "") != c.provider.lower())
            or (c.user_id and record.user_id != c.user_id)
            or (c.chat_id and record.chat_id != c.chat_id)
            or (c.chat_type and record.chat_type != c.chat_type)
            or (c.branch_like and not _like(record.git_branch, c.branch_like)))

    @staticmethod
    def _number_match(record: SessionRecord, c: PruneCriteria) -> bool:
        return (_bounded(record.message_count, c.min_messages, c.max_messages)
                and _bounded(record.input_tokens + record.output_tokens, c.min_tokens, c.max_tokens)
                and _bounded(_cost(record), c.min_cost, c.max_cost)
                and _bounded(record.tool_call_count, c.min_tool_calls, c.max_tool_calls))

    def _selected(self, conn: Connection, c: PruneCriteria, source: Optional[str], *,
                  open_rows: bool = False) -> list[tuple[float, SessionRecord]]:
        """The prune filter over every row, ordered by recency then ``started_at``; ``open_rows`` inverts the ended
        guard."""
        tail = set() if c.include_pinned else self._pinned_tail(conn)
        where = "true" if c.include_open else ("s.ended_at IS NULL" if open_rows else "s.ended_at IS NOT NULL")
        out = []
        for last_active, record, _row in self._ranked(conn, where):
            if ((c.lineage_tips_only and (record.end_reason == END_REASON_COMPRESSION or record.title == CANONICAL_BOT_CHAT_TITLE))
                    or (isinstance(c.archived, bool) and record.archived != c.archived)
                    or (not c.include_pinned and (record.pinned or record.session_id in tail))
                    or not self._times_match(record, last_active, c) or not self._text_match(record, c, source)
                    or not self._number_match(record, c)):
                continue
            out.append((last_active, record))
        out.sort(key=lambda pair: (pair[0], pair[1].started_at))
        return out

    def _whole_lineages(self, conn: Connection, selected: list[tuple[float, SessionRecord]]) -> list[tuple[float, SessionRecord]]:
        """``selected`` minus every compression ancestor of a row it does not hold."""
        kept = self._kept_ancestors(conn, [str(record.session_id) for _, record in selected])
        return [(last_active, record) for last_active, record in selected if record.session_id not in kept]

    def _prune_selection(self, conn: Connection, older_than_days: Optional[float], source: Optional[str],
                         criteria: PruneCriteria, whole_lineages: bool) -> list[tuple[float, SessionRecord]]:
        selected = self._selected(conn, self._resolved(older_than_days, criteria), source)
        return self._whole_lineages(conn, selected) if whole_lineages else selected

    # ── prune ──

    def list_prune_candidates(self, older_than_days: Optional[float] = None, source: Optional[str] = None, *,
                              criteria: PruneCriteria = PruneCriteria(), whole_lineages: bool = False) -> tuple[PruneCandidate, ...]:
        with self._store.transaction() as conn:
            selected = self._prune_selection(conn, older_than_days, source, criteria, whole_lineages)
        return tuple(PruneCandidate(
            session_id=record.session_id, source=record.source, title=record.title, model=record.model,
            started_at=record.started_at, last_active=last_active, ended_at=record.ended_at,
            message_count=record.message_count, archived=int(record.archived)) for last_active, record in selected)

    def count_prune_matches(self, older_than_days: Optional[float] = None, source: Optional[str] = None, *,
                            criteria: PruneCriteria = PruneCriteria(), pinned_only: bool = False) -> int:
        if pinned_only:
            criteria = dataclasses.replace(criteria, include_pinned=True)
        with self._store.transaction() as conn:
            selected = self._selected(conn, self._resolved(older_than_days, criteria), source)
        return sum(1 for _, record in selected if record.pinned or not pinned_only)

    def count_open_prune_matches(self, older_than_days: Optional[float] = None, source: Optional[str] = None, *,
                                 criteria: PruneCriteria = PruneCriteria()) -> int:
        if criteria.include_open:
            return 0
        with self._store.transaction() as conn:
            return len(self._selected(conn, self._resolved(older_than_days, criteria), source, open_rows=True))

    def prune_reclaim_estimate(self, older_than_days: Optional[float], source: Optional[str] = None, *,
                               criteria: PruneCriteria = PruneCriteria(), whole_lineages: bool = True,
                               measure_total_bytes: bool = True) -> PruneReclaimEstimate:
        with self._store.transaction() as conn:
            prunable = [str(record.session_id) for _, record in
                        self._prune_selection(conn, older_than_days, source, criteria, whole_lineages)]
            total = int(scalar(conn, "SELECT count(*) FROM sessions"))
            if not prunable:
                return PruneReclaimEstimate(prunable=0, total=total, prunable_bytes=0, total_bytes=None)

            def measured(session_ids: Optional[Sequence[str]]) -> int:
                ids = session_ids if session_ids is not None else [
                    item["session_id"] for item in _rows(conn, "SELECT DISTINCT session_id FROM messages")]
                rows = transcript_store.load(conn, list(ids))
                return sum(_stored_bytes(row.as_transcript_row()) for row in rows)

            return PruneReclaimEstimate(prunable=len(prunable), total=total, prunable_bytes=measured(prunable),
                                        total_bytes=measured(None) if measure_total_bytes else None)

    def prune_sessions(self, older_than_days: Optional[float] = 90, source: Optional[str] = None, *,
                       criteria: PruneCriteria = PruneCriteria(), exclude_active_write_guards: bool = False,
                       media_root: str) -> PruneResult:
        if criteria.include_open:
            raise ValueError("prune_sessions() never deletes open sessions: include_open is archive-only")
        with self._store.transaction() as conn:
            selected = self._whole_lineages(conn, self._selected(conn, self._resolved(older_than_days, criteria), source))
            ids = sorted(str(record.session_id) for _, record in selected)
            _rows(conn, "SELECT id FROM sessions WHERE id = ANY(%s) ORDER BY id FOR UPDATE", (ids,))
            doomed = [sid for sid in ids
                      if not (exclude_active_write_guards and _leases.lease_guarded(conn, sid))
                      and not _retirement.live_ledger_states(conn, sid)
                      and scalar(conn, "SELECT 1 FROM sessions WHERE id = %s", (sid,)) is not None]
            _retirement.retire(conn, doomed, media_root=media_root)
        return PruneResult(count=len(doomed), deleted_ids=tuple(SessionId(sid) for sid in doomed))

    def prune_empty_ghost_sessions(self, *, media_root: str) -> tuple[SessionId, ...]:
        cutoff = time.time() - GHOST_SESSION_MIN_AGE.total_seconds()
        with self._store.transaction() as conn:
            candidates = [item["id"] for item in _rows(conn, """
                SELECT s.id FROM sessions s WHERE s.source = %s AND s.title IS NULL AND s.ended_at IS NOT NULL
                  AND s.started_at < %s AND NOT EXISTS (SELECT 1 FROM messages m WHERE m.session_id = s.id)
                ORDER BY s.id FOR UPDATE""", (GHOST_SESSION_SOURCE, cutoff))]
            ghosts = [sid for sid in candidates if not _retirement.live_ledger_states(conn, sid)]
            _retirement.retire(conn, ghosts, media_root=media_root)
        return tuple(SessionId(sid) for sid in ghosts)

    # ── orphan sweep ──

    @staticmethod
    def _guarded(conn: Connection, session_id: str) -> bool:
        """A live turn lease or compression lock protects the row; expired guards are reclaimed."""
        for _kind, table, column, key, live in _leases.lease_rows(conn, session_id):
            if live:
                return True
            execute(conn, f"DELETE FROM {table} WHERE {column} = %s", (key,))
        return False

    def sweep_orphaned_sessions(self, *, max_idle_seconds: float, sources: tuple[str, ...] = ('tui', 'desktop', 'subagent'),
                                exclude_ids: tuple[str, ...] = (), exclude_pinned: bool = False,
                                heartbeat_staleness_seconds: Optional[float] = None,
                                heartbeat_ownership_grace_seconds: Optional[float] = None,
                                respect_gateway_heartbeats: bool = True) -> tuple[SessionId, ...]:
        wanted = sorted({source for source in sources if source})
        if max_idle_seconds <= 0 or not wanted:
            return ()
        staleness = (heartbeat_staleness_seconds if heartbeat_staleness_seconds and heartbeat_staleness_seconds > 0
                     else max_idle_seconds * 2)
        grace = (heartbeat_ownership_grace_seconds if heartbeat_ownership_grace_seconds is not None
                 and heartbeat_ownership_grace_seconds >= 0 else staleness)
        now = time.time()
        cutoff = now - max_idle_seconds
        excluded = {str(sid) for sid in exclude_ids if sid}
        with self._store.transaction() as conn:
            tail = self._pinned_tail(conn) if exclude_pinned else set()
            beats = _rows(conn, "SELECT started_at, last_heartbeat FROM gateway_heartbeats") if respect_gateway_heartbeats else []
            victims = []
            for recency, record, row in self._ranked(conn, "s.ended_at IS NULL AND s.source = ANY(%s) ORDER BY s.started_at, s.id",
                                                     (wanted,)):
                sid = row["id"]
                if ((exclude_pinned and (row["pinned"] or sid in tail)) or not row["started_at"] < cutoff
                        or not recency < cutoff or sid in excluded
                        or any(hb["last_heartbeat"] >= now - staleness and hb["started_at"] <= row["started_at"] + grace
                               for hb in beats)
                        or self._guarded(conn, sid)):
                    continue
                victims.append(sid)
            execute(conn, "UPDATE sessions SET ended_at = %s, end_reason = %s WHERE id = ANY(%s) AND ended_at IS NULL",
                    (time.time(), ORPHAN_REAP_END_REASON, victims))
        return tuple(SessionId(sid) for sid in victims)

    # ── idle archive ──

    def list_idle_archive_candidates(self, idle_days: float, *, exclude_pinned: bool = True) -> tuple[SessionId, ...]:
        if idle_days is None or idle_days < 0:
            return ()
        cutoff = time.time() - float(idle_days) * _DAY
        with self._store.transaction() as conn:
            tail = self._pinned_tail(conn) if exclude_pinned else set()
            found = [(row["started_at"], row["id"]) for recency, _record, row in self._ranked(
                conn, "NOT s.archived AND coalesce(s.end_reason, '') <> %s AND NOT (s.hidden AND coalesce(s.title, '') = %s)",
                (END_REASON_COMPRESSION, CANONICAL_BOT_CHAT_TITLE))
                if not (exclude_pinned and (row["pinned"] or row["id"] in tail)) and recency < cutoff]
        return tuple(SessionId(sid) for _started, sid in sorted(found, key=lambda pair: pair[0]))

    def auto_archive_lineage(self, session_id: SessionId) -> bool:
        with self._store.transaction() as conn:
            if scalar(conn, "SELECT 1 FROM sessions WHERE id = %s", (str(session_id),)) is None:
                return False
            ids = _lineage.lineage_ids(conn, str(session_id)) or [str(session_id)]
            execute(conn, "UPDATE sessions SET archived = true, auto_archived = true WHERE id = ANY(%s) AND NOT archived", (ids,))
        return True

    # ── export ──

    def _messages(self, session_id: SessionId, include_compacted: bool, include_inactive: bool) -> tuple[Any, ...]:
        view = TranscriptView.ALL if include_inactive else (TranscriptView.DISPLAY if include_compacted else TranscriptView.ACTIVE)
        return tuple(self._store.messages.read_transcript(session_id, TranscriptRead(view=view)))

    def export_session(self, session_id: SessionId, include_compacted: bool = False,
                       include_inactive: bool = False) -> Optional[SessionExport]:
        record = self._store.sessions.get(session_id)
        if record is None:
            return None
        return SessionExport(session=_export_row(record), messages=self._messages(session_id, include_compacted, include_inactive))

    def export_session_lineage(self, session_id: SessionId, include_compacted: bool = False,
                               include_inactive: bool = False) -> Optional[SessionLineageExport]:
        segments = tuple(segment for segment in (self.export_session(sid, include_compacted, include_inactive)
                                                 for sid in self._store.lineage.lineage(session_id)) if segment is not None)
        if not segments:
            return None
        messages = tuple(message for segment in segments for message in segment.messages)
        return SessionLineageExport(segments=segments, lineage_session_ids=tuple(segment.session.id for segment in segments),
                                    message_count=len(messages), messages=messages)

    def export_all(self, source: Optional[str] = None, include_compacted: bool = False,
                   include_inactive: bool = False) -> tuple[SessionExport, ...]:
        with self._store.transaction() as conn:
            ranked = self._ranked(conn, "(%s::text IS NULL OR %s = '' OR s.source = %s)", (source, source, source))
        ranked.sort(key=lambda item: (item[0], item[1].started_at, str(item[1].session_id)), reverse=True)
        return tuple(SessionExport(session=_export_row(record, last_active=last_active),
                                   messages=self._messages(record.session_id, include_compacted, include_inactive))
                     for last_active, record, _row in ranked[:100000])

    def assert_export_safe(self, session_id: SessionId, max_messages: int) -> int:
        if max_messages < 0:
            raise ValueError("max_messages must be non-negative")
        if max_messages == 0:
            return 0
        with self._store.transaction() as conn:
            count = int(scalar(conn, "SELECT count(*) FROM (SELECT 1 FROM messages WHERE session_id = %s LIMIT %s) t",
                               (str(session_id), max_messages + 1)))
        if count > max_messages:
            raise ExportTooLarge(session_id, count, max_messages)
        return count

    # ── import ──

    def _insert_imported(self, conn: Connection, raw: Mapping[str, Any], messages: Sequence[Mapping[str, Any]],
                         session_id: str) -> None:
        """One normalized session and its messages; the counters count live rows."""
        started = policy.coerce_epoch(raw["started_at"]) if raw.get("started_at") is not None else None
        flags = {flag: bool(_coerce_or(raw.get(flag), int, 0)) for flag in _IMPORT_FLAGS}
        flags["auto_archived"] = flags["auto_archived"] and flags["archived"]
        config = _decoded(raw.get("model_config")) if raw.get("model_config") else None
        values: dict[str, Any] = {
            "id": session_id, "source": str(raw.get("source") or "import"),
            "system_prompt_hash": store_text(conn, raw.get("system_prompt")),
            "started_at": time.time() if started is None else started,
            "model_config": json.dumps(config) if isinstance(config, dict) and config else None,
            **{column: raw.get(column) for column in _IMPORT_PASSTHROUGH},
            **{column: _finite_or_none(raw.get(column)) for column in _IMPORT_FLOATS},
            **{column: _int_or_zero(raw.get(column)) for column in _IMPORT_INTS},
            **flags,
        }
        execute(conn, f"INSERT INTO sessions ({', '.join(values)}) VALUES ({', '.join(['%s'] * len(values))})",
                tuple(values.values()))
        stored = transcript_store.insert_rows(conn, session_id, [_imported_message(message) for message in messages])
        archived = []
        for message, row in zip(messages, stored):
            if not _coerce_or(message.get("active"), int, 1):
                row.visibility = (RowVisibility.COMPACTION_ARCHIVED if _coerce_or(message.get("compacted"), int, 0)
                                  else RowVisibility.REWOUND)
                archived.append(row)
        transcript_store.save_visibility(conn, archived)
        transcript_store.recount(conn, session_id)

    @staticmethod
    def _attach_parents(conn: Connection, parent_updates: list[tuple[str, str]]) -> int:
        """Re-attach imported children whose parent exists without closing a cycle; returns the detached count."""
        parent_by_child = dict(parent_updates)

        def closes_cycle(session_id: str, parent_id: str) -> bool:
            seen, current = {session_id}, parent_id
            while current:
                if current in seen:
                    return True
                seen.add(current)
                current = (parent_by_child[current] if current in parent_by_child
                           else scalar(conn, "SELECT parent_session_id FROM sessions WHERE id = %s", (current,)))
            return False

        detached = 0
        for session_id, parent_id in parent_updates:
            if scalar(conn, "SELECT 1 FROM sessions WHERE id = %s", (parent_id,)) is not None and not closes_cycle(session_id, parent_id):
                execute(conn, "UPDATE sessions SET parent_session_id = %s WHERE id = %s", (parent_id, session_id))
            else:
                parent_by_child.pop(session_id, None)
                detached += 1
        return detached

    def import_sessions(self, sessions: Sequence[Mapping[str, Any]]) -> ImportResult:
        if not isinstance(sessions, (list, tuple)):
            raise ValueError("sessions must be a list")
        if len(sessions) > IMPORT_MAX_SESSIONS:
            raise ValueError(f"sessions must contain at most {IMPORT_MAX_SESSIONS} entries")
        normalized, errors = validate_import_payload(sessions)
        if errors:
            return ImportResult(ok=False, imported=0, skipped=0, detached=0, errors=tuple(errors))
        ids = [str(item["session"].get("id") or "").strip() for item in normalized]
        imported: list[SessionId] = []
        skipped: list[SessionId] = []
        parent_updates: list[tuple[str, str]] = []
        with self._store.transaction() as conn:
            retired = {item["session_id"] for item in _rows(conn, "SELECT session_id FROM retired_sessions WHERE session_id = ANY(%s)", (ids,))}
            if deleted := [sid for sid in ids if sid in retired]:
                return ImportResult(ok=False, imported=0, skipped=0, detached=0, errors=tuple(
                    f"session {sid} was deleted here; a deleted session is not restored" for sid in deleted))
            for item, session_id in zip(normalized, ids):
                if scalar(conn, "SELECT 1 FROM sessions WHERE id = %s", (session_id,)) is not None:
                    skipped.append(SessionId(session_id))
                    continue
                self._insert_imported(conn, item["session"], item["messages"], session_id)
                if parent_id := str(item["session"].get("parent_session_id") or "").strip():
                    parent_updates.append((session_id, parent_id))
                imported.append(SessionId(session_id))
            detached = self._attach_parents(conn, parent_updates)
        return ImportResult(ok=True, imported=len(imported), skipped=len(skipped), detached=detached,
                            imported_ids=tuple(imported), skipped_ids=tuple(skipped))

    def _import_in_transaction(self, conn: Connection, session_id: str, payload: Mapping[str, Any],
                               owner: Any) -> tuple[set[str], dict[str, Any]]:
        """The ``Import`` mutation's handler, in the mutation's transaction: a parent outside the payload links only
        when its history binding names the child's owner under ``owner``."""
        items = payload["sessions"]
        if len(items) > IMPORT_MAX_SESSIONS:
            raise AuthorityRefusal("invalid_params")
        normalized, errors = validate_import_payload(items)
        if errors or not normalized or session_id != normalized[0]["session"]["id"]:
            raise AuthorityRefusal("invalid_params")
        imported: list[str] = []
        skipped: list[str] = []
        parents: list[tuple[str, str]] = []
        for item in normalized:
            raw = item["session"]
            sid = raw["id"]
            if scalar(conn, "SELECT 1 FROM retired_sessions WHERE session_id = %s", (sid,)) is not None:
                raise AuthorityRefusal("admission_conflict", field="sessions",
                                        detail=f"session {sid} was deleted here; a deleted session is not restored")
            if scalar(conn, "SELECT 1 FROM sessions WHERE id = %s", (sid,)) is not None:
                skipped.append(sid)
                continue
            self._insert_imported(conn, raw, item["messages"], sid)
            if raw.get("parent_session_id"):
                parents.append((sid, raw["parent_session_id"]))
            imported.append(sid)
        fresh = set(imported)

        def binding(sid: str) -> Optional[str]:
            return binding_text(_meta.get(conn, HISTORY_PREFIX + sid))

        foreign = [(sid, parent) for sid, parent in parents if parent not in fresh and (
            binding(parent) is None or not bindings_agree(binding(parent), binding(sid), owner))]
        detached = self._attach_parents(conn, [link for link in parents if link not in foreign]) + len(foreign)
        return set(imported), {"ok": True, "imported": len(imported), "skipped": len(skipped), "imported_ids": imported,
                               "skipped_ids": skipped, "detached": detached, "errors": []}

    # ── foreign import ──

    @staticmethod
    def _find_foreign(conn: Connection, origin: ForeignOrigin) -> Optional[SessionId]:
        for item in _rows(conn, "SELECT id, origin_json FROM sessions WHERE source = %s AND origin_json IS NOT NULL "
                                "ORDER BY started_at, id", (bound_id(origin.tool),)):
            parsed = _decoded(item["origin_json"])
            imported = (parsed.get("imported_from") or {}) if isinstance(parsed, dict) else {}
            if not isinstance(imported, dict) or imported.get("tool") != origin.tool:
                continue
            if (origin.foreign_session_id and origin.foreign_session_id == imported.get("foreign_session_id")) or (
                    not origin.foreign_session_id and imported.get("path") == origin.path):
                return SessionId(item["id"])
        return None

    def find_foreign_import(self, origin: ForeignOrigin) -> Optional[SessionId]:
        with self._store.transaction() as conn:
            return self._find_foreign(conn, origin)

    def import_foreign_history(self, origin: ForeignOrigin, messages: Sequence[ForeignTurn], *, title: str,
                               cwd: Optional[str], profile: str) -> ForeignImportResult:
        session_id = new_session_id(hex_len=12)
        turns = [{"role": turn.role, "content": turn.content} for turn in messages]
        normalized, errors = validate_import_payload([{"id": session_id, "source": origin.tool, "title": title, "cwd": cwd,
                                                       "messages": turns}])
        if errors:
            raise ValueError(errors[0]["error"])
        with self._store.transaction() as conn:
            # One foreign import at a time: the dedupe check and the insert must not interleave.
            execute(conn, "SELECT pg_advisory_xact_lock(hashtext('hermes.foreign_import'))")
            if (existing := self._find_foreign(conn, origin)) is not None:
                return ForeignImportResult(session_id=existing, already_imported=True)
            if scalar(conn, "SELECT 1 FROM sessions WHERE id = %s", (session_id,)) is not None:
                raise Conflict("session", session_id)
            item = normalized[0]
            if scalar(conn, "SELECT 1 FROM sessions WHERE title = %s", (title,)) is not None:
                item["session"]["title"] = f"{title} ({session_id[-12:]})"
            self._insert_imported(conn, item["session"], item["messages"], session_id)
            execute(conn, "UPDATE sessions SET origin_json = %s, profile_name = %s WHERE id = %s",
                    (json.dumps({"imported_from": origin.as_dict()}), profile, session_id))
        return ForeignImportResult(session_id=SessionId(session_id), already_imported=False)

    # ── backend heartbeats ──

    def register_backend_heartbeat(self, *, backend_id: str, pid: int, started_at: float,
                                   last_heartbeat: Optional[float] = None, profile: str = '', host: str = '') -> None:
        if not backend_id:
            return
        stamp = time.time() if last_heartbeat is None else float(last_heartbeat)
        with self._store.transaction() as conn:
            execute(conn, "INSERT INTO gateway_heartbeats (backend_id, pid, started_at, last_heartbeat, profile, host) "
                          "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (backend_id) DO UPDATE SET pid = excluded.pid, "
                          "started_at = excluded.started_at, last_heartbeat = excluded.last_heartbeat, "
                          "profile = excluded.profile, host = excluded.host",
                    (str(backend_id), int(pid), float(started_at), stamp, str(profile), str(host)))

    def clear_backend_heartbeat(self, backend_id: str) -> bool:
        if not backend_id:
            return False
        with self._store.transaction() as conn:
            return bool(execute(conn, "DELETE FROM gateway_heartbeats WHERE backend_id = %s", (backend_id,)))

    # ── read-only probes ──

    def session_projection_digest(self, *, busy_timeout_s: float = 0.05) -> Optional[tuple[tuple[str, ...], bytes]]:
        with self._store.transaction() as conn:
            records = [record for _recency, record, _row in self._ranked(conn, "true ORDER BY s.id COLLATE \"C\"")]
        digest = hashlib.blake2b(digest_size=16)
        for record in records:
            values = _export_row(record).as_dict()
            digest.update(repr(tuple(values[field] for field in SESSION_PROJECTION_FIELDS)).encode("utf-8", "backslashreplace"))
            digest.update(b"\0")
        return SESSION_PROJECTION_FIELDS, digest.digest()

    @staticmethod
    def _inside(conn: Connection, worktree_path: str) -> list[dict[str, Any]]:
        inside = _inside_tree(worktree_path)
        return [item for item in _rows(conn, "SELECT id, cwd, ended_at, chat_id, message_count FROM sessions "
                                             "WHERE cwd IS NOT NULL AND cwd <> ''") if inside(item["cwd"])]

    def retained_tree_is_live(self, worktree_path: str, *, pending_only: bool = False) -> bool:
        with self._store.transaction() as conn:
            pending = {item["target_session_id"] for item in _rows(
                conn, "SELECT DISTINCT target_session_id FROM session_admissions WHERE status <> 'terminal'")}
            return any(item["id"] in pending or (not pending_only and item["message_count"] > 0)
                       for item in self._inside(conn, worktree_path))

    def tree_served_sessions(self, worktree_path: str) -> TreeServeFacts:
        with self._store.transaction() as conn:
            found = self._inside(conn, worktree_path)
            for item in found:
                sid = item["id"]
                states = [scalar(conn, "SELECT value FROM state_meta WHERE key = %s", (f"{kind}:{sid}",))
                          for kind in ("loop", "heartbeat", "goal")]
                if item["ended_at"] is None or any(_active_state(value) for value in states):
                    return TreeServeFacts(served=True)
        return TreeServeFacts(served=False, chat_ids=frozenset(item["chat_id"] for item in found if item["chat_id"] is not None))


register_repository("maintenance", PostgresMaintenance)
