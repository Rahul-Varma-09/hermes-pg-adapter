"""``SessionRepository`` on Postgres: session rows, their lifecycle, titles, model config and lineage flags.

Every verb runs in one ``store.transaction()``; a read-then-write verb locks the rows it decides on (``FOR UPDATE``).
The prompt and the ``tools[]`` pin are stored once in ``system_prompts`` and named by sha256: a full read (``get``)
resolves both, a listing resolves only the prompt and carries the pin's hash. Fork markers live in ``model_config``
and the lineage kind is read from them, never stored apart.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Mapping, Optional

from psycopg import Connection
from psycopg.types.json import Jsonb

from persistence import JsonValue, RetiredSessionError, SessionId
from persistence.conversation.sessions import (
    ACTIVITY_PROVENANCE_UNKNOWN, CANONICAL_BOT_CHAT_TITLE, END_REASON_COMPRESSION, END_REASON_SESSION_RESET,
    ONBOARDING_RUN_KEY, RECOVERABLE_END_REASONS, RUNTIME_LOCK_KEY, YOLO_KEY, ArchiveChange, CanonicalTitleLocked,
    DeclaredScopeIdentity, NewSession, OnboardingAdvance, ReplaceSessionConfig, RuntimeLock, SessionExists,
    SessionLineageKind, SessionRecord, TitleConflict, TitleSource, bound_activity_description, is_corrupt_model_config,
    is_explicit_close_reason, normalize_activity_provenance, parse_model_config, sanitize_title,
)
from persistence.core.conventions import UNTRUSTED_INSTANT, stored_instant

from .._backend import PostgresStore, register_repository
from .._sql import execute, has_marker, row as _row, rows as _rows, scalar, store_text
from . import _lineage
from .sessions_delete import SessionDeleteMixin
from .sessions_listing import SessionListingMixin

__all__ = ["PostgresSessionRepository", "session_record"]

_MARKERS = (
    ("_delegate_from", SessionLineageKind.DELEGATE),
    ("_branched_from", SessionLineageKind.BRANCH),
    ("_reset_from", SessionLineageKind.RESET),
)
_MARKER_OF_KIND = {kind: key for key, kind in _MARKERS}
_TITLE_SOURCES = {source.value: source for source in TitleSource}
# A title without provenance ranks as the user's.
_TITLE_RANK = {None: 2, TitleSource.DERIVED: 0, TitleSource.LLM: 1, TitleSource.USER: 2}
_NUMBERED_TITLE_RE = re.compile(r"^(.*?) #(\d+)$")

# NewSession fields copied into a fresh row and filled on upsert only while unset.
_SPEC_FIELDS = ("model", "cwd", "parent_session_id", "session_key", "chat_id", "chat_type", "thread_id",
                "git_repo_root", "display_name", "transport_profile")
_ROUTING_FIELDS = ("user_id", "session_key", "chat_id", "chat_type", "thread_id", "display_name", "origin_json",
                   "transport_profile")
# created_source values that are no creation provenance.
_UNSET_CREATED = (None, "", "unknown", "local")
_QUEUED_PROMPT_KEY = "_queued_prompt"

# The full row: the prompt resolved, the pin resolved too when ``{pin}`` joins it.
_SELECT = """
    SELECT s.*, sp.prompt AS system_prompt, {pin} AS tool_pin_text
    FROM sessions s
    LEFT JOIN system_prompts sp ON sp.hash = s.system_prompt_hash
    {pin_join}"""
_SELECT_RESOLVED = _SELECT.format(pin="tp.prompt", pin_join="LEFT JOIN system_prompts tp ON tp.hash = s.tool_names")
_SELECT_LISTED = _SELECT.format(pin="NULL::text", pin_join="")


def _stripped(value: Any) -> Optional[str]:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _json_value(text: Any) -> Any:
    if not isinstance(text, str) or not text:
        return None
    try:
        return json.loads(text)
    except ValueError:
        return None


def session_record(row: Mapping[str, Any], *, resolve_tool_pin: bool = True) -> SessionRecord:
    """The record of one selected session row (``_SELECT_RESOLVED`` or ``_SELECT_LISTED``)."""
    stored_config = row.get("model_config")
    config = dict(parse_model_config(stored_config if stored_config is not None else None) or {})
    kind, origin_id = SessionLineageKind.NONE, None
    for key, marker_kind in _MARKERS:
        if _stripped(config.get(key)):
            kind, origin_id = marker_kind, config[key]
            break
    origin = _json_value(row.get("origin_json"))
    started = stored_instant(row.get("started_at"), session_id=row["id"], field="started_at")
    ended = stored_instant(row.get("ended_at"), session_id=row["id"], field="ended_at")
    pin_text = row.get("tool_pin_text")
    return SessionRecord(
        session_id=SessionId(row["id"]),
        source=row.get("source") or "",
        created_source=row.get("created_source"),
        user_id=row.get("user_id"),
        session_key=row.get("session_key"),
        chat_id=row.get("chat_id"),
        chat_type=row.get("chat_type"),
        thread_id=row.get("thread_id"),
        display_name=row.get("display_name"),
        origin=dict(origin) if isinstance(origin, Mapping) else None,
        expiry_finalized=bool(row.get("expiry_finalized")),
        model=row.get("model"),
        model_config=config,
        system_prompt=row.get("system_prompt"),
        system_prompt_hash=row.get("system_prompt_hash"),
        parent_session_id=row.get("parent_session_id"),
        lineage_kind=kind,
        reset_from=_stripped(config.get("_reset_from")),
        branched_from=_stripped(config.get("_branched_from")),
        started_at=(started or UNTRUSTED_INSTANT).timestamp(),
        ended_at=None if ended is None else ended.timestamp(),
        end_reason=row.get("end_reason"),
        message_count=row.get("message_count", 0),
        tool_call_count=row.get("tool_call_count", 0),
        input_tokens=row.get("input_tokens", 0),
        output_tokens=row.get("output_tokens", 0),
        cache_read_tokens=row.get("cache_read_tokens", 0),
        cache_write_tokens=row.get("cache_write_tokens", 0),
        reasoning_tokens=row.get("reasoning_tokens", 0),
        api_call_count=row.get("api_call_count", 0),
        estimated_cost_usd=row.get("estimated_cost_usd"),
        actual_cost_usd=row.get("actual_cost_usd"),
        billing_provider=row.get("billing_provider"),
        billing_base_url=row.get("billing_base_url"),
        billing_mode=row.get("billing_mode"),
        cost_status=row.get("cost_status"),
        cost_source=row.get("cost_source"),
        pricing_version=row.get("pricing_version"),
        cwd=row.get("cwd"),
        git_branch=row.get("git_branch"),
        git_repo_root=row.get("git_repo_root"),
        git_metadata_generation=int(row.get("git_metadata_generation") or 0),
        title=row.get("title"),
        title_source=_TITLE_SOURCES.get(row.get("title_source")),
        last_activity_at=row.get("last_activity_at"),
        last_activity_description=row.get("last_activity_description"),
        last_activity_provenance=row.get("last_activity_provenance"),
        handoff_state=row.get("handoff_state"),
        handoff_platform=row.get("handoff_platform"),
        handoff_error=row.get("handoff_error"),
        compression_failure_cooldown_until=row.get("compression_failure_cooldown_until"),
        compression_failure_error=row.get("compression_failure_error"),
        compression_fallback_streak=int(row.get("compression_fallback_streak") or 0),
        compression_ineffective_count=int(row.get("compression_ineffective_count") or 0),
        compression_recovery_deadline=row.get("compression_recovery_deadline"),
        compression_overload_streak=int(row.get("compression_overload_streak") or 0),
        profile_name=row.get("profile_name"),
        transport_profile=row.get("transport_profile"),
        rewind_count=int(row.get("rewind_count") or 0),
        archived=bool(row.get("archived")),
        auto_archived=bool(row.get("auto_archived")),
        pinned=bool(row.get("pinned")),
        hidden=bool(row.get("hidden")),
        last_read_at=row.get("last_read_at"),
        tool_pin=_json_value(pin_text) if resolve_tool_pin else None,
        transcript_revision=int(row.get("transcript_revision") or 0),
        runtime_revision=int(row.get("runtime_revision") or 0),
        runtime_generation=int(row.get("runtime_generation") or 0),
        lineage_origin_id=origin_id,
        model_config_corrupt=is_corrupt_model_config(stored_config),
        model_config_json=None if stored_config is None else json.dumps(stored_config),
        origin_json=row.get("origin_json"),
        tool_names=pin_text if resolve_tool_pin and row.get("tool_names") else row.get("tool_names"),
        prompt_cache_clock=row.get("prompt_cache_clock"),
    )


def _key_namespace(key: str) -> str:
    """The ``agent:<ns>:`` head of a routing key (the first six characters when no namespace follows)."""
    return key[: key[6:].find(":") + 7]


def _same_key_namespace(parent_key: Optional[str], child_key: Optional[str]) -> bool:
    return parent_key is None or child_key is None or _key_namespace(parent_key) == _key_namespace(child_key)


def _merged(config: Mapping[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    """``patch`` merged into a copy of ``config``; ``None`` removes a key."""
    merged = dict(config)
    for key, value in patch.items():
        if value is None:
            merged.pop(key, None)
        else:
            merged[key] = value
    return merged


def bump_conversation_generation(conn: Connection, session_id: str, end_reason: str) -> None:
    """Advance the peer's conversation generation past a reset boundary, in the transaction that writes it."""
    if end_reason not in _lineage.RESET_END_REASONS:
        return
    peer = _row(conn, "SELECT source, session_key FROM sessions WHERE id = %s", (session_id,))
    if peer is None:
        return
    source, session_key = (str(peer[key] or "").strip() for key in ("source", "session_key"))
    if source and session_key:
        execute(conn, "INSERT INTO conversation_generations (source, session_key, generation) VALUES (%s, %s, 1) "
                      "ON CONFLICT (source, session_key) DO UPDATE "
                      "SET generation = conversation_generations.generation + 1", (source, session_key))


def is_retired(conn: Connection, session_id: str) -> bool:
    return scalar(conn, "SELECT 1 FROM retired_sessions WHERE session_id = %s", (session_id,)) is not None


class PostgresSessionRepository(SessionDeleteMixin, SessionListingMixin):
    """Session rows of the bound profile's schema."""

    def __init__(self, store: PostgresStore) -> None:
        self._store = store

    # ── plumbing ──

    def _locked(self, conn: Connection, session_id: str) -> Optional[dict[str, Any]]:
        return _row(conn, "SELECT * FROM sessions WHERE id = %s FOR UPDATE", (session_id,))

    def _update(self, conn: Connection, session_id: str, values: Mapping[str, Any]) -> int:
        if not values:
            return 0
        assignments = ", ".join(f"{column} = %s" for column in values)
        return execute(conn, f"UPDATE sessions SET {assignments} WHERE id = %s",
                       (*(_cell(value) for value in values.values()), session_id))

    def _set_config(self, conn: Connection, session_id: str, config: Mapping[str, Any]) -> None:
        self._update(conn, session_id, {"model_config": Jsonb(dict(config))})

    def _record(self, conn: Connection, session_id: str, *, resolve_tool_pin: bool = True) -> Optional[SessionRecord]:
        query = _SELECT_RESOLVED if resolve_tool_pin else _SELECT_LISTED
        found = _row(conn, query + " WHERE s.id = %s", (session_id,))
        return None if found is None else session_record(found, resolve_tool_pin=resolve_tool_pin)

    def _has_messages(self, conn: Connection, session_id: str) -> bool:
        return scalar(conn, "SELECT 1 FROM messages WHERE session_id = %s LIMIT 1", (session_id,)) is not None

    def _is_empty_ended(self, conn: Connection, row: Mapping[str, Any]) -> bool:
        """The empty-sweep predicate: ended, unarchived, no stored rows."""
        return row["ended_at"] is not None and not row["archived"] and not self._has_messages(conn, row["id"])

    # ── lifecycle ──

    def create(self, spec: NewSession) -> SessionId:
        config = dict(spec.model_config or {})
        if marker := _MARKER_OF_KIND.get(spec.lineage_kind):
            config[marker] = spec.parent_session_id
        origin_json = json.dumps(dict(spec.origin)) if spec.origin else None
        owner = self._store.owner_profile
        with self._store.transaction() as conn:
            existing = self._locked(conn, spec.session_id)
            if existing is not None:
                self._upsert(conn, existing, spec, config, origin_json, owner)
            else:
                if is_retired(conn, spec.session_id):
                    raise RetiredSessionError(spec.session_id, "not_found")
                values: dict[str, Any] = {field: getattr(spec, field) for field in _SPEC_FIELDS}
                values.update(
                    id=spec.session_id, source=spec.source, created_source=spec.source, user_id=spec.user_id,
                    origin_json=origin_json, started_at=time.time(),
                    model_config=Jsonb(config) if config else None,
                    system_prompt_hash=store_text(conn, spec.system_prompt),
                    profile_name=spec.profile_name if (spec.profile_name or "").strip() else owner,
                )
                columns = ", ".join(values)
                execute(conn, f"INSERT INTO sessions ({columns}) VALUES ({', '.join(['%s'] * len(values))})",
                        tuple(_cell(value) for value in values.values()))
            if spec.parent_session_id:
                self._inherit_from_parent(conn, spec.session_id)
            if existing is None:
                _lineage.stamp_ancestors(conn, spec.session_id)
        return SessionId(spec.session_id)

    def _upsert(self, conn: Connection, row: Mapping[str, Any], spec: NewSession, config: dict[str, Any],
                origin_json: Optional[str], owner: Optional[str]) -> None:
        """Fill what an earlier writer left unset: a placeholder ``unknown`` source takes the spec's, an unset
        creating surface a real one, and a config holding only a ``_reset_from`` marker the spec's config."""
        values: dict[str, Any] = {field: getattr(spec, field) for field in _SPEC_FIELDS
                                  if row[field] is None and getattr(spec, field) is not None}
        if row["origin_json"] is None and origin_json is not None:
            values["origin_json"] = origin_json
        if row["system_prompt_hash"] is None and spec.system_prompt is not None:
            values["system_prompt_hash"] = store_text(conn, spec.system_prompt)
        if row["source"] == "unknown":
            values["source"] = spec.source
        if row["created_source"] in _UNSET_CREATED and spec.source != "local":
            values["created_source"] = spec.source
        if row["profile_name"] is None:
            values["profile_name"] = spec.profile_name if (spec.profile_name or "").strip() else owner
        current = row["model_config"]
        if config and isinstance(current, dict) and set(current) == {"_reset_from"}:
            values["model_config"] = Jsonb({**config, "_reset_from": current["_reset_from"]})
        elif current is None and config:
            values["model_config"] = Jsonb(config)
        self._update(conn, row["id"], values)

    def _inherit_from_parent(self, conn: Connection, session_id: str) -> None:
        """Fill the unset columns a child inherits: cwd and git always, the owner within the same key namespace, and
        the routing and peer columns when the parent ended by compression and the child is no delegate or branch
        fork of it."""
        row = _row(conn, "SELECT * FROM sessions WHERE id = %s", (session_id,))
        parent = _row(conn, "SELECT * FROM sessions WHERE id = %s", (row["parent_session_id"],))
        if parent is None:
            return
        values = {field: parent[field] for field in ("cwd", "git_repo_root", "git_branch")
                  if row[field] is None and parent[field] is not None}
        if row["profile_name"] is None and _same_key_namespace(parent["session_key"], row["session_key"]):
            values["profile_name"] = parent["profile_name"]
        if parent["end_reason"] == END_REASON_COMPRESSION:
            config = _lineage.config_of(row)
            if parent["id"] not in (config.get("_delegate_from"), config.get("_branched_from")):
                values.update({field: parent[field] for field in _ROUTING_FIELDS
                               if row[field] is None and parent[field] is not None})
        self._update(conn, session_id, values)

    def create_exclusive(self, session_id: SessionId, *, source: str, model: Optional[str] = None,
                         model_config: Optional[Mapping[str, JsonValue]] = None, system_prompt: Optional[str] = None,
                         title: Optional[str] = None) -> SessionRecord:
        with self._store.transaction() as conn:
            if self._locked(conn, session_id) is not None:
                raise SessionExists(self._record(conn, session_id))
            if is_retired(conn, session_id):
                raise RetiredSessionError(session_id, "session_deleted")
            clean = sanitize_title(title)
            if clean:
                self._resolve_title_conflict(conn, session_id, clean)
            execute(conn, "INSERT INTO sessions (id, source, model, model_config, system_prompt_hash, title, started_at) "
                          "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (session_id, source, model, Jsonb(dict(model_config)) if model_config else None,
                     store_text(conn, system_prompt), clean, time.time()))
            return self._record(conn, session_id)

    def end(self, session_id: SessionId, reason: str) -> None:
        with self._store.transaction() as conn:
            if execute(conn, "UPDATE sessions SET ended_at = %s, end_reason = %s WHERE id = %s AND ended_at IS NULL",
                       (time.time(), reason, session_id)):
                bump_conversation_generation(conn, session_id, reason)

    def reopen(self, session_id: SessionId) -> None:
        with self._store.transaction() as conn:
            self._retire_undrained(conn, session_id)
            row = self._locked(conn, session_id)
            if row is None:
                return
            if row["end_reason"] in _lineage.RESET_END_REASONS and row["session_key"]:
                # Freeze markerless legacy reset children before the parent stops reading as reset-ended.
                markers = " AND ".join(f"NOT {has_marker('c', marker)}" for marker in _lineage.FORK_MARKERS)
                execute(conn, f"""
                    UPDATE sessions c SET model_config = coalesce(c.model_config, '{{}}'::jsonb)
                        || jsonb_build_object('_reset_from', %s::text)
                    WHERE c.parent_session_id = %s AND c.session_key = %s AND c.started_at >= %s
                      AND coalesce(c.source, '') <> 'tool' AND {markers}""",
                        (row["id"], row["id"], row["session_key"], row["started_at"]))
            self._update(conn, session_id, {"ended_at": None, "end_reason": None})
            self._unarchive_auto_archived_lineage(conn, session_id)

    def _retire_undrained(self, conn: Connection, session_id: str) -> int:
        return execute(conn, "UPDATE messages SET active = false WHERE session_id = %s AND role = 'user' AND active "
                             "AND display_metadata ->> %s = '1'", (session_id, _QUEUED_PROMPT_KEY))

    def retire_undrained_queue_rows(self, session_id: SessionId) -> int:
        with self._store.transaction() as conn:
            return self._retire_undrained(conn, session_id)

    def promote_to_reset(self, session_id: SessionId, reason: str = END_REASON_SESSION_RESET) -> bool:
        if not session_id:
            return False
        try:
            with self._store.transaction() as conn:
                row = self._locked(conn, session_id)
                if row is None or (row["ended_at"] is not None and row["end_reason"] not in RECOVERABLE_END_REASONS):
                    return False
                self._update(conn, session_id, {"ended_at": time.time(), "end_reason": reason})
                bump_conversation_generation(conn, session_id, reason)
        except Exception:
            return False
        return True

    def _has_continuation_child(self, conn: Connection, session_id: str) -> bool:
        """Whether a child continues the row: no fork marker naming it, not a tool row."""
        named = " AND ".join(f"coalesce(c.model_config ->> '{marker}', '') <> %s" for marker in _lineage.FORK_MARKERS)
        return scalar(conn, f"SELECT 1 FROM sessions c WHERE c.parent_session_id = %s AND coalesce(c.source, '') <> 'tool' "
                            f"AND {named} LIMIT 1", (session_id, *([session_id] * len(_lineage.FORK_MARKERS)))) is not None

    def reopen_if_explicitly_closed(self, session_id: SessionId, *, provenance: str,
                                    patience_s: Optional[float] = None) -> Optional[str]:
        if not session_id:
            return None
        with self._store.transaction() as conn:
            row = self._locked(conn, session_id)
            if row is None or row["ended_at"] is None:
                return None
            reason = row["end_reason"]
            if not is_explicit_close_reason(reason) or self._has_continuation_child(conn, session_id):
                return None
            self._update(conn, session_id, {"ended_at": None, "end_reason": None})
            return str(reason)

    def touch_activity(self, session_id: SessionId, at: Optional[float] = None, *, description: Optional[str] = None,
                       provenance: Optional[str] = None) -> None:
        if not session_id:
            return
        stamp = float(at if at is not None else time.time())
        with self._store.transaction() as conn:
            # Monotonic: only a strictly newer stamp lands, and the labels ride only with it.
            execute(conn, "UPDATE sessions SET last_activity_at = %s, last_activity_description = %s, "
                          "last_activity_provenance = %s WHERE id = %s "
                          "AND (last_activity_at IS NULL OR last_activity_at < %s)",
                    (stamp, bound_activity_description(description), normalize_activity_provenance(provenance),
                     session_id, stamp))

    def clear_activity_labels(self, session_id: SessionId) -> None:
        if not session_id:
            return
        with self._store.transaction() as conn:
            # The stamp stays: idle and watchdog clocks are continuous across the clear.
            execute(conn, "UPDATE sessions SET last_activity_description = '', last_activity_provenance = %s "
                          "WHERE id = %s AND (coalesce(last_activity_description, '') <> '' "
                          "OR coalesce(last_activity_provenance, %s) <> %s)",
                    (ACTIVITY_PROVENANCE_UNKNOWN, session_id, ACTIVITY_PROVENANCE_UNKNOWN, ACTIVITY_PROVENANCE_UNKNOWN))

    # ── reads ──

    def get(self, session_id: SessionId) -> Optional[SessionRecord]:
        with self._store.transaction() as conn:
            return self._record(conn, session_id)

    def records_created_surface(self) -> bool:
        return True

    def last_active(self, session_id: SessionId) -> Optional[float]:
        with self._store.transaction() as conn:
            row = _lineage.lineage_row(conn, session_id)
            return None if row is None else _lineage.last_active(conn, row)

    def owns_session(self, session_id: SessionId) -> bool:
        with self._store.transaction() as conn:
            found = _row(conn, "SELECT profile_name FROM sessions WHERE id = %s", (session_id,))
        return found is not None and found["profile_name"] in (None, self._store.owner_profile)

    def store_owner(self) -> Optional[str]:
        return self._store.owner_profile

    def resolve_id(self, id_or_prefix: str) -> Optional[SessionId]:
        with self._store.transaction() as conn:
            if scalar(conn, "SELECT 1 FROM sessions WHERE id = %s", (id_or_prefix,)) is not None:
                return SessionId(id_or_prefix)
            # The prefix match folds ASCII case only, as the SQLite store's LIKE does.
            matches = _rows(conn, "SELECT id FROM sessions WHERE starts_with(translate(id, %s, %s), translate(%s, %s, %s)) "
                                  "LIMIT 2", (_UPPER, _LOWER, id_or_prefix, _UPPER, _LOWER))
        return SessionId(matches[0]["id"]) if len(matches) == 1 else None

    def is_kanban_owned(self, session_id: SessionId) -> bool:
        with self._store.transaction() as conn:
            ids = _lineage.resume_lineage_ids(conn, str(session_id))
            return scalar(conn, "SELECT 1 FROM sessions WHERE id = ANY(%s) AND 'kanban' IN (source, created_source) "
                                "LIMIT 1", (ids,)) is not None

    def declared_scope_identity(self, session_id: SessionId) -> DeclaredScopeIdentity:
        with self._store.transaction() as conn:
            row = _lineage.lineage_row(conn, session_id)
        if row is None:
            return DeclaredScopeIdentity(is_fork_child=False, source="")
        return DeclaredScopeIdentity(is_fork_child=_lineage.is_fork_child(row), source=str(row["source"] or "").strip())

    # ── titles ──

    def _bot_chat_title_released(self, conn: Connection, row: Mapping[str, Any]) -> bool:
        """Whether an archived row no longer holds the canonical name: archived deliberately and its compression tip's
        ``end_reason`` not recoverable."""
        if not row["archived"] or row["auto_archived"]:
            return False
        tip = _lineage.lineage_row(conn, _lineage.compression_tip(conn, row["id"])) or row
        return (tip["end_reason"] or "") not in RECOVERABLE_END_REASONS

    def _resolve_title_conflict(self, conn: Connection, session_id: str, title: str) -> Optional[str]:
        """Free ``title`` for ``session_id`` or raise :class:`TitleConflict`: a compression ancestor hands it to its
        continuation, a retired canonical Bot Chat and an ended empty visible ghost yield it. Returns the id whose
        title was cleared."""
        holder = _row(conn, "SELECT * FROM sessions WHERE title = %s AND id <> %s FOR UPDATE", (title, session_id))
        if holder is None:
            return None
        if _lineage.is_compression_ancestor(conn, holder["id"], session_id):
            self._update(conn, holder["id"], {"title": None})
            return holder["id"]
        retired_bot_chat = title == CANONICAL_BOT_CHAT_TITLE and self._bot_chat_title_released(conn, holder)
        if retired_bot_chat or (self._is_empty_ended(conn, holder) and not holder["hidden"]):
            self._update(conn, holder["id"], {"title": None, "title_source": None})
            return holder["id"]
        raise TitleConflict(title, SessionId(holder["id"]))

    def _write_title(self, session_id: SessionId, title: Optional[str], source: TitleSource) -> bool:
        """Write a title under provenance precedence: a user write always lands, an automatic one only over an
        untitled row or strictly lower authority; nothing moves the hidden canonical Bot Chat off its title."""
        with self._store.transaction() as conn:
            return bool(self._title_write(conn, session_id, title, source))

    def _title_write(self, conn: Connection, session_id: str, title: Optional[str], source: TitleSource) -> list[str]:
        """``_write_title`` in the caller's transaction; the ids it changed (the row, and one the uniqueness rule
        freed the name from), empty when it wrote nothing."""
        clean = sanitize_title(title)
        row = self._locked(conn, session_id)
        if row is None:
            return []
        is_user = source is TitleSource.USER
        if row["title"] == CANONICAL_BOT_CHAT_TITLE and row["hidden"] and clean != CANONICAL_BOT_CHAT_TITLE:
            if is_user:
                raise CanonicalTitleLocked(SessionId(session_id))
            return []
        current_rank = _TITLE_RANK[_TITLE_SOURCES.get(row["title_source"])]
        if not is_user and row["title"] is not None and current_rank >= _TITLE_RANK[source]:
            return []
        freed = self._resolve_title_conflict(conn, session_id, clean) if clean else None
        self._update(conn, session_id, {"title": clean, "title_source": source.value if clean else None})
        return [session_id, *([freed] if freed else [])]

    def set_title(self, session_id: SessionId, title: Optional[str]) -> bool:
        return self._write_title(session_id, title, TitleSource.USER)

    def set_auto_title(self, session_id: SessionId, title: str, *, source: TitleSource) -> bool:
        if source is TitleSource.USER:
            raise ValueError(f"invalid automatic title source: {source.value!r}")
        return self._write_title(session_id, title, source)

    def set_title_source(self, session_id: SessionId, source: TitleSource) -> bool:
        with self._store.transaction() as conn:
            return bool(execute(conn, "UPDATE sessions SET title_source = %s WHERE id = %s AND title IS NOT NULL",
                                (source.value, session_id)))

    def retitle_canonical_bot_chat(self, session_id: SessionId, title: str) -> bool:
        with self._store.transaction() as conn:
            return bool(execute(conn, "UPDATE sessions SET title = %s, title_source = %s, hidden = false "
                                      "WHERE id = %s AND title = %s",
                                (title, TitleSource.USER.value, session_id, CANONICAL_BOT_CHAT_TITLE)))

    def get_title(self, session_id: SessionId) -> Optional[str]:
        with self._store.transaction() as conn:
            return scalar(conn, "SELECT title FROM sessions WHERE id = %s", (session_id,))

    def get_title_source(self, session_id: SessionId) -> Optional[TitleSource]:
        with self._store.transaction() as conn:
            found = _row(conn, "SELECT title, title_source FROM sessions WHERE id = %s", (session_id,))
        if found is None or found["title"] is None:
            return None
        return _TITLE_SOURCES.get(found["title_source"])

    def _titled(self, conn: Connection, title: str) -> Optional[dict[str, Any]]:
        """The row holding ``title``; a retired canonical Bot Chat reads as absent."""
        found = _row(conn, "SELECT * FROM sessions WHERE title = %s", (title,))
        if found is not None and title == CANONICAL_BOT_CHAT_TITLE and self._bot_chat_title_released(conn, found):
            return None
        return found

    def get_by_title(self, title: str) -> Optional[SessionRecord]:
        with self._store.transaction() as conn:
            found = self._titled(conn, title)
            # The title lookup joins only the prompt store, so the pin stays its unresolved hash.
            return None if found is None else self._record(conn, found["id"], resolve_tool_pin=False)

    def resolve_by_title(self, title: str) -> Optional[SessionId]:
        with self._store.transaction() as conn:
            exact = self._titled(conn, title)
            if title == CANONICAL_BOT_CHAT_TITLE:
                return None if exact is None else SessionId(exact["id"])
            numbered = _row(conn, "SELECT id FROM sessions WHERE title IS NOT NULL "
                                  "AND starts_with(translate(title, %s, %s), translate(%s, %s, %s)) "
                                  "ORDER BY started_at DESC, id DESC LIMIT 1",
                            (_UPPER, _LOWER, title + " #", _UPPER, _LOWER))
        if numbered is not None:
            return SessionId(numbered["id"])
        return None if exact is None else SessionId(exact["id"])

    def next_title_in_lineage(self, base_title: str) -> str:
        match = _NUMBERED_TITLE_RE.match(base_title)
        base = match.group(1) if match else base_title
        with self._store.transaction() as conn:
            titles = [found["title"] for found in _rows(
                conn, "SELECT title FROM sessions WHERE title = %s "
                      "OR starts_with(translate(title, %s, %s), translate(%s, %s, %s))",
                (base, _UPPER, _LOWER, base + " #", _UPPER, _LOWER))]
        if not titles:
            return base
        # The unnumbered original counts as #1.
        numbers = [int(found.group(2)) for found in map(_NUMBERED_TITLE_RE.match, titles) if found]
        return f"{base} #{max([1, *numbers]) + 1}"

    def title_holder(self, title: str, *, excluding: SessionId) -> Optional[SessionId]:
        with self._store.transaction() as conn:
            holder = scalar(conn, "SELECT id FROM sessions WHERE title = %s AND id <> %s LIMIT 1", (title, excluding))
        return None if holder is None else SessionId(holder)

    # ── model config ──

    def _config_write(self, session_id: str, change) -> bool:
        """Apply ``change(row) -> (config or None, extra column values)`` to the locked row; False without it."""
        with self._store.transaction() as conn:
            row = self._locked(conn, session_id)
            if row is None:
                return False
            config, values = change(row)
            if config is not None:
                values = {**values, "model_config": Jsonb(config)}
            self._update(conn, session_id, values)
            return True

    def replace_session_config(self, session_id: SessionId, command: ReplaceSessionConfig) -> None:
        extra = {} if command.model is None else {"model": command.model}
        self._config_write(session_id, lambda row: (dict(command.model_config), extra))

    def patch_model_config(self, session_id: SessionId, patch: Mapping[str, Optional[JsonValue]]) -> None:
        if not session_id or not patch:
            return
        self._config_write(session_id, lambda row: (_merged(_lineage.config_of(row), patch), {}))

    def advance_onboarding_run(self, session_id: SessionId, event: Any, *, now: float) -> OnboardingAdvance:
        from session_rules.onboarding_flow import Facts, apply, is_active

        with self._store.transaction() as conn:
            tip = _lineage.compression_tip(conn, str(session_id))
            row = self._locked(conn, tip)
            if row is None:
                return OnboardingAdvance(session_id=None, was_active=False, step=None)
            config = _lineage.config_of(row)
            current = config.get(ONBOARDING_RUN_KEY)
            if not isinstance(current, dict):
                return OnboardingAdvance(session_id=SessionId(tip), was_active=False, step=None)
            step = apply(current, event, Facts.from_stored(current["facts"]), now=now)
            self._set_config(conn, tip, _merged(config, {ONBOARDING_RUN_KEY: json.loads(json.dumps(step.record))}))
            return OnboardingAdvance(session_id=SessionId(tip), was_active=is_active(current), step=step)

    def get_model_config_value(self, session_id: SessionId, key: str, default: JsonValue = None) -> JsonValue:
        with self._store.transaction() as conn:
            found = _row(conn, "SELECT model_config FROM sessions WHERE id = %s", (session_id,))
        if found is None:
            return default
        return _lineage.config_of(found).get(key, default)

    def session_ids_with_model_config_key(self, key: str) -> tuple[SessionId, ...]:
        with self._store.transaction() as conn:
            found = _rows(conn, "SELECT id FROM sessions WHERE coalesce(model_config -> %s, 'null'::jsonb) <> 'null'::jsonb "
                                "ORDER BY started_at DESC, id DESC", (key,))
        return tuple(SessionId(item["id"]) for item in found)

    def update_model(self, session_id: SessionId, model: str, provider: Optional[str] = None, *,
                     base_url: Optional[str] = None, api_mode: Optional[str] = None) -> None:
        patch: dict[str, Any] = {RUNTIME_LOCK_KEY: None}
        if model:
            patch["model"] = model
        if provider:
            route = {"provider": provider, "base_url": base_url or None, "api_mode": api_mode or None}
            # The nested route names its model: resume must still pair this provider with this model.
            patch.update(route, gateway_runtime={**route, "model": model or None})
        self._config_write(session_id, lambda row: (_merged(_lineage.config_of(row), patch), {"model": model}))

    def set_runtime_lock(self, session_id: SessionId, lock: RuntimeLock) -> None:
        payload: dict[str, JsonValue] = {**lock.as_mapping(), "updated_at": time.time()}
        extra = {} if lock.model is None else {"model": lock.model}
        self._config_write(session_id, lambda row: ({**_lineage.config_of(row), RUNTIME_LOCK_KEY: payload}, extra))

    def set_yolo(self, session_id: SessionId, enabled: bool) -> None:
        if session_id:
            self._config_write(session_id, lambda row: ({**_lineage.config_of(row), YOLO_KEY: bool(enabled)}, {}))

    def update_system_prompt(self, session_id: SessionId, system_prompt: Optional[str]) -> None:
        with self._store.transaction() as conn:
            self._update(conn, session_id, {"system_prompt_hash": store_text(conn, system_prompt)})

    def request_prompt_refresh(self, session_id: Optional[SessionId] = None) -> int:
        key = "" if session_id is None else str(session_id)
        with self._store.transaction() as conn:
            conn.execute("LOCK TABLE prompt_refresh_epochs IN EXCLUSIVE MODE")
            epoch = int(scalar(conn, "SELECT coalesce(max(epoch), 0) + 1 FROM prompt_refresh_epochs"))
            execute(conn, "INSERT INTO prompt_refresh_epochs (session_id, epoch) VALUES (%s, %s) "
                          "ON CONFLICT (session_id) DO UPDATE SET epoch = excluded.epoch", (key, epoch))
        return epoch

    def prompt_refresh_epoch(self, session_id: SessionId) -> int:
        with self._store.transaction() as conn:
            return int(scalar(conn, "SELECT coalesce(max(epoch), 0) FROM prompt_refresh_epochs "
                                    "WHERE session_id IN ('', %s)", (str(session_id),)))

    def count_with_stored_prompt(self) -> int:
        with self._store.transaction() as conn:
            return int(scalar(conn, "SELECT count(*) FROM sessions WHERE system_prompt_hash IS NOT NULL"))

    def update_tool_pin(self, session_id: SessionId, tool_pin: Optional[JsonValue]) -> None:
        with self._store.transaction() as conn:
            text = None if tool_pin is None else json.dumps(tool_pin)
            self._update(conn, session_id, {"tool_names": store_text(conn, text)})

    # ── working directory and git metadata ──

    def update_cwd(self, session_id: SessionId, cwd: str, git_branch: Optional[str] = None,
                   git_repo_root: Optional[str] = None, *, replace_git_meta: bool = False) -> Optional[int]:
        if not session_id or not cwd:
            return None
        captured = {"git_branch": (git_branch or "").strip(), "git_repo_root": (git_repo_root or "").strip()}
        with self._store.transaction() as conn:
            row = self._locked(conn, session_id)
            if row is None:
                return None
            if row["cwd"] != cwd or replace_git_meta:
                values = {field: value or None for field, value in captured.items()}
            else:
                # Same cwd: only a captured value overwrites, so a failed probe never clobbers one.
                values = {field: value for field, value in captured.items() if value}
            generation = int(row["git_metadata_generation"] or 0) + 1
            self._update(conn, session_id, {**values, "cwd": cwd, "git_metadata_generation": generation})
            return generation

    def publish_git_metadata(self, session_id: SessionId, cwd: str, generation: int, git_branch: Optional[str] = None,
                             git_repo_root: Optional[str] = None) -> bool:
        valid_generation = isinstance(generation, int) and not isinstance(generation, bool) and generation >= 1
        if not session_id or not cwd or not valid_generation:
            return False
        fields = {field: value for field, value in (("git_branch", (git_branch or "").strip()),
                                                   ("git_repo_root", (git_repo_root or "").strip())) if value}
        if not fields:
            return False
        assignments = ", ".join(f"{column} = %s" for column in fields)
        with self._store.transaction() as conn:
            return bool(execute(conn, f"UPDATE sessions SET {assignments} WHERE id = %s AND cwd = %s "
                                      f"AND git_metadata_generation = %s",
                                (*fields.values(), session_id, cwd, generation)))

    # ── lineage flags ──

    def _set_lineage_flag(self, conn: Connection, session_id: str, values: Mapping[str, Any]) -> list[str]:
        """Set ``values`` on every row of the session's compression lineage; returns the ids written."""
        if scalar(conn, "SELECT 1 FROM sessions WHERE id = %s", (session_id,)) is None:
            return []
        ids = _lineage.lineage_ids(conn, session_id)
        assignments = ", ".join(f"{column} = %s" for column in values)
        execute(conn, f"UPDATE sessions SET {assignments} WHERE id = ANY(%s)", (*values.values(), ids))
        return ids

    def _retire_canonical_bot_chat(self, conn: Connection, session_id: str) -> list[str]:
        """The deliberate archive's retirement of the canonical Bot Chat: every lineage row titled it loses title and
        source; when any does, the lineage is un-hidden and its recoverable end reasons cleared."""
        ids = _lineage.lineage_ids(conn, session_id)
        held = [found["id"] for found in _rows(conn, "SELECT id FROM sessions WHERE id = ANY(%s) AND title = %s",
                                               (ids, CANONICAL_BOT_CHAT_TITLE))]
        if not held:
            return []
        execute(conn, "UPDATE sessions SET hidden = false, end_reason = CASE WHEN end_reason = ANY(%s) THEN NULL "
                      "ELSE end_reason END WHERE id = ANY(%s)", (list(RECOVERABLE_END_REASONS), ids))
        execute(conn, "UPDATE sessions SET title = NULL, title_source = NULL WHERE id = ANY(%s)", (held,))
        return held

    def set_archived(self, session_id: SessionId, archived: bool, *, notify: bool = True) -> ArchiveChange:
        with self._store.transaction() as conn:
            row = self._locked(conn, session_id)
            was_archived = row is not None and row["archived"]
            # The deliberate archive clears the idle sweep's provenance either way.
            ids = self._set_lineage_flag(conn, session_id, {"archived": bool(archived), "auto_archived": False})
            if archived and ids:
                self._retire_canonical_bot_chat(conn, session_id)
        newly_archived = bool(archived and ids and not was_archived)
        return ArchiveChange(changed=bool(ids),
                             archived_ids=tuple(SessionId(sid) for sid in ids) if newly_archived and notify else ())

    def set_pinned(self, session_id: SessionId, pinned: bool) -> bool:
        with self._store.transaction() as conn:
            ids = self._set_lineage_flag(conn, session_id, {"pinned": bool(pinned)})
            row = _row(conn, "SELECT hidden, title FROM sessions WHERE id = %s", (session_id,))
            canonical_bot_chat = row is not None and row["hidden"] and row["title"] == CANONICAL_BOT_CHAT_TITLE
            if pinned and not canonical_bot_chat:
                self._set_lineage_flag(conn, session_id, {"hidden": False})
        return bool(ids)

    def set_hidden(self, session_id: SessionId, hidden: bool) -> bool:
        with self._store.transaction() as conn:
            return bool(self._set_lineage_flag(conn, session_id, {"hidden": bool(hidden)}))

    def set_read(self, session_id: SessionId, read: bool = True) -> bool:
        with self._store.transaction() as conn:
            return bool(self._set_lineage_flag(conn, session_id, {"last_read_at": time.time() if read else 0.0}))

    def _unarchive_auto_archived_lineage(self, conn: Connection, session_id: str) -> bool:
        """Un-hide a lineage only the idle sweep archived; any deliberately archived row keeps it archived."""
        ids = _lineage.lineage_ids(conn, session_id)
        if scalar(conn, "SELECT 1 FROM sessions WHERE id = ANY(%s) AND archived AND NOT auto_archived LIMIT 1",
                  (ids,)) is not None:
            return False
        return bool(execute(conn, "UPDATE sessions SET archived = false, auto_archived = false "
                                  "WHERE id = ANY(%s) AND archived", (ids,)))

    def unarchive_recoverable(self, session_id: SessionId) -> bool:
        if not session_id:
            return False
        try:
            with self._store.transaction() as conn:
                row = self._locked(conn, session_id)
                if row is None or not row["archived"]:
                    return False
                # The accidental stamp lives on the live tip.
                tip_id = _lineage.compression_tip(conn, session_id)
                tip = _lineage.lineage_row(conn, tip_id) or row
                if (tip["end_reason"] or "") not in RECOVERABLE_END_REASONS:
                    return False
                if not self._set_lineage_flag(conn, session_id, {"archived": False, "auto_archived": False}):
                    return False
                self._update(conn, tip["id"], {"ended_at": None, "end_reason": None})
        except Exception:
            return False
        return True


# ASCII-only case fold, the SQLite store's LIKE comparison.
_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_LOWER = "abcdefghijklmnopqrstuvwxyz"


def _cell(value: Any) -> Any:
    """A Python value as the cell it is written as: a mapping becomes jsonb."""
    if isinstance(value, Mapping):
        return Jsonb(dict(value))
    return value


register_repository("sessions", PostgresSessionRepository)
