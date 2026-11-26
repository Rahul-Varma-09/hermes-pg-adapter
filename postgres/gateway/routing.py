"""``RoutingRepository`` on Postgres: the gateway routing index, its route fences, routing peers, orphan repair,
never-active keyed rows and Telegram DM topics.

Routing entries are ``gateway_routing`` rows holding the gateway's exact entry JSON text, with the string
``session_id`` it names alongside for the per-session sweeps. Route fences are ``state_meta`` values under
``gateway.retired_route.v1.<namespace>:<id>``; this store's own retirement markers are ``retired_sessions``. The peer,
orphan and never-active verbs select candidate rows in SQL and judge them with the reference rules in Python.
"""

from __future__ import annotations

import json
import time
from typing import Any, Iterable, Mapping, Optional

from psycopg import Connection

from persistence import JsonValue, PermanentStoreError, SessionArtifacts, SessionId, SessionRowMissing
from persistence.authority.vocabulary import RETRY_HORIZON_S
from persistence.conversation.sessions import RECOVERABLE_END_REASONS, SessionRecord
from persistence.gateway.routing import (
    END_REASON_SUPERSEDED_BY_REPAIR, ORPHAN_ADOPTION_MAX_GAP_S, PEER_RESET_END_REASONS, TOPIC_SCHEMA_META_KEY,
    NeverActivePrune, NeverActiveSession, OrphanReport, RecoveredPeerSession, TopicAlreadyLinked, TopicBinding,
    normalize_topic_profile_name, route_namespace,
)

from .._backend import PostgresStore, register_repository
from .._sql import execute, row as _row, rows as _rows, scalar
from ..conversation import _lineage
from ..conversation.sessions import _SELECT_RESOLVED, session_record

ROUTE_FENCE_PREFIX = "gateway.retired_route.v1."
_PEER_COLUMNS = ("user_id", "chat_id", "chat_type", "thread_id")
_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_LOWER = "abcdefghijklmnopqrstuvwxyz"
_HAS_MESSAGES = "EXISTS (SELECT 1 FROM messages m WHERE m.session_id = s.id)"


def _blank(value: Any) -> str:
    return "" if value is None else str(value)


def _parse(text: Any) -> Any:
    try:
        return json.loads(text or "{}")
    except (ValueError, TypeError, RecursionError):
        return None


def _session_id_text(entry_json: str) -> Optional[str]:
    """The entry's ``session_id`` as text; malformed JSON refuses with ``PermanentStoreError('malformed JSON')``."""
    try:
        parsed = json.loads(entry_json)
    except ValueError as exc:
        raise PermanentStoreError("malformed JSON") from exc
    value = parsed.get("session_id") if isinstance(parsed, dict) else None
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value) if isinstance(value, (str, int, float)) else None


def _string_target(entry_json: str) -> Optional[str]:
    """The entry's ``session_id`` when it names one as a string (the sweep column)."""
    parsed = _parse(entry_json)
    value = parsed.get("session_id") if isinstance(parsed, dict) else None
    return value if isinstance(value, str) else None


def _fence_expired(value: Any, cutoff: float) -> bool:
    """A sweep fence (``expires``) retired before ``cutoff``; a value that is not JSON never expires."""
    parsed = _parse(value)
    if not isinstance(parsed, dict) or not parsed.get("expires"):
        return False
    retired_at = parsed.get("retired_at")
    return isinstance(retired_at, (int, float)) and retired_at < cutoff


def _recency(record: SessionRecord) -> float:
    return record.started_at if record.last_activity_at is None else record.last_activity_at


def _recoverable(record: SessionRecord) -> bool:
    return record.ended_at is None or record.end_reason in RECOVERABLE_END_REASONS or record.handoff_state == "completed"


def _peer_tuple(record: SessionRecord) -> tuple[str, ...]:
    return tuple(_blank(getattr(record, column)) for column in _PEER_COLUMNS)


def _fenced(candidate: SessionRecord, boundaries: Iterable[SessionRecord]) -> bool:
    recency = _recency(candidate)
    return any(boundary.ended_at is not None and boundary.end_reason in PEER_RESET_END_REASONS
               and boundary.ended_at > recency for boundary in boundaries)


def _topic(found: Mapping[str, Any]) -> TopicBinding:
    return TopicBinding(profile_name=found["profile_name"], chat_id=found["chat_id"], thread_id=found["thread_id"],
                        user_id=found["user_id"], session_key=found["session_key"],
                        session_id=SessionId(found["session_id"]), managed_mode=found["managed_mode"],
                        linked_at=found["linked_at"], updated_at=found["updated_at"])


class PostgresRoutingRepository:
    """The routing index and peers of the bound profile's schema."""

    def __init__(self, store: PostgresStore) -> None:
        self._store = store

    # ── session rows ──

    def _records(self, conn: Connection, where: str, params: Any = ()) -> list[tuple[SessionRecord, bool]]:
        """``(record, has stored messages)`` of the rows matching ``where`` (over alias ``s``)."""
        found = _rows(conn, f"{_SELECT_RESOLVED} WHERE {where}", params)
        return [(session_record(item), bool(scalar(conn, "SELECT 1 FROM messages WHERE session_id = %s LIMIT 1",
                                                   (item["id"],)))) for item in found]

    @staticmethod
    def _message_rows(conn: Connection, session_id: str) -> int:
        return int(scalar(conn, "SELECT count(*) FROM messages WHERE session_id = %s", (session_id,)))

    # ── the routing index ──

    @staticmethod
    def _route_retired(conn: Connection, session_key: str, entry_json: str, now: float) -> bool:
        """A launch (``main``) route whose session this store deleted, or an unexpired route fence, which on a launch
        route counts only while the session row is gone."""
        session_id = _session_id_text(entry_json)
        if session_id is None:
            return False
        namespace = route_namespace(session_key)
        if namespace == "main" and scalar(conn, "SELECT 1 FROM retired_sessions WHERE session_id = %s",
                                          (session_id,)) is not None:
            return True
        fence = scalar(conn, "SELECT value FROM state_meta WHERE key = %s", (f"{ROUTE_FENCE_PREFIX}{namespace}:{session_id}",))
        if fence is None or _fence_expired(fence, now - RETRY_HORIZON_S):
            return False
        return namespace != "main" or scalar(conn, "SELECT 1 FROM sessions WHERE id = %s", (session_id,)) is None

    @staticmethod
    def _put_route(conn: Connection, routing_scope: str, session_key: str, entry_json: str) -> None:
        execute(conn, "INSERT INTO gateway_routing (scope, session_key, entry_json, session_id, updated_at) "
                      "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (scope, session_key) DO UPDATE "
                      "SET entry_json = excluded.entry_json, session_id = excluded.session_id, updated_at = excluded.updated_at",
                (routing_scope, session_key, entry_json, _string_target(entry_json), time.time()))

    def route_retired(self, session_key: str, entry_json: str) -> bool:
        with self._store.transaction() as conn:
            return self._route_retired(conn, session_key, entry_json, time.time())

    def upsert_routing_entry(self, session_key: str, entry_json: str, *, routing_scope: str = "") -> None:
        if not session_key or not entry_json:
            return
        with self._store.transaction() as conn:
            if not self._route_retired(conn, session_key, entry_json, time.time()):
                self._put_route(conn, routing_scope, session_key, entry_json)

    def replace_routing_entries(self, entries: Mapping[str, str], *, routing_scope: str = "") -> set[str]:
        now = time.time()
        pairs = [(key, entry) for key, entry in entries.items() if key and entry]
        with self._store.transaction() as conn:
            skipped = {key for key, entry in pairs if self._route_retired(conn, key, entry, now)}
            execute(conn, "DELETE FROM gateway_routing WHERE scope = %s", (routing_scope,))
            for key, entry in pairs:
                if key not in skipped:
                    self._put_route(conn, routing_scope, key, entry)
        return skipped

    def load_routing_entries(self, *, routing_scope: str = "") -> dict[str, str]:
        with self._store.transaction() as conn:
            found = _rows(conn, "SELECT session_key, entry_json FROM gateway_routing WHERE scope = %s ORDER BY session_key",
                          (routing_scope,))
        return {item["session_key"]: item["entry_json"] for item in found}

    def routing_entry_for_session(self, session_id: str) -> Mapping[str, JsonValue] | None:
        with self._store.transaction() as conn:
            found = _rows(conn, "SELECT entry_json FROM gateway_routing WHERE session_id = %s ORDER BY scope, session_key",
                          (session_id,))
        for item in found:
            entry = _parse(item["entry_json"])
            if isinstance(entry, dict) and entry.get("session_id") == session_id:
                return entry
        return None

    @staticmethod
    def _mark_fence(conn: Connection, key: str, *, expires: bool) -> None:
        """Write a route fence: a sweep's fence never shortens a permanent one."""
        value: dict[str, Any] = {"retired_at": time.time()}
        if expires:
            existing = scalar(conn, "SELECT value FROM state_meta WHERE key = %s FOR UPDATE", (key,))
            parsed = _parse(existing) if existing is not None else None
            if existing is None or (isinstance(parsed, dict) and parsed.get("expires")):
                value["expires"] = True
            else:
                return
        execute(conn, "INSERT INTO state_meta (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value = excluded.value",
                (key, json.dumps(value)))

    def fence_retired_routes(self, session_ids: Iterable[str], *, namespace: str, expiring: Iterable[str] = ()) -> None:
        ids = sorted({str(sid) for sid in session_ids if sid})
        if not ids:
            return
        expiring = set(expiring)
        with self._store.transaction() as conn:
            cutoff = time.time() - RETRY_HORIZON_S
            stale = [item["key"] for item in _rows(conn, "SELECT key, value FROM state_meta WHERE starts_with(key, %s)",
                                                   (ROUTE_FENCE_PREFIX,)) if _fence_expired(item["value"], cutoff)]
            execute(conn, "DELETE FROM state_meta WHERE key = ANY(%s)", (stale,))
            present = {item["id"] for item in _rows(conn, "SELECT id FROM sessions WHERE id = ANY(%s)", (ids,))}
            fenced: dict[str, set[str]] = {}
            for sid in ids:
                names = {namespace} if sid in present else {namespace, "main"}
                for name in sorted(names):
                    self._mark_fence(conn, f"{ROUTE_FENCE_PREFIX}{name}:{sid}", expires=sid in expiring)
                fenced[sid] = names
            for item in _rows(conn, "SELECT scope, session_key, session_id FROM gateway_routing WHERE session_id = ANY(%s)", (ids,)):
                if route_namespace(item["session_key"]) in fenced[item["session_id"]]:
                    execute(conn, "DELETE FROM gateway_routing WHERE scope = %s AND session_key = %s",
                            (item["scope"], item["session_key"]))

    def sweep_routing_entries_for_sessions(self, session_ids: Iterable[str]) -> int:
        ids = sorted({str(sid) for sid in session_ids})
        with self._store.transaction() as conn:
            return execute(conn, "DELETE FROM gateway_routing WHERE session_id = ANY(%s)", (ids,))

    # ── routing peers ──

    @staticmethod
    def _compression_lineage(conn: Connection, session_id: str) -> list[str]:
        """The row and every parent that ended ``compression`` above a child with no branch, delegate or reset marker
        and not a ``tool`` row."""
        query = "SELECT id, parent_session_id, source, model_config, end_reason FROM sessions WHERE id = %s"
        current = _row(conn, query, (session_id,))
        if current is None:
            return []
        ids = [session_id]
        while True:
            parent_id = current["parent_session_id"]
            parent = _row(conn, query, (parent_id,)) if parent_id else None
            config = _lineage.config_of(current)
            if (parent is None or parent["end_reason"] != "compression"
                    or any(config.get(marker) is not None for marker in ("_branched_from", "_delegate_from", "_reset_from"))
                    or (current["source"] or "") == "tool" or parent_id in ids):
                return ids
            ids.append(parent_id)
            current = parent

    def record_session_peer(self, session_id: str, *, source: str, user_id: str | None = None,
                            session_key: str | None = None, chat_id: str | None = None, chat_type: str | None = None,
                            thread_id: str | None = None, display_name: str | None = None,
                            origin_json: str | None = None, include_compression_ancestors: bool = False,
                            messaging_lane: bool = False, transport_profile: str | None = None) -> None:
        if not session_id or not session_key:
            return
        with self._store.transaction() as conn:
            if include_compression_ancestors:
                targets = self._compression_lineage(conn, session_id)
            else:
                targets = [session_id]
                execute(conn, "INSERT INTO sessions (id, source, created_source, profile_name, started_at) "
                              "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
                        (session_id, source, source, self._store.owner_profile, time.time()))
            execute(conn, "UPDATE sessions SET session_key = %s, source = %s, user_id = %s, chat_id = %s, chat_type = %s, "
                          "thread_id = %s, display_name = coalesce(%s, display_name), "
                          "origin_json = coalesce(%s, origin_json), transport_profile = coalesce(%s, transport_profile) "
                          "WHERE id = ANY(%s)",
                    (session_key, source, user_id, chat_id, chat_type, thread_id, display_name, origin_json,
                     transport_profile, targets))
            if include_compression_ancestors and messaging_lane:
                # An explicit repoint to a messaging lane moved these rows: the local conversation whose receipt names
                # one of them goes with it in the same write.
                bindings = self._store.bindings
                moved = {owner: found for owner, found in bindings._moved_receipts(conn).items() if found[0] in targets}
                bindings._hand_off_conversations(conn, moved)

    def latest_session_for_peer(self, *, source: str, user_id: str | None = None, session_key: str | None = None,
                                chat_id: str | None = None, chat_type: str | None = None,
                                thread_id: str | None = None) -> RecoveredPeerSession | None:
        if not session_key:
            return None
        with self._store.transaction() as conn:
            keyed = self._records(conn, "s.session_key = %s AND s.source = %s", (session_key, source))
            records = [record for record, _has in keyed]
            exact = [(has or record.message_count > 0, _recency(record), record) for record, has in keyed
                     if _recoverable(record) and not _fenced(record, records)]
            if exact:
                has_messages, _when, winner = max(exact, key=lambda item: (item[0], item[1]))
                return RecoveredPeerSession(session=winner, has_messages=has_messages)
            if chat_id is None or chat_type is None:
                return None
            wanted = tuple(_blank(value) for value in (user_id, chat_id, chat_type, thread_id))
            peers = self._records(conn, "s.source = %s AND coalesce(s.user_id, '') = %s AND coalesce(s.chat_id, '') = %s "
                                        "AND coalesce(s.chat_type, '') = %s AND coalesce(s.thread_id, '') = %s",
                                  (source, *wanted))
        owner = self._store.owner_profile
        owned = [(record, has) for record, has in peers
                 if owner is None or (owner if record.profile_name is None else record.profile_name) == owner]
        boundaries = [record for record, _has in owned]
        found = [record for record, has in owned if _peer_tuple(record) == wanted and _recoverable(record)
                 and (has or record.message_count > 0) and not _fenced(record, boundaries)]
        if not found:
            return None
        return RecoveredPeerSession(session=max(found, key=_recency), has_messages=True)

    def session_for_origin(self, *, platform: str, chat_id: str, thread_id: str | None = None,
                           user_id: str | None = None) -> str | None:
        if not platform or chat_id in (None, ""):
            return None
        query = ("SELECT id, user_id FROM sessions WHERE translate(source, %s, %s) = translate(%s, %s, %s) "
                 "AND session_key IS NOT NULL AND chat_id = %s AND ended_at IS NULL")
        params: list[Any] = [_UPPER, _LOWER, platform, _UPPER, _LOWER, str(chat_id)]
        if thread_id is not None:
            query += " AND coalesce(thread_id, '') = %s"
            params.append(str(thread_id))
        with self._store.transaction() as conn:
            found = _rows(conn, query + " ORDER BY started_at DESC", params)
        if not found:
            return None
        if user_id:
            exact = [item for item in found if _blank(item["user_id"]) == str(user_id)]
            if exact:
                return str(exact[0]["id"])
            if len(found) > 1:
                return None
        elif len({user for user in (_blank(item["user_id"]).strip() for item in found) if user}) > 1:
            return None
        return str(found[0]["id"])

    # ── orphan repair ──

    def _profile_fence(self, donor: Mapping[str, Any], orphan_owner: Optional[str]) -> bool:
        owner = self._store.owner_profile
        return owner is None or (owner if donor["profile_name"] is None else donor["profile_name"]) == orphan_owner

    def _contiguity_donors(self, conn: Connection, orphan: Mapping[str, Any], last_active: Optional[float], gap: float,
                           orphan_owner: Optional[str]) -> list[dict[str, Any]]:
        if last_active is None:
            return []
        started = orphan["started_at"] or 0
        donors = []
        for donor in _rows(conn, "SELECT * FROM sessions WHERE session_key IS NOT NULL AND id <> %s "
                                 "AND coalesce(source, '') = %s", (orphan["id"], orphan["source"] or "")):
            if not self._profile_fence(donor, orphan_owner):
                continue
            if _blank(donor["user_id"]) and _blank(orphan["user_id"]) and donor["user_id"] != orphan["user_id"]:
                continue
            donor_active = _lineage.last_active(conn, donor)
            if donor_active is not None and started - gap <= donor_active <= started + gap and donor_active < last_active:
                donors.append((donor_active, donor))
        donors.sort(key=lambda item: item[0], reverse=True)
        return [donor for _active, donor in donors[:2]]

    def _orphan_report(self, conn: Connection, orphan: Mapping[str, Any], gap: float) -> dict[str, Any]:
        orphan_owner = orphan["profile_name"] or self._store.owner_profile
        last_active = _lineage.last_active(conn, orphan)
        donor: Optional[Mapping[str, Any]] = None
        reason = ""
        if orphan["parent_session_id"]:
            evidence = "lineage"
            parent = _row(conn, "SELECT * FROM sessions WHERE id = %s", (orphan["parent_session_id"],))
            if (parent is not None and parent["session_key"] is not None
                    and (parent["source"] or "") == (orphan["source"] or "") and self._profile_fence(parent, orphan_owner)):
                donor = parent
            else:
                reason = "parent session carries no gateway identity of this source"
        else:
            evidence = "contiguity"
            candidates = self._contiguity_donors(conn, orphan, last_active, gap, orphan_owner)
            if not candidates:
                reason = f"no keyed predecessor fell quiet within {gap:.0f}s of this session's start"
            elif len(candidates) > 1:
                reason = "ambiguous: more than one keyed predecessor matches this window"
            else:
                donor = candidates[0]
        return {"orphan_id": SessionId(orphan["id"]), "source": orphan["source"],
                "message_count": self._message_rows(conn, orphan["id"]), "started_at": orphan["started_at"],
                "last_active": last_active, "donor_id": SessionId(donor["id"]) if donor else None,
                "session_key": donor["session_key"] if donor else None, "evidence": evidence if donor else "",
                "adoptable": donor is not None, "reason": reason}

    def orphaned_sessions(self, *, max_gap_s: float | None = None) -> tuple[OrphanReport, ...]:
        gap = ORPHAN_ADOPTION_MAX_GAP_S if max_gap_s is None else float(max_gap_s)
        with self._store.transaction() as conn:
            orphans = _rows(conn, f"""
                SELECT * FROM sessions s WHERE s.session_key IS NULL AND coalesce(s.source, '') <> 'tool'
                  AND coalesce(s.model_config -> '_branched_from', 'null'::jsonb) = 'null'::jsonb
                  AND coalesce(s.model_config -> '_delegate_from', 'null'::jsonb) = 'null'::jsonb
                  AND {_HAS_MESSAGES} ORDER BY s.started_at""")
            reports = [self._orphan_report(conn, orphan, gap) for orphan in orphans]
        claims: dict[str, int] = {}
        for report in reports:
            if report["donor_id"] is not None:
                claims[report["donor_id"]] = claims.get(report["donor_id"], 0) + 1
        for report in reports:
            if report["donor_id"] is not None and claims[report["donor_id"]] > 1:
                report["adoptable"] = False
                report["reason"] = "ambiguous: more than one unkeyed session claims this predecessor"
        return tuple(OrphanReport(**report) for report in reports)

    def adopt_orphan(self, orphan_id: str, donor_id: str) -> bool:
        if not orphan_id or not donor_id or orphan_id == donor_id:
            return False
        owner = self._store.owner_profile
        with self._store.transaction() as conn:
            donor = _row(conn, "SELECT * FROM sessions WHERE id = %s FOR UPDATE", (donor_id,))
            orphan = _row(conn, "SELECT * FROM sessions WHERE id = %s FOR UPDATE", (orphan_id,))
            if (donor is None or orphan is None or not donor["session_key"] or orphan["session_key"]
                    or (donor["source"] or "") != (orphan["source"] or "")
                    or (owner is not None and (donor["profile_name"] or owner) != (orphan["profile_name"] or owner))):
                return False
            if orphan["session_key"] is None:
                copied = {column: donor[column] for column in ("chat_id", "chat_type", "thread_id", "user_id",
                                                               "origin_json", "display_name") if orphan[column] is None}
                if orphan["parent_session_id"] is None:
                    copied["parent_session_id"] = donor_id
                copied["session_key"] = donor["session_key"]
                execute(conn, f"UPDATE sessions SET {', '.join(f'{c} = %s' for c in copied)} WHERE id = %s",
                        (*copied.values(), orphan_id))
            execute(conn, "UPDATE sessions SET ended_at = coalesce(ended_at, %s), end_reason = %s WHERE id = %s",
                    (time.time(), END_REASON_SUPERSEDED_BY_REPAIR, donor_id))
        return True

    # ── never-active keyed rows ──

    def never_active_keyed_sessions(self, *, older_than_days: float) -> tuple[NeverActiveSession, ...]:
        if older_than_days < 0:
            raise ValueError(
                f"older_than_days must be >= 0, got {older_than_days!r}: a negative "
                "retention builds a future cutoff that matches every never-active keyed row.")
        cutoff = time.time() - float(older_than_days) * 86400.0
        with self._store.transaction() as conn:
            found = _rows(conn, f"""
                SELECT * FROM sessions s WHERE s.session_key IS NOT NULL AND s.ended_at IS NULL AND s.title IS NULL
                  AND s.last_activity_at IS NULL AND s.message_count = 0 AND s.tool_call_count = 0
                  AND s.api_call_count = 0 AND s.input_tokens = 0 AND s.output_tokens = 0 AND NOT s.pinned
                  AND NOT s.archived AND s.started_at < %s AND NOT {_HAS_MESSAGES}
                ORDER BY s.started_at, s.id""", (cutoff,))
        return tuple(NeverActiveSession(session_id=SessionId(item["id"]), session_key=str(item["session_key"]),
                                        source=item["source"], chat_id=item["chat_id"], chat_type=item["chat_type"],
                                        user_id=item["user_id"], started_at=item["started_at"]) for item in found)

    def prune_never_active(self, *, older_than_days: float, artifacts: SessionArtifacts | None = None,
                           media_root: str) -> NeverActivePrune:
        candidates = self.never_active_keyed_sessions(older_than_days=older_than_days)
        sessions = self._store.sessions
        with self._store.transaction() as conn:
            ids = {str(found.session_id) for found in candidates
                   if sessions._owned_lineage_ids(conn, str(found.session_id)) == [str(found.session_id)]}
            routed_before = [item["session_id"] for item in _rows(
                conn, "SELECT session_id FROM gateway_routing WHERE session_id = ANY(%s)", (sorted(ids),))]
        if not ids:
            return NeverActivePrune(sessions_deleted=0, routing_entries_deleted=0, sessions_skipped=0, deleted_ids=(),
                                    removed_ids=())
        outcome = sessions.delete_many([SessionId(sid) for sid in sorted(ids)], skip_guarded=True, media_root=media_root)
        if artifacts is not None:
            artifacts.remove_for(outcome.removed_ids)
        kept = {str(sid) for sid in outcome.skipped_guarded}
        return NeverActivePrune(
            sessions_deleted=outcome.deleted,
            routing_entries_deleted=sum(1 for sid in routed_before if sid not in kept),
            sessions_skipped=len(kept), deleted_ids=tuple(SessionId(sid) for sid in sorted(ids - kept)),
            removed_ids=tuple(SessionId(sid) for sid in sorted(map(str, outcome.removed_ids))))

    # ── Telegram DM topics ──

    @staticmethod
    def _prepare(conn: Connection) -> None:
        execute(conn, "INSERT INTO state_meta (key, value) VALUES (%s, '3') ON CONFLICT (key) DO UPDATE SET value = '3'",
                (TOPIC_SCHEMA_META_KEY,))

    def prepare_topic_storage(self) -> None:
        with self._store.transaction() as conn:
            self._prepare(conn)

    def enable_topic_mode(self, *, chat_id: str, user_id: str, profile_name: str = "default",
                          has_topics_enabled: bool | None = None, allows_users_to_create_topics: bool | None = None) -> None:
        now = time.time()
        with self._store.transaction() as conn:
            self._prepare(conn)
            execute(conn, """
                INSERT INTO telegram_dm_topic_mode (profile_name, chat_id, user_id, enabled, activated_at, updated_at,
                                                    has_topics_enabled, allows_users_to_create_topics, capability_checked_at)
                VALUES (%s, %s, %s, true, %s, %s, %s, %s, %s)
                ON CONFLICT (profile_name, chat_id) DO UPDATE SET user_id = excluded.user_id, enabled = true,
                    updated_at = excluded.updated_at, has_topics_enabled = excluded.has_topics_enabled,
                    allows_users_to_create_topics = excluded.allows_users_to_create_topics,
                    capability_checked_at = excluded.capability_checked_at""",
                    (normalize_topic_profile_name(profile_name), str(chat_id), str(user_id), now, now,
                     has_topics_enabled, allows_users_to_create_topics, now))

    def disable_topic_mode(self, *, chat_id: str, profile_name: str = "default", clear_bindings: bool = True) -> None:
        profile, chat = normalize_topic_profile_name(profile_name), str(chat_id)
        with self._store.transaction() as conn:
            execute(conn, "UPDATE telegram_dm_topic_mode SET enabled = false, updated_at = %s "
                          "WHERE profile_name = %s AND chat_id = %s", (time.time(), profile, chat))
            if clear_bindings:
                execute(conn, "DELETE FROM telegram_dm_topic_bindings WHERE profile_name = %s AND chat_id = %s",
                        (profile, chat))

    def is_topic_mode_enabled(self, *, chat_id: str, user_id: str, profile_name: str = "default") -> bool:
        with self._store.transaction() as conn:
            found = _row(conn, "SELECT user_id, enabled FROM telegram_dm_topic_mode WHERE profile_name = %s AND chat_id = %s",
                         (normalize_topic_profile_name(profile_name), str(chat_id)))
        return found is not None and found["user_id"] == str(user_id) and bool(found["enabled"])

    def get_topic_binding(self, *, chat_id: str, thread_id: str, profile_name: str = "default") -> TopicBinding | None:
        with self._store.transaction() as conn:
            found = _row(conn, "SELECT * FROM telegram_dm_topic_bindings WHERE profile_name = %s AND chat_id = %s "
                               "AND thread_id = %s", (normalize_topic_profile_name(profile_name), str(chat_id), str(thread_id)))
        return _topic(found) if found is not None else None

    def list_topic_bindings_for_chat(self, *, chat_id: str, profile_name: str = "default") -> tuple[TopicBinding, ...]:
        with self._store.transaction() as conn:
            found = _rows(conn, "SELECT * FROM telegram_dm_topic_bindings WHERE profile_name = %s AND chat_id = %s "
                                "ORDER BY updated_at DESC", (normalize_topic_profile_name(profile_name), str(chat_id)))
        return tuple(_topic(item) for item in found)

    def get_topic_binding_for_session(self, *, session_id: str) -> TopicBinding | None:
        with self._store.transaction() as conn:
            found = _row(conn, "SELECT * FROM telegram_dm_topic_bindings WHERE session_id = %s", (str(session_id),))
        return _topic(found) if found is not None else None

    def is_session_linked_to_topic(self, *, session_id: str) -> bool:
        return self.get_topic_binding_for_session(session_id=session_id) is not None

    def delete_topic_binding(self, *, chat_id: str, thread_id: str, profile_name: str = "default") -> int:
        profile, chat = normalize_topic_profile_name(profile_name), str(chat_id)
        with self._store.transaction() as conn:
            if not execute(conn, "DELETE FROM telegram_dm_topic_bindings WHERE profile_name = %s AND chat_id = %s "
                                 "AND thread_id = %s", (profile, chat, str(thread_id))):
                return 0
            if scalar(conn, "SELECT 1 FROM telegram_dm_topic_bindings WHERE profile_name = %s AND chat_id = %s LIMIT 1",
                      (profile, chat)) is None:
                execute(conn, "UPDATE telegram_dm_topic_mode SET enabled = false, updated_at = %s "
                              "WHERE profile_name = %s AND chat_id = %s", (time.time(), profile, chat))
            return 1

    def bind_topic(self, *, chat_id: str, thread_id: str, user_id: str, session_key: str, session_id: str,
                   managed_mode: str = "auto", profile_name: str = "default", expected_session_id: str | None = None,
                   move_session: bool = False) -> bool:
        now = time.time()
        key = (normalize_topic_profile_name(profile_name), str(chat_id), str(thread_id))
        sid = str(session_id)
        # The tables are prepared by the first bind, even one the checks below refuse.
        self.prepare_topic_storage()
        with self._store.transaction() as conn:
            if expected_session_id is not None:
                current = scalar(conn, "SELECT session_id FROM telegram_dm_topic_bindings WHERE profile_name = %s "
                                       "AND chat_id = %s AND thread_id = %s FOR UPDATE", key)
                if current is None or str(current) != str(expected_session_id):
                    return False
            linked = _row(conn, "SELECT profile_name, chat_id, thread_id FROM telegram_dm_topic_bindings "
                                "WHERE session_id = %s FOR UPDATE", (sid,))
            old_key = (linked["profile_name"], linked["chat_id"], linked["thread_id"]) if linked else key
            if old_key != key and (not move_session or old_key[0] != key[0]):  # never across profiles
                raise TopicAlreadyLinked(sid)
            if scalar(conn, "SELECT 1 FROM sessions WHERE id = %s", (sid,)) is None:
                raise SessionRowMissing(SessionId(sid))
            if old_key != key:
                execute(conn, "DELETE FROM telegram_dm_topic_bindings WHERE profile_name = %s AND chat_id = %s "
                              "AND thread_id = %s", old_key)
            execute(conn, """
                INSERT INTO telegram_dm_topic_bindings (profile_name, chat_id, thread_id, user_id, session_key, session_id,
                                                        managed_mode, linked_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (profile_name, chat_id, thread_id) DO UPDATE SET user_id = excluded.user_id,
                    session_key = excluded.session_key, session_id = excluded.session_id,
                    managed_mode = excluded.managed_mode, updated_at = excluded.updated_at""",
                    (*key, str(user_id), str(session_key), sid, managed_mode, now, now))
        return True


register_repository("routing", PostgresRoutingRepository)
