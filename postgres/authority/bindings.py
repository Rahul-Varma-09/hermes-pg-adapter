"""``SessionBindingRepository`` on Postgres: local creation receipts, API bindings and the local lineage reads.

Creation receipts are ``authority_meta`` values under ``binding:<id>``, legacy bindings under
``gateway.local_legacy.v1:<id>``, history-control records under ``history_control:<id>``; API binding receipts and the
declared conversation keys are ``authority_meta`` values too. Each write is one transaction that stages the session
row, the routing entry and the receipt together.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import Any, Mapping, Optional

from psycopg import Connection

from persistence.authority.admissions import AdmissionGate, raise_gate
from persistence.authority.bindings import (
    ApiBinding, BindingMarks, LegacyAdoptionScan, LocalLineage, NewApiBinding, NewSessionBinding, ReservedBinding,
    SessionBinding, api_generation_id,
)
from persistence.authority.vocabulary import LEGACY_LOCAL_SOURCES, OwnerRule
from persistence.conversation.lineage import CompressionChild
from persistence.core.canonical import canonical_json
from persistence.core.errors import AuthorityRefusal, PermanentStoreError, ProofRejected
from persistence.core.identity import JsonValue, SessionId
from session_rules.legacy_local import (
    binding_names, client_history_row, legacy_account_label, legacy_history_row, not_client_labels, same_owner,
    validate_legacy_row,
)
from session_rules.local_create import same_create_retry

from .._backend import PostgresStore, register_repository
from .._sql import execute, row as _row, rows as _rows, scalar
from ..conversation import _lineage
from ..conversation.sessions import bump_conversation_generation
from . import _leases, _meta
from ._bindings_lineage import (
    HISTORY_PREFIX, LEGACY_PREFIX, RECEIPT_PREFIX, adoptable_compression, advance_local_target, commit_local_target,
    decode, local_conversation_ids, local_lineage_owner, route_exists, store_route, validate_local_lineage,
    with_adopted_compression,
)
from .admissions import bound_id
from .runtime import VIEWER_CLOSE_PREFIX, require_epoch

logger = logging.getLogger(__name__)

API_BINDING_PREFIX = "gateway.api.binding.v1."
API_DECLARED_PREFIX = "gateway.api.conversation.v1."
_IDENTITY = ("profile_id", "principal_id", "request_id", "session_id", "route")
_LANE_IDENTITY = ("session_key", "source", "user_id", "chat_id", "chat_type", "thread_id", "display_name", "origin_json",
                  "transport_profile")
_TITLE_RANK = {"derived": 0, "llm": 1, "user": 2}


# ── receipts ──

def to_binding(stored: Any) -> SessionBinding:
    """The stored receipt as a detached record; one that does not decode to an object is ``storage_unavailable``."""
    try:
        receipt = decode(stored)
        if not isinstance(receipt, dict):
            raise ValueError("invalid receipt")
        receipt = json.loads(json.dumps(receipt))
        lineage = receipt.get("lineage", [receipt["session_id"]])
        return SessionBinding(
            profile_id=receipt["profile_id"], principal_id=receipt["principal_id"], request_id=receipt["request_id"],
            session_id=SessionId(receipt["session_id"]), route=receipt["route"], entry=receipt["entry"],
            policy=receipt["policy"], lineage=tuple(SessionId(sid) for sid in lineage),
            legacy_session_id=receipt.get("legacy_session_id"))
    except (TypeError, ValueError, KeyError, AttributeError) as exc:
        raise AuthorityRefusal("storage_unavailable") from exc


def receipt_of(binding: SessionBinding) -> dict[str, Any]:
    """The receipt dict a ``SessionBinding`` was read from (``lineage`` present unless it is the default)."""
    receipt: dict[str, Any] = {
        "profile_id": binding.profile_id, "principal_id": binding.principal_id, "request_id": binding.request_id,
        "session_id": binding.session_id, "route": binding.route, "entry": dict(binding.entry),
        "policy": dict(binding.policy)}
    if binding.legacy_session_id is not None:
        receipt["legacy_session_id"] = binding.legacy_session_id
    if binding.lineage != (binding.session_id,):
        receipt["lineage"] = list(binding.lineage)
    return json.loads(json.dumps(receipt))


def _new_receipt(binding: NewSessionBinding) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "profile_id": binding.profile_id, "principal_id": binding.principal_id, "request_id": binding.request_id,
        "session_id": binding.session_id, "route": binding.route, "entry": dict(binding.entry),
        "policy": dict(binding.policy)}
    if binding.legacy_session_id is not None:
        receipt["legacy_session_id"] = binding.legacy_session_id
    if binding.lineage is not None:
        receipt["lineage"] = list(binding.lineage)
    return receipt


def _row_cwd(policy: Mapping[str, Any]) -> Any:
    """The create request's ``row_cwd`` (``""`` = none) when it sent one, else the policy cwd."""
    try:
        request = json.loads(policy.get("request_json") or "{}")
    except (TypeError, ValueError):
        request = {}
    value = request.get("row_cwd") if isinstance(request, dict) else None
    return policy["cwd"] if not isinstance(value, str) else (value or None)


def _policy_target(stored: Any) -> Any:
    """The stored receipt's ``entry['session_id']``, None when it does not decode to an object holding an object
    ``entry``."""
    try:
        value = decode(stored)
    except (TypeError, ValueError):
        return None
    entry = value.get("entry") if isinstance(value, dict) else None
    return entry.get("session_id") if isinstance(entry, dict) else None


def binding_text(saved: Any) -> Optional[str]:
    """A stored history binding as compact JSON text, None when absent."""
    return None if saved is None else json.dumps(saved, separators=(",", ":"))


# ── session-row rules ──

def legacy_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """The columns the legacy-history rules read."""
    return {key: row.get(key) for key in ("source", "user_id", "session_key", "chat_id", "origin_json", "profile_name")}


def _routed(row: Mapping[str, Any]) -> bool:
    return bool(row.get("session_key") or row.get("chat_id") or row.get("origin_json"))


def _owner_user(row: Mapping[str, Any]) -> Any:
    """The row's owner identity; the Desktop Nous-account label is none."""
    return None if legacy_account_label(legacy_row(row)) else row.get("user_id")


def history_row(row: Mapping[str, Any], owner: Optional[OwnerRule]) -> bool:
    """Owner history a sidebar edit may claim: legacy history or an import."""
    return row.get("source") == "import" or legacy_history_row(legacy_row(row), owner)


def _title_rank(source: Any) -> int:
    """Title provenance rank: a pre-provenance (NULL) title ranks as the user's."""
    return 2 if source is None else _TITLE_RANK.get(str(source), 0)


def _ledger_states(conn: Connection, session_id: str, admission_states: tuple[str, ...]) -> set[str]:
    return {item["status"] for item in _rows(
        conn, "SELECT status FROM session_admissions WHERE target_session_id = %s AND status = ANY(%s) "
              "UNION SELECT status FROM worker_executions WHERE session_id = %s AND status IN ('registered', 'running', 'unknown')",
        (session_id, list(admission_states), session_id))}


def _refuse_states(states: set[str]) -> None:
    if "unknown" in states:
        raise AuthorityRefusal("unknown_execution")
    if states:
        raise AuthorityRefusal("session_busy")


def require_not_executing(conn: Connection, session_ids: list[str]) -> None:
    """A started or unknown admission or a live worker on any id refuses."""
    for sid in session_ids:
        _refuse_states(_ledger_states(conn, sid, ("started", "unknown")))


class PostgresSessionBindingRepository:
    """Local creation receipts and API bindings of the bound profile's schema."""

    def __init__(self, store: PostgresStore) -> None:
        self._store = store

    def _require_idle(self, conn: Connection, session_ids: list[str]) -> None:
        """Queued, started or unknown admissions and live workers refuse, then the live transcript leases of a row not
        ended by compression."""
        for sid in session_ids:
            _refuse_states(_ledger_states(conn, sid, ("queued", "started", "unknown")))
            if scalar(conn, "SELECT end_reason FROM sessions WHERE id = %s", (sid,)) != "compression":
                self._store.messages._refuse_live_leases(conn, sid, ("compaction", "turn"))

    # ── local reservation ──

    def reserve(self, *, epoch: int, binding: NewSessionBinding, client_request: Mapping[str, JsonValue] | None = None,
                owner: OwnerRule | None) -> ReservedBinding:
        request: Mapping[str, Any] = client_request if client_request is not None else {}
        if client_request is None and binding.policy.get("client_env_json") is not None:
            raise AuthorityRefusal("invalid_params")
        receipt = json.loads(canonical_json(_new_receipt(binding)))
        sid, route = receipt["session_id"], receipt["route"]
        with self._store.transaction() as conn:
            require_epoch(conn, epoch)
            old = _meta.get(conn, RECEIPT_PREFIX + sid, lock=True)
            if old is not None:
                try:
                    saved = decode(old)
                    same = all(saved[k] == receipt[k] for k in _IDENTITY)
                    same = same and same_create_retry(saved["policy"], receipt["policy"]["request_json"], request)
                except (ValueError, TypeError, KeyError) as exc:
                    raise AuthorityRefusal("storage_unavailable") from exc
                if not same:
                    raise AuthorityRefusal("invalid_params")
                return ReservedBinding(binding=to_binding(saved), created=False)
            if scalar(conn, "SELECT 1 FROM retired_sessions WHERE session_id = %s", (sid,)) is not None:
                raise AuthorityRefusal("not_found")
            exists = scalar(conn, "SELECT 1 FROM sessions WHERE id = %s", (sid,)) is not None
            if "legacy_session_id" not in receipt and exists:
                raise AuthorityRefusal("storage_unavailable")
            if route_exists(conn, route):
                raise AuthorityRefusal("admission_conflict")
            if "legacy_session_id" in receipt:
                self._bind_legacy_target(conn, receipt, owner)
            if not exists:
                self._insert_reserved_row(conn, binding, receipt)
            store_route(conn, route, receipt["entry"])
            _meta.put(conn, RECEIPT_PREFIX + sid, receipt)
        return ReservedBinding(binding=to_binding(receipt), created=True)

    def _insert_reserved_row(self, conn: Connection, binding: NewSessionBinding, receipt: dict[str, Any]) -> None:
        policy, entry = receipt["policy"], receipt["entry"]
        chat_id = bound_id(entry["origin"]["chat_id"])
        started = datetime.fromisoformat(entry["created_at"]).timestamp()
        execute(conn, "INSERT INTO sessions (id, source, created_source, user_id, session_key, chat_id, chat_type, model, cwd, "
                      "profile_name, origin_json, started_at) VALUES (%s, %s, %s, %s, %s, %s, 'dm', %s, %s, %s, %s, %s)",
                (receipt["session_id"], policy["source"], binding.created_source or None, receipt["principal_id"],
                 receipt["route"], chat_id, policy["model"], _row_cwd(policy), self._store.owner_profile,
                 canonical_json(entry["origin"]), started))

    def _bind_legacy_target(self, conn: Connection, receipt: dict[str, Any], owner: Optional[OwnerRule]) -> None:
        """Adopt a legacy local session as the receipt's target after the ownership, idle, lease and not-yet-bound
        checks: the row takes the receipt's route and origin and the legacy binding record is written."""
        target = receipt["legacy_session_id"]
        if target != receipt["session_id"]:
            raise AuthorityRefusal("storage_unavailable")
        row = _row(conn, "SELECT * FROM sessions WHERE id = %s FOR UPDATE", (target,))
        if row is None:
            raise AuthorityRefusal("not_found")
        validate_legacy_row(self._store.owner_profile, legacy_row(row), subject=receipt["principal_id"], rule=owner)
        imported = _meta.get(conn, HISTORY_PREFIX + target)
        if imported is not None and not binding_names(binding_text(imported), receipt["principal_id"], owner):
            raise ProofRejected("session", target)
        if _lineage.chain_step(conn, target) is not None:
            raise AuthorityRefusal("admission_conflict")
        self._require_idle(conn, [target])
        if scalar(conn, "SELECT 1 FROM session_turn_leases WHERE conversation_id = %s AND expires_at > %s",
                  (_leases.conversation_key(conn, target), time.time())) is not None:
            raise AuthorityRefusal("runtime_coordination_required")
        if _meta.has(conn, LEGACY_PREFIX + target):
            raise AuthorityRefusal("storage_unavailable")
        label = "desktop" if client_history_row(legacy_row(row), owner) else row["source"]
        execute(conn, "UPDATE sessions SET source = %s, session_key = %s, chat_id = %s, user_id = %s, chat_type = 'dm', "
                      "origin_json = %s WHERE id = %s",
                (label, receipt["route"], receipt["session_id"], receipt["principal_id"],
                 canonical_json(receipt["entry"]["origin"]), target))
        _meta.put(conn, LEGACY_PREFIX + target, {key: receipt[key] for key in ("session_id", "profile_id", "principal_id")}
                  | {"legacy_session_id": target})

    # ── lineage writes ──

    def adopt_untracked_compression(self, *, epoch: int, session_id: SessionId, expected: SessionBinding) -> None:
        with self._store.transaction() as conn:
            require_epoch(conn, epoch)
            saved = _meta.get(conn, RECEIPT_PREFIX + str(session_id), lock=True)
            if saved is None:
                raise TypeError("the JSON object must be str, bytes or bytearray, not NoneType")
            if to_binding(saved) != expected:
                return
            receipt = receipt_of(expected)
            adoptable = adoptable_compression(conn, receipt)
            if adoptable is None:
                return
            children, routed = adoptable
            receipt = with_adopted_compression(receipt, children)
            validate_local_lineage(conn, receipt)
            commit_local_target(conn, receipt, dict(routed, session_id=children[-1]))

    def reset_target(self, *, epoch: int, parent_session_id: SessionId, entry: Mapping[str, JsonValue]) -> None:
        parent_id = str(parent_session_id)
        with self._store.transaction() as conn:
            require_epoch(conn, epoch)
            parent = _row(conn, "SELECT * FROM sessions WHERE id = %s FOR UPDATE", (parent_id,))
            if parent is None or parent["end_reason"] == "compression":
                raise AuthorityRefusal("admission_conflict")
            saved = _meta.get(conn, RECEIPT_PREFIX + str(parent["chat_id"]), lock=True) if parent["chat_id"] else None
            if saved is None:
                raise AuthorityRefusal("storage_unavailable")
            receipt = json.loads(json.dumps(decode(saved)))
            logical_id = receipt["session_id"]
            if validate_local_lineage(conn, receipt) != parent_id:
                raise AuthorityRefusal("admission_conflict")
            require_not_executing(conn, list(dict.fromkeys([logical_id, parent_id])))
            child_id = str(entry["session_id"])
            if scalar(conn, "SELECT 1 FROM sessions WHERE id = %s", (child_id,)) is not None:
                raise PermanentStoreError("UNIQUE constraint failed: sessions.id")
            policy = receipt["policy"]
            self._store.lineage._insert_child(conn, parent, CompressionChild(
                session_id=SessionId(child_id), source=policy["source"], model=policy["model"],
                model_config={"_reset_from": parent_id}, cwd=policy["cwd"], profile_name=parent["profile_name"]))
            execute(conn, "UPDATE sessions SET ended_at = %s, end_reason = 'session_reset' WHERE id = %s", (time.time(), parent_id))
            bump_conversation_generation(conn, parent_id, "session_reset")
            advance_local_target(conn, parent_id, child_id, entry=json.loads(json.dumps(dict(entry))))
            execute(conn, "UPDATE sessions SET runtime_generation = runtime_generation + 1 WHERE id = %s", (logical_id,))

    def end_idle(self, *, epoch: int, session_id: SessionId, target_id: SessionId, reason: str) -> int:
        with self._store.transaction() as conn:
            require_epoch(conn, epoch)
            if scalar(conn, "SELECT 1 FROM session_admissions WHERE target_session_id = %s "
                            "AND status IN ('queued', 'started', 'unknown') LIMIT 1", (str(session_id),)) is not None:
                return 0
            if scalar(conn, "SELECT 1 FROM worker_executions WHERE session_id = ANY(%s) "
                            "AND status IN ('registered', 'running', 'unknown') LIMIT 1",
                      ([str(session_id), str(target_id)],)) is not None:
                return 0
            ended = execute(conn, "UPDATE sessions SET ended_at = %s, end_reason = %s WHERE id = %s AND ended_at IS NULL",
                            (time.time(), reason, str(target_id)))
            if ended:
                bump_conversation_generation(conn, str(target_id), reason)
            return 1 if ended else 0

    def release_handed_off_receipts(self, *, epoch: int, lanes: frozenset[str]) -> dict[str, tuple[str, str]]:
        with self._store.transaction() as conn:
            require_epoch(conn, epoch)
            moved = {owner: pair for owner, pair in self._moved_receipts(conn).items() if pair[1] in lanes}
            return self._hand_off_conversations(conn, moved)

    def _moved_receipts(self, conn: Connection) -> dict[str, tuple[str, str]]:
        """Owner -> (current target, its source) for receipts whose target row now carries another route."""
        found = _rows(conn, "SELECT substr(m.key, %s) AS owner, s.id, s.source FROM authority_meta m "
                            "JOIN sessions s ON s.id = m.value -> 'entry' ->> 'session_id' "
                            "WHERE starts_with(m.key, %s) AND jsonb_typeof(m.value) = 'object' "
                            "AND jsonb_typeof(m.value -> 'entry') = 'object' AND s.session_key IS NOT NULL "
                            "AND s.session_key IS DISTINCT FROM m.value ->> 'route' ORDER BY m.key COLLATE \"C\"",
                      (len(RECEIPT_PREFIX) + 1, RECEIPT_PREFIX))
        return {item["owner"]: (item["id"], item["source"]) for item in found}

    def _hand_off_conversations(self, conn: Connection, moved: dict[str, tuple[str, str]]) -> dict[str, tuple[str, str]]:
        """Retire each moved receipt and give its conversation's other segments the lane identity of its current
        target, the title following the head; a receipt with a malformed lineage stays."""
        handed: dict[str, tuple[str, str]] = {}
        columns = ", ".join(_LANE_IDENTITY)
        for owner, (target, source) in moved.items():
            saved = _meta.get(conn, RECEIPT_PREFIX + owner)
            try:
                receipt = decode(saved) if saved is not None else None
            except ValueError:
                receipt = None
            lineage = (receipt if isinstance(receipt, dict) else {}).get("lineage", [owner])
            if not isinstance(lineage, list) or not all(isinstance(sid, str) for sid in lineage):
                continue
            segments = list(dict.fromkeys([owner, *lineage]))
            execute(conn, f"UPDATE sessions SET ({columns}) = (SELECT {columns} FROM sessions WHERE id = %s) "
                          "WHERE id = ANY(%s) AND id <> %s", (target, segments, target))
            self._title_follows_head(conn, owner, self._segment_root(conn, segments, target))
            handed[owner] = (target, source)
        for owner in handed:
            _meta.delete(conn, RECEIPT_PREFIX + owner)
            _meta.delete(conn, LEGACY_PREFIX + owner)
        return handed

    @staticmethod
    def _segment_root(conn: Connection, segments: list[str], target: str) -> str:
        """The first row of ``target``'s ``/new`` segment."""
        if target not in segments:
            return target
        index = segments.index(target)
        while index > 0 and scalar(conn, "SELECT end_reason FROM sessions WHERE id = %s", (segments[index - 1],)) == "compression":
            index -= 1
        return segments[index]

    @staticmethod
    def _title_follows_head(conn: Connection, owner: str, head: str) -> None:
        """The conversation's name moves to its head over a title of lower provenance there."""
        row = _row(conn, "SELECT title, title_source, hidden FROM sessions WHERE id = %s", (owner,))
        if head == owner or row is None or not row["title"]:
            return
        kept = _row(conn, "SELECT title, title_source FROM sessions WHERE id = %s", (head,))
        if kept is None or (kept["title"] and _title_rank(kept["title_source"]) >= _title_rank(row["title_source"])):
            return
        execute(conn, "UPDATE sessions SET title = NULL WHERE id = ANY(%s)", ([owner, head],))
        execute(conn, "UPDATE sessions SET title = %s, title_source = %s, hidden = %s WHERE id = %s",
                (row["title"], row["title_source"], row["hidden"], head))

    # ── API bindings ──

    def bind_api(self, *, epoch: int, binding: NewApiBinding, gate: AdmissionGate = None,
                 walked_from: SessionId | None = None) -> None:
        sid, storage_source = str(binding.session_id), binding.storage_source
        with self._store.transaction() as conn:
            require_epoch(conn, epoch)
            raise_gate(gate)
            if scalar(conn, "SELECT 1 FROM retired_sessions WHERE session_id = %s", (sid,)) is not None:
                raise AuthorityRefusal("not_found")
            if walked_from and not self._keep_passed_fences(conn, str(walked_from), sid):
                raise AuthorityRefusal("chain_moved")
            saved = _meta.get(conn, API_BINDING_PREFIX + sid, lock=True)
            if saved is not None:
                stored = decode(saved)
                if stored.get("storage_source", "api_server") != storage_source:
                    raise ProofRejected("session", sid)
                if binding.declared_key and stored.get("declared_key") != binding.declared_key:
                    raise AuthorityRefusal("admission_conflict")
                return
            if binding.declared_key:
                self._reserve_key(conn, binding.declared_key, sid)
            self._bind_row(conn, binding)

    @staticmethod
    def _keep_passed_fences(conn: Connection, walked_from: str, session_id: str) -> bool:
        """Whether every ``api_generation_id(walked_from, n)`` before ``session_id`` is retired; each passed marker
        stops expiring."""
        generation = 0
        while (sid := api_generation_id(walked_from, generation)) != session_id:
            if not execute(conn, "UPDATE retired_sessions SET expires = false WHERE session_id = %s", (sid,)):
                return False
            generation += 1
        return True

    @staticmethod
    def _reserve_key(conn: Connection, key: str, session_id: str) -> None:
        holder = _meta.get(conn, API_DECLARED_PREFIX + key, lock=True)
        if holder is not None and holder != session_id:
            raise AuthorityRefusal("admission_conflict")
        _meta.put_new(conn, API_DECLARED_PREFIX + key, session_id)

    def _bind_row(self, conn: Connection, binding: NewApiBinding) -> None:
        """The row, route and receipt half of ``bind_api``."""
        sid, route, storage_source = str(binding.session_id), binding.route, binding.storage_source
        receipt: dict[str, Any] = json.loads(json.dumps(dict(binding.receipt)))
        row = _row(conn, "SELECT source, title, session_key FROM sessions WHERE id = %s FOR UPDATE", (sid,))
        if row is not None and row["source"] != storage_source:
            raise ProofRejected("session", sid)
        if storage_source == "bot_room":
            if row is not None and row["title"] != binding.title:
                raise AuthorityRefusal("admission_conflict")
            if scalar(conn, "SELECT 1 FROM sessions WHERE title = %s AND id <> %s", (binding.title, sid)) is not None:
                raise AuthorityRefusal("admission_conflict")
        legacy_key = binding.declared_key or binding.caller_key
        if row is not None and row["session_key"] not in (None, "", route):
            if not legacy_key or row["session_key"] != legacy_key:
                raise AuthorityRefusal("admission_conflict")
            self._reserve_key(conn, legacy_key, sid)
            receipt["declared_key"] = legacy_key
        existing = scalar(conn, "SELECT entry_json FROM gateway_routing WHERE scope = '' AND session_key = %s FOR UPDATE", (route,))
        held = json.loads(existing)["session_id"] if existing is not None else sid
        if held != sid:
            if _meta.has(conn, API_BINDING_PREFIX + str(held)):
                raise AuthorityRefusal("admission_conflict")
            logger.warning("API route of %s taken back from heal replacement %s", sid, held)
        if row is None:
            execute(conn, "INSERT INTO sessions (id, source, title, hidden, started_at) VALUES (%s, %s, %s, %s, %s)",
                    (sid, storage_source, binding.title, storage_source == "bot_room", binding.started_at))
        execute(conn, "UPDATE sessions SET session_key = %s, chat_id = %s, user_id = %s, chat_type = 'dm', origin_json = %s "
                      "WHERE id = %s", (route, sid, binding.origin["user_id"], canonical_json(dict(binding.origin)), sid))
        store_route(conn, route, dict(binding.entry))
        _meta.put(conn, API_BINDING_PREFIX + sid, json.loads(canonical_json(receipt)))

    def api_binding(self, session_id: SessionId) -> Optional[ApiBinding]:
        with self._store.transaction() as conn:
            saved = _meta.get(conn, API_BINDING_PREFIX + str(session_id))
        if saved is None:
            return None
        receipt = decode(saved)
        identity = receipt.get("room_identity")
        return ApiBinding(
            profile_id=receipt["profile_id"], session_id=SessionId(receipt["session_id"]), route=receipt["route"],
            entry=receipt["entry"], declared_key=receipt.get("declared_key"),
            storage_source=receipt.get("storage_source", "api_server"),
            room_identity=tuple(identity) if identity is not None else None,
            room_generation=int(receipt.get("room_generation", 0)))

    def declared_session(self, declared_key: str) -> Optional[SessionId]:
        with self._store.transaction() as conn:
            holder = _meta.get(conn, API_DECLARED_PREFIX + declared_key)
        return None if holder is None else SessionId(holder)

    # ── reads ──

    def target_has_history(self, session_id: str) -> bool:
        sid = bound_id(session_id)
        with self._store.transaction() as conn:
            return scalar(conn, "SELECT 1 FROM messages WHERE session_id = %s UNION ALL SELECT 1 FROM session_admissions "
                                "WHERE target_session_id = %s UNION ALL SELECT 1 FROM worker_executions WHERE session_id = %s "
                                "UNION ALL SELECT 1 FROM sessions WHERE parent_session_id = %s LIMIT 1",
                          (sid, sid, sid, sid)) is not None

    def receipt(self, session_id: SessionId) -> SessionBinding:
        with self._store.transaction() as conn:
            stored = _meta.get(conn, RECEIPT_PREFIX + str(session_id))
        if stored is None:
            raise AuthorityRefusal("storage_unavailable")
        return to_binding(stored)

    def _receipts(self) -> list[tuple[str, Any]]:
        with self._store.transaction() as conn:
            return _meta.scan(conn, RECEIPT_PREFIX)

    def local_session_ids(self) -> list[SessionId]:
        return [SessionId(key[len(RECEIPT_PREFIX):]) for key, _value in self._receipts()]

    def list_lineages(self) -> list[LocalLineage]:
        found: list[LocalLineage] = []
        for _key, raw in self._receipts():
            try:
                receipt = decode(raw)
            except (TypeError, ValueError):
                continue
            owner = receipt.get("session_id") if isinstance(receipt, dict) else None
            lineage = receipt.get("lineage") if isinstance(owner, str) else None
            if isinstance(lineage, list):
                found.append(LocalLineage(owner=SessionId(owner),
                                          lineage=tuple(SessionId(sid) for sid in lineage if isinstance(sid, str))))
        return found

    def binding_marks(self, session_id: SessionId, *, lineage_of: SessionId | None = None) -> BindingMarks:
        with self._store.transaction() as conn:
            policy = _meta.has(conn, RECEIPT_PREFIX + str(session_id))
            legacy = _meta.has(conn, LEGACY_PREFIX + str(session_id))
            owner = local_lineage_owner(conn, str(lineage_of)) if lineage_of and (policy or legacy) else None
        return BindingMarks(policy=policy, legacy=legacy, lineage_owner=None if owner is None else SessionId(owner))

    def lineage_owner(self, session_id: SessionId) -> SessionId:
        with self._store.transaction() as conn:
            return SessionId(local_lineage_owner(conn, str(session_id)))

    def conversation_ids(self, session_id: SessionId) -> list[SessionId]:
        with self._store.transaction() as conn:
            return [SessionId(sid) for sid in local_conversation_ids(conn, str(session_id))]

    def validate_lineage(self, receipt: SessionBinding) -> SessionId:
        with self._store.transaction() as conn:
            return SessionId(validate_local_lineage(conn, receipt_of(receipt)))

    def compression_adoptable(self, receipt: SessionBinding) -> bool:
        with self._store.transaction() as conn:
            return adoptable_compression(conn, receipt_of(receipt)) is not None

    def restorable(self, receipt: SessionBinding) -> SessionBinding:
        judged = receipt_of(receipt)
        with self._store.transaction() as conn:
            adoptable = adoptable_compression(conn, judged)
            if adoptable is not None:
                judged = with_adopted_compression(judged, adoptable[0])
            validate_local_lineage(conn, judged)
        return to_binding(judged)

    def legacy_adoption_scan(self, owner: OwnerRule) -> LegacyAdoptionScan:
        refused = sorted(not_client_labels(owner))
        with self._store.transaction() as conn:
            ids = [item["id"] for item in _rows(
                conn, "SELECT id FROM sessions WHERE source = ANY(%s) OR (source <> ALL(%s) AND coalesce(session_key, '') = '' "
                      "AND coalesce(chat_id, '') = '' AND coalesce(origin_json, '') = '') ORDER BY started_at, id COLLATE \"C\"",
                (sorted(LEGACY_LOCAL_SOURCES), refused))]
            marked = frozenset(key[len(prefix):] for prefix in (RECEIPT_PREFIX, LEGACY_PREFIX) for key, _value in _meta.scan(conn, prefix))
        return LegacyAdoptionScan(session_ids=tuple(SessionId(sid) for sid in ids), marked=marked)

    def history_binding(self, session_id: SessionId) -> Optional[str]:
        with self._store.transaction() as conn:
            return binding_text(_meta.get(conn, HISTORY_PREFIX + str(session_id)))

    def handoff_head(self, session_id: SessionId) -> tuple[Optional[SessionId], SessionId]:
        with self._store.transaction() as conn:
            owner = local_lineage_owner(conn, str(session_id))
            saved = _meta.get(conn, RECEIPT_PREFIX + owner)
            target = _policy_target(saved) if saved is not None else None
            start = str(target or session_id)
            head = _lineage.compression_tip(conn, start) or start
        return (SessionId(owner) if target else None), SessionId(head)

    def boot_session_ids(self) -> list[SessionId]:
        with self._store.transaction() as conn:
            found = _rows(conn, """
                SELECT substr(m.key, %(start)s) AS owner FROM authority_meta m
                LEFT JOIN sessions s ON jsonb_typeof(m.value) = 'object' AND jsonb_typeof(m.value -> 'entry') = 'object'
                                    AND s.id = m.value -> 'entry' ->> 'session_id'
                WHERE starts_with(m.key, %(prefix)s)
                  AND (s.id IS NULL OR s.ended_at IS NULL OR s.end_reason = 'compression'
                       OR EXISTS (SELECT 1 FROM session_admissions a WHERE a.target_session_id = substr(m.key, %(start)s)
                                  AND a.status IN ('queued', 'started', 'unknown'))
                       OR EXISTS (SELECT 1 FROM authority_meta c WHERE c.key = %(close)s || substr(m.key, %(start)s)))
                ORDER BY m.key COLLATE "C"
            """, {"start": len(RECEIPT_PREFIX) + 1, "prefix": RECEIPT_PREFIX, "close": VIEWER_CLOSE_PREFIX})
        return [SessionId(item["owner"]) for item in found]

    def authorize_history(self, *, session_id: SessionId, profile_id: str, subject: str, may_create: bool,
                          native_transport: bool | None = None, owner: OwnerRule | None) -> bool:
        sid = str(session_id)
        with self._store.transaction() as conn:
            row = _row(conn, "SELECT * FROM sessions WHERE id = %s", (sid,))
            saved = _meta.get(conn, HISTORY_PREFIX + sid)
        if native_transport is not None:
            self._require_history_claim(saved is not None, row, sid, subject, native_transport, owner)
        if row is None or _routed(row):
            return False
        if saved is not None:
            if not binding_names(binding_text(saved), subject, owner):
                raise ProofRejected("session", sid)
            return True
        if not history_row(row, owner):
            return False
        if _owner_user(row) and not same_owner(row["user_id"], subject, owner):
            raise ProofRejected("session", sid)
        if not may_create:
            raise ProofRejected("session", sid)
        return True

    def _require_history_claim(self, claimed: bool, row: Optional[Mapping[str, Any]], sid: str, subject: str,
                               native_transport: bool, owner: Optional[OwnerRule]) -> None:
        """An unrouted, unclaimed row outside this subject's own history refuses ``ProofRejected`` unless
        ``native_transport`` holds."""
        if row is None or _routed(row) or claimed:
            return
        own_profile = row.get("profile_name") in (None, "", self._store.owner_profile or "default")
        if own_profile and _owner_user(row) and same_owner(row["user_id"], subject, owner):
            return
        if not native_transport:
            raise ProofRejected("session", sid)


register_repository("bindings", PostgresSessionBindingRepository, {"receipted_writes": True})
