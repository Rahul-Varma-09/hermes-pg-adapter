"""``AdmissionRepository`` on Postgres: the fail-closed admission lifecycle.

A live admission is a ``session_admissions`` row (``lineage`` and ``payload`` jsonb, ``digest`` over the canonical
payload); a deleted session's admissions survive as ``admission_terminals`` tombstones of the same shape. Settled
results, queued-replacement bindings, API target settings and the media and spillover retirement candidates are
``authority_meta`` values. Each verb is one transaction: it takes the session row's lock where it decides on that
session's queue, reads and locks only the admission rows it decides on, and writes back what changed. A session's
execution generation and revision are its ``runtime_generation`` / ``runtime_revision`` columns.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
import uuid
from dataclasses import fields
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Optional, Sequence

from psycopg import Connection
from psycopg.types.json import Jsonb

from persistence.authority.admissions import (
    ANY_HEAD, ActiveReplace, AdmissionGate, AdmissionRecord, AdmissionResult, ApiTargetClaim, CapturedResult,
    CarrierRewind, ClaimedAdmission, QueuedFold, QueuedReplacement, RetryRewriteRefused, TranscriptRewrite, TurnBoundary,
    UnheldMedia, raise_gate, steer_replayed,
)
from persistence.authority.vocabulary import (
    ADMISSION_QUEUE_MAX, ADMITTED_REQUEST_CONFLICT, STEERING, admission_fingerprint, retry_input,
)
from persistence.conversation.messages import NewMessage, ReplaceMode
from persistence.core.canonical import canonical_json
from persistence.core.errors import (
    STORE_FAILURES, AuthorityRefusal, CapabilityUnsupported, FenceStale, NotFound, PermanentStoreError,
    StaleFenceKind, UnknownExecution,
)
from persistence.core.identity import MessageId
from session_rules.api_payloads import compacted_api_payload
from session_rules.automation import folded_payload
from session_rules.media import hosted_document_references
from session_rules.turns import FAILED_TURN_DISPLAY_KIND, PARTIAL_FAILED_TURN_NOTICE

from .._backend import PostgresStore, register_repository
from .._sql import execute, row as _row, rows as _rows, scalar
from ..conversation import _lineage, _policy as policy, _transcript as transcript_store
from ..conversation._retirement import MEDIA_PREFIX, SPILLOVER_PREFIX
from . import _meta
from ._bindings_lineage import local_physical_target, runtime_lineage
from .runtime import encode, require_epoch, text

REPLACEMENT_PREFIX = "gateway.queued_replacement.v1."
API_SETTINGS_PREFIX = "gateway.api.settings.v1."
_PENDING = ("queued", "started", "unknown")
_STORED = "_stored"
# The columns an admission write can change.
_MUTABLE = ("target_session_id", "lineage", "payload", "digest", "status", "outcome", "owner_epoch", "generation")


# ── row helpers ──

def bound_id(value: Any) -> Any:
    """The text an id compares as (an int as its digits, a float as its TEXT spelling); an unbindable type is a
    ``PermanentStoreError``."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        value = int.__int__(value)
        if not -(2**63) <= value < 2**63:
            raise OverflowError("Python int too large to convert to SQLite INTEGER")
        return str(value)
    if isinstance(value, float):
        return policy.real_as_text(value)
    raise PermanentStoreError(f"Error binding parameter: type '{type(value).__name__}' is not supported")


def digest(session_id: str, payload: Any, intent: str) -> str:
    """The admission digest: target, payload (without the owner's client stamp) and intent."""
    return admission_fingerprint(canonical_target=session_id, payload={"input": retry_input(payload), "intent": intent})


def stale(entity: str, entity_id: Any) -> FenceStale:
    return FenceStale(StaleFenceKind.STALE_GENERATION, entity, str(entity_id))


def record(row: Mapping[str, Any]) -> AdmissionRecord:
    """The row without its digest and lineage, the payload a detached copy."""
    return AdmissionRecord(
        seq=row["seq"], admission_id=row["admission_id"], request_id=row["request_id"], principal_id=row["principal_id"],
        target_session_id=row["target_session_id"], payload=copy.deepcopy(row["payload"]), intent=row["intent"],
        status=row["status"], outcome=row.get("outcome"), owner_epoch=row.get("owner_epoch"), generation=row.get("generation"))


def _result_key(admission_id: str) -> tuple[str, str]:
    return ("admission_result", admission_id)


def _replacement_binding(principal_id: str, session_id: str, request_id: str) -> str:
    return REPLACEMENT_PREFIX + admission_fingerprint(canonical_target=session_id,
                                                      payload={"principal": principal_id, "request": request_id})


def live_row(conn: Connection, admission_id: Any, *, lock: bool = True) -> Optional[dict[str, Any]]:
    return _row(conn, "SELECT * FROM session_admissions WHERE admission_id = %s" + (" FOR UPDATE" if lock else ""),
                (bound_id(admission_id),))


def tombstone(conn: Connection, admission_id: Any) -> Optional[dict[str, Any]]:
    stored = scalar(conn, "SELECT row FROM admission_terminals WHERE admission_id = %s", (bound_id(admission_id),))
    return dict(stored) if stored is not None else None


def admission(conn: Connection, admission_id: Any, *, lock: bool = True) -> dict[str, Any]:
    """The live row, else its tombstone, else ``not_found``."""
    found = live_row(conn, admission_id, lock=lock) or tombstone(conn, admission_id)
    if found is None:
        raise AuthorityRefusal("not_found")
    return found


def save(conn: Connection, row: Mapping[str, Any]) -> None:
    """Write a live row's mutable columns back."""
    values = [Jsonb(row[c]) if c in ("lineage", "payload") else row[c] for c in _MUTABLE]
    execute(conn, f"UPDATE session_admissions SET {', '.join(f'{c} = %s' for c in _MUTABLE)} WHERE admission_id = %s",
            (*values, row["admission_id"]))


def session_rows(conn: Connection, session_id: Any, *, lock: bool = True) -> list[dict[str, Any]]:
    return _rows(conn, "SELECT * FROM session_admissions WHERE target_session_id = %s ORDER BY seq"
                 + (" FOR UPDATE" if lock else ""), (bound_id(session_id),))


def require_session(conn: Connection, session_id: Any, *, lock: bool = False) -> dict[str, Any]:
    found = _row(conn, "SELECT * FROM sessions WHERE id = %s" + (" FOR UPDATE" if lock else ""), (bound_id(session_id),))
    if found is None:
        raise AuthorityRefusal("not_found")
    return found


def runtime_generation(conn: Connection, session_id: str) -> int:
    return int(scalar(conn, "SELECT runtime_generation FROM sessions WHERE id = %s", (session_id,)) or 0)


def bump_revision(conn: Connection, session_id: str) -> None:
    execute(conn, "UPDATE sessions SET runtime_revision = runtime_revision + 1 WHERE id = %s", (session_id,))


def retire_workers(conn: Connection, row: Mapping[str, Any], owner_epoch: Any) -> None:
    """The workers at the row's generation and ``owner_epoch`` on its target or its lineage tail go terminal."""
    lineage = row.get("lineage") or [row["target_session_id"]]
    execute(conn, "UPDATE worker_executions SET status = 'terminal' WHERE session_id = ANY(%s) AND generation = %s "
                  "AND owner_epoch = %s", (list({row["target_session_id"], lineage[-1]}), row["generation"], owner_epoch))


def requeue_transfer(conn: Connection, admission_id: Any, epoch: int, generation: Any) -> bool:
    """Requeue a steering transfer still terminal at this epoch and generation; whether it was requeued."""
    return bool(execute(conn, "UPDATE session_admissions SET status = 'queued', outcome = NULL, generation = NULL "
                              "WHERE admission_id = %s AND status = 'terminal' AND outcome = %s AND owner_epoch = %s "
                              "AND generation = %s", (bound_id(admission_id), STEERING, epoch, generation)))


def settle_transfers(conn: Connection, *, epoch: int, generation: Any, transfers: Optional[Mapping[str, bool]],
                     outcome: str) -> None:
    """An undelivered steering transfer is requeued; a delivered one's outcome becomes ``steered`` when the turn
    completed, else ``interrupted``."""
    for admission_id, delivered in (transfers or {}).items():
        if not delivered:
            requeue_transfer(conn, admission_id, epoch, generation)
            continue
        execute(conn, "UPDATE session_admissions SET outcome = %s WHERE admission_id = %s AND status = 'terminal' "
                      "AND outcome = %s AND owner_epoch = %s AND generation = %s",
                ("steered" if outcome == "completed" else "interrupted", bound_id(admission_id), STEERING, epoch, generation))


def chain_tip(conn: Connection, session_id: str) -> str:
    """The tip of the compression continuation walk, ``admission_conflict`` on a cycle or past 100 steps."""
    if _lineage.lineage_row(conn, session_id) is None:
        raise AuthorityRefusal("not_found")
    chain = [session_id]
    for _ in range(100):
        child = _lineage.chain_step(conn, chain[-1])
        if child is None:
            return chain[-1]
        if child in chain:
            raise AuthorityRefusal("admission_conflict")
        chain.append(child)
    raise AuthorityRefusal("admission_conflict")


def turn_target(conn: Connection, session_id: str) -> str:
    """The closer's transcript: the compression tip of the local physical target."""
    return chain_tip(conn, local_physical_target(conn, session_id))


# ── stored results and the discarded-turn closer ──

def _reanchor(messages: list[Any], user_message: Any) -> int:
    fallback = -1
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if not (isinstance(msg, dict) and msg.get("role") == "user"):
            continue
        if msg.get("content") == user_message:
            return i
        view = policy.user_originated_turn_view(msg)
        if view is None:
            continue
        if view.get("content") == user_message:
            return i
        if fallback < 0:
            fallback = i
    return fallback


def compact_result(result: Mapping[str, Any], *, user_message: Any = None) -> dict[str, Any]:
    """Keep replay and accounting data and this turn's output, never cumulative history."""
    out = dict(result)
    value = out.get("result")
    if not isinstance(value, dict):
        return out
    value = dict(value)
    value.pop("tools", None)
    messages = value.get("messages")
    if isinstance(messages, list) and not value.get("_compressed") and not value.get("_messages_are_turn_suffix"):
        anchor = value.get("current_turn_user_idx")
        if not (type(anchor) is int and 0 <= anchor < len(messages) and isinstance(messages[anchor], dict)
                and messages[anchor].get("role") == "user"):
            anchor = _reanchor(messages, user_message)
        value["messages"] = messages[anchor + 1:]
        if anchor >= 0 and isinstance(messages[anchor], dict) and messages[anchor].get("api_content"):
            value["_turn_user_row"] = {k: messages[anchor][k] for k in ("role", "content", "api_content")}
        value["_messages_are_turn_suffix"] = True
        value.pop("current_turn_user_idx", None)
    out["result"] = value
    return out


def encode_result(result: Any) -> Any:
    """Store the reply once: the closing assistant message's content becomes null with a ``reply_ref``."""
    if not isinstance(result, dict):
        return result
    if _STORED in result:
        return {**result, _STORED: {"v": 1, "raw": result[_STORED]}}
    value = result.get("result")
    if not isinstance(value, dict) or not isinstance(value.get("messages"), list):
        return result
    reply, messages = value.get("final_response"), list(value["messages"])
    last = next((i for i in range(len(messages) - 1, -1, -1)
                 if isinstance(messages[i], dict) and messages[i].get("role") == "assistant"), None)
    if not (isinstance(reply, str) and reply and last is not None and messages[last].get("content") == reply):
        return result
    messages[last] = {**messages[last], "content": None}
    return {**result, "result": {**value, "messages": messages}, _STORED: {"v": 1, "reply_ref": last}}


def load_result(text_value: str) -> Any:
    """Decode a stored result; only ``encode_result``'s exact envelope is decoded."""
    result = json.loads(text_value)
    if not isinstance(result, dict) or not isinstance(envelope := result.get(_STORED), dict):
        return result
    if envelope.keys() == {"v", "raw"} and type(envelope["v"]) is int and envelope["v"] == 1:
        result[_STORED] = envelope["raw"]
        return result
    value, ref = result.get("result"), envelope.get("reply_ref")
    messages = value.get("messages") if isinstance(value, dict) else None
    if not (envelope.keys() == {"v", "reply_ref"} and type(envelope["v"]) is int and envelope["v"] == 1 and type(ref) is int
            and isinstance(messages, list) and 0 <= ref < len(messages) and isinstance(value.get("final_response"), str)
            and isinstance(messages[ref], dict) and messages[ref].get("role") == "assistant"
            and messages[ref].get("content", 0) is None):
        return result
    messages[ref] = {k: value["final_response"] if k == "content" else v for k, v in messages[ref].items()}
    del result[_STORED]
    return result


def close_discarded_turn(conn: Connection, target: str, notice: Optional[str] = None, *, replace_after: Any = None,
                         display_metadata: Optional[Mapping[str, Any]] = None) -> bool:
    """Close a discarded turn on the physical transcript ``target`` like a failed turn."""
    from dataclasses import replace

    live = [row for row in transcript_store.load(conn, [target], lock=True)
            if row.visibility.name == "ACTIVE" and row.message.role not in ("session_meta", "system")]
    tail = max(live, key=lambda row: row.id) if live else None
    if (notice is not None and type(replace_after) is int and tail is not None and tail.message.role == "assistant"
            and tail.message.display_kind == FAILED_TURN_DISPLAY_KIND and tail.id > replace_after):
        tail.message = replace(tail.message, content=notice, display_metadata=dict(display_metadata) if display_metadata else None)
        transcript_store.save(conn, tail)
        return True
    open_tail = tail is not None and (tail.message.role == "user" or (tail.message.role == "assistant" and bool(tail.message.tool_calls)))
    if notice is None and not open_tail:
        return False
    boundary = NewMessage(role="assistant", content=notice or PARTIAL_FAILED_TURN_NOTICE, timestamp=time.time(),
                          display_kind=FAILED_TURN_DISPLAY_KIND, display_metadata=dict(display_metadata) if display_metadata else None)
    transcript_store.insert_rows(conn, target, [boundary])
    return True


# ── reads helpers ──

def _api_turn(payload: Any) -> Any:
    return payload.get("api_turn_v1") if isinstance(payload, dict) else None


def _compaction_candidate(row: Mapping[str, Any]) -> bool:
    if row["status"] != "terminal" or row["principal_id"] != "api":
        return False
    data = _api_turn(row["payload"])
    if not isinstance(data, dict):
        return False
    inline = "media" in data and "data:image/" in canonical_json(row["payload"]).lower()
    history = isinstance(data.get("history"), list) and not str(row["request_id"]).lower().startswith("responses:")
    return inline or history


def _webhook_source(payload: Any) -> Optional[Mapping[str, Any]]:
    native = payload.get("native_text_v1") if isinstance(payload, dict) else None
    source = native.get("source") if isinstance(native, dict) else None
    if not isinstance(source, dict) or source.get("platform") != "webhook" or native.get("automation") is not None:
        return None
    return source


def _media(payload: Any, holder: str) -> list[Any]:
    value = payload.get(holder) if isinstance(payload, dict) else None
    media = value.get("media") if isinstance(value, dict) else None
    return list(media) if isinstance(media, list) else []


def _has_media(payload: Any, holder: str) -> bool:
    value = payload.get(holder) if isinstance(payload, dict) else None
    return isinstance(value, dict) and "media" in value


def _paths(references: Sequence[Any]) -> set[str]:
    try:
        return {reference["path"] for reference in references}
    except (KeyError, TypeError) as exc:
        raise PermanentStoreError(f"admission media reference is corrupt: {exc!r}") from exc


def _new_message(message: Mapping[str, Any]) -> NewMessage:
    known = {f.name for f in fields(NewMessage)}
    values = {key: value for key, value in message.items() if key in known}
    if isinstance(values.get("tool_calls"), list):
        values["tool_calls"] = tuple(values["tool_calls"])
    elif values.get("tool_calls") is None:
        values.pop("tool_calls", None)
    return NewMessage(**values)


class PostgresAdmissionRepository:
    """Admissions of the bound profile's schema."""

    def __init__(self, store_handle: PostgresStore) -> None:
        self._store = store_handle

    # ── admit ──

    def _tombstone_replay(self, conn: Connection, epoch: Any, principal_id: str, session_id: str, request_id: str,
                          payload: Any, intent: str) -> Optional[AdmissionRecord]:
        """The stored tombstone replay of a retired target past the input checks: ``None`` for a live target."""
        require_epoch(conn, epoch)
        if scalar(conn, "SELECT 1 FROM retired_sessions WHERE session_id = %s", (session_id,)) is None:
            return None
        saved = scalar(conn, "SELECT row FROM admission_terminals WHERE principal_id = %s AND target_session_id = %s "
                             "AND request_id = %s", (principal_id, session_id, request_id))
        if saved is None:
            raise AuthorityRefusal("not_found")
        if saved["digest"] != digest(session_id, payload, intent):
            raise AuthorityRefusal("admission_conflict", **ADMITTED_REQUEST_CONFLICT)
        return record(saved)

    def retired_replay(self, *, epoch: int, principal_id: str, session_id: str, request_id: str,
                       payload: Mapping[str, Any], intent: str = "queue") -> Optional[AdmissionRecord]:
        for value in (principal_id, session_id, request_id):
            text(value)
        canonical = encode(payload)
        with self._store.transaction() as conn:
            return self._tombstone_replay(conn, epoch, principal_id, session_id, request_id, canonical, intent)

    def admit(self, *, epoch: int, principal_id: str, session_id: str, request_id: str, payload: Mapping[str, Any],
              intent: str = "queue", gate: AdmissionGate = None, replaces: Optional[QueuedReplacement] = None,
              supersedes: Optional[str] = None, api_target: Optional[ApiTargetClaim] = None,
              rewrite: Optional[TranscriptRewrite] = None, hosted: Any = None,
              fold: Optional[QueuedFold] = None) -> tuple[AdmissionRecord, bool]:
        for value in (principal_id, session_id, request_id):
            text(value)
        if intent not in ("queue", "steer", "redirect"):
            raise AuthorityRefusal("invalid_params")
        canonical = encode(payload)
        with self._store.transaction() as conn:
            retired = self._tombstone_replay(conn, epoch, principal_id, session_id, request_id, canonical, intent)
            if retired is not None:
                return retired, False
            row_digest = digest(session_id, canonical, intent)
            require_session(conn, session_id, lock=True)
            queue = session_rows(conn, session_id)
            old = next((row for row in queue if (row["principal_id"], row["request_id"]) == (principal_id, request_id)), None)
            if old is not None:
                if old["digest"] != row_digest:
                    raise AuthorityRefusal("admission_conflict", **ADMITTED_REQUEST_CONFLICT)
                if replaces is not None:
                    saved = _meta.get(conn, _replacement_binding(principal_id, session_id, request_id))
                    if saved is None or json.loads(saved) != replaces.admission_id:
                        raise AuthorityRefusal("admission_conflict")
                return record(old), False
            if sum(1 for row in queue if row["status"] == "queued") >= ADMISSION_QUEUE_MAX:
                raise AuthorityRefusal("queue_full")
            raise_gate(gate)
            self._run_extras(conn, principal_id, session_id, request_id, replaces, supersedes, api_target, rewrite, hosted, fold)
            seq = int(scalar(conn, "SELECT nextval(pg_get_serial_sequence('session_admissions', 'seq'))"))
            row = {"seq": seq, "admission_id": uuid.uuid4().hex, "request_id": request_id, "principal_id": principal_id,
                   "target_session_id": session_id, "lineage": [session_id], "payload": canonical, "digest": row_digest,
                   "intent": intent, "status": "queued", "outcome": None, "owner_epoch": epoch, "generation": None}
            execute(conn, "INSERT INTO session_admissions (seq, admission_id, request_id, principal_id, target_session_id, "
                          "lineage, payload, digest, intent, status, outcome, owner_epoch, generation) "
                          "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (seq, row["admission_id"], request_id, principal_id, session_id, Jsonb(row["lineage"]), Jsonb(canonical),
                     row_digest, intent, "queued", None, epoch, None))
            return record(row), True

    def _run_extras(self, conn: Connection, principal_id: str, session_id: str, request_id: str,
                    replaces: Optional[QueuedReplacement], supersedes: Optional[str], api_target: Optional[ApiTargetClaim],
                    rewrite: Optional[TranscriptRewrite], hosted: Any, fold: Optional[QueuedFold]) -> None:
        if replaces is not None:
            replaced = admission(conn, replaces.admission_id)
            if replaced["status"] != "queued":
                raise AuthorityRefusal("admission_started")
            replaced.update(status="terminal", outcome="cancelled")
            save(conn, replaced)
            _meta.put(conn, _replacement_binding(principal_id, session_id, request_id), json.dumps(replaces.admission_id))
        if supersedes is not None:
            # The replaced tick must still be queued on this target.
            row = live_row(conn, supersedes)
            if row is None or row["target_session_id"] != session_id or row["status"] != "queued":
                raise stale("admission", supersedes)
            row.update(status="terminal", outcome="cancelled")
            save(conn, row)
        if api_target is not None:
            self._claim_api_target(conn, session_id, request_id, api_target)
        if rewrite is not None:
            self._rewrite(conn, rewrite)
        if hosted is not None:
            self._hosted_check(conn, hosted)
        if fold is not None:
            # The folded completion must still be the queued tail on this target, but for the skipped ticks.
            row = live_row(conn, fold.admission_id)
            later = [] if row is None else _rows(conn, "SELECT admission_id FROM session_admissions WHERE target_session_id = %s "
                                                       "AND status = 'queued' AND seq > %s", (session_id, row["seq"]))
            if (row is None or row["target_session_id"] != session_id or row["status"] != "queued"
                    or any(other["admission_id"] not in fold.skip_ids for other in later)):
                raise stale("admission", fold.admission_id)
            row.update(status="terminal", outcome="cancelled", payload=folded_payload(copy.deepcopy(row["payload"])))
            save(conn, row)

    def _api_tombstones(self, conn: Connection, request_id: Any) -> list[dict[str, Any]]:
        return [dict(item["row"]) for item in _rows(conn, "SELECT row FROM admission_terminals WHERE principal_id = 'api' "
                                                          "AND request_id = %s", (bound_id(request_id),))]

    def _claim_api_target(self, conn: Connection, session_id: str, request_id: str, claim: ApiTargetClaim) -> None:
        """An API admission of the request on another target, or with a client key its tombstone there, refuses."""
        if scalar(conn, "SELECT 1 FROM session_admissions WHERE principal_id = 'api' AND request_id = %s "
                        "AND target_session_id <> %s LIMIT 1", (request_id, session_id)) is not None:
            raise AuthorityRefusal("admission_conflict")
        if claim.client_key and any(row["target_session_id"] != session_id for row in self._api_tombstones(conn, request_id)):
            raise AuthorityRefusal("admission_conflict")
        _meta.put(conn, API_SETTINGS_PREFIX + session_id, canonical_json(dict(claim.settings)))

    def _rewrite(self, conn: Connection, rewrite: TranscriptRewrite) -> None:
        """The ``/retry`` rewrite in the admission's transaction: refusals and store failures keep their class, any
        other failure is ``RetryRewriteRefused``."""
        messages = self._store.messages
        try:
            if isinstance(rewrite, CarrierRewind):
                messages._rewind_rows(conn, rewrite.session_id, int(rewrite.target_row_id),
                                      tuple(MessageId(i) for i in rewrite.expected_active_ids),
                                      expected_content=rewrite.expected_target_content, preserve_handoff=True)
            elif isinstance(rewrite, ActiveReplace):
                active = tuple(item["id"] for item in _rows(conn, "SELECT id FROM messages WHERE session_id = %s AND active "
                                                                  "ORDER BY id", (rewrite.session_id,)))
                if active != tuple(rewrite.expected_active_ids):
                    raise RuntimeError("active transcript changed before the replacement could be persisted")
                messages._replace_on(conn, rewrite.session_id, [_new_message(m) for m in rewrite.messages],
                                     ReplaceMode.PRESERVE_EXISTING_ARCHIVE, reject_active_turn_lease=True)
        except (AuthorityRefusal, *STORE_FAILURES):
            raise
        except Exception as exc:
            raise RetryRewriteRefused(rewrite.session_id, str(exc)) from exc

    def _hosted_check(self, conn: Connection, guard: Any) -> None:
        check = getattr(getattr(self._store, "hosted_rooms", None), "hosted_member_check", None)
        if check is None:
            raise CapabilityUnsupported("hosted_member_check")
        if not check(guard, operation="submit"):
            raise AuthorityRefusal("permission_denied")

    # ── the lifecycle writes ──

    def _worker_states(self, conn: Connection, session_id: str) -> set[str]:
        """The live worker statuses on the session's runtime lineage."""
        lineage = sorted(runtime_lineage(conn, session_id))
        return {item["status"] for item in _rows(
            conn, "SELECT status FROM worker_executions WHERE session_id = ANY(%s) AND status IN ('registered', 'running', 'unknown')",
            (lineage,))}

    def claim(self, *, epoch: int, session_id: str, gate: AdmissionGate = None,
              head: Any = ANY_HEAD) -> ClaimedAdmission | Literal[False] | None:
        with self._store.transaction() as conn:
            require_epoch(conn, epoch)
            raise_gate(gate)
            require_session(conn, session_id, lock=True)
            queue = session_rows(conn, session_id)
            blocked = [row for row in queue if row["status"] in ("started", "unknown")]
            if any(row["status"] == "unknown" for row in blocked):
                raise UnknownExecution("admission", str(session_id))
            workers = self._worker_states(conn, bound_id(session_id))
            if "unknown" in workers:
                raise UnknownExecution("worker_execution", str(session_id))
            if blocked or workers:
                return None
            first = next((row for row in queue if row["status"] == "queued"), None)
            if head is not ANY_HEAD and (first["admission_id"] if first is not None else None) != head:
                return False
            if first is None:
                return None
            queued_owner_epoch = first["owner_epoch"]
            generation = int(scalar(conn, "UPDATE sessions SET runtime_generation = runtime_generation + 1 WHERE id = %s "
                                          "RETURNING runtime_generation", (bound_id(session_id),)))
            first.update(status="started", owner_epoch=epoch, generation=generation)
            save(conn, first)
            mark = int(scalar(conn, "SELECT coalesce(max(id), 0) FROM messages"))
            base = record(first)
            return ClaimedAdmission(**{f.name: getattr(base, f.name) for f in fields(AdmissionRecord)},
                                    transcript_mark=mark, queued_owner_epoch=queued_owner_epoch)

    def return_claim(self, *, epoch: int, admission_id: str, generation: int,
                     queued_owner_epoch: Optional[int]) -> AdmissionRecord:
        with self._store.transaction() as conn:
            require_epoch(conn, epoch)
            row = admission(conn, admission_id)
            if row["status"] == "queued" and row["generation"] is None:
                return record(row)
            target = row["target_session_id"]
            targets = list({target, (row.get("lineage") or [target])[-1]})
            bound_worker = type(generation) is int and scalar(
                conn, "SELECT 1 FROM worker_executions WHERE owner_epoch = %s AND generation = %s AND session_id = ANY(%s) LIMIT 1",
                (epoch, generation, targets)) is not None
            bound_steer = type(generation) is int and scalar(
                conn, "SELECT 1 FROM session_admissions WHERE status = 'terminal' AND outcome = %s AND owner_epoch = %s "
                      "AND generation = %s AND target_session_id = %s LIMIT 1", (STEERING, epoch, generation, target)) is not None
            if (row["status"] != "started" or row["owner_epoch"] != epoch or type(generation) is not int
                    or row["generation"] != generation or runtime_generation(conn, target) != generation
                    or bound_worker or bound_steer):
                raise stale("admission", admission_id)
            row.update(status="queued", owner_epoch=queued_owner_epoch, generation=None)
            save(conn, row)
            execute(conn, "UPDATE sessions SET runtime_generation = %s WHERE id = %s", (generation - 1, target))
            return record(row)

    def _store_result(self, conn: Connection, row: Mapping[str, Any], result: Any) -> None:
        user_message = row["payload"].get("text") if isinstance(row["payload"], dict) else None
        _meta.put(conn, _result_key(row["admission_id"]), encode(encode_result(compact_result(result, user_message=user_message))))

    def settle(self, *, epoch: int, admission_id: str, generation: int, outcome: str,
               result: Optional[Mapping[str, Any]] = None, transfers: Optional[Mapping[str, bool]] = None,
               boundary: Optional[TurnBoundary] = None) -> AdmissionRecord:
        if outcome not in ("completed", "interrupted", "rejected", "failed"):
            raise AuthorityRefusal("invalid_params")
        with self._store.transaction() as conn:
            require_epoch(conn, epoch)
            row = admission(conn, admission_id)
            target = row["target_session_id"]
            require_session(conn, target)
            if (row["status"] != "started" or row["owner_epoch"] != epoch or type(generation) is not int
                    or row["generation"] != generation or runtime_generation(conn, target) != generation):
                raise stale("admission", admission_id)
            retire_workers(conn, row, epoch)
            if boundary is not None:
                close_discarded_turn(conn, turn_target(conn, target), boundary.notice, replace_after=boundary.replace_after,
                                     display_metadata=boundary.display_metadata)
            if result is not None:
                self._store_result(conn, row, result)
            row.update(status="terminal", outcome=str(outcome))
            save(conn, row)
            settle_transfers(conn, epoch=epoch, generation=generation, transfers=transfers, outcome=outcome)
            bump_revision(conn, target)
            return record(row)

    def cancel(self, *, epoch: int, admission_id: str) -> AdmissionRecord:
        with self._store.transaction() as conn:
            require_epoch(conn, epoch)
            row = admission(conn, admission_id)
            if row["status"] == "unknown":
                raise UnknownExecution("admission", str(admission_id))
            if row["status"] == "started" or row["outcome"] == STEERING:
                raise stale("admission", admission_id)
            if row["status"] == "queued":
                row.update(status="terminal", outcome="cancelled")
                save(conn, row)
            return record(row)

    def steer(self, *, epoch: int, admission_id: str, generation: int) -> tuple[AdmissionRecord, bool]:
        with self._store.transaction() as conn:
            require_epoch(conn, epoch)
            row = admission(conn, admission_id)
            if steer_replayed(row):
                return record(row), False
            if row["status"] != "queued":
                raise AuthorityRefusal("admission_started")
            if type(generation) is not int or scalar(
                    conn, "SELECT 1 FROM session_admissions WHERE target_session_id = %s AND status = 'started' "
                          "AND owner_epoch = %s AND generation = %s LIMIT 1",
                    (row["target_session_id"], epoch, generation)) is None:
                raise AuthorityRefusal("session_idle")
            row.update(status="terminal", outcome=STEERING, owner_epoch=epoch, generation=generation)
            save(conn, row)
            return record(row), True

    def revert_steer(self, *, epoch: int, admission_id: str, generation: int) -> bool:
        with self._store.transaction() as conn:
            require_epoch(conn, epoch)
            return requeue_transfer(conn, admission_id, epoch, generation)

    def mark_unknown(self, *, epoch: int, admission_id: str, generation: int,
                     transfers: Optional[Mapping[str, bool]] = None) -> str:
        with self._store.transaction() as conn:
            require_epoch(conn, epoch)
            row = admission(conn, admission_id)
            if row["owner_epoch"] != epoch or row["generation"] != generation:
                raise stale("admission", admission_id)
            if row["status"] in {"terminal", "unknown"}:
                return row["status"]
            if row["status"] != "started":
                raise stale("admission", admission_id)
            row["status"] = "unknown"
            save(conn, row)
            settle_transfers(conn, epoch=epoch, generation=generation, transfers=transfers, outcome="unknown")
            bump_revision(conn, row["target_session_id"])
            return "unknown"

    def rehome_queued(self, *, epoch: int, old_session_id: str, new_session_id: str, principal_prefix: str) -> list[str]:
        for value in (old_session_id, new_session_id, principal_prefix):
            text(value)
        with self._store.transaction() as conn:
            require_epoch(conn, epoch)
            require_session(conn, new_session_id)
            old_rows = session_rows(conn, old_session_id)
            if any(row["status"] == "unknown" for row in old_rows):
                return []
            moved = [row for row in old_rows
                     if row["status"] == "queued" and row["principal_id"][:len(principal_prefix)] == principal_prefix]
            for row in moved:
                row.update(target_session_id=new_session_id, lineage=[new_session_id],
                           digest=digest(new_session_id, row["payload"], row["intent"]))
                save(conn, row)
            if moved:
                for sid in (old_session_id, new_session_id):
                    bump_revision(conn, sid)
            return [row["admission_id"] for row in moved]

    def retarget_local_automation(self, *, epoch: int, session_id: str, principal_id: str, old_targets: Iterable[str],
                                  new_target: str) -> list[str]:
        old_targets = list(old_targets)
        for value in (session_id, principal_id, new_target, *old_targets):
            text(value)
        targets = set(old_targets)
        with self._store.transaction() as conn:
            require_epoch(conn, epoch)
            require_session(conn, new_target)
            queue = session_rows(conn, session_id)
            if any(row["status"] == "unknown" for row in queue):
                return []
            moved = []
            for row in queue:
                if row["status"] != "queued" or row["principal_id"] != principal_id:
                    continue
                payload = row["payload"]
                descriptor = payload.get("local_automation_v1") if isinstance(payload, dict) else None
                if not isinstance(descriptor, dict) or descriptor.get("target") not in targets:
                    continue
                descriptor["target"] = new_target
                row.update(payload=encode(payload), digest=digest(session_id, payload, row["intent"]))
                save(conn, row)
                moved.append(row["admission_id"])
            if moved:
                bump_revision(conn, session_id)
            return moved

    def recover(self, *, epoch: int) -> int:
        with self._store.transaction() as conn:
            require_epoch(conn, epoch)
            # A child worker of another epoch dies with its owner.
            ended = _rows(conn, "UPDATE worker_executions SET status = 'terminal' WHERE kind = 'child' AND status <> 'terminal' "
                                "AND owner_epoch <> %s RETURNING session_id", (epoch,))
            execute(conn, "UPDATE sessions SET ended_at = %s, end_reason = 'agent_close' WHERE id = ANY(%s) AND ended_at IS NULL",
                    (time.time(), [item["session_id"] for item in ended]))
            execute(conn, "UPDATE worker_executions SET status = 'unknown' WHERE status IN ('registered', 'running') "
                          "AND owner_epoch <> %s", (epoch,))
            execute(conn, "UPDATE session_admissions SET outcome = 'interrupted' WHERE status = 'terminal' AND outcome = %s "
                          "AND owner_epoch IS DISTINCT FROM %s", (STEERING, epoch))
            return execute(conn, "UPDATE session_admissions SET status = 'unknown' WHERE status = 'started' "
                                 "AND owner_epoch IS DISTINCT FROM %s", (epoch,))

    def resolve_unknown(self, *, epoch: int, admission_id: str, generation: int, notice: Optional[str] = None,
                        captured: Optional[CapturedResult] = None) -> AdmissionRecord:
        with self._store.transaction() as conn:
            require_epoch(conn, epoch)
            row = admission(conn, admission_id)
            if (type(generation) is int and row["generation"] == generation and row["status"] == "terminal"
                    and (row["outcome"] == "interrupted" or _meta.has(conn, _result_key(row["admission_id"])))):
                return record(row)
            if type(generation) is not int or row["status"] != "unknown" or row["generation"] != generation:
                raise stale("admission", admission_id)
            outcome = "interrupted"
            committed = captured is not None and row["owner_epoch"] == epoch
            if committed:
                outcome = captured.outcome
                if outcome not in ("completed", "interrupted", "failed"):
                    raise AuthorityRefusal("invalid_params")
                self._store_result(conn, row, dict(captured.result))
            target = row["target_session_id"]
            close_discarded_turn(conn, turn_target(conn, target), None if committed else notice)
            retire_workers(conn, row, row["owner_epoch"])
            row.update(status="terminal", outcome=outcome)
            save(conn, row)
            bump_revision(conn, target)
            return record(row)

    # ── reads ──

    def _read(self, query: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        with self._store.transaction() as conn:
            return _rows(conn, query, params)

    def find(self, *, principal_id: str, session_id: str, request_id: str) -> Optional[AdmissionRecord]:
        found = self._read("SELECT * FROM session_admissions WHERE principal_id = %s AND target_session_id = %s "
                           "AND request_id = %s ORDER BY seq LIMIT 1",
                           (bound_id(principal_id), bound_id(session_id), bound_id(request_id)))
        return record(found[0]) if found else None

    def match(self, *, request_id: Optional[str] = None, admission_id: Optional[str] = None,
              principal_id: Optional[str] = None, session_id: Optional[str] = None,
              include_retired: bool = False) -> list[AdmissionRecord]:
        given = [(name, bound_id(value)) for name, value in (("request_id", request_id), ("admission_id", admission_id),
                                                             ("principal_id", principal_id), ("target_session_id", session_id))
                 if value is not None]
        where = " AND ".join(f"{name} = %s" for name, _value in given) or "true"
        with self._store.transaction() as conn:
            found = _rows(conn, f"SELECT * FROM session_admissions WHERE {where} ORDER BY seq", [value for _name, value in given])
            if include_retired and request_id is not None:
                found += [t for t in self._api_tombstones(conn, request_id) if all(t.get(name) == value for name, value in given)]
        return [record(row) for row in found]

    def list(self, *, session_id: str, pending_only: bool = True) -> list[AdmissionRecord]:
        with self._store.transaction() as conn:
            selected = session_rows(conn, session_id, lock=False)
        return [record(row) for row in selected if not pending_only or row["status"] in _PENDING]

    def get(self, *, admission_id: str) -> Optional[AdmissionRecord]:
        with self._store.transaction() as conn:
            found = live_row(conn, admission_id, lock=False) or tombstone(conn, admission_id)
        return record(found) if found is not None else None

    def latest_admission(self, *, session_id: str, payload_marker: str) -> Optional[AdmissionRecord]:
        with self._store.transaction() as conn:
            selected = session_rows(conn, session_id, lock=False)
        found = next((row for row in reversed(selected) if isinstance(row["payload"], dict) and payload_marker in row["payload"]), None)
        return record(found) if found is not None else None

    def settled_result(self, *, admission_id: str, epoch: Optional[int] = None,
                       generation: Optional[int] = None) -> Optional[AdmissionResult]:
        if (epoch is None) != (generation is None):
            raise ValueError("settled_result takes epoch and generation together or not at all")
        with self._store.transaction() as conn:
            if epoch is not None:
                require_epoch(conn, epoch)
            row = admission(conn, admission_id, lock=False)
            saved = _meta.get(conn, _result_key(row["admission_id"]))
        if epoch is None:
            if row["status"] != "terminal" or saved is None:
                return None
            return AdmissionResult(record=record(row), result=load_result(json.dumps(saved)))
        if row["owner_epoch"] != epoch or row["generation"] != generation:
            raise stale("admission", admission_id)
        if row["status"] != "terminal":
            return AdmissionResult(record=record(row), result=None)
        if saved is None:
            raise AuthorityRefusal("storage_unavailable")
        return AdmissionResult(record=record(row), result=load_result(json.dumps(saved)))

    def pending_targets(self, *, statuses: frozenset[str], principal_id: Optional[str] = None) -> frozenset[str]:
        found = self._read("SELECT DISTINCT target_session_id FROM session_admissions WHERE status = ANY(%s) "
                           "AND (%s::text IS NULL OR principal_id = %s)", (list(statuses), principal_id, principal_id))
        return frozenset(item["target_session_id"] for item in found)

    def list_unknown(self) -> list[AdmissionRecord]:
        return [record(row) for row in self._read("SELECT * FROM session_admissions WHERE status = 'unknown' ORDER BY seq")]

    def has_pending(self, *, session_ids: Sequence[str] = (), statuses: Optional[frozenset[str]] = None) -> bool:
        found = self._read("SELECT 1 FROM session_admissions WHERE (cardinality(%s::text[]) = 0 OR target_session_id = ANY(%s)) "
                           "AND (%s::text[] IS NULL OR status = ANY(%s)) LIMIT 1",
                           (list(session_ids), list(session_ids), None if statuses is None else list(statuses),
                            None if statuses is None else list(statuses)))
        return bool(found)

    def ledger_busy(self, session_id: str, target: str) -> bool:
        return bool(self._read(
            "SELECT 1 FROM session_admissions WHERE target_session_id = %s AND status IN ('queued', 'started', 'unknown') "
            "UNION ALL SELECT 1 FROM worker_executions WHERE session_id = ANY(%s) AND status IN ('registered', 'running', 'unknown') "
            "LIMIT 1", (session_id, [session_id, target])))

    def webhook_ingress_payloads(self, *, chat_id: str) -> list[Mapping[str, Any]]:
        out = []
        for row in self._read("SELECT * FROM session_admissions ORDER BY seq"):
            source = _webhook_source(row["payload"])
            if source is not None and source.get("chat_id") == chat_id:
                out.append(record(row).payload)
        return out

    def unfinalized_webhook_admissions(self) -> list[str]:
        found = self._read("SELECT a.* FROM session_admissions a JOIN sessions s ON s.id = a.target_session_id "
                           "WHERE s.source = 'webhook' AND s.ended_at IS NULL AND a.status = 'terminal' "
                           "AND a.generation IS NOT NULL ORDER BY a.seq")
        return [row["admission_id"] for row in found if _webhook_source(row["payload"]) is not None]

    def terminal_ids(self, admission_ids: Sequence[str]) -> frozenset[str]:
        ids = [str(i) for i in admission_ids]
        live = self._read("SELECT admission_id FROM session_admissions WHERE admission_id = ANY(%s) AND status = 'terminal'", (ids,))
        retired = self._read("SELECT admission_id FROM admission_terminals WHERE admission_id = ANY(%s)", (ids,))
        return frozenset(item["admission_id"] for item in (*live, *retired))

    def api_compaction_candidates(self) -> list[str]:
        return [row["admission_id"] for row in self._read(
            "SELECT * FROM session_admissions WHERE status = 'terminal' AND principal_id = 'api' ORDER BY seq")
                if _compaction_candidate(row)]

    def compact_api_payloads(self, *, admission_ids: Sequence[str]) -> int:
        compacted = 0
        with self._store.transaction() as conn:
            for admission_id in admission_ids:
                row = live_row(conn, admission_id)
                if row is None or not _compaction_candidate(row):
                    continue
                payload = compacted_api_payload(row["payload"], row["request_id"])
                if payload is not None:
                    row["payload"] = encode(payload)
                    save(conn, row)
                    compacted += 1
        return compacted

    # ── media ──

    def _mentions(self, references: Sequence[Mapping[str, Any]], after_id: int = 0) -> tuple[int, set[str]]:
        needles: dict[str, str] = {}
        for reference in references:
            raw = reference["path"]
            for tail in (reference["sha256"] + sep + Path(raw).name for sep in ("/", "\\")):
                needles[tail] = needles[json.dumps(tail)[1:-1]] = raw
        with self._store.transaction() as conn:
            watermark = int(scalar(conn, "SELECT coalesce(max(id), 0) FROM messages"))
            rows = transcript_store.load(conn, [item["id"] for item in _rows(conn, "SELECT id FROM sessions")]) if needles else []
        mentioned: set[str] = set()
        for row in rows:
            if row.id <= after_id:
                continue
            content = row.message.content
            body = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            mentioned.update(raw for needle, raw in needles.items() if needle in body)
        return watermark, mentioned

    def history_mentions(self, references: Sequence[Mapping[str, Any]]) -> tuple[int, frozenset[str]]:
        watermark, mentioned = self._mentions(references)
        return watermark, frozenset(mentioned)

    def _held_paths(self, media_root: str) -> set[str]:
        """Media paths admissions still hold: API-turn media of every row, plus attachment, native-text and
        hosted-document media of non-terminal rows."""
        held: set[str] = set()
        for row in self._read("SELECT request_id, payload, status FROM session_admissions"):
            payload = row["payload"]
            references = _media(payload, "api_turn_v1")
            if row["status"] != "terminal":
                references += _media(payload, "attachments_v1") + _media(payload, "native_text_v1")
                references += hosted_document_references(row["request_id"], payload, media_root)
            held |= _paths(references)
        return held

    def unheld_media(self, references: Sequence[Mapping[str, Any]], *, after_id: int, retain_history: bool,
                     media_root: str) -> UnheldMedia:
        held = self._held_paths(media_root)
        if retain_history:
            held |= self._mentions(references, after_id)[1]
        candidates = tuple(dict(reference) for reference in references if reference["path"] not in held)
        return UnheldMedia(candidates=candidates, held=frozenset(held))

    def record_retired_media(self, references: Sequence[Mapping[str, Any]]) -> None:
        with self._store.transaction() as conn:
            for reference in references:
                _meta.put_new(conn, MEDIA_PREFIX + hashlib.sha256(reference["path"].encode()).hexdigest(), json.dumps(reference))

    def drop_retired_candidates(self, keys: Sequence[str]) -> None:
        with self._store.transaction() as conn:
            for key in keys:
                _meta.delete(conn, key)

    def settled_media(self) -> list[Sequence[Mapping[str, Any]]]:
        out = []
        for row in self._read("SELECT payload FROM session_admissions WHERE status = 'terminal' ORDER BY seq"):
            payload = row["payload"]
            if _has_media(payload, "attachments_v1") or _has_media(payload, "native_text_v1"):
                out.append(_media(payload, "attachments_v1") + _media(payload, "native_text_v1"))
        return out

    def retired_candidates(self, kind: str) -> list[tuple[str, Any]]:
        prefix = MEDIA_PREFIX if kind == "media" else SPILLOVER_PREFIX
        with self._store.transaction() as conn:
            return [(key, json.loads(value)) for key, value in _meta.scan(conn, prefix)]

    # ── the worker family's lookup ──

    def _admission(self, conn: Connection, admission_id: Any) -> dict[str, Any]:
        """The live row or its tombstone for the worker receipt steps, which refuse an absent one as ``NotFound``."""
        try:
            return admission(conn, admission_id)
        except AuthorityRefusal as exc:
            raise NotFound("admission", str(admission_id)) from exc


register_repository("admissions", PostgresAdmissionRepository, {"durable_admissions": True})
