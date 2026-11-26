"""``WorkerExecutionRepository`` on Postgres and the store's ``worker_executions`` repository.

``PostgresWorkerExecutionRepository`` implements the seven worker protocols: the session, compression, compaction,
read and delegation verbs live in their sibling modules as mixins. Every receipted verb builds the wire payload and
calls the one receipt transaction of :mod:`.receipt` with its handler; ``persist_operation`` takes the handler from
:data:`OPERATION_HANDLERS`. Registration, adoption and the reservations are their own transactions. Execution rows are
``worker_executions`` (``adoption_digest`` holds the secret's digest); receipts are ``worker_receipts``.
"""

from __future__ import annotations

import hmac
import json
import secrets
from typing import Any, Mapping, Unpack

from psycopg.types.json import Jsonb

from persistence.authority.vocabulary import OwnerRule, admission_fingerprint
from persistence.conversation.sessions import NewSession, SessionLineageKind
from persistence.core.errors import FenceStale, StaleFenceKind, UnknownExecution
from persistence.core.fences import AuthorityEpoch, ExecutionGeneration
from persistence.core.identity import JsonValue, SessionId
from persistence.workers.delegations import WorkerDelegationPayload
from persistence.workers.executions import (
    AdmissionWorkerScope, ChildExecution, ChildReservation, WorkerAuxiliaryUsage, WorkerExecutionId,
    WorkerExecutionRecord, WorkerKind, WorkerMainUsage, WorkerStatus, WorkerTurnLease,
)
from persistence.workers.vocabulary import CHILD_SEPARATOR
from session_rules.legacy_local import same_owner
from session_rules.session_ids import new_session_id

from .._backend import PostgresStore, register_repository
from .._sql import execute, row as _row, rows as _rows, scalar
from ..authority import admissions as admission_rows
from ..authority._bindings_lineage import local_physical_target, runtime_lineage
from ..authority._leases import conversation_key
from ..authority.runtime import require_epoch
from . import compaction, compression, compression_reads, receipt, sessions
from .compaction import WorkerCompactionVerbs
from .compression import WorkerCompressionVerbs
from .compression_reads import WorkerCompressionReads, authorize
from .delegations import WorkerDelegationLedger
from .receipt import Step, denied, refuse, require_fields, secret_digest, stale_generation, text
from .sessions import WorkerSessionVerbs
from .usage import TOKEN_COUNTERS, apply_auxiliary_usage, apply_main_usage, stamp_created_source

_KINDS = tuple(kind.value for kind in WorkerKind)
_MODEL_USAGE_FIELDS = frozenset({
    "model", "billing_provider", "billing_base_url", "billing_mode", *TOKEN_COUNTERS, "estimated_cost_usd",
    "actual_cost_usd", "cost_status", "cost_source", "api_call_count",
})
_MAIN_USAGE_KEYS = _MODEL_USAGE_FIELDS | {"pricing_version", "absolute", "task", "prompt_cache_clock"}
_AUX_USAGE_KEYS = (_MODEL_USAGE_FIELDS - {"billing_mode", "actual_cost_usd", "cost_status", "cost_source"}) | {"task"}
_LIVE = ("registered", "running", "unknown")


def record(found: Mapping[str, Any]) -> WorkerExecutionRecord:
    """The row without its digest."""
    return WorkerExecutionRecord(
        execution_id=WorkerExecutionId(found["execution_id"]), session_id=SessionId(found["session_id"]),
        kind=WorkerKind(found["kind"]), owner_epoch=AuthorityEpoch(found["owner_epoch"]),
        generation=ExecutionGeneration(found["generation"]), status=WorkerStatus(found["status"]),
        last_sequence=found["last_sequence"])


def _insert_worker(conn: Any, execution_id: str, session_id: str, kind: str, epoch: Any, generation: int, status: str,
                   digest: str) -> dict[str, Any]:
    return _row(conn, "INSERT INTO worker_executions (execution_id, session_id, kind, owner_epoch, generation, status, "
                      "adoption_digest, last_sequence) VALUES (%s, %s, %s, %s, %s, %s, %s, 0) RETURNING *",
                (execution_id, session_id, kind, epoch, generation, status, digest))


def _valid_usage_value(key: str, value: Any) -> bool:
    if key in (*TOKEN_COUNTERS, "api_call_count"):
        return type(value) is int and 0 <= value <= 2**53
    if key in ("estimated_cost_usd", "actual_cost_usd"):
        return value is None or (type(value) in (int, float) and 0 <= value <= 1e12)
    if key == "absolute":
        return type(value) is bool
    return value is None or (isinstance(value, str) and len(value) <= 4096)


def _usage_handler(auxiliary: bool) -> receipt.Handler:
    def handle(step: Step, payload: dict[str, Any]) -> dict[str, Any]:
        source = None
        if not auxiliary:
            source = payload.get("source") if isinstance(payload.get("source"), str) else None
            payload = {key: value for key, value in payload.items() if key != "source"}
        if set(payload) - (_AUX_USAGE_KEYS if auxiliary else _MAIN_USAGE_KEYS):
            refuse("invalid_params")
        if not all(_valid_usage_value(key, value) for key, value in payload.items()):
            refuse("invalid_params")
        step.wrote = True
        if auxiliary:
            if not payload.get("task"):
                refuse("invalid_params")
            apply_auxiliary_usage(step.conn, step.session_id, payload)
        else:
            apply_main_usage(step.conn, step.session_id, payload)
            stamp_created_source(step.conn, step.session_id, source)
        return {"value": None}
    return handle


def _turn_handler(operation: str) -> receipt.Handler:
    """The turn-lease handler for ``turn.acquire`` and ``turn.renew``, over the coordination verbs."""
    def handle(step: Step, payload: dict[str, Any]) -> dict[str, Any]:
        if set(payload) != {"holder", "ttl_seconds"}:
            refuse("invalid_params")
        holder = text(payload["holder"])
        ttl = payload.get("ttl_seconds", 300)
        if type(ttl) not in (float, int) or not 0.1 <= ttl <= 3600:
            refuse("invalid_params")
        coordination, sid = step.store.coordination, SessionId(step.session_id)
        if operation == "turn.acquire":
            return {"value": coordination.try_acquire_turn_lease(session_id=sid, holder=holder, ttl_seconds=ttl)}
        return {"value": coordination.refresh_turn_lease(session_id=sid, holder=holder, ttl_seconds=ttl)}
    return handle


def _turn_cleanup(step: Step, payload: dict[str, Any]) -> dict[str, Any]:
    """Release ``holder``'s turn lease on the conversation; ``target`` must share the worker session's conversation."""
    require_fields(payload, ("target", "holder"))
    target, holder = text(payload["target"]), text(payload["holder"])
    authorize(step, target)
    key = conversation_key(step.conn, step.session_id)
    if key != conversation_key(step.conn, target):
        raise denied("session", target)
    execute(step.conn, "DELETE FROM session_turn_leases WHERE conversation_id = %s AND holder = %s", (key, holder))
    return {"released": True}


def _finish(step: Step, payload: dict[str, Any]) -> dict[str, Any]:
    if payload:
        refuse("invalid_params")
    return {"status": "terminal"}


#: Every operation a typed worker verb runs, except ``child.reserve`` and the ``delegation.*`` ledger.
OPERATION_HANDLERS: dict[str, receipt.Handler] = {
    "session.title": sessions._title,
    "session.title_source": sessions._title_source,
    "session.next_title": sessions._next_title,
    "session.activity": sessions._activity,
    "session.activity_clear": sessions._activity_clear,
    "session.billing_route": sessions._billing_route,
    "session.api_content": sessions._api_content,
    "session.message_api_content": sessions._message_api_content,
    "session.user_content": sessions._user_content,
    "session.create": sessions._create,
    "session.end": sessions._end,
    "session.lifecycle": sessions._lifecycle,
    "session.prompt": sessions._prompt,
    "session.sidecars": sessions._sidecars,
    "session.tools": sessions._tools,
    "session.tail_role": sessions._tail_role,
    "session.row_role": sessions._row_role,
    "transcript.append": sessions._append,
    "turn.cleanup": _turn_cleanup,
    "compression.append": compaction._append,
    "compression.reopen": compression._reopen,
    "compression.cleanup": compression._cleanup,
    "compression.watermark": compression._watermark,
    "compression.archive": compaction._archive,
    "compression.publish": compaction._publish,
    "compression.context": compression_reads._context,
    "compression.lineage": compression_reads._lineage,
    "compression.boundary": compression._boundary,
    "compression.history": compression_reads._history,
    **{"compression.lock." + action: compression._lock(action) for action in ("acquire", "renew", "holder")},
    "compression.cooldown.record": compression._cooldown_record,
    "compression.cooldown.restore": compression._cooldown_restore,
    "compression.cooldown.clear": compression._cooldown_clear,
    "compression.fallback": compression._counter("compression_fallback_streak"),
    "compression.ineffective": compression._counter("compression_ineffective_count"),
    "compression.recovery": compression._counter("compression_recovery_deadline"),
    "execution.finish": _finish,
    "usage.main": _usage_handler(False),
    "usage.auxiliary": _usage_handler(True),
    "turn.acquire": _turn_handler("turn.acquire"),
    "turn.renew": _turn_handler("turn.renew"),
}


def _child_handler(*, epoch: Any, execution_id: str, identity: dict[str, Any], principal: str, limits: tuple[int, int],
                   refused: str | None) -> receipt.Handler:
    """The child-execution reservation handler with its trusted inputs bound."""
    def handle(step: Step, payload: dict[str, Any]) -> dict[str, Any]:
        secret = payload.get("secret") if isinstance(payload, dict) else None
        if set(payload) != {"secret"} or not isinstance(secret, str) or not 16 <= len(secret) <= 256:
            refuse("invalid_params")
        max_children, max_depth = limits
        prefix = execution_id + CHILD_SEPARATOR
        code = refused
        if code is None and prefix.count(CHILD_SEPARATOR) > max_depth:
            code = "delegation_depth_exceeded"
        if code is None and receipt.live_children(step.conn, execution_id) >= max_children:
            code = "delegation_capacity"
        if code is not None:
            return {"refused": code}
        parent = step.session() or {}
        child_sid, child_eid = new_session_id(), prefix + secrets.token_hex(8)
        step.store.sessions.create(NewSession(session_id=SessionId(child_sid), source="subagent",
                                              parent_session_id=SessionId(step.session_id),
                                              lineage_kind=SessionLineageKind.DELEGATE, profile_name=parent.get("profile_name")))
        scope = dict(identity, session_id=child_sid, execution_id=child_eid, generation=0, secret=secret)
        claim = admission_fingerprint(canonical_target=child_sid, payload=scope | {"principal": principal})
        _insert_worker(step.conn, child_eid, child_sid, "child", epoch, 0, "running", secret_digest(claim))
        return {"child": {"session_id": child_sid, "execution_id": child_eid, "generation": 0}}
    return handle


class PostgresWorkerExecutionRepository(
    WorkerSessionVerbs, WorkerCompressionVerbs, WorkerCompactionVerbs, WorkerCompressionReads, WorkerDelegationLedger,
):
    """Worker executions of the bound profile's schema; see ``persistence.workers.executions``."""

    def __init__(self, store: PostgresStore) -> None:
        self._store = store

    def _apply(self, operation: str, payload: Any, handler: receipt.Handler, *, epoch: Any, execution_id: Any,
               session_id: Any, generation: Any, sequence: Any) -> dict[str, Any]:
        return receipt.apply(self._store, operation, payload, handler, epoch=epoch, execution_id=execution_id,
                             session_id=session_id, generation=generation, sequence=sequence)

    # ── registration and adoption ──

    def register(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId,
                 generation: ExecutionGeneration, kind: WorkerKind, adoption_secret: str,
                 require_idle: bool = False) -> WorkerExecutionRecord:
        for value in (execution_id, session_id):
            text(value)
        if kind not in _KINDS:
            refuse("invalid_params")
        digest = secret_digest(adoption_secret)
        with self._store.transaction() as conn:
            require_epoch(conn, epoch)
            session = admission_rows.require_session(conn, session_id, lock=True)
            if type(generation) is not int or int(session["runtime_generation"]) != generation:
                raise stale_generation("session", session_id)
            old = receipt.worker_row(conn, execution_id)
            if old is not None:
                if (old["session_id"], old["generation"], old["kind"], old["owner_epoch"], old["adoption_digest"]) != (
                        session_id, generation, getattr(kind, "value", kind), epoch, digest):
                    refuse("admission_conflict")
                return record(old)
            targets = sorted(runtime_lineage(conn, session_id))
            linked = {item["status"] for item in _rows(conn, "SELECT status FROM session_admissions WHERE target_session_id = ANY(%s)",
                                                       (targets,))}
            if require_idle and linked & {"queued", "started", "unknown"}:
                raise stale_generation("session", session_id)
            if scalar(conn, "SELECT 1 FROM worker_executions WHERE session_id = ANY(%s) AND status = ANY(%s) LIMIT 1",
                      (targets, list(_LIVE))) is not None:
                raise stale_generation("session", session_id)
            if "unknown" in linked:
                raise UnknownExecution("admission", session_id)
            return record(_insert_worker(conn, execution_id, session_id, getattr(kind, "value", kind), epoch, generation, "registered", digest))

    def adopt(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId,
              generation: ExecutionGeneration, adoption_secret: str) -> WorkerExecutionRecord:
        digest = secret_digest(adoption_secret)
        with self._store.transaction() as conn:
            require_epoch(conn, epoch)
            found = receipt.assignment(conn, execution_id, session_id, generation)
            if found["status"] == "terminal":
                raise stale_generation("worker_execution", execution_id)
            if not hmac.compare_digest(found["adoption_digest"], digest):
                raise denied("worker_execution", execution_id)
            if str(execution_id).startswith("admission-worker:") and found["owner_epoch"] != epoch:
                raise FenceStale(StaleFenceKind.STALE_EPOCH, "worker_execution", str(execution_id))
            linked = [item for item in _rows(
                conn, "SELECT admission_id, target_session_id, lineage FROM session_admissions WHERE generation = %s "
                      "AND owner_epoch = %s AND status IN ('started', 'unknown') FOR UPDATE", (generation, found["owner_epoch"]))
                      if item["target_session_id"] == session_id or session_id in (item["lineage"] or ())]
            if len(linked) > 1:
                refuse("admission_conflict")
            if linked:
                execute(conn, "UPDATE session_admissions SET owner_epoch = %s, status = 'started' WHERE admission_id = %s",
                        (epoch, linked[0]["admission_id"]))
            return record(_row(conn, "UPDATE worker_executions SET owner_epoch = %s, status = 'running' WHERE execution_id = %s "
                                     "RETURNING *", (epoch, found["execution_id"])))

    def verify_claim(self, *, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration,
                     adoption_secret: str) -> WorkerExecutionRecord:
        with self._store.transaction() as conn:
            found = receipt.worker_row(conn, execution_id, lock=False)
        if found is None:
            refuse("not_found")
        if found["session_id"] != session_id:
            raise denied("worker_execution", execution_id)
        if type(generation) is not int or found["generation"] != generation:
            raise stale_generation("worker_execution", execution_id)
        if not hmac.compare_digest(found["adoption_digest"], secret_digest(adoption_secret)):
            raise denied("worker_execution", execution_id)
        return record(found)

    def execution_session(self, *, execution_id: WorkerExecutionId) -> SessionId | None:
        with self._store.transaction() as conn:
            found = receipt.worker_row(conn, execution_id, lock=False)
        return None if found is None else SessionId(found["session_id"])

    def has_live_worker(self) -> bool:
        with self._store.transaction() as conn:
            return scalar(conn, "SELECT 1 FROM worker_executions WHERE status <> 'terminal' LIMIT 1") is not None

    # ── reservations ──

    def reserve_for_admission(self, *, epoch: AuthorityEpoch, admission_id: Any, principal_id: str, profile_id: str,
                              pid: int, birth: float, secret: str, owner: OwnerRule | None,
                              worker_subject: str | None = None) -> AdmissionWorkerScope:
        with self._store.transaction() as conn:
            require_epoch(conn, epoch)
            admission = admission_rows.admission(conn, admission_id)
            if admission["status"] != "started" or admission["owner_epoch"] != epoch:
                refuse("producer_not_started")
            if not same_owner(admission["principal_id"], principal_id, owner):
                raise denied("admission", admission_id)
            target, generation = admission["target_session_id"], admission["generation"]
            session = admission_rows.require_session(conn, target, lock=True)
            if int(session["runtime_generation"]) != generation:
                raise stale_generation("session", target)
            physical = local_physical_target(conn, target)
            execution_id = "admission-worker:" + str(admission_id)
            if receipt.worker_row(conn, execution_id) is not None:
                refuse("admission_conflict")
            if scalar(conn, "SELECT 1 FROM worker_executions WHERE session_id = %s AND status <> 'terminal' LIMIT 1",
                      (physical,)) is not None:
                raise stale_generation("session", physical)
            execute(conn, "UPDATE sessions SET runtime_generation = %s WHERE id = %s", (generation, physical))
            if physical != target and admission_rows.live_row(conn, admission_id) is not None:
                execute(conn, "UPDATE session_admissions SET lineage = %s WHERE admission_id = %s",
                        (Jsonb([*(admission.get("lineage") or [target]), physical]), admission["admission_id"]))
            scope = dict(profile_id=profile_id, session_id=physical, execution_id=execution_id, generation=generation,
                         pid=pid, birth=birth, secret=secret)
            claim = admission_fingerprint(canonical_target=physical, payload=scope | {"principal": worker_subject or principal_id})
            _insert_worker(conn, execution_id, physical, "compute", epoch, generation, "registered", secret_digest(claim))
            return AdmissionWorkerScope(**scope, epoch=epoch)

    def lose_reservation(self, *, epoch: AuthorityEpoch, admission_id: Any, execution_id: WorkerExecutionId,
                         generation: ExecutionGeneration) -> None:
        with self._store.transaction() as conn:
            require_epoch(conn, epoch)
            current = admission_rows.admission(conn, admission_id)
            if current["status"] != "started" or current["owner_epoch"] != epoch or current["generation"] != generation:
                raise stale_generation("admission", admission_id)
            found = receipt.worker_row(conn, execution_id)
            if found is None or found["generation"] != generation or found["owner_epoch"] != epoch:
                raise stale_generation("worker_execution", execution_id)
            execute(conn, "UPDATE worker_executions SET status = 'terminal' WHERE execution_id = %s", (found["execution_id"],))
            receipt.compact_terminal_receipts(conn, found["execution_id"])
            execute(conn, "UPDATE session_admissions SET status = 'unknown' WHERE admission_id = %s", (current["admission_id"],))
            admission_rows.bump_revision(conn, current["target_session_id"])

    def reserve_child(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId,
                      generation: ExecutionGeneration, sequence: int, secret: str | WorkerDelegationPayload, profile_id: str,
                      pid: int, birth: float, principal: str, max_concurrent_children: int, max_spawn_depth: int,
                      refused: str | None = None) -> ChildReservation:
        handler = _child_handler(epoch=epoch, execution_id=str(execution_id),
                                 identity={"profile_id": profile_id, "pid": pid, "birth": birth}, principal=principal,
                                 limits=(max_concurrent_children, max_spawn_depth), refused=refused)
        payload = secret.payload if isinstance(secret, WorkerDelegationPayload) else {"secret": secret}
        result = self._apply("child.reserve", payload, handler, epoch=epoch, execution_id=execution_id,
                             session_id=session_id, generation=generation, sequence=sequence)
        child = result.get("child")
        if child is None:
            return ChildReservation(refused=result["refused"])
        return ChildReservation(child=ChildExecution(session_id=SessionId(child["session_id"]),
                                                     execution_id=WorkerExecutionId(child["execution_id"]),
                                                     generation=ExecutionGeneration(child["generation"])))

    # ── receipted execution operations ──

    def persist_operation(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId,
                          generation: ExecutionGeneration, sequence: int, operation: str,
                          payload: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
        if not isinstance(operation, str) or operation not in OPERATION_HANDLERS:
            refuse("invalid_params")
        return self._apply(operation, payload, OPERATION_HANDLERS[operation], epoch=epoch, execution_id=execution_id,
                           session_id=session_id, generation=generation, sequence=sequence)

    def finish(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId,
               generation: ExecutionGeneration, sequence: int) -> None:
        self._apply("execution.finish", {}, _finish, epoch=epoch, execution_id=execution_id, session_id=session_id,
                    generation=generation, sequence=sequence)

    def record_usage(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId,
                     generation: ExecutionGeneration, sequence: int, **usage: Unpack[WorkerMainUsage]) -> None:
        self._apply("usage.main", dict(usage), _usage_handler(False), epoch=epoch, execution_id=execution_id,
                    session_id=session_id, generation=generation, sequence=sequence)

    def record_auxiliary_usage(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId,
                               generation: ExecutionGeneration, sequence: int, task: str,
                               **usage: Unpack[WorkerAuxiliaryUsage]) -> None:
        self._apply("usage.auxiliary", dict(usage, task=task), _usage_handler(True), epoch=epoch, execution_id=execution_id,
                    session_id=session_id, generation=generation, sequence=sequence)

    def acquire_turn_lease(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId,
                           generation: ExecutionGeneration, sequence: int, lease: WorkerTurnLease) -> bool:
        payload = {"holder": lease.holder, "ttl_seconds": lease.ttl_seconds}
        return bool(self._apply("turn.acquire", payload, _turn_handler("turn.acquire"), epoch=epoch, execution_id=execution_id,
                                session_id=session_id, generation=generation, sequence=sequence)["value"])

    def renew_turn_lease(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId,
                         generation: ExecutionGeneration, sequence: int, lease: WorkerTurnLease) -> bool:
        payload = {"holder": lease.holder, "ttl_seconds": lease.ttl_seconds}
        return bool(self._apply("turn.renew", payload, _turn_handler("turn.renew"), epoch=epoch, execution_id=execution_id,
                                session_id=session_id, generation=generation, sequence=sequence)["value"])

    def cleanup_turn_lease(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId,
                           generation: ExecutionGeneration, sequence: int, target_session_id: SessionId, holder: str) -> None:
        self._apply("turn.cleanup", {"target": target_session_id, "holder": holder}, _turn_cleanup, epoch=epoch,
                    execution_id=execution_id, session_id=session_id, generation=generation, sequence=sequence)

    # ── receipt reads ──

    def claim_origin(self, *, execution_id: WorkerExecutionId, session_id: SessionId) -> SessionId:
        with self._store.transaction() as conn:
            found = _rows(conn, "SELECT sequence, result_json FROM worker_receipts WHERE execution_id = %s ORDER BY sequence",
                          (str(execution_id),))
        for item in found:
            result = json.loads(item["result_json"])
            moved = result.get("worker_assignment") if isinstance(result, dict) else None
            if isinstance(moved, dict) and moved.get("parent") is not None:
                return SessionId(moved["parent"])
        return session_id

    def publication_target(self, *, execution_id: WorkerExecutionId, session_id: SessionId,
                           generation: ExecutionGeneration, sequence: int, operation: str,
                           payload: Mapping[str, JsonValue]) -> SessionId:
        if operation != "compression.publish":
            return session_id
        digest = admission_fingerprint(canonical_target=session_id,
                                       payload={"operation": operation, "payload": json.loads(json.dumps(payload))})
        with self._store.transaction() as conn:
            found = receipt.receipt_assignment(conn, execution_id, session_id, generation, sequence, digest)
        return SessionId(found["session_id"])


register_repository("worker_executions", PostgresWorkerExecutionRepository, {"receipted_writes": True})
