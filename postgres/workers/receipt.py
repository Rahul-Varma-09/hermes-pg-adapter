"""The worker receipt transaction and the checks its operations share.

:func:`apply` is the one dispatcher every receipted worker verb calls. Inside one transaction it runs: the payload's
encoding and size, the epoch fence, the receipt-aware assignment (the execution row locked), the owner-epoch fence,
the stored receipt (replay, conflict or retired), the live status, the strict next sequence, the operation, then the
receipt write with the sequence advance, the status and the revision bump. The operation calls the sibling
repositories' own verbs; their transactions join this one (``PostgresStore.transaction`` is re-entrant), so the
operation's writes publish with its receipt or not at all.

An operation changed the store when it is not a read and either its handler says it wrote (a write that matched a row,
even with the same values) or this transaction inserted, updated or deleted a row while it ran
(``pg_stat_xact_user_tables`` of this schema, which counts a matched update even when the values stay the same).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any, Callable, Mapping, NoReturn, Optional

from psycopg import Connection

from persistence.authority.vocabulary import admission_fingerprint
from persistence.core.canonical import canonical_json
from persistence.core.errors import AuthorityRefusal, FenceStale, ProofRejected, StaleFenceKind
from persistence.workers.vocabulary import CHILD_SEPARATOR, RECEIPT_PAYLOAD_MAX_BYTES

from .._sql import execute, row as _row, rows as _rows, scalar
from ..authority.admissions import bound_id
from ..authority.runtime import require_epoch

#: What a read receipt shrinks to when its execution ends.
RETIRED_RECEIPT = {"retired": True}
#: The operations whose handler writes nothing, so they never count as a change.
READ_ONLY = frozenset({
    "compression.context", "compression.lineage", "compression.history", "compression.watermark",
    "compression.lock.holder", "compression.boundary", "session.lifecycle", "session.tail_role", "session.row_role",
    "session.next_title", "delegation.read",
})
_WRITES = ("SELECT coalesce(sum(n_tup_ins + n_tup_upd + n_tup_del), 0) FROM pg_stat_xact_user_tables "
           "WHERE schemaname = current_schema()")


class Step:
    """One operation's view inside the receipt transaction: the connection, the store handle, the addressed session
    and the execution row. ``wrote`` marks a write the handler made, even one that left the values as they were."""

    __slots__ = ("conn", "session_id", "store", "worker", "wrote")

    def __init__(self, conn: Connection, store: Any, session_id: str, worker: dict[str, Any]) -> None:
        self.conn, self.store, self.session_id, self.worker, self.wrote = conn, store, session_id, worker, False

    def session(self, session_id: Optional[str] = None) -> Optional[dict[str, Any]]:
        return _row(self.conn, "SELECT * FROM sessions WHERE id = %s", (session_id or self.session_id,))


Handler = Callable[[Step, dict[str, Any]], dict[str, Any]]


def refuse(reason: str) -> NoReturn:
    raise AuthorityRefusal(reason)


def stale_generation(entity: str, entity_id: object) -> FenceStale:
    return FenceStale(StaleFenceKind.STALE_GENERATION, entity, str(entity_id))


def denied(entity: str, entity_id: object) -> ProofRejected:
    return ProofRejected(entity, str(entity_id))


def text(value: Any) -> str:
    """A non-empty string of at most 1024 characters, else ``invalid_params``."""
    if not isinstance(value, str) or not value or len(value) > 1024:
        refuse("invalid_params")
    return value


def number(value: Any) -> float:
    """A finite int or float, else ``invalid_params``."""
    if type(value) not in (int, float) or not math.isfinite(value):
        refuse("invalid_params")
    return value  # type: ignore[no-any-return]


def require_fields(payload: Any, required: tuple[str, ...], optional: tuple[str, ...] = ()) -> None:
    """Exactly the required keys plus any of the optional ones."""
    if not isinstance(payload, dict) or set(payload) - set(required) - set(optional) or set(required) - set(payload):
        refuse("invalid_params")


def secret_digest(secret: Any) -> str:
    """The SHA-256 hex digest of the secret text."""
    return hashlib.sha256(text(secret).encode("utf-8")).hexdigest()


def worker_row(conn: Connection, execution_id: Any, *, lock: bool = True) -> Optional[dict[str, Any]]:
    """The execution row of ``execution_id`` (the id bound as text, an unbindable id ``PermanentStoreError``)."""
    return _row(conn, "SELECT * FROM worker_executions WHERE execution_id = %s" + (" FOR UPDATE" if lock else ""),
                (bound_id(execution_id),))


def runtime_generation(conn: Connection, session_id: str) -> int:
    found = scalar(conn, "SELECT runtime_generation FROM sessions WHERE id = %s", (session_id,))
    if found is None:
        refuse("not_found")
    return int(found)


def assignment(conn: Connection, execution_id: Any, session_id: Any, generation: Any) -> dict[str, Any]:
    """The execution row assigned to ``session_id`` at ``generation``: ``not_found``, denied for another session,
    else a stale generation unless the row's and the session's runtime generation both match."""
    found = worker_row(conn, execution_id)
    if found is None:
        refuse("not_found")
    if found["session_id"] != session_id:
        raise denied("worker_execution", execution_id)
    if type(generation) is not int or found["generation"] != generation or runtime_generation(conn, session_id) != generation:
        raise stale_generation("worker_execution", execution_id)
    return found


def stored_receipt(conn: Connection, execution_id: str, sequence: int) -> Optional[dict[str, Any]]:
    found = _row(conn, "SELECT payload_digest, result_json FROM worker_receipts WHERE execution_id = %s AND sequence = %s",
                 (execution_id, sequence))
    return None if found is None else {"digest": found["payload_digest"], "result": json.loads(found["result_json"])}


def receipt_assignment(conn: Connection, execution_id: Any, session_id: Any, generation: Any, sequence: int,
                       digest: str) -> dict[str, Any]:
    """The receipt-aware assignment: only an identical committed publication may still address the parent it left."""
    stored = stored_receipt(conn, bound_id(execution_id), sequence)
    if stored is not None and stored["digest"] == digest and isinstance(stored["result"], dict):
        moved = stored["result"].get("worker_assignment")
        if moved and moved["parent"] == session_id:
            current = worker_row(conn, execution_id)
            return assignment(conn, execution_id, current["session_id"], generation)  # type: ignore[index]
    return assignment(conn, execution_id, session_id, generation)


def payload_digest(operation: str, payload: Any, session_id: Any) -> tuple[dict[str, Any], str]:
    """The payload as the receipt sees it (its canonical JSON decoded) and its receipt digest; an unencodable or
    oversized payload is ``invalid_params``."""
    try:
        encoded = canonical_json(payload)
    except (TypeError, ValueError):
        refuse("invalid_params")
    if len(encoded.encode("utf-8", errors="surrogatepass")) > RECEIPT_PAYLOAD_MAX_BYTES:
        refuse("invalid_params")
    decoded = json.loads(encoded)
    return decoded, admission_fingerprint(canonical_target=session_id, payload={"operation": operation, "payload": decoded})


def apply(store: Any, operation: str, payload: Any, handler: Handler, *, epoch: Any, execution_id: Any, session_id: Any,
          generation: Any, sequence: Any) -> dict[str, Any]:
    """The stored result of an exact retry, else ``handler``'s result committed with its receipt."""
    if type(sequence) is not int or sequence < 1:
        refuse("invalid_params")
    decoded, digest = payload_digest(operation, payload, session_id)
    with store.transaction() as conn:
        require_epoch(conn, epoch)
        worker = receipt_assignment(conn, execution_id, session_id, generation, sequence, digest)
        if worker["owner_epoch"] != epoch:
            raise FenceStale(StaleFenceKind.STALE_EPOCH, "worker_execution", str(execution_id))
        eid = worker["execution_id"]
        old = stored_receipt(conn, eid, sequence)
        if old is not None:
            if old["digest"] != digest:
                refuse("admission_conflict")
            if old["result"] == RETIRED_RECEIPT:
                raise stale_generation("worker_receipt", execution_id)
            return old["result"]  # type: ignore[no-any-return]
        if worker["status"] not in ("registered", "running"):
            raise stale_generation("worker_execution", execution_id)
        if sequence != worker["last_sequence"] + 1:
            refuse("invalid_params")
        step = Step(conn, store, session_id, worker)
        before = scalar(conn, _WRITES)
        result = handler(step, decoded)
        changed = (operation not in READ_ONLY and (step.wrote or scalar(conn, _WRITES) != before)) or operation == "execution.finish"
        stored = json.dumps(result, ensure_ascii=True, allow_nan=False)
        execute(conn, "INSERT INTO worker_receipts (execution_id, sequence, payload_digest, result_json) VALUES (%s, %s, %s, %s)",
                (eid, sequence, digest, stored))
        execute(conn, "UPDATE worker_executions SET last_sequence = %s, status = %s WHERE execution_id = %s",
                (sequence, "terminal" if operation == "execution.finish" else "running", eid))
        if changed:
            execute(conn, "UPDATE sessions SET runtime_revision = runtime_revision + 1 WHERE id = %s", (session_id,))
        if operation == "execution.finish":
            compact_terminal_receipts(conn, eid)
        return copy.deepcopy(json.loads(stored))


def compact_terminal_receipts(conn: Connection, execution_id: str) -> None:
    """Retire the live descendants, shrink this execution's read receipts (results carrying ``messages`` or
    ``session``) to the retired marker, and end a child's own session row."""
    for child in _rows(conn, "SELECT execution_id FROM worker_executions WHERE starts_with(execution_id, %s) "
                             "AND status <> 'terminal' ORDER BY execution_id", (execution_id + CHILD_SEPARATOR,)):
        execute(conn, "UPDATE worker_executions SET status = 'terminal' WHERE execution_id = %s", (child["execution_id"],))
        _compact(conn, child["execution_id"])
        _end_child_row(conn, child["execution_id"])
    _compact(conn, execution_id)
    if CHILD_SEPARATOR in execution_id:
        _end_child_row(conn, execution_id)


def _compact(conn: Connection, execution_id: str) -> None:
    execute(conn, "UPDATE worker_receipts SET result_json = %s WHERE execution_id = %s "
                  "AND jsonb_typeof(result_json::jsonb) = 'object' AND (result_json::jsonb ? 'messages' OR result_json::jsonb ? 'session')",
            (json.dumps(RETIRED_RECEIPT), execution_id))


def _end_child_row(conn: Connection, execution_id: str) -> None:
    """The execution's session row ends ``agent_close`` unless it already ended."""
    import time

    execute(conn, "UPDATE sessions SET ended_at = %s, end_reason = 'agent_close' WHERE ended_at IS NULL AND id = "
                  "(SELECT session_id FROM worker_executions WHERE execution_id = %s)", (time.time(), execution_id))


def live_children(conn: Connection, execution_id: str) -> int:
    """The live direct children of an execution."""
    prefix = execution_id + CHILD_SEPARATOR
    found = _rows(conn, "SELECT execution_id FROM worker_executions WHERE starts_with(execution_id, %s) AND status <> 'terminal'",
                  (prefix,))
    return sum(1 for item in found if "/" not in item["execution_id"][len(prefix):])


def changed_flag(step: Step, value: Any) -> dict[str, Any]:
    step.wrote = bool(value)
    return {"value": int(bool(value))}


def mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}
