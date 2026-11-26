"""``SessionRuntimeRepository`` on Postgres: the store-wide authority epoch and the runtime verbs beside the receipted
mutation.

The epoch is the ``runtime_epoch`` singleton; a session's revision and execution generation are its
``runtime_revision`` / ``runtime_generation`` columns; mutation receipts, the deferred-pick slot and viewer closes are
``authority_meta`` values; the ledger is ``session_admissions`` and ``worker_executions``. Every verb is one
transaction, and a fenced verb checks the epoch under the singleton's row lock.
"""

from __future__ import annotations

import json
import time
from typing import Any, Mapping, Optional, Sequence

from psycopg import Connection

from persistence import FenceStale, SessionId, StaleFenceKind
from persistence.authority.bindings import api_generation_id, generation_request_id, local_session_id, room_session_id
from persistence.authority.runtime import (
    ApiGenerations, ChatPickParams, DeferredPick, GenerationChain, GenerationPick, GenerationRow, PinnedIdentity,
    RoomGenerations, SessionRuntimeState, StoredDeferredPick, mutation_receipt_key, parse_mutation,
)
from persistence.core.canonical import canonical_json
from persistence.core.errors import AuthorityRefusal

from .._backend import PostgresStore, register_repository
from .._sql import execute, row as _row, rows as _rows, scalar
from ..conversation.sessions import bump_conversation_generation
from . import _meta

RECEIPT_PREFIX = "gateway.mutation.v1."
DEFERRED_PICK_PREFIX = "gateway.deferred_pick.v1."
VIEWER_CLOSE_PREFIX = "gateway.viewer_close.v1:"


def text(value: Any) -> str:
    """A non-empty string of at most 1024 characters, else ``invalid_params``."""
    if not isinstance(value, str) or not value or len(value) > 1024:
        raise AuthorityRefusal("invalid_params")
    return value


def encode(value: Any) -> Any:
    """The canonical round trip of a value."""
    return json.loads(canonical_json(value))


def revision_conflict(entity_id: Any) -> FenceStale:
    return FenceStale(StaleFenceKind.REVISION_CONFLICT, "session", str(entity_id))


def stale(entity: str, entity_id: Any) -> FenceStale:
    return FenceStale(StaleFenceKind.STALE_GENERATION, entity, str(entity_id))


def require_epoch(conn: Connection, epoch: object) -> int:
    """Refuse any epoch but the current one: a non-int (bool and float included), a never-issued zero and any mismatch
    are the epoch-kind stale fence. Locks the singleton for the caller's transaction."""
    current = scalar(conn, "SELECT epoch FROM runtime_epoch FOR SHARE")
    if type(epoch) is not int or not current or epoch != current:
        raise FenceStale(StaleFenceKind.STALE_EPOCH, "runtime_epoch", "singleton") from AuthorityRefusal("stale_epoch")
    return int(current)


def require_session(conn: Connection, session_id: Any, *, lock: bool = False) -> dict[str, Any]:
    """The session row, else ``not_found``."""
    found = _row(conn, "SELECT * FROM sessions WHERE id = %s" + (" FOR UPDATE" if lock else ""), (str(session_id),))
    if found is None:
        raise AuthorityRefusal("not_found")
    return found


def ledger_status(conn: Connection, session_id: str) -> Optional[str]:
    """The status of the first non-terminal admission targeted at the id, else of the first non-terminal worker on it."""
    status = scalar(conn, "SELECT status FROM session_admissions WHERE target_session_id = %s AND status <> 'terminal' "
                          "ORDER BY seq LIMIT 1", (session_id,))
    if status is None:
        status = scalar(conn, "SELECT status FROM worker_executions WHERE session_id = %s AND status <> 'terminal' "
                              "ORDER BY execution_id LIMIT 1", (session_id,))
    return status


def local_lineage_owner(conn: Connection, session_id: str) -> str:
    """The row's local owner (its ``chat_id``) when the owner's receipt lineage holds the row, else the id itself."""
    owner = scalar(conn, "SELECT chat_id FROM sessions WHERE id = %s", (session_id,))
    if not owner or owner == session_id:
        return session_id
    receipt = _meta.local_receipt(conn, owner)
    lineage = receipt.get("lineage") if receipt is not None else None
    return owner if isinstance(lineage, list) and session_id in lineage else session_id


def _derive(chain: GenerationChain, generation: int) -> str:
    if isinstance(chain, ApiGenerations):
        return str(api_generation_id(chain.session_id, generation))
    if isinstance(chain, RoomGenerations):
        return str(room_session_id(chain.identity, generation))
    return str(local_session_id(chain.profile_id, chain.principal_id,
                                generation_request_id(chain.base_request_id, generation)))


class PostgresSessionRuntimeRepository:
    """Runtime authority of the bound profile's schema, plus the suites' seeding seams."""

    def __init__(self, store_handle: PostgresStore) -> None:
        self._store = store_handle

    # ── the suites' seeding seams ──

    def seed_local_receipt(self, session_id: str, receipt: Mapping[str, Any]) -> None:
        from ..conversation.sessions_delete import LOCAL_RECEIPT_PREFIX

        with self._store.transaction() as conn:
            _meta.put(conn, LOCAL_RECEIPT_PREFIX + session_id, encode(dict(receipt)))

    def seed_retirement(self, session_id: str, *, expires: bool = False) -> None:
        from ..conversation._retirement import mark_retired

        with self._store.transaction() as conn:
            mark_retired(conn, [session_id], expiring=[session_id] if expires else [])

    # ── the epoch ──

    def begin_epoch(self, *, instance_id: str) -> int:
        text(instance_id)
        with self._store.transaction() as conn:
            return int(scalar(conn, "INSERT INTO runtime_epoch (singleton, epoch, instance_id) VALUES (true, 1, %s) "
                                    "ON CONFLICT (singleton) DO UPDATE SET epoch = runtime_epoch.epoch + 1, "
                                    "instance_id = excluded.instance_id RETURNING epoch", (instance_id,)))

    def current_epoch(self) -> Optional[int]:
        with self._store.transaction() as conn:
            epoch = scalar(conn, "SELECT epoch FROM runtime_epoch")
        return int(epoch) if epoch else None

    def has_mutation_receipt(self, principal_id: str, session_id: str, request_id: str) -> bool:
        for value in (principal_id, session_id, request_id):
            text(value)
        key, _digest = mutation_receipt_key(principal_id, session_id, request_id, operation="model", payload={},
                                            expected_revision=0)
        with self._store.transaction() as conn:
            return _meta.has(conn, key)

    @staticmethod
    def _replay(conn: Connection, key: str, digest: str) -> Any:
        """The stored receipt's result for ``key``, None when absent; another digest is ``admission_conflict``."""
        stored = _meta.get(conn, key)
        if stored is None:
            return None
        receipt = json.loads(stored) if isinstance(stored, str) else stored
        if receipt["digest"] != digest:
            raise AuthorityRefusal("admission_conflict")
        return receipt["result"]

    # ── deferred and chat picks ──

    def defer_model_pick(self, epoch: int, *, key: str, digest: str, session_id: str,
                         pick: DeferredPick) -> dict[str, Any]:
        parse_mutation(pick.operation, dict(pick.payload))
        entry = {"principal_id": pick.principal_id, "request_id": pick.request_id, "payload": encode(dict(pick.payload)),
                 "application_id": "deferred-pick:" + pick.request_id}
        with self._store.transaction() as conn:
            require_epoch(conn, epoch)
            replay = self._replay(conn, key, digest)
            if replay is not None:
                return replay
            session = require_session(conn, session_id, lock=True)
            slot = DEFERRED_PICK_PREFIX + session_id
            stored = _meta.get(conn, slot, lock=True)
            current = json.loads(stored) if stored is not None else {}
            if pick.operation == "reasoning":
                current.get("payload", {}).pop("reasoning", None)
                value, picked = {**current, "reasoning": entry}, {"effort": pick.payload["effort"]}
            else:
                older = "reasoning" in current and "reasoning" not in pick.payload
                value = {**entry, **({"reasoning": current["reasoning"]} if older else {})}
                picked = {"model": pick.shown or pick.payload["model"]}
            result = {"session_id": session_id, "operation": pick.operation, "status": "deferred", **picked,
                      "revision": int(session["runtime_revision"]), "execution_generation": int(session["runtime_generation"])}
            _meta.put(conn, slot, canonical_json(value))
            _meta.put(conn, key, canonical_json({"digest": digest, "result": result}))
        return json.loads(canonical_json(result))

    def deferred_pick(self, session_id: str) -> Optional[StoredDeferredPick]:
        with self._store.transaction() as conn:
            stored = _meta.get(conn, DEFERRED_PICK_PREFIX + session_id)
        return None if stored is None else StoredDeferredPick(pick=json.loads(stored), stored=stored)

    def forget_deferred_pick(self, session_id: str, *, stored: str) -> None:
        with self._store.transaction() as conn:
            _meta.delete(conn, DEFERRED_PICK_PREFIX + session_id, only_if=stored)

    def mark_deferred_pick_asked(self, session_id: str, *, stored: str, confirm: str) -> None:
        try:
            value = canonical_json({**json.loads(stored), "awaiting_confirm": confirm})
        except (TypeError, ValueError) as exc:
            raise AuthorityRefusal("invalid_params") from exc
        self._replace_slot(session_id, stored, value)

    def replace_deferred_pick(self, session_id: str, *, stored: str, pick: Mapping[str, Any]) -> str:
        value = canonical_json(dict(pick))
        self._replace_slot(session_id, stored, value)
        return value

    def _replace_slot(self, session_id: str, stored: str, value: str) -> None:
        with self._store.transaction() as conn:
            if _meta.get(conn, DEFERRED_PICK_PREFIX + session_id, lock=True) == stored:
                _meta.put(conn, DEFERRED_PICK_PREFIX + session_id, value)

    @staticmethod
    def _fenced_mount(conn: Connection, session_id: str, params: ChatPickParams) -> dict[str, Any]:
        """The session row when its revision and generation match the pick's expectations."""
        session = require_session(conn, session_id, lock=True)
        if int(session["runtime_revision"]) != params.expected_revision:
            raise revision_conflict(session_id)
        if int(session["runtime_generation"]) != params.expected_generation:
            raise stale("session", session_id)
        return session

    def prepare_chat_pick(self, epoch: int, *, key: str, digest: str, session_id: str,
                          params: ChatPickParams) -> dict[str, Any] | SessionRuntimeState:
        with self._store.transaction() as conn:
            require_epoch(conn, epoch)
            replay = self._replay(conn, key, digest)
            if replay is not None:
                return replay
            session = self._fenced_mount(conn, session_id, params)
        return SessionRuntimeState(revision=int(session["runtime_revision"]), generation=int(session["runtime_generation"]))

    def commit_chat_pick(self, epoch: int, *, key: str, digest: str, session_id: str, params: ChatPickParams,
                         result: Mapping[str, Any]) -> dict[str, Any]:
        committed = encode(dict(result))
        with self._store.transaction() as conn:
            require_epoch(conn, epoch)
            replay = self._replay(conn, key, digest)
            if replay is not None:
                return replay
            self._fenced_mount(conn, session_id, params)
            committed["revision"] = int(scalar(conn, "UPDATE sessions SET runtime_revision = runtime_revision + 1 "
                                                     "WHERE id = %s RETURNING runtime_revision", (session_id,)))
            _meta.put(conn, key, canonical_json({"digest": digest, "result": committed}))
        return committed

    def _receipt_results(self, conn: Connection) -> list[dict[str, Any]]:
        out = []
        for _key, value in _meta.scan(conn, RECEIPT_PREFIX):
            try:
                receipt = json.loads(value) if isinstance(value, str) else value
            except ValueError:
                continue
            if isinstance(receipt, dict) and isinstance(receipt.get("result"), dict):
                out.append(receipt["result"])
        return out

    def _picks_on(self, conn: Connection, results: list[dict[str, Any]], session_id: str, before: Any = None) -> set[str]:
        """The policy operations picked on the session (before revision ``before`` when given): committed receipts plus
        the local receipt's ``picks``, else the picks inherited from the branch parent."""
        picked = {r["operation"] for r in results
                  if r.get("operation") in ("model", "reasoning", "workspace") and (r.get("status") or "") != "deferred"
                  and r.get("session_id") == session_id and (before is None or (r.get("revision") is not None and r["revision"] < before))}
        receipt = _meta.local_receipt(conn, session_id)
        if receipt is None:
            return picked
        if "picks" in receipt:
            return picked | set(receipt["picks"])
        branched = next((r for r in results if r.get("operation") == "branch" and r.get("branched_session_id") == session_id), None)
        return picked | self._picks_on(conn, results, branched.get("session_id"), branched.get("revision")) if branched else picked

    def policy_picks(self, session_id: str) -> set[str]:
        with self._store.transaction() as conn:
            return self._picks_on(conn, self._receipt_results(conn), session_id)

    # ── viewer closes and retirement reads ──

    def store_viewer_close(self, epoch: int, *, session_id: str, record: str) -> None:
        with self._store.transaction() as conn:
            require_epoch(conn, epoch)
            _meta.put(conn, VIEWER_CLOSE_PREFIX + session_id, record)

    def forget_viewer_close(self, epoch: int, *, session_id: str, record: str) -> bool:
        with self._store.transaction() as conn:
            require_epoch(conn, epoch)
            return _meta.delete(conn, VIEWER_CLOSE_PREFIX + session_id, only_if=record)

    def viewer_close(self, session_id: str) -> Optional[str]:
        with self._store.transaction() as conn:
            return _meta.get(conn, VIEWER_CLOSE_PREFIX + session_id)

    def viewer_closes(self) -> list[tuple[str, str]]:
        with self._store.transaction() as conn:
            return [(key[len(VIEWER_CLOSE_PREFIX):], value) for key, value in _meta.scan(conn, VIEWER_CLOSE_PREFIX)]

    def retired_session(self, session_id: str) -> bool:
        with self._store.transaction() as conn:
            return scalar(conn, "SELECT 1 FROM retired_sessions WHERE session_id = %s", (session_id,)) is not None

    def expiring_retirements(self, session_ids: Sequence[str]) -> set[str]:
        with self._store.transaction() as conn:
            found = _rows(conn, "SELECT session_id FROM retired_sessions WHERE session_id = ANY(%s) AND expires",
                          (list(session_ids),))
        return {item["session_id"] for item in found}

    def first_generation(self, chain: GenerationChain, *, pinned: Optional[PinnedIdentity] = None,
                         start: int = 0) -> GenerationPick:
        generation = start
        with self._store.transaction() as conn:
            while True:
                sid = _derive(chain, generation)
                if scalar(conn, "SELECT 1 FROM retired_sessions WHERE session_id = %s", (sid,)) is None:
                    found = _row(conn, "SELECT title, source, archived FROM sessions WHERE id = %s", (sid,))
                    return GenerationPick(generation=generation, session_id=sid, row=None if found is None else GenerationRow(
                        title=found["title"], source=found["source"], archived=bool(found["archived"])))
                if pinned is not None and scalar(
                        conn, "SELECT 1 FROM admission_terminals WHERE principal_id = %s AND target_session_id = %s "
                              "AND request_id = %s", (pinned.principal_id, sid, pinned.request_id)) is not None:
                    return GenerationPick(generation=generation, session_id=sid, row=None)
                generation += 1

    # ── the ledger-gated ends ──

    def pending_ledger_status(self, session_key: str) -> Optional[str]:
        with self._store.transaction() as conn:
            return ledger_status(conn, session_key)

    def _end(self, conn: Connection, session_id: str, reason: str) -> None:
        execute(conn, "UPDATE sessions SET ended_at = %s, end_reason = %s WHERE id = %s", (time.time(), reason, session_id))
        bump_conversation_generation(conn, session_id, reason)

    def end_unless_ledger_busy(self, session_id: str, *, session_key: Optional[str], end_reason: str) -> bool:
        with self._store.transaction() as conn:
            session = _row(conn, "SELECT chat_id, ended_at FROM sessions WHERE id = %s FOR UPDATE", (session_id,))
            ids = {session_id, str(session_key or "")}
            if session is not None and str(session["chat_id"] or "").startswith("local-"):
                ids.add(str(session["chat_id"]))
            ids.add(local_lineage_owner(conn, session_id))
            if any(ledger_status(conn, sid) is not None for sid in filter(None, ids)):
                return False
            if session is not None and session["ended_at"] is None:
                self._end(conn, session_id, end_reason)
            return True

    def end_after_settlement(self, epoch: int, *, session_id: str, admission_id: str, generation: int,
                             admission_epoch: int, reason: str) -> bool:
        with self._store.transaction() as conn:
            session = _row(conn, "SELECT ended_at, runtime_generation FROM sessions WHERE id = %s FOR UPDATE", (session_id,))
            if session is None or session["ended_at"] is not None or int(session["runtime_generation"]) != generation:
                return False
            current = scalar(conn, "SELECT epoch FROM runtime_epoch")
            if not current or current != epoch:
                return False
            settled = _row(conn, "SELECT target_session_id, status, generation, owner_epoch FROM session_admissions "
                                 "WHERE admission_id = %s", (admission_id,))
            if (settled is None or settled["target_session_id"] != session_id or settled["status"] != "terminal"
                    or settled["generation"] != generation or settled["owner_epoch"] != admission_epoch):
                return False
            if scalar(conn, "SELECT 1 FROM session_admissions WHERE target_session_id = %s AND status <> 'terminal' "
                            "LIMIT 1", (session_id,)) is not None:
                return False
            self._end(conn, session_id, reason)
            return True


# Registered with its mutation verbs as ``mutation.PostgresRuntimeRepository``.
