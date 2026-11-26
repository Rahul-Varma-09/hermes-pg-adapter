"""``prepare_mutation``, ``mutate``, ``commit_refreshed_policy`` and ``delete_footprint`` of the Postgres runtime
repository, over the closed ``MutationCommand`` union.

A receipt is the ``authority_meta`` value under its ``gateway.mutation.v1.*`` key: the canonical JSON text of
``{"digest", "result"}``. Each ``mutate`` is one transaction: the fences (epoch, receipt replay, existence, revision,
generation, the live guard), the history-control write, the command's handler, the revision bumps and the receipt. The
handlers compose the session, message and lineage repositories' connection-level steps, which join this transaction.

``Compress`` commits the owner's prepared summary through the message and lineage repositories' own compaction and
publication bodies; ``Import`` runs the maintenance repository's import steps.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Iterable, Mapping, Optional

from psycopg import Connection

from persistence.authority.admissions import GateRefusal
from persistence.authority.bindings import local_session_id
from persistence.authority.runtime import (
    _GENERATION_REQUIRED, COMPACTED_TARGET_DETAIL, ClaimHistory, DeleteFootprint, HistoryWrite, ImportControl,
    LocalSnapshot, MutationCommand, MutationOutcome, PreparedMutation, SessionGone, mutation_receipt_key,
    parse_mutation,
)
from persistence.authority.vocabulary import OwnerRule
from persistence.conversation.lineage import CompressionChild
from persistence.conversation.sessions import TitleSource, sanitize_title
from persistence.core.canonical import canonical_json
from persistence.core.conventions import RowVisibility
from persistence.core.errors import AuthorityRefusal, CapabilityUnsupported, ProofRejected
from persistence.core.identity import SessionId
from session_rules.legacy_local import binding_names, bindings_agree, owners_agree, same_owner
from session_rules.route_entries import Platform, SessionEntry, SessionSource, _now, build_session_key
from session_rules.turns import typed_prompt_content

from .._sql import execute, row as _row, rows as _rows, scalar
from ..conversation import _lineage, _policy as policy, _retirement, _transcript as transcript_store
from ..conversation.sessions import bump_conversation_generation
from . import _meta
from ._bindings_lineage import (
    HISTORY_PREFIX, RECEIPT_PREFIX as LOCAL_RECEIPT_PREFIX, advance_local_target, local_conversation_ids,
    local_lineage_owner, local_physical_target, runtime_lineage, store_route, validate_local_lineage,
)
from .bindings import _owner_user, _routed, binding_text, history_row
from .runtime import (
    DEFERRED_PICK_PREFIX, PostgresSessionRuntimeRepository, encode, require_epoch, revision_conflict, stale, text,
)
from .._backend import register_repository

_POLICY_OPERATIONS = ("model", "reasoning", "workspace", "editor_mcp", "edit_policy")


def _checked(principal_id: Any, session_id: Any, request_id: Any, expected_revision: Any, command: Any,
             expected_generation: Any) -> tuple[MutationCommand, dict[str, Any], str, str]:
    for value in (principal_id, session_id, request_id):
        text(value)
    if type(expected_revision) is not int or expected_revision < 0:
        raise AuthorityRefusal("invalid_params")
    operation = getattr(command, "operation", None)
    payload_of = getattr(command, "payload", None)
    parsed = parse_mutation(operation, payload_of() if callable(payload_of) else None)
    if expected_generation is not None and (type(expected_generation) is not int or expected_generation < 0):
        raise AuthorityRefusal("invalid_params")
    if parsed.operation in _GENERATION_REQUIRED and expected_generation is None:
        raise AuthorityRefusal("invalid_params")
    payload = encode(parsed.payload())
    key, digest = mutation_receipt_key(principal_id, session_id, request_id, operation=parsed.operation, payload=payload,
                                       expected_revision=expected_revision, expected_generation=expected_generation)
    return parsed, payload, key, digest


def _profile_from_session_key(session_key: Any) -> Optional[str]:
    """The profile namespace of an ``agent:<ns>:...`` gateway session key, else None."""
    if not session_key:
        return None
    parts = str(session_key).split(":")
    if len(parts) < 2 or parts[0] != "agent":
        return None
    namespace = parts[1] or "main"
    if namespace == "main":
        return "default"
    return "main" if namespace == "main~" else namespace


def _load_text(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


class MutationMixin:
    """The receipted mutation verbs; the repository supplies ``_store``."""

    _store: Any

    # ── guards ──

    @staticmethod
    def _ledger_states(conn: Connection, session_id: str, admission_states: tuple[str, ...]) -> set[str]:
        return {item["status"] for item in _rows(
            conn, "SELECT status FROM session_admissions WHERE target_session_id = %s AND status = ANY(%s) "
                  "UNION SELECT status FROM worker_executions WHERE session_id = %s "
                  "AND status IN ('registered', 'running', 'unknown')", (session_id, list(admission_states), session_id))}

    @staticmethod
    def _refuse(states: set[str]) -> None:
        if "unknown" in states:
            raise AuthorityRefusal("unknown_execution")
        if states:
            raise AuthorityRefusal("session_busy")

    def _unleased(self, conn: Connection, session_id: str) -> None:
        """A live compaction or turn lease refuses, except on a row compression closed."""
        found = _row(conn, "SELECT ended_at, end_reason FROM sessions WHERE id = %s", (session_id,))
        if found is not None and found["end_reason"] == "compression" and found["ended_at"] is not None:
            return
        self._store.messages._refuse_live_leases(conn, session_id, ("compaction", "turn"))

    def _require_target_advanceable(self, conn: Connection, session_ids: Iterable[str]) -> None:
        """Executing work refuses, then a live transcript lease."""
        ids = list(session_ids)
        for sid in ids:
            self._refuse(self._ledger_states(conn, sid, ("started", "unknown")))
        for sid in ids:
            self._unleased(conn, sid)

    def _require_idle(self, conn: Connection, session_ids: Iterable[str]) -> None:
        """Any queued, started or unknown admission or live worker refuses, then a live transcript lease, per id."""
        for sid in session_ids:
            self._refuse(self._ledger_states(conn, sid, ("queued", "started", "unknown")))
            self._unleased(conn, sid)

    def _local_snapshot(self, conn: Connection, session_id: str) -> LocalSnapshot:
        """The receipt (``runtime_coordination_required`` without one), its validated target, the advanceable fence,
        the target's revision."""
        receipt = _meta.local_receipt(conn, session_id)
        if receipt is None:
            raise AuthorityRefusal("runtime_coordination_required")
        target = validate_local_lineage(conn, receipt)
        self._require_target_advanceable(conn, list(dict.fromkeys([session_id, target])))
        revision = int(scalar(conn, "SELECT runtime_revision FROM sessions WHERE id = %s", (target,)) or 0)
        return LocalSnapshot(receipt=receipt, target=target, target_revision=revision)

    def _validate_prepared(self, conn: Connection, session_id: str, prepared: Optional[PreparedMutation]) -> LocalSnapshot:
        """No prepared commit is ``runtime_coordination_required``; a snapshot that moved is ``revision_conflict``."""
        if not isinstance(prepared, PreparedMutation):
            raise AuthorityRefusal("runtime_coordination_required")
        current = self._local_snapshot(conn, session_id)
        if current != prepared.snapshot:
            raise revision_conflict(session_id)
        return current

    # ── the delete walk ──

    @staticmethod
    def _continuation_edges(conn: Connection, session_id: str) -> list[str]:
        """The continuation children of ``session_id`` when compression ended it, and its parent when compression
        ended that."""
        edges = [item["id"] for item in _rows(
            conn, f"SELECT c.id FROM sessions p JOIN sessions c ON c.parent_session_id = p.id "
                  f"WHERE p.id = %s AND {_lineage.CONTINUATION_EDGE_SQL} ORDER BY c.id", (session_id,))]
        parent = scalar(conn, f"SELECT p.id FROM sessions c JOIN sessions p ON p.id = c.parent_session_id "
                              f"WHERE c.id = %s AND {_lineage.CONTINUATION_EDGE_SQL}", (session_id,))
        return [*edges, *([parent] if parent else [])]

    @staticmethod
    def _same_history_owner(conn: Connection, anchor_id: str, session_id: str, rule: Any) -> bool:
        """Whether a member shares the anchor's history owner."""
        columns = "SELECT id, source, user_id, session_key, chat_id, origin_json, profile_name FROM sessions WHERE id = %s"
        anchor, found = _row(conn, columns, (anchor_id,)), _row(conn, columns, (session_id,))
        if anchor is None or found is None:
            return False
        if _routed(found):
            return all(found[k] == anchor[k] for k in ("user_id", "session_key", "chat_id"))
        binding = binding_text(_meta.get(conn, HISTORY_PREFIX + session_id))
        if not bindings_agree(binding, binding_text(_meta.get(conn, HISTORY_PREFIX + anchor_id)), rule):
            return False
        routeless_peer = found["source"] == anchor["source"] and not _routed(anchor)
        return binding is not None or ((history_row(found, rule) or routeless_peer)
                                       and (_owner_user(found) in (None, "")
                                            or owners_agree(_owner_user(found), anchor["user_id"], rule)))

    def _delete_targets(self, conn: Connection, session_id: str, rule: Any = None) -> list[str]:
        """The id, its local owner and the owner's validated receipt lineage, then compression relatives of the same
        history owner and delegate subtrees to a fixed point; the id first, the rest sorted."""
        targets = {session_id}
        owner = local_lineage_owner(conn, session_id)
        targets.add(owner)
        receipt = _meta.local_receipt(conn, owner)
        if receipt is not None:
            validate_local_lineage(conn, receipt)
            targets.update(receipt.get("lineage", [session_id]))
        anchor = dict.fromkeys(targets, session_id)
        frontier = list(targets)
        for _ in range(_lineage.CHAIN_CAP):
            found: dict[str, str] = {}
            for sid in frontier:
                for nxt in self._continuation_edges(conn, sid):
                    if nxt not in targets and nxt not in found and self._same_history_owner(conn, anchor[sid], nxt, rule):
                        found[nxt] = anchor[sid]
            for sid in self._store.sessions._delegate_children(conn, frontier):
                if sid not in targets:
                    found.setdefault(sid, sid)
            if not found:
                break
            anchor.update(found)
            targets.update(found)
            frontier = list(found)
        return [session_id, *sorted(targets - {session_id})]

    def delete_footprint(self, session_id: SessionId, owner: Any) -> DeleteFootprint:
        with self._store.transaction() as conn:
            targets = self._delete_targets(conn, str(session_id), owner)
            children = [item["id"] for item in _rows(conn, "SELECT id FROM sessions WHERE parent_session_id = ANY(%s)",
                                                     (targets,))]
        return DeleteFootprint(targets=tuple(SessionId(sid) for sid in targets),
                               children=tuple(SessionId(sid) for sid in children))

    # ── the fences ──

    def _fenced(self, conn: Connection, epoch: int, session_id: str, parsed: MutationCommand, key: str, digest: str,
                expected_revision: int, expected_generation: Optional[int], live_guard: Optional[GateRefusal],
                owner: Optional[OwnerRule] = None) -> MutationOutcome | dict[str, Any] | None:
        """Epoch, receipt replay, existence, revision, generation and the live guard, in that order; the replayed
        outcome, or the session row."""
        require_epoch(conn, epoch)
        stored = _meta.get(conn, key)
        if stored is not None:
            receipt = _load_text(stored)
            if receipt["digest"] != digest:
                raise AuthorityRefusal("admission_conflict")
            return MutationOutcome(result=receipt["result"], replayed=True, gone=None)
        session = _row(conn, "SELECT * FROM sessions WHERE id = %s FOR UPDATE", (session_id,))
        if session is None and parsed.operation != "import":
            raise AuthorityRefusal("not_found")
        if (session["runtime_revision"] if session is not None else 0) != expected_revision:
            raise revision_conflict(session_id)
        if expected_generation is not None and (session is None or session["runtime_generation"] != expected_generation):
            raise stale("session", session_id)
        if live_guard is not None:
            if parsed.operation == "delete":
                self._delete_targets(conn, session_id, owner)
            raise live_guard.refusal()
        return session

    def prepare_mutation(self, epoch: int, *, principal_id: str, session_id: str, request_id: str, expected_revision: int,
                         command: MutationCommand, expected_generation: int | None = None,
                         live_guard: GateRefusal | None = None) -> LocalSnapshot | MutationOutcome:
        parsed, _payload, key, digest = _checked(principal_id, session_id, request_id, expected_revision, command,
                                                 expected_generation)
        with self._store.transaction() as conn:
            fenced = self._fenced(conn, epoch, session_id, parsed, key, digest, expected_revision, expected_generation,
                                  live_guard)
            if isinstance(fenced, MutationOutcome):
                return fenced
            snapshot = self._local_snapshot(conn, session_id)
            if parsed.operation == "branch":
                self._require_idle(conn, list(dict.fromkeys([session_id, snapshot.target])))
            return snapshot

    def mutate(self, epoch: int, *, principal_id: str, session_id: str, request_id: str, expected_revision: int,
               command: MutationCommand, expected_generation: int | None = None, live_guard: GateRefusal | None = None,
               history: HistoryWrite | None = None, prepared: PreparedMutation | None = None,
               media_root: str | None = None, owner: OwnerRule | None) -> MutationOutcome:
        parsed, payload, key, digest = _checked(principal_id, session_id, request_id, expected_revision, command,
                                                expected_generation)
        with self._store.transaction() as conn:
            fenced = self._fenced(conn, epoch, session_id, parsed, key, digest, expected_revision, expected_generation,
                                  live_guard, owner)
            if isinstance(fenced, MutationOutcome):
                return fenced
            session = fenced
            if isinstance(history, ClaimHistory):
                self._authorize_history(conn, history.actor, session_id, owner)
            elif isinstance(history, ImportControl):
                for sid in history.session_ids:
                    if scalar(conn, "SELECT 1 FROM sessions WHERE id = %s", (sid,)) is None:
                        _meta.put(conn, HISTORY_PREFIX + sid, [history.actor.profile_id, history.actor.subject])
            affected, projection = self._apply(conn, session_id, parsed, payload, prepared, media_root, owner)
            gone = None
            if parsed.operation == "delete":
                gone = SessionGone(tuple(projection.get("deleted_ids") or (session_id,)), "delete")
            elif payload.get("archived") is True and session is not None and not session["archived"]:
                gone = SessionGone(tuple(affected) or (session_id,), "archive")
            if affected:
                execute(conn, "UPDATE sessions SET runtime_revision = runtime_revision + 1 WHERE id = ANY(%s)",
                        (sorted(affected),))
            updated = _row(conn, "SELECT runtime_revision, runtime_generation FROM sessions WHERE id = %s", (session_id,))
            result = {"session_id": session_id,
                      "revision": updated["runtime_revision"] if updated else expected_revision + 1,
                      "execution_generation": updated["runtime_generation"] if updated else expected_generation,
                      "operation": parsed.operation, **projection}
            _meta.put(conn, key, canonical_json({"digest": digest, "result": result}))
        return MutationOutcome(result=json.loads(canonical_json(result)), replayed=False, gone=gone)

    def _authorize_history(self, conn: Connection, actor: Any, session_id: str, owner: Optional[OwnerRule]) -> None:
        """Authorize and claim the row's history, refusing a row that holds no claimable history."""
        found = _row(conn, "SELECT * FROM sessions WHERE id = %s", (session_id,))
        if found is None or _routed(found):
            raise ProofRejected("session", session_id)
        key, binding = HISTORY_PREFIX + session_id, [actor.profile_id, actor.subject]
        saved = _meta.get(conn, key)
        if saved is not None:
            if not binding_names(binding_text(saved), actor.subject, owner):
                raise ProofRejected("session", session_id)
            return
        if not history_row(found, owner):
            raise ProofRejected("session", session_id)
        if _owner_user(found) and not same_owner(found["user_id"], actor.subject, owner):
            raise ProofRejected("session", session_id)
        if not actor.may_create:
            raise ProofRejected("session", session_id)
        _meta.put(conn, key, binding)

    # ── the command handlers ──

    def _apply(self, conn: Connection, session_id: str, command: MutationCommand, payload: dict[str, Any],
               prepared: Optional[PreparedMutation], media_root: Optional[str],
               owner: Any) -> tuple[set[str], dict[str, Any]]:
        operation = command.operation
        if operation in _POLICY_OPERATIONS:
            return getattr(self, f"_{operation}")(conn, session_id, payload, prepared)
        if operation == "reset":
            return self._reset(conn, session_id)
        if operation == "branch":
            return self._branch(conn, session_id, payload, prepared)
        if operation == "compress":
            return self._compress(conn, session_id, prepared)
        if operation == "delete":
            return self._delete(conn, session_id, media_root, owner)
        if operation == "import":
            return self._import(conn, session_id, payload, owner)
        if operation == "rewind":
            return self._rewind(conn, session_id, payload)
        return self._metadata(conn, session_id, payload)

    def _metadata(self, conn: Connection, session_id: str, payload: Mapping[str, Any]) -> tuple[set[str], dict[str, Any]]:
        sessions = self._store.sessions
        affected: set[str] = set()
        result: dict[str, Any] = {}
        for key, value in payload.items():
            if key == "title":
                # The listing shows a compressed conversation under its live tip's title, so the rename lands there;
                # every segment can rename that tip, so a committed write advances the whole lineage's revision.
                tip = _lineage.compression_tip(conn, session_id) or session_id
                written = sessions._title_write(conn, tip, value, TitleSource.USER)
                affected.update(written)
                if written:
                    lineage = _lineage.lineage_ids(conn, session_id) or [session_id]
                    affected.update(lineage)
                    if not sanitize_title(value):
                        # An untitled tip lists under an ancestor's title, so a clear clears the lineage.
                        for sid in lineage:
                            affected.update(sessions._title_write(conn, sid, value, TitleSource.USER))
                result[key] = scalar(conn, "SELECT title FROM sessions WHERE id = %s", (tip,))
                continue
            column = "last_read_at" if key == "unread" else key
            values: dict[str, Any] = {column: (0.0 if value else time.time()) if key == "unread" else bool(value)}
            if key == "archived":
                values["auto_archived"] = False
            affected.update(sessions._set_lineage_flag(conn, session_id, values))
            if key == "archived" and value:
                affected.update(sessions._retire_canonical_bot_chat(conn, session_id))
            result[key] = value
        return affected, result

    def _delete(self, conn: Connection, session_id: str, media_root: Optional[str],
                owner: Any) -> tuple[set[str], dict[str, Any]]:
        targets = self._delete_targets(conn, session_id, owner)
        self._require_idle(conn, targets)
        for sid in targets:
            bump_conversation_generation(conn, sid, "session_reset")
        execute(conn, "UPDATE sessions SET runtime_revision = runtime_revision + 1 WHERE parent_session_id = ANY(%s) "
                      "AND NOT id = ANY(%s)", (targets, targets))
        _retirement.retire(conn, targets, media_root=media_root)
        return set(), {"deleted_ids": targets}

    @staticmethod
    def _summarized_by_compaction(conn: Connection, physical: str, message_id: int) -> bool:
        """Whether ``message_id`` is a user row compaction summarized: archived on the physical target, or shown on
        another session of its compression lineage."""
        found = _row(conn, "SELECT session_id, role, active, compacted FROM messages WHERE id = %s", (message_id,))
        if found is None or found["role"] != "user":
            return False
        if found["session_id"] == physical:
            return not found["active"] and bool(found["compacted"])
        if not (found["active"] or found["compacted"]):
            return False
        return found["session_id"] in (_lineage.lineage_ids(conn, physical) or [physical])

    def _rewind(self, conn: Connection, session_id: str, payload: Mapping[str, Any]) -> tuple[set[str], dict[str, Any]]:
        physical = local_physical_target(conn, session_id)
        self._require_idle(conn, list(dict.fromkeys([session_id, physical])))
        target_id = payload["target_message_id"]
        rows = transcript_store.load(conn, [physical], lock=True)
        row = next((r for r in rows if r.id == target_id), None)
        if row is None or row.visibility is not RowVisibility.ACTIVE or row.message.role != "user":
            if self._summarized_by_compaction(conn, physical, target_id):
                raise AuthorityRefusal("invalid_params", field="target_message_id", detail=COMPACTED_TARGET_DETAIL)
            raise AuthorityRefusal("invalid_params", field="target_message_id",
                                    detail="target user message is no longer in session history")
        m = row.message
        handoff, live = policy.split_user_originated_turn(
            {"role": m.role, "content": m.content, "display_kind": m.display_kind, "display_metadata": m.display_metadata,
             "_compressed_summary": m.compressed_summary})
        carrier = live if handoff is not None else None
        preserve = carrier is not None or bool(payload.get("preserve_compaction_handoff", False))
        target = transcript_store._replay_dict(row, include_row_ids=True, include_summary_markers=True,
                                               stamp_snapshot=False)
        active = tuple(r.id for r in rows if r.visibility is RowVisibility.ACTIVE)
        rewound, scaffold, _target = self._store.messages._rewind_rows(conn, physical, int(target_id), active,
                                                                        preserve_handoff=preserve)
        if carrier is not None:
            target["content"] = carrier.get("content")
        target["content"] = typed_prompt_content(target.get("content"), target)
        execute(conn, "UPDATE sessions SET runtime_generation = runtime_generation + 1 WHERE id = %s", (session_id,))
        head = scalar(conn, "SELECT max(id) FROM messages WHERE session_id = %s AND active", (physical,))
        return {session_id, physical}, {"rewound_count": rewound, "target_message": target, "new_head_id": head,
                                        "replacement_message_id": scaffold.id if scaffold is not None else None}

    def _import(self, conn: Connection, session_id: str, payload: Mapping[str, Any],
                owner: Any) -> tuple[set[str], dict[str, Any]]:
        importer = getattr(getattr(self._store, "maintenance", None), "_import_in_transaction", None)
        if importer is None:
            raise CapabilityUnsupported("mutation.import")
        return importer(conn, session_id, payload, owner)

    def _compress(self, conn: Connection, session_id: str,
                  prepared: Optional[PreparedMutation]) -> tuple[set[str], dict[str, Any]]:
        """The prepared summary committed in place on the target, or published as a fresh compression child the local
        receipt and the conversation's name move to."""
        from ..workers.sessions import new_message

        snapshot = self._validate_prepared(conn, session_id, prepared)
        policy_, target = snapshot.receipt["policy"], snapshot.target
        messages = [new_message(m) for m in prepared.values["messages"]]
        if prepared.values["in_place"]:
            child = target
            self._store.messages.compact(SessionId(target), messages)
            affected = {session_id, target}
        else:
            child = uuid.uuid4().hex
            self._store.lineage.publish_compression_child(
                SessionId(target), CompressionChild(session_id=SessionId(child), source=policy_["source"], model=policy_["model"],
                                                    model_config={"_compressed_from": target}, cwd=policy_["cwd"]),
                messages, require_compression_lease=False)
            advance_local_target(conn, target, child)
            self._store.bindings._title_follows_head(conn, target, child)
            affected = set()  # The local receipt's advance already moved the logical revision.
        generation = int(scalar(conn, "UPDATE sessions SET runtime_generation = runtime_generation + 1 WHERE id = %s "
                                      "RETURNING runtime_generation", (session_id,)))
        return affected, {"target_session_id": child, "previous_target_session_id": target, "execution_generation": generation,
                          "message_count": len(messages), "summary": prepared.values.get("summary")}

    # ── the policy commands ──

    def _commit_policy(self, conn: Connection, session_id: str, prepared: Optional[PreparedMutation]) -> tuple[LocalSnapshot, int]:
        """Validate the prepared snapshot, store the prepared policy on the local receipt and bump the execution
        generation."""
        snapshot = self._validate_prepared(conn, session_id, prepared)
        receipt = dict(snapshot.receipt)
        receipt["policy"] = json.loads(json.dumps(dict(prepared.values["policy"])))
        _meta.put(conn, LOCAL_RECEIPT_PREFIX + session_id, receipt)
        generation = int(scalar(conn, "UPDATE sessions SET runtime_generation = runtime_generation + 1 WHERE id = %s "
                                      "RETURNING runtime_generation", (session_id,)))
        return snapshot, generation

    def _model(self, conn: Connection, session_id: str, payload: Mapping[str, Any],
               prepared: Optional[PreparedMutation]) -> tuple[set[str], dict[str, Any]]:
        """The target row takes the chosen model and provider; a deferred pick keeps only its reasoning level, dropped
        when the payload sets one."""
        snapshot, generation = self._commit_policy(conn, session_id, prepared)
        chosen = prepared.values["policy"]
        model, config = chosen["model"], json.loads(chosen["config_json"])["model"]
        current = _lineage.config_of(_row(conn, "SELECT model_config FROM sessions WHERE id = %s", (snapshot.target,)) or {})
        execute(conn, "UPDATE sessions SET model = %s, model_config = %s WHERE id = %s",
                (model, json.dumps({**current, "provider": config["provider"]}), snapshot.target))
        key = DEFERRED_PICK_PREFIX + session_id
        saved = _meta.get(conn, key)
        level = _load_text(saved).get("reasoning") if saved is not None and "reasoning" not in payload else None
        if level is None:
            _meta.delete(conn, key)
        else:
            _meta.put(conn, key, canonical_json({"reasoning": level}))
        return {session_id, snapshot.target}, {"model": model, "provider": config["provider"],
                                               "execution_generation": generation,
                                               **({"effort": payload["reasoning"]} if "reasoning" in payload else {})}

    def _reasoning(self, conn: Connection, session_id: str, payload: Mapping[str, Any],
                   prepared: Optional[PreparedMutation]) -> tuple[set[str], dict[str, Any]]:
        """The deferred pick loses its reasoning level; the slot goes when it holds no application."""
        snapshot, generation = self._commit_policy(conn, session_id, prepared)
        key = DEFERRED_PICK_PREFIX + session_id
        saved = _meta.get(conn, key)
        if saved is not None:
            pick = _load_text(saved)
            pick.pop("reasoning", None)
            pick.get("payload", {}).pop("reasoning", None)
            if "application_id" in pick:
                _meta.put(conn, key, canonical_json(pick))
            else:
                _meta.delete(conn, key)
        return {session_id, snapshot.target}, {"effort": payload["effort"], "execution_generation": generation}

    def _workspace(self, conn: Connection, session_id: str, payload: Mapping[str, Any],
                   prepared: Optional[PreparedMutation]) -> tuple[set[str], dict[str, Any]]:
        """Every row of the local conversation and the target take the cwd and git metadata, the git generation
        bumped."""
        snapshot, generation = self._commit_policy(conn, session_id, prepared)
        cwd, git = prepared.values["policy"]["cwd"], prepared.values["git"]
        ids = list(dict.fromkeys([*local_conversation_ids(conn, session_id), snapshot.target]))
        execute(conn, "UPDATE sessions SET cwd = %s, git_branch = %s, git_repo_root = %s, "
                      "git_metadata_generation = git_metadata_generation + 1 WHERE id = ANY(%s)",
                (cwd, git["branch"] or None, git["root"] or None, ids))
        return {session_id, snapshot.target}, {"cwd": cwd, "branch": git["branch"] or None,
                                               "git_repo_root": git["root"] or None, "execution_generation": generation}

    def _editor_mcp(self, conn: Connection, session_id: str, payload: Mapping[str, Any],
                    prepared: Optional[PreparedMutation]) -> tuple[set[str], dict[str, Any]]:
        snapshot, generation = self._commit_policy(conn, session_id, prepared)
        return {session_id, snapshot.target}, {"tools_changed": prepared.values["tools_changed"],
                                               "mcp_servers": prepared.values["mcp_servers"],
                                               "execution_generation": generation}

    def _edit_policy(self, conn: Connection, session_id: str, payload: Mapping[str, Any],
                     prepared: Optional[PreparedMutation]) -> tuple[set[str], dict[str, Any]]:
        snapshot, generation = self._commit_policy(conn, session_id, prepared)
        return {session_id, snapshot.target}, {"edit_approval_policy": payload["policy"],
                                               "execution_generation": generation}

    def commit_refreshed_policy(self, epoch: int, *, session_id: str, old: Mapping[str, Any], new: Mapping[str, Any],
                                generation: int) -> int:
        with self._store.transaction() as conn:
            require_epoch(conn, epoch)
            snapshot = self._local_snapshot(conn, session_id)
            self._require_idle(conn, list(dict.fromkeys([session_id, snapshot.target])))
            current = scalar(conn, "SELECT runtime_generation FROM sessions WHERE id = %s", (session_id,))
            if snapshot.receipt["policy"] != old or current is None or current != generation:
                raise revision_conflict(session_id)
            receipt = dict(snapshot.receipt, policy=encode(dict(new)))
            _meta.put(conn, LOCAL_RECEIPT_PREFIX + session_id, receipt)
            return int(scalar(conn, "UPDATE sessions SET runtime_generation = runtime_generation + 1 WHERE id = %s "
                                    "RETURNING runtime_generation", (session_id,)))

    # ── the publications ──

    def _local_receipt_or_refuse(self, conn: Connection, session_id: str) -> tuple[dict[str, Any], str]:
        receipt = _meta.local_receipt(conn, session_id)
        if receipt is None:
            raise AuthorityRefusal("runtime_coordination_required")
        return receipt, validate_local_lineage(conn, receipt)

    def _reset(self, conn: Connection, session_id: str) -> tuple[set[str], dict[str, Any]]:
        """A fresh child of the local conversation's target, the receipt and route advanced to it."""
        receipt, target = self._local_receipt_or_refuse(conn, session_id)
        # An explicit reset revokes unlinked unknown compute; admission-linked work needs its own discard.
        execute(conn, "UPDATE worker_executions SET status = 'terminal' WHERE session_id = ANY(%s) AND status = 'unknown' "
                      "AND NOT starts_with(execution_id, 'admission-worker:')", (sorted(runtime_lineage(conn, session_id)),))
        self._require_target_advanceable(conn, list(dict.fromkeys([session_id, target])))
        parent = _row(conn, "SELECT * FROM sessions WHERE id = %s FOR UPDATE", (target,))
        policy_ = receipt["policy"]
        previous = SessionEntry.from_dict(receipt["entry"])
        now = _now()
        child_id = uuid.uuid4().hex
        entry = SessionEntry(previous.session_key, child_id, now, now, origin=previous.origin, platform=previous.platform,
                             chat_type=previous.chat_type, display_name=previous.display_name, is_fresh_reset=True)
        self._store.lineage._insert_child(conn, parent, CompressionChild(
            session_id=SessionId(child_id), source=policy_["source"], model=policy_["model"],
            model_config={"_reset_from": target}, cwd=policy_["cwd"], profile_name=parent["profile_name"]))
        execute(conn, "UPDATE sessions SET ended_at = %s, end_reason = 'session_reset' WHERE id = %s", (time.time(), target))
        bump_conversation_generation(conn, target, "session_reset")
        advance_local_target(conn, target, child_id, entry=entry.to_dict())
        generation = int(scalar(conn, "UPDATE sessions SET runtime_generation = runtime_generation + 1 WHERE id = %s "
                                      "RETURNING runtime_generation", (session_id,)))
        # advance_local_target already advanced the logical owner's revision.
        return set(), {"target_session_id": child_id, "previous_target_session_id": target,
                       "execution_generation": generation}

    @staticmethod
    def _branch_boundary(conn: Connection, target: str, through: Any) -> int:
        """The highest row id the branch copies; the tool results right after an assistant boundary belong to it."""
        if through is None:
            return int(scalar(conn, "SELECT coalesce(max(id), 0) FROM messages WHERE session_id = %s AND active", (target,)))
        role = scalar(conn, "SELECT role FROM messages WHERE id = %s AND session_id = %s AND active", (through, target))
        if role is None:
            raise AuthorityRefusal("invalid_params")
        if role == "assistant":
            for later in _rows(conn, "SELECT id, role FROM messages WHERE session_id = %s AND active AND id > %s ORDER BY id",
                               (target, through)):
                if later["role"] != "tool":
                    break
                through = later["id"]
        return int(through)

    @staticmethod
    def _child_policy(session_id: str, parent_policy: Mapping[str, Any], child: str,
                      prepared: Optional[PreparedMutation]) -> dict[str, Any]:
        """The parent's frozen policy at Default edit approval, with the child's own prepared editor selection."""
        child_policy = dict(parent_policy)
        if "edit_approval_policy" in child_policy:
            child_policy["edit_approval_policy"] = "ask"
        if prepared is None:
            if child_policy.get("editor_mcp_json"):
                raise AuthorityRefusal("runtime_coordination_required")
            return child_policy
        values = prepared.values
        if child != values["child"] or child_policy.get("editor_mcp_json") != values["parent_editor_mcp_json"]:
            raise revision_conflict(session_id)
        return {**child_policy, "editor_mcp_json": values["editor_mcp_json"], "toolsets": values["toolsets"],
                "config_secret_ref": values["config_secret_ref"]}

    def _branch(self, conn: Connection, session_id: str, payload: Mapping[str, Any],
                prepared: Optional[PreparedMutation]) -> tuple[set[str], dict[str, Any]]:
        """An independent local route with the target's active transcript (through the boundary) and frozen policy."""
        saved, target = self._local_receipt_or_refuse(conn, session_id)
        self._require_idle(conn, list(dict.fromkeys([session_id, target])))
        request_id = prepared.values["request_id"] if prepared is not None else "branch:" + uuid.uuid4().hex
        child = local_session_id(saved["profile_id"], saved["principal_id"], request_id)
        child_policy = self._child_policy(session_id, saved["policy"], child, prepared)
        boundary = self._branch_boundary(conn, target, payload.get("through_message_id"))
        source = SessionSource(Platform.LOCAL, child, user_id=saved["principal_id"], chat_type="dm",
                               profile=saved["entry"]["origin"].get("profile"))
        route = build_session_key(source, profile=_profile_from_session_key(saved["route"]))
        now = _now()
        entry = SessionEntry(route, child, now, now, origin=source, platform=Platform.LOCAL)
        parent = _row(conn, "SELECT * FROM sessions WHERE id = %s", (target,))
        self._store.lineage._insert_child(conn, parent, CompressionChild(
            session_id=SessionId(child), source=child_policy["source"], model=child_policy["model"],
            model_config={"_branched_from": target}, cwd=child_policy["cwd"], profile_name=parent["profile_name"]))
        execute(conn, "UPDATE sessions SET session_key = %s, chat_id = %s, origin_json = %s, system_prompt_hash = %s "
                      "WHERE id = %s", (route, child, canonical_json(source.to_dict()), parent["system_prompt_hash"], child))
        copied = [r for r in transcript_store.load(conn, [target])
                  if r.visibility is RowVisibility.ACTIVE and r.id <= boundary]
        transcript_store.clone_rows(conn, child, copied)
        transcript_store.recount(conn, child)
        sessions = self._store.sessions
        if payload.get("title"):
            sessions._title_write(conn, child, payload["title"], TitleSource.USER)
        else:
            # Untitled branch: the next name in the parent's lineage, its own titled sidebar row.
            sessions._title_write(conn, child, sessions.next_title_in_lineage(parent["title"] or "branch"),
                                  TitleSource.DERIVED)
        receipt = dict(profile_id=saved["profile_id"], principal_id=saved["principal_id"], request_id=request_id,
                       session_id=child, route=route, entry=entry.to_dict(), policy=child_policy)
        # The copied policy holds the parent's picks; the child's refresh must keep them as picks.
        receipt["picks"] = sorted(self._picks_on(conn, self._receipt_results(conn), session_id))
        store_route(conn, route, json.loads(canonical_json(entry.to_dict())))
        _meta.put(conn, LOCAL_RECEIPT_PREFIX + child, json.loads(canonical_json(receipt)))
        return {session_id}, {"branched_session_id": child, "copied_messages": len(copied)}


class PostgresRuntimeRepository(MutationMixin, PostgresSessionRuntimeRepository):
    """The runtime repository: the epoch, picks, closes and ledger ends, and the receipted mutations."""


register_repository("runtime", PostgresRuntimeRepository, {"authority_fences": True})
