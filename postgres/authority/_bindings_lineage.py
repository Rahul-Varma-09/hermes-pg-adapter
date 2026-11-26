"""The local-conversation lineage walks the authority repositories share: a creation receipt's validated tip (the
physical target a turn lands on), the row's local owner, and a session's runtime lineage.

Receipts are ``authority_meta`` values under ``binding:<creation id>``; a legacy binding's record sits under
``gateway.local_legacy.v1:<id>``. A receipt that does not validate is ``storage_unavailable``.
"""

from __future__ import annotations

import json
import time
from itertools import pairwise
from typing import Any, Optional

from psycopg import Connection

from persistence.authority.bindings import local_session_id
from persistence.core.errors import AuthorityRefusal

from .._sql import execute, row as _row, scalar
from ..conversation import _lineage
from . import _meta

RECEIPT_PREFIX = "binding:"
LEGACY_PREFIX = "gateway.local_legacy.v1:"
HISTORY_PREFIX = "history_control:"


def unavailable() -> AuthorityRefusal:
    return AuthorityRefusal("storage_unavailable")


def decode(value: Any) -> Any:
    """A stored receipt decoded: text is read as JSON (``ValueError`` when malformed), a stored value as is."""
    return json.loads(value) if isinstance(value, str) else value


def legacy_lineage_root(conn: Connection, receipt: dict[str, Any]) -> str:
    """The receipt's lineage root: its legacy session id when it binds one, else its session id."""
    root = receipt.get("legacy_session_id", receipt["session_id"])
    if "legacy_session_id" in receipt:
        if root != receipt["session_id"]:
            raise unavailable()
        expected = {k: receipt[k] for k in ("session_id", "profile_id", "principal_id", "legacy_session_id")}
        if decode(_meta.get(conn, LEGACY_PREFIX + root)) != expected:
            raise unavailable()
    return str(root)


def validate_local_lineage(conn: Connection, receipt: dict[str, Any]) -> str:
    """The validated receipt lineage's tip, else ``storage_unavailable``."""
    if ("legacy_session_id" not in receipt
            and receipt["session_id"] != local_session_id(receipt["profile_id"], receipt["principal_id"], receipt["request_id"])):
        raise unavailable()
    root = legacy_lineage_root(conn, receipt)
    lineage = receipt.get("lineage", [root])
    if (not isinstance(lineage, list) or not lineage or len(set(lineage)) != len(lineage)
            or lineage[0] != root or lineage[-1] != receipt["entry"]["session_id"]):
        raise unavailable()
    for parent_id, child_id in pairwise(lineage):
        parent, child = _lineage.lineage_row(conn, parent_id), _lineage.lineage_row(conn, child_id)
        if parent is None or child is None or child["parent_session_id"] != parent_id:
            raise unavailable()
        if parent["end_reason"] == "session_reset":
            valid = _lineage.config_of(child).get("_reset_from") == parent_id
        else:
            valid = parent["end_reason"] == "compression" and _lineage.chain_step(conn, parent_id) == child_id
        if not valid:
            raise unavailable()
    if _lineage.chain_step(conn, lineage[-1]) is not None:
        raise unavailable()
    return str(lineage[-1])


def local_physical_target(conn: Connection, session_id: str) -> str:
    """The receipt's validated tip; the id itself without a receipt."""
    saved = _meta.get(conn, RECEIPT_PREFIX + session_id)
    if saved is None:
        return session_id
    receipt = decode(saved)
    if receipt["session_id"] != session_id:
        raise unavailable()
    return validate_local_lineage(conn, receipt)


def local_lineage_owner(conn: Connection, session_id: str) -> str:
    """The row's local owner (its ``chat_id``) when the owner's receipt lineage holds the row, else the id itself."""
    owner = scalar(conn, "SELECT chat_id FROM sessions WHERE id = %s", (session_id,))
    if not owner or owner == session_id:
        return session_id
    saved = _meta.get(conn, RECEIPT_PREFIX + owner)
    try:
        lineage = decode(saved).get("lineage") if saved is not None else None
    except (TypeError, ValueError, AttributeError):
        return session_id
    return str(owner) if isinstance(lineage, list) and session_id in lineage else session_id


def runtime_lineage(conn: Connection, session_id: str) -> set[str]:
    """The session, its physical target, its local owner when the owner's receipt lineage holds it, and every
    compression relative of those."""
    targets = {session_id, local_physical_target(conn, session_id)}
    owner = scalar(conn, "SELECT chat_id FROM sessions WHERE id = %s", (session_id,))
    if owner:
        saved = _meta.get(conn, RECEIPT_PREFIX + owner)
        receipt: Optional[dict[str, Any]] = decode(saved) if saved is not None else None
        if receipt is not None and session_id in receipt.get("lineage", [owner]):
            targets.update({owner, local_physical_target(conn, owner)})
    expanded: set[str] = set()
    for sid in targets:
        expanded.update(_lineage.lineage_ids(conn, sid) or [sid])
    return targets | expanded


def local_conversation_ids(conn: Connection, session_id: str) -> list[str]:
    """The local owner and its receipt lineage, in order without repeats; ``[session_id]`` when no lineage list
    decodes."""
    owner = local_lineage_owner(conn, session_id)
    saved = _meta.get(conn, RECEIPT_PREFIX + owner)
    try:
        lineage = decode(saved).get("lineage") if saved is not None else None
    except (TypeError, ValueError, AttributeError):
        lineage = None
    if not isinstance(lineage, list):
        return [session_id]
    return list(dict.fromkeys([owner, *(sid for sid in lineage if isinstance(sid, str))]))


# ── the routing index rows the local writes commit with their receipt ──

def entry_text(entry: dict[str, Any]) -> str:
    return json.dumps(entry, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def route_exists(conn: Connection, route: str) -> bool:
    return scalar(conn, "SELECT 1 FROM gateway_routing WHERE scope = '' AND session_key = %s", (route,)) is not None


def routed_entry(conn: Connection, route: str, tip: str) -> Optional[dict[str, Any]]:
    """``route``'s routing entry while it still targets ``tip``, else None."""
    stored = scalar(conn, "SELECT entry_json FROM gateway_routing WHERE scope = '' AND session_key = %s FOR UPDATE", (route,))
    routed = json.loads(stored) if stored is not None else None
    return routed if isinstance(routed, dict) and routed.get("session_id") == tip else None


def store_route(conn: Connection, route: str, entry: dict[str, Any]) -> None:
    """Write ``entry`` as the routing index row of ``route`` in the ``""`` namespace."""
    target = entry.get("session_id")
    execute(conn, "INSERT INTO gateway_routing (scope, session_key, entry_json, session_id, updated_at) "
                  "VALUES ('', %s, %s, %s, %s) ON CONFLICT (scope, session_key) DO UPDATE "
                  "SET entry_json = excluded.entry_json, session_id = excluded.session_id",
            (route, entry_text(entry), target if isinstance(target, str) else None, time.time()))


def commit_local_target(conn: Connection, receipt: dict[str, Any], routed: dict[str, Any]) -> None:
    """Commit the route's entry (while the route exists), the receipt and the logical revision, together."""
    if route_exists(conn, receipt["route"]):
        store_route(conn, receipt["route"], routed)
    _meta.put(conn, RECEIPT_PREFIX + receipt["session_id"], receipt)
    execute(conn, "UPDATE sessions SET runtime_revision = runtime_revision + 1 WHERE id = %s", (receipt["session_id"],))


def advance_local_target(conn: Connection, parent_session_id: str, child_session_id: str, *,
                         entry: Optional[dict[str, Any]] = None) -> None:
    """Advance the parent's local receipt to ``child_session_id`` and commit it with the route entry:
    ``storage_unavailable`` for a receipt that does not match the parent or fails validation, ``admission_conflict``
    when the route no longer targets the parent."""
    parent = _row(conn, "SELECT chat_id, user_id, session_key FROM sessions WHERE id = %s", (parent_session_id,))
    logical_id = parent["chat_id"] if parent else None
    if not logical_id:
        return
    raw = _meta.get(conn, RECEIPT_PREFIX + logical_id, lock=True)
    if raw is None:
        if logical_id.startswith("local-") or _meta.has(conn, LEGACY_PREFIX + logical_id):
            raise unavailable()
        return
    try:
        receipt = json.loads(json.dumps(decode(raw)))
        if (receipt["session_id"] != logical_id or receipt["entry"]["session_id"] != parent_session_id
                or receipt["principal_id"] != parent["user_id"] or receipt["route"] != parent["session_key"]):
            raise unavailable()
        routed = routed_entry(conn, receipt["route"], parent_session_id)
        if routed is None:
            raise AuthorityRefusal("admission_conflict")
        candidate = dict(entry or routed, session_id=child_session_id)
        receipt["entry"] = candidate
        receipt["lineage"] = [*receipt.get("lineage", [logical_id]), child_session_id]
        validate_local_lineage(conn, receipt)
    except AuthorityRefusal:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise unavailable() from exc
    commit_local_target(conn, receipt, candidate)


def untracked_steps(conn: Connection, tip: str, lineage: list[str]) -> Optional[list[str]]:
    """Every compression step past ``tip`` in order; None when the chain leads back into itself."""
    steps: list[str] = []
    seen = {*lineage, tip}
    current = tip
    while (step := _lineage.chain_step(conn, current)) is not None:
        if step in seen:
            return None
        seen.add(step)
        steps.append(step)
        current = step
    return steps


def adoptable_compression(conn: Connection, receipt: dict[str, Any]) -> Optional[tuple[list[str], dict[str, Any]]]:
    """``(steps, routed)`` when every compression step past the tip carries the receipt's route, owner and principal
    and the route still targets the tip, else None."""
    tip = receipt["entry"]["session_id"]
    children = untracked_steps(conn, tip, receipt.get("lineage", [receipt["session_id"]]))
    if not children:
        return None
    identity = (receipt["route"], receipt["session_id"], receipt["principal_id"])
    for child_id in children:
        child = _row(conn, "SELECT session_key, chat_id, user_id FROM sessions WHERE id = %s", (child_id,))
        if child is None or (child["session_key"], child["chat_id"], child["user_id"]) != identity:
            return None
    routed = routed_entry(conn, receipt["route"], tip)
    return (children, routed) if routed is not None else None


def with_adopted_compression(receipt: dict[str, Any], children: list[str]) -> dict[str, Any]:
    """``receipt`` with ``children`` appended to its lineage and its entry moved to the last one; the caller
    validates the copy."""
    return dict(receipt, entry=dict(receipt["entry"], session_id=children[-1]),
                lineage=[*receipt.get("lineage", [receipt["session_id"]]), *children])
