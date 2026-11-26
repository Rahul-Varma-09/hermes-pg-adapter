"""Lineage walks over the ``sessions`` table, shared by the conversation repositories.

A child is a *fork* of its parent when it is a tool row or a fork marker in its ``model_config`` names the parent; it
is a *compression continuation* when it is no fork (reset included) and the parent ended by compression. A
conversation's compression lineage is the connected component over those continuation edges, and its *tip* is reached
by following each node's preferred continuation forward.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from psycopg import Connection

from persistence.conversation.sessions import END_REASON_COMPRESSION
from persistence.core.conventions import stored_instant

from .._sql import has_marker, row as _row, rows as _rows, scalar

FORK_MARKERS = ("_branched_from", "_delegate_from", "_reset_from")
RESET_END_REASONS = ("session_reset", "session_switch", "idle", "daily", "suspended", "resume_pending_expired")
CHAIN_CAP = 1000
WALK_NODES = 100

_LINEAGE_COLUMNS = ("id, parent_session_id, source, model_config, end_reason, ended_at, started_at, "
                    "last_activity_at, session_key")

# A compression continuation edge (child c of parent p), as SQL.
CONTINUATION_EDGE_SQL = (
    "p.end_reason = 'compression' AND coalesce(c.source, '') <> 'tool' AND NOT ("
    + " OR ".join(has_marker("c", marker) for marker in FORK_MARKERS) + ")")


def trusted(expression: str) -> str:
    """SQL: ``expression`` when it is a trusted epoch instant (``persistence.core.conventions``'s window), else NULL."""
    return f"(CASE WHEN {expression} >= 0 AND {expression} <= 4200000000 THEN {expression} END)"


def last_active_sql(alias: str) -> str:
    """SQL: the newest trusted of ``alias``'s activity stamp and its newest stored transcript row (any visibility),
    else its trusted ``started_at``; NULL when no cell is trusted."""
    newest = f"(SELECT max(lm.timestamp) FROM messages lm WHERE lm.session_id = {alias}.id)"
    return (f"coalesce(greatest({trusted(f'{alias}.last_activity_at')}, {trusted(newest)}), "
            f"{trusted(f'{alias}.started_at')})")


def forward_members_sql(alias: str) -> str:
    """SQL subquery: ``alias`` and every row reachable from it forward over compression continuation edges, as
    ``fwd(id)``; use it as ``WITH RECURSIVE {forward_members_sql('s')}``."""
    return (f"fwd(id) AS (SELECT {alias}.id UNION SELECT c.id FROM fwd JOIN sessions p ON p.id = fwd.id "
            f"JOIN sessions c ON c.parent_session_id = p.id WHERE {CONTINUATION_EDGE_SQL})")


def config_of(row: Mapping[str, Any]) -> dict[str, Any]:
    config = row.get("model_config")
    return config if isinstance(config, dict) else {}


def lineage_row(conn: Connection, session_id: Optional[str]) -> Optional[dict[str, Any]]:
    if not session_id:
        return None
    return _row(conn, f"SELECT {_LINEAGE_COLUMNS} FROM sessions WHERE id = %s", (session_id,))


def is_fork_child(row: Mapping[str, Any], *, include_reset: bool = False) -> bool:
    """A branch, delegate or tool child of its parent (``include_reset``: a reset fork too); a marker counts only when
    it names the parent, since a continuation inherits its parent's markers."""
    if row.get("source") == "tool":
        return True
    config = config_of(row)
    keys = FORK_MARKERS if include_reset else ("_branched_from", "_delegate_from")
    markers = [config.get(key) for key in keys]
    parent = row.get("parent_session_id")
    return parent in markers if parent else any(marker is not None for marker in markers)


def is_compression_child(conn: Connection, row: Mapping[str, Any]) -> bool:
    parent = lineage_row(conn, row.get("parent_session_id"))
    return (parent is not None and not is_fork_child(row, include_reset=True)
            and parent["end_reason"] == END_REASON_COMPRESSION)


def _descent_child(parent: Mapping[str, Any], row: Mapping[str, Any]) -> bool:
    """A child a resume descends into: no fork marker at all, no legacy reset child of a reset-ended parent on the
    same routing key, not a tool row."""
    config = config_of(row)
    if row.get("source") == "tool" or any(config.get(marker) is not None for marker in FORK_MARKERS):
        return False
    return not (parent["end_reason"] in RESET_END_REASONS and row.get("session_key")
                and row["session_key"] == parent.get("session_key"))


def last_active(conn: Connection, row: Mapping[str, Any]) -> Optional[float]:
    """The newest trusted of the activity stamp and the newest stored transcript row (any visibility), else a trusted
    ``started_at``; None when no cell is trusted."""
    sid = row["id"]
    newest = scalar(conn, "SELECT max(timestamp) FROM messages WHERE session_id = %s", (sid,))
    trusted = [instant for stamp in (row.get("last_activity_at"), newest)
               if (instant := stored_instant(stamp, session_id=sid, field="last_active")) is not None]
    chosen = max(trusted) if trusted else stored_instant(row.get("started_at"), session_id=sid, field="started_at")
    return None if chosen is None else chosen.timestamp()


def chain_step(conn: Connection, current: str) -> Optional[str]:
    """The preferred continuation of ``current``: one that compressed again first, then an open one, then the most
    recently active."""
    parent = lineage_row(conn, current)
    if parent is None or parent["end_reason"] != END_REASON_COMPRESSION:
        return None
    candidates = [child for child in _rows(conn, f"SELECT {_LINEAGE_COLUMNS} FROM sessions WHERE parent_session_id = %s",
                                           (current,))
                  if _descent_child(parent, child)]
    if not candidates:
        return None

    def rank(child: Mapping[str, Any]) -> int:
        if child["end_reason"] == END_REASON_COMPRESSION:
            return 0
        return 1 if child["ended_at"] is None else 2

    best = min(rank(child) for child in candidates)

    def recency(child: Mapping[str, Any]) -> tuple:
        active = last_active(conn, child)
        return (active is not None, active or 0.0, child["started_at"], child["id"])

    return str(max((child for child in candidates if rank(child) == best), key=recency)["id"])


def compression_tip(conn: Connection, session_id: str) -> str:
    """The live continuation of ``session_id``: its chain walked forward to the end."""
    walked = [session_id]
    for _ in range(CHAIN_CAP):
        child = chain_step(conn, walked[-1])
        if child is None or child in walked:
            break
        walked.append(child)
    return walked[-1]


UNSET_CREATED = (None, "", "unknown", "local")


def stamp_ancestors(conn: Connection, session_id: str) -> int:
    """Copy the session's creating surface onto compression ancestors without one, each hop the parent's preferred
    continuation; returns the rows stamped."""
    row = _row(conn, "SELECT created_source, parent_session_id FROM sessions WHERE id = %s", (session_id,))
    surface = row["created_source"] if row is not None else None
    if row is None or surface in UNSET_CREATED:
        return 0
    stamped = 0
    current, parent_id = session_id, row["parent_session_id"]
    for _ in range(CHAIN_CAP):
        parent = _row(conn, "SELECT created_source, parent_session_id FROM sessions WHERE id = %s", (parent_id,)) if parent_id else None
        if (parent is None or parent["created_source"] not in (*UNSET_CREATED, surface)
                or chain_step(conn, parent_id) != current):
            break
        if parent["created_source"] != surface:
            conn.execute("UPDATE sessions SET created_source = %s WHERE id = %s", (surface, parent_id))
            stamped += 1
        current, parent_id = parent_id, parent["parent_session_id"]
    return stamped


def is_compression_ancestor(conn: Connection, ancestor_id: str, descendant_id: str) -> bool:
    """Whether ``ancestor_id`` is a compression predecessor of ``descendant_id`` over continuation edges."""
    current = lineage_row(conn, descendant_id)
    seen = {descendant_id}
    while current is not None:
        parent = lineage_row(conn, current["parent_session_id"])
        if parent is None or parent["end_reason"] != END_REASON_COMPRESSION or parent["id"] in seen:
            return False
        if is_fork_child(current, include_reset=True):
            return False
        if parent["id"] == ancestor_id:
            return True
        seen.add(parent["id"])
        current = parent
    return False


def lineage_ids(conn: Connection, session_id: str) -> list[str]:
    """The connected component over compression continuation edges containing ``session_id``."""
    found = _rows(conn, f"""
        WITH RECURSIVE edges AS (
            SELECT c.id AS child, c.parent_session_id AS parent
            FROM sessions c JOIN sessions p ON p.id = c.parent_session_id
            WHERE {CONTINUATION_EDGE_SQL}
        ), component(id) AS (
            SELECT %s::text
            UNION
            SELECT CASE WHEN e.child = component.id THEN e.parent ELSE e.child END
            FROM edges e JOIN component ON component.id IN (e.child, e.parent)
        )
        SELECT component.id FROM component JOIN sessions s ON s.id = component.id""", (session_id,))
    return [found_row["id"] for found_row in found]


def resume_lineage_ids(conn: Connection, session_id: str) -> list[str]:
    """The rows a display resume materializes: the session and its verified compression ancestors."""
    row = lineage_row(conn, session_id)
    if row is None or is_fork_child(row):
        return [session_id]
    chain = [session_id]
    while len(chain) < WALK_NODES and not is_fork_child(row, include_reset=True):
        parent = lineage_row(conn, row["parent_session_id"])
        if parent is None or parent["end_reason"] != END_REASON_COMPRESSION or parent["id"] in chain:
            break
        chain.append(parent["id"])
        row = parent
    return chain
