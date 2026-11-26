"""Listings, counts and multi-row reads of the Postgres session repository, mixed into ``PostgresSessionRepository``.

The listing filters run in SQL (sources, archive and hidden state, cwd and workspace prefixes, the children rule,
message counts and the local reset lineages a listing leaves out), so a start-ordered page is windowed by Postgres.
A last-active page ranks each row by the newest activity over its forward compression chain, also in SQL. Projection
of a compression root to its live tip, previews and lineage names are filled per listed row.
"""

from __future__ import annotations

import dataclasses
import json
import re
import time
from typing import Any, Mapping, Optional, Sequence

from psycopg import Connection

from persistence import SessionId
from persistence.authority.vocabulary import OwnerRule
from persistence.conversation.sessions import (
    AUTO_LOAD_SCAFFOLD_PREFIX, BOUNDARY_END_REASONS, CANONICAL_BOT_CHAT_TITLE, END_REASON_COMPRESSION,
    ERROR_FINISH_REASONS, INTERNAL_LISTING_SOURCES, LOCAL_CHAT_PREFIX, SKILL_SCAFFOLD_PREFIX, CronRunEntry,
    CronRunIdentity, LiveListingFacts, LiveListingRow, PendingAdmissionRef, PickerFacts, PickerRoot, PickerSegment,
    RecentBrowseQuery, RecentBrowseTimeout, RecentSessionEntry, RosterSession, RosterSessions, RosterWorker,
    SessionCountQuery, SessionCwdUsage, SessionEdge, SessionFacetEntry, SessionLifecycleStatus, SessionListEntry,
    SessionListRow, SessionOrder, SessionQuery, SessionRecencyEntry, SessionRecord, SessionWindow,
    SkillScaffoldedSession,
)
from session_rules import turns
from session_rules.legacy_local import same_owner

from .._sql import has_marker, row as _row, rows as _rows, scalar
from . import _lineage
from .sessions_delete import LOCAL_RECEIPT_PREFIX

_PREVIEW_MAX_CHARS = 60
_CRON_PRINCIPAL_ID = "cron-owner"
_CRON_REQUEST_PREFIX = "cron:"
_LEGACY_CRON_RUN_ID = re.compile(r"^cron_(.+)_\d{8}_\d{6}$")
_LEGACY_CRON_STAMP = re.compile(r"[0-9]{8}_[0-9]{6}")
_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_LOWER = "abcdefghijklmnopqrstuvwxyz"
SOURCE_OR_BLANK = "coalesce(s.source, '')"
CHAT_ID_OR_BLANK = "coalesce(s.chat_id, '')"


def _sql_list(values: Sequence[str]) -> str:
    return "ARRAY[" + ", ".join("'" + value.replace("'", "''") + "'" for value in values) + "]::text[]"


_RESET_REASONS = _sql_list(_lineage.RESET_END_REASONS)
_BOUNDARY_REASONS = _sql_list(sorted(BOUNDARY_END_REASONS))
_LAST_ACTIVE = _lineage.last_active_sql("s")
# Listing rows select the prompt but not the pin, and the row's last active.
_LISTED = f"""
    SELECT s.*, sp.prompt AS system_prompt, NULL::text AS tool_pin_text, {_LAST_ACTIVE} AS last_active
    FROM sessions s LEFT JOIN system_prompts sp ON sp.hash = s.system_prompt_hash"""


def _fold(expression: str) -> str:
    """SQL ASCII-only case fold, the SQLite store's ``LIKE`` comparison."""
    return f"translate({expression}, '{_UPPER}', '{_LOWER}')"


def _ascii_lower(text: str) -> str:
    return text.translate(str.maketrans(_UPPER, _LOWER))


def _under_sql(column: str, prefix: str) -> tuple[str, list[Any]]:
    """The cwd-prefix match: the prefix itself, or below it (either separator, ASCII case folded)."""
    head = prefix.rstrip("/\\") or prefix
    return (f"({column} = %s OR starts_with({_fold(column)}, %s) OR starts_with({_fold(column)}, %s))",
            [head, _ascii_lower(head + "/"), _ascii_lower(head + "\\")])


def _skill_preview_text(text: str) -> str:
    described = turns.describe_skill_invocation(text)
    return str(described) if described is not None else text.split(turns.SKILL_EXCERPT_JOINT)[0]


def shape_preview(raw: Any) -> str:
    """The short first-turn preview a listing shows."""
    text = str(raw or "").strip()
    if not text:
        return ""
    text = _skill_preview_text(text.replace("\n", " ").replace("\r", " "))
    return text[:_PREVIEW_MAX_CHARS] + "..." if len(text) > _PREVIEW_MAX_CHARS else text


def _recency(last_active: Optional[float]) -> tuple[bool, float]:
    return (last_active is not None, last_active or 0.0)


def _window_sql(window: SessionWindow) -> tuple[str, list[Any]]:
    return " LIMIT %s OFFSET %s", [window.limit, window.offset]


def _classify(role: str, has_tool_calls: bool, finish_reason: Optional[str]) -> SessionLifecycleStatus:
    if (finish_reason or "").strip().lower() in ERROR_FINISH_REASONS:
        return SessionLifecycleStatus.ERROR
    normalized = (role or "").strip().lower()
    if normalized in {"user", "tool"} or (normalized == "assistant" and has_tool_calls):
        return SessionLifecycleStatus.INTERRUPTED
    return SessionLifecycleStatus.COMPLETE


def _cron_run_from_request(request_id: Any, job_id: Optional[str] = None) -> Optional[tuple[str, str]]:
    """``(job_id, execution_id)`` of a cron fire's request id."""
    if not isinstance(request_id, str) or not request_id.startswith(_CRON_REQUEST_PREFIX):
        return None
    if job_id is None:
        rest = request_id[len(_CRON_REQUEST_PREFIX):]
        if rest.count(":") != 1:
            return None
        job_id, _, execution_id = rest.partition(":")
    else:
        prefix = f"{_CRON_REQUEST_PREFIX}{job_id}:"
        execution_id = request_id[len(prefix):] if request_id.startswith(prefix) else ""
    return (job_id, execution_id) if job_id and execution_id else None


class SessionListingMixin:
    """Listing and read verbs; the repository supplies ``_store``."""

    _store: Any

    # ── filters ──

    @staticmethod
    def _admitted_child_sql(include_subagents: bool) -> str:
        """The children filter: a root, a branch or reset child (marker, or the legacy heuristics); delegate runs only
        with ``include_subagents``."""
        listable = f"""(s.parent_session_id IS NULL OR {has_marker('s', '_branched_from')}
            OR {has_marker('s', '_reset_from')} OR EXISTS (
                SELECT 1 FROM sessions p WHERE p.id = s.parent_session_id AND (
                    (p.end_reason = 'branched' AND p.ended_at IS NOT NULL AND s.started_at >= p.ended_at)
                    OR (p.end_reason = ANY({_RESET_REASONS}) AND coalesce(s.session_key, '') <> ''
                        AND s.session_key = p.session_key))))"""
        delegate = has_marker("s", "_delegate_from")
        return f"({listable} OR {delegate})" if include_subagents else f"({listable} AND NOT {delegate})"

    def _filters(self, query: Any, *, exclude_children: bool, superseded: Sequence[str] = (),
                 hidden_ok: bool = True) -> tuple[list[str], list[Any]]:
        """The shared listing and count filters over a :class:`SessionQuery` or :class:`SessionCountQuery`."""
        where: list[str] = []
        params: list[Any] = []
        if exclude_children:
            where.append(self._admitted_child_sql(query.include_subagents))
        if superseded:
            where.append("NOT s.id = ANY(%s)")
            params.append(list(superseded))
        if not hidden_ok:
            where.append("NOT s.hidden")
        if query.sources:
            if query.local_created_sources:
                where.append("(CASE WHEN s.source = 'local' THEN s.created_source = ANY(%s) ELSE s.source = ANY(%s) END)")
                params += [list(query.local_created_sources), list(query.sources)]
            else:
                where.append("s.source = ANY(%s)")
                params.append(list(query.sources))
        session_key = getattr(query, "session_key", None)
        if session_key:
            where.append("s.session_key = %s")
            params.append(session_key)
        if query.exclude_sources:
            where.append("NOT coalesce(s.source = ANY(%s), false)")
            params.append(list(query.exclude_sources))
        if query.cwd_prefix:
            clause, values = _under_sql("s.cwd", query.cwd_prefix)
            where.append(clause)
            params += values
        workspace = getattr(query, "workspace_key", None)
        if workspace:
            head = workspace.rstrip("/\\") or workspace
            clause, values = _under_sql("s.cwd", head)
            where.append(f"(CASE WHEN coalesce(s.git_repo_root, '') <> '' THEN s.git_repo_root = %s ELSE {clause} END)")
            params += [head, *values]
        if query.min_message_count > 0:
            where.append("s.message_count >= %s")
            params.append(query.min_message_count)
        if query.archived_only:
            where.append("s.archived")
        elif not query.include_archived:
            where.append("NOT s.archived")
        return where, params

    def _local_lineage_index(self, conn: Connection) -> tuple[list[str], dict[str, list[str]]]:
        """``(superseded, heads)`` of the local reset lineages: earlier segments a listing leaves out, and the latest
        segment's whole lineage."""
        from ..authority._meta import scan

        parsed = []
        for _key, receipt in scan(conn, LOCAL_RECEIPT_PREFIX):
            lineage = receipt.get("lineage") if isinstance(receipt, dict) else None
            if (isinstance(receipt, dict) and isinstance(receipt.get("session_id"), str) and isinstance(lineage, list)
                    and len(lineage) >= 2 and all(isinstance(sid, str) for sid in lineage)):
                parsed.append(lineage)
        reasons = {item["id"]: item["end_reason"] for item in _rows(
            conn, "SELECT id, end_reason FROM sessions WHERE id = ANY(%s)",
            ([sid for lineage in parsed for sid in lineage],))}
        superseded: list[str] = []
        heads: dict[str, list[str]] = {}
        for lineage in parsed:
            starts = [sid for i, sid in enumerate(lineage) if i == 0 or reasons.get(lineage[i - 1]) == "session_reset"]
            superseded += starts[:-1]
            heads[starts[-1]] = lineage
        return superseded, heads

    # ── listing entries ──

    def _preview(self, conn: Connection, session_id: str) -> str:
        """The shaped first user turn with content that is not hidden, by (timestamp, row id)."""
        content = scalar(conn, "SELECT content FROM messages WHERE session_id = %s AND role = 'user' "
                               "AND content IS NOT NULL AND coalesce(display_kind, '') <> 'hidden' "
                               "ORDER BY timestamp, id LIMIT 1", (session_id,))
        return shape_preview(content) if content is not None else ""

    def _listed_row(self, conn: Connection, session_id: str) -> Optional[dict[str, Any]]:
        return _row(conn, _LISTED + " WHERE s.id = %s", (session_id,))

    @staticmethod
    def _projected(root: SessionRecord, tip: SessionRecord) -> SessionRecord:
        """A compression root carrying its tip's surfaced fields; the root keeps its ``started_at`` and accounting."""
        return dataclasses.replace(
            root, session_id=tip.session_id, ended_at=tip.ended_at, end_reason=tip.end_reason,
            message_count=tip.message_count, tool_call_count=tip.tool_call_count,
            title=tip.title if tip.title is not None else root.title, model=tip.model, system_prompt=tip.system_prompt,
            cwd=tip.cwd, git_branch=tip.git_branch, git_repo_root=tip.git_repo_root)

    def _entry(self, conn: Connection, row: Mapping[str, Any], *, project: bool = False, compact: bool = False,
               heads: Optional[Mapping[str, list[str]]] = None) -> SessionListEntry:
        """One listing row; ``project`` projects a compression root to its tip, ``heads`` names a local lineage's
        segments on its head row."""
        from .sessions import session_record

        record = session_record(row, resolve_tool_pin=False)
        surfaced = row
        root_id: Optional[str] = None
        chain: list[str] = []
        if project and row["end_reason"] == END_REASON_COMPRESSION:
            chain = [row["id"]]
            while (child := _lineage.chain_step(conn, chain[-1])) is not None and child not in chain:
                chain.append(child)
            tip = self._listed_row(conn, chain[-1]) if len(chain) > 1 else None
            if tip is not None:
                surfaced = tip
                record = self._projected(record, session_record(tip, resolve_tool_pin=False))
                root_id = row["id"]
            else:
                chain = []
        found = (heads or {}).get(root_id or row["id"])
        lineage_ids = list(dict.fromkeys([*found, *chain])) if found is not None else chain
        if compact:
            record = dataclasses.replace(record, system_prompt=None, system_prompt_hash=None, git_metadata_generation=0)
        last_active = surfaced["last_active"]
        return SessionListEntry(
            session=record,
            preview=self._preview(conn, surfaced["id"]),
            # Unread: activity (else started_at) postdates the lineage-stamped last_read_at (unset reads as read).
            unread=record.last_read_at is not None and (last_active or record.started_at) > record.last_read_at,
            last_active=last_active,
            lineage_root_id=SessionId(root_id) if root_id else None,
            lineage_ids=tuple(SessionId(sid) for sid in lineage_ids),
            continuation_kind="compression" if root_id else None,
            compact=compact,
        )

    def list_sessions(self, query: SessionQuery, window: SessionWindow = SessionWindow(limit=20), *,
                      compact_rows: bool = False) -> tuple[SessionListEntry, ...]:
        exclude_children = not query.include_children
        # The archived-only view is the recovery surface, so it admits hidden rows.
        hidden_ok = query.include_hidden or query.archived_only
        with self._store.transaction() as conn:
            superseded, heads = self._local_lineage_index(conn) if exclude_children else ([], {})
            where, params = self._filters(query, exclude_children=exclude_children, superseded=superseded,
                                          hidden_ok=hidden_ok)
            if query.order is SessionOrder.LAST_ACTIVE:
                listed = self._last_active_page(conn, query, where, params, window)
            else:
                limit, limit_params = _window_sql(window)
                listed = _rows(conn, f"{_LISTED} WHERE {' AND '.join(where) or 'true'} "
                                     f"ORDER BY s.started_at DESC, s.id DESC{limit}", (*params, *limit_params))
            if query.include_pinned:
                # A pin is an explicit keep: the back-fill ignores the archived filter.
                pin_query = dataclasses.replace(query, include_archived=True, archived_only=False)
                pin_where, pin_params = self._filters(pin_query, exclude_children=exclude_children,
                                                      superseded=superseded, hidden_ok=hidden_ok)
                listed += _rows(conn, f"{_LISTED} WHERE {' AND '.join(pin_where) or 'true'} AND s.pinned "
                                      f"AND NOT s.id = ANY(%s) ORDER BY s.started_at DESC, s.id DESC",
                                (*pin_params, [item["id"] for item in listed]))
            project = query.project_compression_tips and exclude_children
            return tuple(self._entry(conn, item, project=project, compact=compact_rows, heads=heads)
                         for item in listed)

    def _last_active_page(self, conn: Connection, query: SessionQuery, where: list[str], params: list[Any],
                          window: SessionWindow) -> list[dict[str, Any]]:
        """A LAST_ACTIVE page: the ``id_query``/``search_text`` filters and the ranking both read the row's forward
        compression chain."""
        member_last_active = _lineage.last_active_sql("m")
        where, params = list(where), list(params)
        id_needle = (query.id_query or "").strip().lower()
        search = (query.search_text or "").strip().lower()
        if id_needle:
            where.append(f"EXISTS (SELECT 1 FROM unnest(chain.members) member WHERE strpos({_fold('member')}, %s) > 0)")
            params.append(id_needle)
        if search:
            compact = re.sub(r"[\W_]+", "", search)
            title = _fold("coalesce(m.title, '')")
            stripped = f"replace(replace(replace(replace({title}, '-', ''), '_', ''), '.', ''), ' ', '')"
            where.append(f"""EXISTS (SELECT 1 FROM sessions m WHERE m.id = ANY(chain.members) AND (
                strpos({title}, %s) > 0 OR strpos({_fold('m.id')}, %s) > 0
                OR (%s <> '' AND strpos({stripped}, %s) > 0)))""")
            params += [search, search, compact, compact]
        limit, limit_params = _window_sql(window)
        return _rows(conn, f"""
            {_LISTED}
            CROSS JOIN LATERAL (
                WITH RECURSIVE {_lineage.forward_members_sql('s')}
                SELECT array_agg(fwd.id) AS members, max({member_last_active}) AS effective
                FROM fwd JOIN sessions m ON m.id = fwd.id
            ) chain
            WHERE {' AND '.join(where) or 'true'}
            ORDER BY chain.effective IS NOT NULL DESC, chain.effective DESC NULLS LAST, s.started_at DESC, s.id DESC
            {limit}""", (*params, *limit_params))

    def search_by_id(self, text: str, *, limit: int = 20, include_archived: bool = True,
                     sources: Optional[Sequence[str]] = None,
                     exclude_sources: Sequence[str] = ()) -> tuple[SessionListEntry, ...]:
        needle = (text or "").strip().lower()
        if not needle or limit <= 0:
            return ()
        query = SessionQuery(sources=tuple(sources) if sources else None, exclude_sources=tuple(exclude_sources),
                             include_archived=include_archived, order=SessionOrder.LAST_ACTIVE, id_query=needle)
        candidates = self.list_sessions(query, SessionWindow(limit=limit * 4))

        def rank(entry: SessionListEntry) -> int:
            ids = [str(value).lower() for value in (entry.session.session_id, entry.lineage_root_id) if value]
            if needle in ids:
                return 0
            return 1 if any(value.startswith(needle) for value in ids) else 2

        ranked = sorted(enumerate(candidates), key=lambda item: (rank(item[1]), item[0]))
        return tuple(entry for _, entry in ranked[:limit])

    def summary(self, session_id: SessionId, *, compact_rows: bool = False) -> Optional[SessionListEntry]:
        with self._store.transaction() as conn:
            found = self._listed_row(conn, session_id)
            return None if found is None else self._entry(conn, found, compact=compact_rows)

    def list_row(self, session_id: SessionId) -> Optional[SessionListRow]:
        from .sessions import session_record

        with self._store.transaction() as conn:
            found = self._listed_row(conn, session_id)
        if found is None:
            return None
        record = session_record(found, resolve_tool_pin=False)
        return SessionListRow(session_id=record.session_id, source=record.source, title=record.title,
                              started_at=record.started_at, ended_at=record.ended_at,
                              message_count=record.message_count, model=record.model, hidden=record.hidden,
                              archived=record.archived)

    # ── counts ──

    def _count_where(self, conn: Connection, query: SessionCountQuery) -> tuple[list[str], list[Any]]:
        superseded = self._local_lineage_index(conn)[0] if query.exclude_children else []
        return self._filters(query, exclude_children=query.exclude_children, superseded=superseded)

    def count_sessions(self, query: SessionCountQuery = SessionCountQuery()) -> int:
        with self._store.transaction() as conn:
            where, params = self._count_where(conn, query)
            return int(scalar(conn, f"SELECT count(*) FROM sessions s WHERE {' AND '.join(where) or 'true'}", params))

    def count_at_least(self, minimum: int = 1) -> bool:
        with self._store.transaction() as conn:
            return int(scalar(conn, "SELECT count(*) FROM (SELECT 1 FROM sessions LIMIT %s) found",
                              (max(minimum, 0),))) >= minimum

    def count_by_source(self, *, include_archived: bool = False, archived_only: bool = False,
                        exclude_children: bool = False) -> Mapping[str, int]:
        query = SessionCountQuery(include_archived=include_archived, archived_only=archived_only,
                                  exclude_children=exclude_children)
        with self._store.transaction() as conn:
            where, params = self._count_where(conn, query)
            found = _rows(conn, f"SELECT coalesce(nullif(s.source, ''), 'cli') AS source, count(*) AS n "
                                f"FROM sessions s WHERE {' AND '.join(where) or 'true'} "
                                f"GROUP BY 1 ORDER BY n DESC, min(s.started_at)", params)
        return {item["source"]: int(item["n"]) for item in found}

    # ── recency reads ──

    _RECENCY_ORDER = (f" ORDER BY {_LAST_ACTIVE} IS NOT NULL DESC, {_LAST_ACTIVE} DESC NULLS LAST, "
                      f"s.started_at DESC, s.id DESC")

    def _recency_entries(self, found: Sequence[Mapping[str, Any]]) -> tuple[SessionRecencyEntry, ...]:
        from .sessions import session_record

        return tuple(SessionRecencyEntry(session=session_record(item, resolve_tool_pin=False),
                                         last_active=item["last_active"]) for item in found)

    def search_sessions(self, *, sources: Optional[Sequence[str]] = None, workspace_key: Optional[str] = None,
                        local_created_sources: Optional[Sequence[str]] = None, skip_launch_stubs: bool = False,
                        window: SessionWindow = SessionWindow(limit=20)) -> tuple[SessionRecencyEntry, ...]:
        query = SessionQuery(sources=tuple(sources) if sources else None, workspace_key=workspace_key,
                             local_created_sources=tuple(local_created_sources) if local_created_sources else None,
                             include_archived=True)
        where, params = self._filters(query, exclude_children=False)
        if skip_launch_stubs:
            # A launch stub has neither messages nor an admission.
            where.append("(s.message_count > 0 OR EXISTS (SELECT 1 FROM session_admissions a "
                         "WHERE a.target_session_id = s.id))")
        limit, limit_params = _window_sql(window)
        with self._store.transaction() as conn:
            found = _rows(conn, f"{_LISTED} WHERE {' AND '.join(where) or 'true'}{self._RECENCY_ORDER}{limit}",
                          (*params, *limit_params))
        return self._recency_entries(found)

    def list_gateway_sessions(self, *, platform: Optional[str] = None,
                              active_only: bool = True) -> tuple[SessionRecencyEntry, ...]:
        # Winner pick first (ties kept), filters second.
        where = ["s.session_key IS NOT NULL",
                 "s.started_at = (SELECT max(k.started_at) FROM sessions k WHERE k.session_key = s.session_key)"]
        params: list[Any] = []
        if platform:
            where.append(f"{_fold(SOURCE_OR_BLANK)} = %s")
            params.append(_ascii_lower(platform))
        if active_only:
            where.append("s.ended_at IS NULL")
        with self._store.transaction() as conn:
            found = _rows(conn, f"{_LISTED} WHERE {' AND '.join(where)}{self._RECENCY_ORDER}", params)
        return self._recency_entries(found)

    def lifecycle_statuses(self, session_ids: Sequence[SessionId]) -> Mapping[SessionId, SessionLifecycleStatus]:
        wanted = [str(sid) for sid in session_ids if sid]
        with self._store.transaction() as conn:
            last = {item["session_id"]: item for item in _rows(conn, """
                SELECT DISTINCT ON (session_id) session_id, role, tool_calls, finish_reason
                FROM messages WHERE session_id = ANY(%s) ORDER BY session_id, id DESC""", (wanted,))}
        statuses: dict[SessionId, SessionLifecycleStatus] = {}
        for sid in wanted:
            item = last.get(sid)
            statuses[SessionId(sid)] = (SessionLifecycleStatus.EMPTY if item is None else _classify(
                item["role"], bool(item["tool_calls"] and item["tool_calls"] not in ("[]", "null")), item["finish_reason"]))
        return statuses

    def descendants(self, session_id: SessionId) -> tuple[SessionEdge, ...]:
        # The edge: no delegate, branch or reset intent (judged by a nonblank marker), not a tool row, not a legacy
        # reset child of its parent.
        no_intent = " AND ".join(
            f"NOT coalesce(jsonb_typeof(c.model_config -> '{marker}') = 'string' "
            f"AND btrim(c.model_config ->> '{marker}') <> '', false)"
            for marker in _lineage.FORK_MARKERS)
        with self._store.transaction() as conn:
            found = _rows(conn, f"""
                WITH RECURSIVE tree(id) AS (
                    SELECT id FROM sessions WHERE id = %s
                    UNION
                    SELECT c.id FROM tree JOIN sessions c ON c.parent_session_id = tree.id
                    JOIN sessions p ON p.id = c.parent_session_id
                    WHERE {no_intent} AND coalesce(c.source, '') <> 'tool'
                      AND NOT coalesce(coalesce(c.session_key, '') <> '' AND p.session_key = c.session_key
                                        AND p.end_reason = ANY({_RESET_REASONS}), false)
                )
                SELECT s.id, s.parent_session_id, s.started_at FROM tree JOIN sessions s ON s.id = tree.id
                ORDER BY s.started_at, s.id""", (session_id,))
        return tuple(SessionEdge(session_id=SessionId(item["id"]), parent_session_id=item["parent_session_id"],
                                 started_at=item["started_at"]) for item in found)

    # ── the bounded recent browse ──

    def list_recent_sessions(self, query: RecentBrowseQuery) -> tuple[RecentSessionEntry, ...]:
        deadline = time.monotonic() + query.timeout_seconds

        def check_deadline() -> None:
            if time.monotonic() >= deadline:
                raise RecentBrowseTimeout(f"recent-session browse exceeded {query.timeout_seconds:g}s deadline")

        with self._store.transaction() as conn:
            check_deadline()
            candidates = [item["id"] for item in _rows(conn, f"""
                SELECT s.id FROM sessions s
                WHERE NOT s.archived AND NOT s.hidden AND NOT {has_marker('s', '_delegate_from')}
                  AND NOT coalesce(s.source = ANY(%s), false)
                ORDER BY coalesce(s.last_activity_at, s.started_at) DESC, s.started_at DESC, s.id DESC
                LIMIT %s""", (list(query.exclude_sources), query.candidate_limit))]
            edges = _rows(conn, f"""SELECT c.id AS child, c.parent_session_id AS parent
                FROM sessions c JOIN sessions p ON p.id = c.parent_session_id WHERE {_lineage.CONTINUATION_EDGE_SQL}""")
            parent_of = {edge["child"]: edge["parent"] for edge in edges}
            children: dict[str, list[str]] = {}
            for edge in edges:
                children.setdefault(edge["parent"], []).append(edge["child"])
            check_deadline()
            # Up to each candidate's root, then down to every tip, each walk under its own row ceiling.
            ancestors = _bounded_walk(candidates, lambda sid: [parent_of[sid]] if sid in parent_of else [],
                                      query.lineage_limit)
            roots = [sid for sid in ancestors if sid not in parent_of]
            chain = _bounded_walk(roots, lambda sid: children.get(sid, []), query.lineage_limit)
            check_deadline()
            tips_by_root: dict[str, list[str]] = {}
            for sid, root_id in chain.items():
                if not children.get(sid):
                    tips_by_root.setdefault(root_id, []).append(sid)
            listed = {item["id"]: item for item in _rows(conn, _LISTED + " WHERE s.id = ANY(%s)",
                                                         (list({*tips_by_root, *chain}),))}
            ranked = []
            for root_id, tips in tips_by_root.items():
                root = listed.get(root_id)
                if root is not None and self._is_listable_root(conn, root):
                    tip = max((listed[sid] for sid in tips if sid in listed),
                              key=lambda item: (*_recency(item["last_active"]), item["id"]))
                    ranked.append((tip["last_active"], root, tip))
            ranked.sort(key=lambda item: (*_recency(item[0]), item[1]["started_at"], item[2]["id"]), reverse=True)
            check_deadline()
            return tuple(self._recent_entry(conn, *item) for item in ranked[: query.limit])

    def _is_listable_root(self, conn: Connection, row: Mapping[str, Any]) -> bool:
        config = _lineage.config_of(row)
        if row["archived"] or row["hidden"] or config.get("_delegate_from") is not None:
            return False
        if row["parent_session_id"] is None:
            return True
        for marker in ("_branched_from", "_reset_from"):
            value = config.get(marker)
            if isinstance(value, str) and value.strip():
                return True
        parent = _row(conn, "SELECT end_reason, session_key FROM sessions WHERE id = %s", (row["parent_session_id"],))
        # Reset continuation: the parent ended at a deliberate boundary and the child rides its routing key.
        return bool(parent is not None and parent["end_reason"] in BOUNDARY_END_REASONS and row["session_key"]
                    and row["session_key"] == parent["session_key"])

    def _recent_entry(self, conn: Connection, last_active: Optional[float], root: Mapping[str, Any],
                      tip: Mapping[str, Any]) -> RecentSessionEntry:
        from .sessions import session_record

        record = session_record(tip, resolve_tool_pin=False)
        return RecentSessionEntry(
            session_id=record.session_id, source=record.source, model=record.model,
            title=record.title if record.title is not None else root["title"], started_at=root["started_at"],
            ended_at=record.ended_at, end_reason=record.end_reason, message_count=record.message_count,
            tool_call_count=record.tool_call_count, last_active=last_active, preview=self._preview(conn, tip["id"]),
            lineage_root_id=SessionId(root["id"]) if root["id"] != tip["id"] else None,
            # The browse selects no last_read_at, so its unread rule always reads False.
            unread=False)

    # ── cron runs ──

    def _policy_cron_job(self, conn: Connection, session_id: str) -> Optional[str]:
        """The cron job a local creation receipt's request names, else ``None``."""
        from ..authority._meta import local_receipt

        receipt = local_receipt(conn, session_id)
        try:
            request = (receipt.get("policy") or {}).get("request_json") if receipt is not None else None
            parsed = json.loads(request) if isinstance(request, str) else request
        except ValueError:
            return None
        job = (parsed.get("cron_job") or {}).get("id") if isinstance(parsed, dict) else None
        return job if isinstance(job, str) else None

    def _cron_admission_run(self, conn: Connection, session_id: str,
                            job_id: Optional[str] = None) -> Optional[tuple[str, str]]:
        """``(job, execution)`` of the first scheduler admission of the session (``job_id``: only that job's)."""
        admissions = _rows(conn, "SELECT request_id FROM session_admissions WHERE principal_id = %s "
                                 "AND target_session_id = %s ORDER BY seq", (_CRON_PRINCIPAL_ID, session_id))
        for admission in admissions if job_id is not None else admissions[:1]:
            run = (_cron_run_from_request(admission["request_id"])
                   or _cron_run_from_request(admission["request_id"], self._policy_cron_job(conn, session_id)))
            if run is not None and job_id in (None, run[0]):
                return run
        return None

    def list_cron_job_runs(self, job_id: str,
                           window: SessionWindow = SessionWindow(limit=20)) -> tuple[CronRunEntry, ...]:
        with self._store.transaction() as conn:
            runs: list[tuple[dict[str, Any], Optional[str]]] = []
            for item in _rows(conn, _LISTED + " WHERE s.source = 'cron' ORDER BY s.started_at DESC, s.id DESC"):
                run = self._cron_admission_run(conn, item["id"], job_id)
                if run is not None:
                    runs.append((item, run[1]))
                elif item["id"].startswith(f"cron_{job_id}_") and _LEGACY_CRON_STAMP.fullmatch(
                        item["id"][len(f"cron_{job_id}_"):]):
                    runs.append((item, None))
            page = runs[window.offset:] if window.limit is None else runs[window.offset: window.offset + window.limit]
            return tuple(CronRunEntry(entry=self._entry(conn, item), cron_job_id=job_id, cron_execution_id=execution)
                         for item, execution in page)

    def cron_run_identity(self, session_id: SessionId) -> Optional[CronRunIdentity]:
        legacy = _LEGACY_CRON_RUN_ID.match(str(session_id or ""))
        if legacy is not None:
            return CronRunIdentity(cron_job_id=legacy.group(1), cron_execution_id=None)
        with self._store.transaction() as conn:
            run = self._cron_admission_run(conn, str(session_id))
        return None if run is None else CronRunIdentity(cron_job_id=run[0], cron_execution_id=run[1])

    # ── facets ──

    def list_skill_scaffolded_sessions(self, *, limit: int = 200) -> tuple[SkillScaffoldedSession, ...]:
        with self._store.transaction() as conn:
            found = _rows(conn, f"""
                SELECT s.id, s.title, first.content FROM sessions s
                CROSS JOIN LATERAL (
                    SELECT m.content FROM messages m WHERE m.session_id = s.id AND m.role = 'user'
                      AND m.content IS NOT NULL ORDER BY m.timestamp, m.id LIMIT 1) first
                WHERE s.title IS NOT NULL
                  AND (starts_with({_fold('first.content')}, %s) OR starts_with({_fold('first.content')}, %s))
                ORDER BY s.started_at DESC, s.id DESC""",
                          (_ascii_lower(SKILL_SCAFFOLD_PREFIX), _ascii_lower(AUTO_LOAD_SCAFFOLD_PREFIX)))
        return tuple(SkillScaffoldedSession(session_id=SessionId(item["id"]), title=item["title"],
                                            content=item["content"]) for item in (found[:limit] if limit >= 0 else found))

    def list_unlinked_telegram_sessions(self, *, user_id: str, profile_name: Optional[str] = None,
                                        limit: int = 10) -> tuple[SessionFacetEntry, ...]:
        from .sessions import session_record

        from persistence.gateway.routing import TOPIC_SCHEMA_META_KEY

        profile = ((self._store.owner_profile if profile_name is None else profile_name) or "").strip() or "default"
        with self._store.transaction() as conn:
            # Until the gateway prepared its topic storage the unscoped read runs: no owner or binding filter. A row
            # without an owner reads as the default profile's; the owner compares with spaces trimmed.
            scoped = scalar(conn, "SELECT 1 FROM state_meta WHERE key = %s", (TOPIC_SCHEMA_META_KEY,)) is not None
            fence = (" AND coalesce(nullif(btrim(s.profile_name, ' '), ''), 'default') = %(profile)s AND NOT EXISTS "
                     "(SELECT 1 FROM telegram_dm_topic_bindings b WHERE b.session_id = s.id)") if scoped else ""
            found = _rows(conn, f"{_LISTED} WHERE s.source = 'telegram' AND s.user_id = %(user)s{fence}"
                                f"{self._RECENCY_ORDER} LIMIT %(limit)s",
                          {"user": str(user_id), "profile": profile, "limit": max(limit, 0)})
            return tuple(SessionFacetEntry(session=session_record(item, resolve_tool_pin=False),
                                           last_active=item["last_active"], preview=self._preview(conn, item["id"]))
                         for item in found)

    def distinct_session_cwds(self, *, include_archived: bool = False) -> tuple[SessionCwdUsage, ...]:
        with self._store.transaction() as conn:
            found = _rows(conn, """
                SELECT cwd, count(*) AS sessions, max(coalesce(ended_at, started_at)) AS last_active FROM sessions
                WHERE cwd IS NOT NULL AND btrim(cwd, ' ') <> '' AND (%s OR NOT archived)
                GROUP BY cwd ORDER BY cwd COLLATE "C" """, (include_archived,))
        return tuple(SessionCwdUsage(cwd=item["cwd"], sessions=int(item["sessions"]), last_active=item["last_active"])
                     for item in found)

    def live_listing_facts(self, session_ids: Sequence[SessionId]) -> LiveListingFacts:
        from .sessions import session_record

        wanted = list(dict.fromkeys(str(sid) for sid in session_ids))
        rows: dict[SessionId, LiveListingRow] = {}
        with self._store.transaction() as conn:
            listed = {item["id"]: item for item in _rows(conn, _LISTED + " WHERE s.id = ANY(%s)", (wanted,))}
            for sid in wanted:
                if sid not in listed:
                    continue
                record = session_record(listed[sid], resolve_tool_pin=False)
                from ..authority._meta import local_receipt

                receipt = local_receipt(conn, sid)
                target_id = (receipt.get("entry") or {}).get("session_id") if receipt is not None else None
                target_ended = (scalar(conn, "SELECT ended_at FROM sessions WHERE id = %s", (target_id,))
                                if isinstance(target_id, str) else None)
                rows[SessionId(sid)] = LiveListingRow(
                    id=record.session_id, source=record.source, title=record.title, started_at=record.started_at,
                    ended_at=record.ended_at, message_count=record.message_count, model=record.model,
                    hidden=record.hidden, archived=record.archived, runtime_revision=record.runtime_revision,
                    runtime_generation=record.runtime_generation, target_ended_at=target_ended)
            pending: dict[SessionId, list[PendingAdmissionRef]] = {}
            for admission in _rows(conn, "SELECT admission_id, seq, status, target_session_id FROM session_admissions "
                                         "WHERE target_session_id = ANY(%s) AND status IN ('queued', 'started', 'unknown') "
                                         "ORDER BY seq", (wanted,)):
                pending.setdefault(SessionId(admission["target_session_id"]), []).append(PendingAdmissionRef(
                    admission_id=admission["admission_id"], seq=int(admission["seq"]), status=admission["status"]))
        return LiveListingFacts(rows=rows, pending={sid: tuple(refs) for sid, refs in pending.items()})

    def sessions_in_workspace(self, session_ids: Sequence[SessionId], workspace: str) -> frozenset[SessionId]:
        where, params = self._filters(SessionQuery(workspace_key=workspace, include_archived=True),
                                      exclude_children=False)
        with self._store.transaction() as conn:
            found = _rows(conn, f"SELECT s.id FROM sessions s WHERE s.id = ANY(%s) AND {' AND '.join(where)}",
                          ([str(sid) for sid in session_ids], *params))
        return frozenset(SessionId(item["id"]) for item in found)

    def picker_session_facts(self, root_ids: Sequence[SessionId], segment_ids: Sequence[SessionId],
                             reply_ids: Sequence[SessionId]) -> PickerFacts:
        with self._store.transaction() as conn:
            roots = {SessionId(item["id"]): PickerRoot(cwd=item["cwd"], git_repo_root=item["git_repo_root"])
                     for item in _rows(conn, "SELECT id, cwd, git_repo_root FROM sessions WHERE id = ANY(%s)",
                                       ([str(sid) for sid in root_ids],))}
            segments = {SessionId(item["id"]): PickerSegment(preview=self._preview(conn, item["id"]),
                                                             git_branch=item["git_branch"], pinned=bool(item["pinned"]),
                                                             last_active=item["last_active"])
                        for item in _rows(conn, _LISTED + " WHERE s.id = ANY(%s)", ([str(sid) for sid in segment_ids],))}
            replies = {SessionId(sid): self._last_reply(conn, str(sid)) for sid in dict.fromkeys(reply_ids)}
        return PickerFacts(roots=roots, segments=segments, last_replies=replies)

    def _last_reply(self, conn: Connection, session_id: str) -> Optional[str]:
        """The newest of the segment's eight newest shown assistant replies whose display projection shows text."""
        found = _rows(conn, """
            SELECT content, compressed_summary FROM messages
            WHERE session_id = %s AND role = 'assistant' AND (active OR compacted)
              AND coalesce(content, '') <> '' AND coalesce(display_kind, '') NOT IN ('hidden', 'failed_turn')
              AND coalesce(display_metadata ->> 'model_only', '') IN ('', 'false', '0', 'null')
            ORDER BY id DESC LIMIT 8""", (session_id,))
        for item in found:
            shown = turns.project_compaction_message_for_display(
                {"role": "assistant", "content": item["content"], "_compressed_summary": item["compressed_summary"]})
            text = shape_preview(turns.flatten_message_text(shown["content"])) if shown is not None else ""
            if text:
                return text
        return None

    def subagent_session_ids(self, subagent_id: str, marker_key: str) -> tuple[SessionId, ...]:
        with self._store.transaction() as conn:
            found = _rows(conn, "SELECT id FROM sessions WHERE source = 'subagent' AND model_config -> %s = to_jsonb(%s::text) "
                                "ORDER BY started_at DESC, id DESC", (marker_key, subagent_id))
        return tuple(SessionId(item["id"]) for item in found)

    def first_session_started_at(self) -> Optional[float]:
        with self._store.transaction() as conn:
            return scalar(conn, "SELECT min(started_at) FROM sessions")

    def longest_session_id(self) -> Optional[SessionId]:
        with self._store.transaction() as conn:
            found = scalar(conn, "SELECT s.id FROM sessions s ORDER BY (SELECT count(*) FROM messages m "
                                 "WHERE m.session_id = s.id) DESC, s.started_at, s.id LIMIT 1")
        return None if found is None else SessionId(found)

    # ── the Bots roster ──

    def _roster_session(self, conn: Connection, root: Mapping[str, Any]) -> RosterSession:
        """A roster conversation: the root, its compression tip's columns and the tip's latest dialogue line."""
        tip = _lineage.compression_tip(conn, root["id"])
        target = _row(conn, "SELECT * FROM sessions WHERE id = %s", (tip,)) or root
        line = scalar(conn, "SELECT content FROM messages WHERE session_id = %s AND active AND role IN ('user', 'assistant') "
                            "AND btrim(coalesce(content, '')) <> '' ORDER BY id DESC LIMIT 1", (tip,))
        text = " ".join(str(line).split()) if line else ""
        return RosterSession(
            id=SessionId(root["id"]), resolved_id=SessionId(tip), title=target["title"] or "",
            root_title=root["title"] or "", preview=text[:80] + "..." if len(text) > 80 else text,
            started_at=target["started_at"], last_active=_activity_or_start(target),
            message_count=int(target["message_count"] or 0))

    def roster_sessions(self, *, owner_user_id: Optional[str], owner: OwnerRule | None) -> RosterSessions:
        operator = owner_user_id is None
        local_chat = f"starts_with({_fold(CHAT_ID_OR_BLANK)}, %s)"
        newest = " ORDER BY coalesce(s.last_activity_at, s.started_at) DESC LIMIT 1"
        hidden = f"coalesce(s.source, '') = ANY({_sql_list(INTERNAL_LISTING_SOURCES)})"
        with self._store.transaction() as conn:
            canonical = _row(conn, f"SELECT s.* FROM sessions s WHERE s.title = %s AND {local_chat}",
                             (CANONICAL_BOT_CHAT_TITLE, LOCAL_CHAT_PREFIX))
            canonical_session = None
            if canonical is not None and not canonical["archived"] and (
                    operator or same_owner(canonical["user_id"], owner_user_id, owner)):
                canonical_session = self._roster_session(conn, canonical)
            # The newest conversation a picker shows: a hidden row never wins.
            owner_filter = "" if operator else "s.user_id = %s AND "
            latest = _row(conn, f"SELECT s.* FROM sessions s WHERE {owner_filter}{local_chat} AND NOT s.archived "
                                f"AND NOT s.hidden AND NOT {hidden}{newest}",
                          (*(() if operator else (owner_user_id,)), LOCAL_CHAT_PREFIX))
            last_session = None
            if latest is not None:
                last_session = (canonical_session if canonical_session is not None and canonical["id"] == latest["id"]
                                else self._roster_session(conn, latest))
            # The freshest worker: no owner filter, a worker's user is its dispatch identity.
            worker = _row(conn, f"SELECT s.* FROM sessions s WHERE {hidden} AND NOT s.archived AND NOT s.hidden{newest}")
        return RosterSessions(
            canonical_session=canonical_session, last_session=last_session,
            worker_session=None if worker is None else RosterWorker(
                id=SessionId(worker["id"]), source=(worker["source"] or "").strip().lower(),
                title=worker["title"] or "", last_active=_activity_or_start(worker)))


def _activity_or_start(row: Mapping[str, Any]) -> float:
    stamp = row["last_activity_at"]
    return stamp if stamp is not None else row["started_at"]


def _bounded_walk(seeds: list[str], step, ceiling: int) -> dict[str, str]:
    """Breadth-first closure over ``step`` from ``seeds`` (each labelled with itself), stopping once ``ceiling`` rows
    are admitted. Returns row id -> seed label."""
    reached: dict[str, str] = {}
    for seed in seeds:
        if len(reached) >= ceiling:
            break
        reached.setdefault(seed, seed)
    queue = list(reached)
    index = 0
    while index < len(queue) and len(reached) < ceiling:
        current = queue[index]
        index += 1
        for nxt in step(current):
            if nxt not in reached and len(reached) < ceiling:
                reached[nxt] = reached[current]
                queue.append(nxt)
    return reached
