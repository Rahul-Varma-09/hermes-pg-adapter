"""``SessionSearch`` on Postgres: message search over the bound profile's transcripts.

The filters every route shares run in SQL (visibility, hidden rows, source include and exclude, role, the owning
session's start window), so only candidate rows reach the matcher; which candidates match, their order, snippets and
paging follow the store's query routes (``_search_matcher``): the word route with its substring and OR-relaxed
retries, the ``tool`` role substring route and the CJK routes. Each hit's context is the previous, own and next row of
its session by (timestamp, id). There is no derived index to rebuild: the stored rows are the index.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Collection, Mapping, Optional, Sequence

from psycopg import Connection

from persistence import MessageId, RowVisibility, SessionId
from persistence.conversation.messages import NewMessage
from persistence.conversation.search import (
    FTS_TOOL_CONTENT_PREFIX_CHARS, SEARCH_RESULT_FIELDS, SearchContext, SearchHit, SearchIndexRebuildStatus, SearchSort,
)
from persistence.conversation.sessions import NewSession

from .._backend import PostgresStore, register_repository
from .._sql import execute, row as _row, rows as _rows, scalar
from . import _policy as policy, _search_matcher as m, _transcript as store

_REBUILD_MARKER_KEY = "search.deferred_rebuild"


@dataclass
class _Row:
    id: int
    session_id: str
    source: Optional[str]
    model: Optional[str]
    started_at: Optional[float]
    delegate_child: bool
    role: str
    content: Optional[str]
    decoded: Any
    timestamp: float
    tool_name: Optional[str]
    tool_calls: Optional[str]

    def word_columns(self) -> list[str]:
        content = (self.content or "")[:FTS_TOOL_CONTENT_PREFIX_CHARS] if self.role == "tool" else self.content or ""
        return [content, self.tool_name or "", self.tool_calls or ""]

    def substring_indexed(self) -> bool:
        return self.role != "tool" and self.source not in m._TRIGRAM_EXCLUDED_SOURCES and not self.delegate_child

    def substring_columns(self) -> list[str]:
        return [self.content or "", self.tool_name or ""]

    def like_columns(self) -> list[Optional[str]]:
        return [self.content, self.tool_name, self.tool_calls]


def _search_fields(fields: Optional[Collection[str]]) -> Optional[tuple[str, ...]]:
    """The requested projection in canonical order."""
    if fields is None:
        return None
    if isinstance(fields, str):
        raise TypeError("search fields must be a collection of field names, not a string")
    requested = set(fields)
    unknown = requested.difference(SEARCH_RESULT_FIELDS)
    if unknown:
        raise ValueError(f"unknown search result field(s): {', '.join(sorted(unknown))}")
    return tuple(field for field in SEARCH_RESULT_FIELDS if field in requested)


def _page(rows: list[Any], limit: int, offset: int) -> list[Any]:
    """SQL ``LIMIT ? OFFSET ?``: a negative limit is no limit, a negative offset is zero."""
    start = max(0, offset)
    return rows[start:] if limit < 0 else rows[start: start + limit]


def _ranked(rows: list[_Row], sort: SearchSort) -> list[_Row]:
    """Word and substring route order: timestamp for NEWEST / OLDEST, id order otherwise and within a timestamp."""
    if sort is SearchSort.NEWEST:
        return sorted(rows, key=lambda r: (-r.timestamp, r.id))
    if sort is SearchSort.OLDEST:
        return sorted(rows, key=lambda r: (r.timestamp, r.id))
    return sorted(rows, key=lambda r: r.id)


def _word_route(pool: list[_Row], tree: Any, sort: SearchSort, limit: int, offset: int) -> list[tuple[_Row, str]]:
    phrases = {row.id: m.matched_phrases(tree, row.word_columns(), m._word_hits) for row in pool}
    matched = [row for row in pool if phrases[row.id]]
    return [(row, m.word_snippet(row.word_columns(), phrases[row.id])) for row in _page(_ranked(matched, sort), limit, offset)]


def _substring_route(pool: list[_Row], tree: Any, sort: SearchSort, limit: int, offset: int) -> list[tuple[_Row, str]]:
    phrases = {row.id: m.matched_phrases(tree, row.substring_columns(), m._substring_hits)
               for row in pool if row.substring_indexed()}
    matched = [row for row in pool if phrases.get(row.id)]
    return [(row, m.substring_snippet(row.substring_columns(), phrases[row.id]))
            for row in _page(_ranked(matched, sort), limit, offset)]


def _substring_retry(pool: list[_Row], text: str, sort: SearchSort, limit: int, offset: int) -> list[tuple[_Row, str]]:
    """The zero-result retry over substrings, for a query whose words all have three or more characters."""
    bare = text.strip('"').strip()
    tokens = m._non_operator_tokens(bare)
    if not tokens or any(len(token) < 3 for token in tokens):
        return []
    try:
        return _substring_route(pool, m.parse_query(m._quote_tokens(bare)), sort, limit, offset)
    except m.QuerySyntaxError:
        return []


def _tool_route(pool: list[_Row], text: str, sort: SearchSort, limit: int, offset: int) -> list[tuple[_Row, Optional[str]]]:
    groups, first = m.like_boolean(text)
    if first is None:
        return []

    def present(row: _Row, term: str) -> bool:
        return any(m.like_contains(column, term) for column in row.like_columns())

    matched = [row for row in pool if any(all(present(row, term) != negated for term, negated in group) for group in groups)]
    ordered = sorted(matched, key=lambda r: (r.timestamp, r.id), reverse=sort is not SearchSort.OLDEST)
    return [(row, m.like_snippet(row.content, first)) for row in _page(ordered, limit, offset)]


def _cjk_route(pool: list[_Row], text: str, unindexed: bool, sort: SearchSort, limit: int,
               offset: int) -> list[tuple[_Row, Optional[str]]]:
    """CJK text: substrings when every CJK word has three or more CJK characters, else any word by LIKE, newest
    first."""
    raw = m._strip_cjk_wildcards(text).strip('"').strip()
    tokens = m._non_operator_tokens(raw)
    short = any(m._count_cjk(token) < 3 for token in tokens if any(m._is_cjk(char) for char in token))
    if m._count_cjk(raw) >= 3 and not short and not unindexed:
        try:
            return _substring_route(pool, m.parse_query(m._quote_tokens(raw)), sort, limit, offset)
        except m.QuerySyntaxError:
            pass
    terms = tokens or [raw]
    matched = [row for row in pool if any(m.like_contains(column, term) for term in terms for column in row.like_columns())]
    ordered = sorted(matched, key=lambda r: (r.timestamp, r.id), reverse=True)
    return [(row, m.like_snippet(row.content, terms[0])) for row in _page(ordered, limit, offset)]


class PostgresSessionSearch:
    """Message search of the bound profile's schema, plus the suites' seeding seam."""

    def __init__(self, store_handle: PostgresStore) -> None:
        self._store = store_handle

    # ── the suites' seeding seam ──

    def seed_session(self, session_id: str, *, source: Optional[str] = None, model: Optional[str] = None,
                     started_at: Optional[float] = None, delegate_child: bool = False) -> SessionId:
        """A stored session with the start time and source the filters read; a delegate child carries a
        ``_delegate_from`` marker."""
        self._store.sessions.create(NewSession(SessionId(session_id), source, model=model))
        with self._store.transaction() as conn:
            if started_at is not None:
                execute(conn, "UPDATE sessions SET started_at = %s WHERE id = %s", (started_at, session_id))
            if delegate_child:
                execute(conn, "UPDATE sessions SET model_config = coalesce(model_config, '{}'::jsonb) "
                              "|| '{\"_delegate_from\": \"seed-parent\"}'::jsonb WHERE id = %s", (session_id,))
        return SessionId(session_id)

    def seed_message(self, session_id: str, role: str, content: Any, *, timestamp: float,
                     tool_name: Optional[str] = None, tool_calls: Sequence[Mapping[str, Any]] = (),
                     display_kind: Optional[str] = None, visibility: RowVisibility = RowVisibility.ACTIVE) -> MessageId:
        """One transcript row of a stored session, in the given visibility."""
        message = NewMessage(role, content, float(timestamp), tool_name=tool_name, tool_calls=tuple(tool_calls),
                             display_kind=display_kind)
        with self._store.transaction() as conn:
            if scalar(conn, "SELECT 1 FROM sessions WHERE id = %s", (session_id,)) is None:
                raise ValueError(f"seed_message before seed_session for {session_id!r}")
            row = store.insert_rows(conn, session_id, [message])[0]
            if visibility is not RowVisibility.ACTIVE:
                row.visibility = visibility
                store.save_visibility(conn, [row])
        return MessageId(row.id)

    def seed_deferred_rebuild(self, total: int, indexed: int) -> None:
        """Plant a deferred-rebuild progress marker."""
        with self._store.transaction() as conn:
            execute(conn, "INSERT INTO state_meta (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE "
                          "SET value = excluded.value", (_REBUILD_MARKER_KEY, json.dumps({"total": int(total),
                                                                                          "indexed": int(indexed)})))

    # ── index administration ──

    def rebuild_search_index(self) -> int:
        # Nothing derived exists; the stored rows are the index.
        return 0

    def retry_deferred_search_recovery(self) -> bool:
        return False

    def search_index_rebuild_status(self) -> Optional[SearchIndexRebuildStatus]:
        with self._store.transaction() as conn:
            raw = scalar(conn, "SELECT value FROM state_meta WHERE key = %s", (_REBUILD_MARKER_KEY,))
        marker = json.loads(raw) if raw else None
        if marker is None or marker["total"] <= 0:
            return None
        return SearchIndexRebuildStatus(total=marker["total"], indexed=marker["indexed"])

    # ── search ──

    def _pool(self, conn: Connection, *, include_inactive: bool, source_filter: Optional[Sequence[str]],
              exclude_sources: Optional[Sequence[str]], role_filter: Optional[Sequence[str]],
              after_ts: Optional[int], before_ts: Optional[int]) -> list[_Row]:
        """The candidate rows every route shares, by the shared filters."""
        where, params = ["coalesce(m.display_kind, '') <> 'hidden'"], []
        if not include_inactive:
            where.append("(m.active OR m.compacted)")
        if source_filter is not None:
            where.append("s.source = ANY(%s)")
            params.append(list(source_filter))
        if exclude_sources:
            where.append("s.source IS NOT NULL AND NOT s.source = ANY(%s)")
            params.append(list(exclude_sources))
        if role_filter:
            where.append("m.role = ANY(%s)")
            params.append(list(role_filter))
        if after_ts is not None:
            where.append("s.started_at >= %s")
            params.append(int(after_ts))
        if before_ts is not None:
            where.append("s.started_at < %s")
            params.append(int(before_ts))
        found = _rows(conn, f"""
            SELECT m.*, s.source AS session_source, s.model AS session_model, s.started_at AS session_started,
                   coalesce(s.model_config ? '_delegate_from', false) AS delegate_child
            FROM messages m JOIN sessions s ON s.id = m.session_id
            WHERE {' AND '.join(where)} ORDER BY m.timestamp, m.id""", params)
        return [self._row(item) for item in found]

    @staticmethod
    def _row(item: dict[str, Any]) -> _Row:
        stored = store.decode(item)
        message = stored.message
        return _Row(id=stored.id, session_id=stored.session_id, source=item["session_source"],
                    model=item["session_model"], started_at=item["session_started"],
                    delegate_child=bool(item["delegate_child"]), role=message.role,
                    content=store.content_text(message.content), decoded=message.content, timestamp=stored.epoch,
                    tool_name=message.tool_name,
                    tool_calls=json.dumps(list(message.tool_calls)) if message.tool_calls else None)

    def search_messages(self, query: str, *, source_filter: Optional[Sequence[str]] = None,
                        exclude_sources: Optional[Sequence[str]] = None, role_filter: Optional[Sequence[str]] = None,
                        limit: int = 20, offset: int = 0, sort: SearchSort = SearchSort.RELEVANCE,
                        include_inactive: bool = False, fields: Optional[Collection[str]] = None,
                        after_ts: Optional[int] = None, before_ts: Optional[int] = None) -> tuple[SearchHit, ...]:
        projection = _search_fields(fields)
        if not query or not query.strip():
            return ()
        text = m.sanitize_query(query)
        if not text:
            return ()
        with self._store.transaction() as conn:
            pool = self._pool(conn, include_inactive=include_inactive, source_filter=source_filter,
                              exclude_sources=exclude_sources, role_filter=role_filter, after_ts=after_ts,
                              before_ts=before_ts)
            if bool(role_filter) and "tool" in (role_filter or ()):
                found = _tool_route(pool, text, sort, limit, offset)
            elif any(m._is_cjk(char) for char in text):
                unindexed = bool(source_filter) and any(src in m._TRIGRAM_EXCLUDED_SOURCES for src in source_filter or ())
                found = _cjk_route(pool, text, unindexed, sort, limit, offset)
            else:
                try:
                    found = _word_route(pool, m.parse_query(text), sort, limit, offset)
                except m.QuerySyntaxError:
                    return ()
                if not found:
                    found = _substring_retry(pool, text, sort, limit, offset)
                relaxed = m._or_relaxed(text) if not found else None
                if relaxed is not None:
                    try:
                        found = _word_route(pool, m.parse_query(relaxed), sort, limit, offset)
                    except m.QuerySyntaxError:
                        found = []
            wanted = SEARCH_RESULT_FIELDS if projection is None else projection
            return tuple(self._hit(conn, row, snippet, wanted) for row, snippet in found)

    def _context(self, conn: Connection, row: _Row) -> tuple[SearchContext, ...]:
        """The previous row, the row and the next row of its session by (timestamp, id), whatever their visibility."""
        before = _row(conn, "SELECT * FROM messages WHERE session_id = %s AND (timestamp, id) < (%s, %s) "
                            "ORDER BY timestamp DESC, id DESC LIMIT 1", (row.session_id, row.timestamp, row.id))
        after = _row(conn, "SELECT * FROM messages WHERE session_id = %s AND (timestamp, id) > (%s, %s) "
                           "ORDER BY timestamp, id LIMIT 1", (row.session_id, row.timestamp, row.id))
        neighbours = [store.decode(item).message for item in (before,) if item is not None]
        own = NewMessage(role=row.role, content=row.decoded)
        neighbours += [own, *(store.decode(item).message for item in (after,) if item is not None)]
        return tuple(SearchContext(message.role, policy.flatten_search_text(message.content)[:200])
                     for message in neighbours)

    def _hit(self, conn: Connection, row: _Row, snippet: Optional[str], wanted: Sequence[str]) -> SearchHit:
        values: dict[str, Any] = {
            "id": MessageId(row.id), "session_id": SessionId(row.session_id), "role": row.role, "snippet": snippet,
            "timestamp": row.timestamp, "tool_name": row.tool_name, "source": row.source, "model": row.model,
            "session_started": row.started_at,
        }
        if "context" in wanted:
            values["context"] = self._context(conn, row)
        names = {"id": "message_id"}
        return SearchHit(**{names.get(name, name): values[name] for name in wanted})


register_repository("search", PostgresSessionSearch, {"full_text_search": True, "substring_search": True})
