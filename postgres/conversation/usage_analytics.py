"""``UsageAnalytics`` on Postgres: the dashboard and insights reads as SQL aggregates over ``sessions``,
``session_model_usage`` and ``messages``.

Aggregates follow SQL: a sum over no rows is NULL (``coalesce(..., 0)`` where the contract documents 0), a count is 0.
The daily read buckets by the calling process's local date, so its grouping runs here rather than in the server's
time zone.
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional

from .._backend import PostgresStore, register_repository
from .._sql import row as _row, rows as _rows

_SESSION_COLUMNS = (
    "id", "source", "model", "started_at", "ended_at", "message_count", "tool_call_count", "input_tokens",
    "output_tokens", "cache_read_tokens", "cache_write_tokens", "billing_provider", "billing_base_url", "billing_mode",
    "estimated_cost_usd", "actual_cost_usd", "cost_status", "cost_source", "api_call_count",
)
_TOKENS = ("input_tokens", "output_tokens", "cache_read_tokens", "reasoning_tokens")
_VOLUME = " ORDER BY sum(input_tokens) + sum(output_tokens) DESC NULLS LAST"
_SINCE = ("s.started_at >= %s AND (coalesce(%s, '') = '' OR s.source = %s "
          "OR (s.source = 'local' AND s.created_source = %s))")


def _since(started_since: float, source: Optional[str]) -> tuple[Any, ...]:
    return (started_since, source, source, source)


def _sums(alias: str = "") -> str:
    return ", ".join(f"sum({alias}{name}) AS {name}" for name in _TOKENS)


class PostgresUsageAnalytics:
    """Read-only analytics over the bound profile's schema."""

    def __init__(self, store_handle: PostgresStore) -> None:
        self._store = store_handle

    def _query(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self._store.transaction() as conn:
            return _rows(conn, sql, params)

    # ── dashboard reads ──

    def daily_usage(self, *, started_after: float) -> list[dict[str, Any]]:
        days: dict[str, list[dict[str, Any]]] = {}
        for row in self._query("SELECT started_at, input_tokens, output_tokens, cache_read_tokens, reasoning_tokens, "
                               "estimated_cost_usd, actual_cost_usd, api_call_count FROM sessions WHERE started_at > %s",
                               (started_after,)):
            days.setdefault(time.strftime("%Y-%m-%d", time.localtime(row["started_at"])), []).append(row)

        def total(rows: list[dict[str, Any]], name: str) -> Any:
            present = [row[name] for row in rows if row[name] is not None]
            return sum(present) if present else None

        return [{"day": day, **{name: total(rows, name) for name in _TOKENS},
                 "estimated_cost": total(rows, "estimated_cost_usd") or 0, "actual_cost": total(rows, "actual_cost_usd") or 0,
                 "sessions": len(rows), "api_calls": total(rows, "api_call_count")}
                for day, rows in sorted(days.items())]

    def session_usage_by_model(self, *, started_after: float) -> list[dict[str, Any]]:
        return self._query(f"""
            SELECT model, sum(input_tokens) AS input_tokens, sum(output_tokens) AS output_tokens,
                   coalesce(sum(estimated_cost_usd), 0) AS estimated_cost, count(*) AS sessions,
                   sum(api_call_count) AS api_calls
            FROM sessions WHERE started_at > %s AND model IS NOT NULL GROUP BY model{_VOLUME}""", (started_after,))

    def session_usage_totals(self, *, started_after: float, model_rows_only: bool = False) -> dict[str, Any]:
        with self._store.transaction() as conn:
            return _row(conn, """
                SELECT sum(input_tokens) AS total_input, sum(output_tokens) AS total_output,
                       sum(cache_read_tokens) AS total_cache_read, sum(reasoning_tokens) AS total_reasoning,
                       coalesce(sum(estimated_cost_usd), 0) AS total_estimated_cost,
                       coalesce(sum(actual_cost_usd), 0) AS total_actual_cost, count(*) AS total_sessions,
                       sum(api_call_count) AS total_api_calls
                FROM sessions WHERE started_at > %s AND (NOT %s OR coalesce(model, '') <> '')""",
                        (started_after, model_rows_only))

    def session_usage_by_route(self, *, started_after: float) -> list[dict[str, Any]]:
        return self._query(f"""
            SELECT model, billing_provider, {_sums()},
                   coalesce(sum(estimated_cost_usd), 0) AS estimated_cost, coalesce(sum(actual_cost_usd), 0) AS actual_cost,
                   count(*) AS sessions, sum(api_call_count) AS api_calls, sum(tool_call_count) AS tool_calls,
                   max(started_at) AS last_used_at, avg(input_tokens + output_tokens)::double precision AS avg_tokens_per_session
            FROM sessions WHERE started_at > %s AND coalesce(model, '') <> ''
            GROUP BY model, billing_provider{_VOLUME}""", (started_after,))

    def session_tool_calls_by_route(self, *, started_after: float) -> list[dict[str, Any]]:
        return self._query("""
            SELECT model, billing_provider, coalesce(sum(tool_call_count), 0) AS tool_calls
            FROM sessions WHERE started_at > %s AND coalesce(model, '') <> '' GROUP BY model, billing_provider""",
                           (started_after,))

    def call_usage_by_route(self, *, started_after: float) -> list[dict[str, Any]]:
        return self._query(f"""
            SELECT u.model, u.billing_provider, {_sums('u.')},
                   coalesce(sum(u.estimated_cost_usd), 0) AS estimated_cost,
                   coalesce(sum(u.actual_cost_usd), 0) AS actual_cost, count(DISTINCT u.session_id) AS sessions,
                   sum(u.api_call_count) AS api_calls, max(u.last_seen) AS last_used_at,
                   avg(u.input_tokens + u.output_tokens)::double precision AS avg_tokens_per_session
            FROM session_model_usage u JOIN sessions s ON s.id = u.session_id
            WHERE s.started_at > %s AND u.task = '' GROUP BY u.model, u.billing_provider
            ORDER BY sum(u.input_tokens) + sum(u.output_tokens) DESC NULLS LAST""", (started_after,))

    def auxiliary_usage_by_model_task(self, *, started_after: float) -> list[dict[str, Any]]:
        return self._query(f"""
            SELECT u.model, u.task, u.billing_provider, {_sums('u.')},
                   coalesce(sum(u.estimated_cost_usd), 0) AS estimated_cost, count(DISTINCT u.session_id) AS sessions,
                   sum(u.api_call_count) AS api_calls, max(u.last_seen) AS last_used_at
            FROM session_model_usage u JOIN sessions s ON s.id = u.session_id
            WHERE s.started_at > %s AND u.task <> '' GROUP BY u.model, u.task, u.billing_provider
            ORDER BY sum(u.input_tokens) + sum(u.output_tokens) DESC NULLS LAST""", (started_after,))

    # ── insights reads ──

    def insight_sessions(self, *, started_since: float, source: Optional[str] = None) -> list[dict[str, Any]]:
        return self._query(f"SELECT {', '.join('s.' + c for c in _SESSION_COLUMNS)} FROM sessions s WHERE {_SINCE} "
                           f"ORDER BY s.started_at DESC", _since(started_since, source))

    def insight_tool_result_counts(self, *, started_since: float, source: Optional[str] = None) -> list[dict[str, Any]]:
        return self._query(f"""
            SELECT m.session_id, m.tool_name, count(*) AS count
            FROM messages m JOIN sessions s ON s.id = m.session_id
            WHERE {_SINCE} AND m.role = 'tool' AND m.tool_name IS NOT NULL
            GROUP BY m.session_id, m.tool_name ORDER BY min(m.id)""", _since(started_since, source))

    def _tool_call_rows(self, started_since: float, source: Optional[str]) -> list[dict[str, Any]]:
        """Assistant rows with tool calls and their ``tool_calls`` text (the JSON array as the SQLite store encodes it)."""
        found = self._query(f"""
            SELECT m.session_id, m.timestamp, m.tool_calls::text AS calls
            FROM messages m JOIN sessions s ON s.id = m.session_id
            WHERE {_SINCE} AND m.role = 'assistant' AND m.tool_calls IS NOT NULL
              AND json_typeof(m.tool_calls) = 'array' AND json_array_length(m.tool_calls) > 0
            ORDER BY s.started_at, s.id, m.id""", _since(started_since, source))
        return [{**row, "calls": json.dumps(json.loads(row["calls"]))} for row in found]

    def insight_tool_call_payloads(self, *, started_since: float, source: Optional[str] = None) -> list[dict[str, Any]]:
        return [{"session_id": row["session_id"], "tool_calls": row["calls"]}
                for row in self._tool_call_rows(started_since, source)]

    def insight_skill_call_payloads(self, *, started_since: float, source: Optional[str] = None) -> list[dict[str, Any]]:
        return [{"tool_calls": row["calls"], "timestamp": row["timestamp"]}
                for row in self._tool_call_rows(started_since, source)
                if "skill_view" in row["calls"] or "skill_manage" in row["calls"]]

    def insight_message_stats(self, *, started_since: float, source: Optional[str] = None) -> dict[str, Any]:
        with self._store.transaction() as conn:
            return _row(conn, f"""
                SELECT count(m.id) AS total_messages,
                       sum(CASE WHEN m.role = 'user' THEN 1 ELSE 0 END) AS user_messages,
                       sum(CASE WHEN m.role = 'assistant' THEN 1 ELSE 0 END) AS assistant_messages,
                       sum(CASE WHEN m.role = 'tool' THEN 1 ELSE 0 END) AS tool_messages
                FROM messages m JOIN sessions s ON s.id = m.session_id WHERE {_SINCE}""", _since(started_since, source))

    def insight_model_usage(self, *, started_since: float, source: Optional[str] = None) -> list[dict[str, Any]]:
        return self._query(f"""
            SELECT u.session_id, u.model, u.billing_provider, u.billing_base_url, u.api_call_count,
                   u.input_tokens, u.output_tokens, u.cache_read_tokens, u.cache_write_tokens, u.reasoning_tokens,
                   u.estimated_cost_usd, u.actual_cost_usd, u.cost_status, u.cost_source, u.billing_mode
            FROM session_model_usage u JOIN sessions s ON s.id = u.session_id WHERE {_SINCE} ORDER BY u.seq""",
                           _since(started_since, source))


register_repository("usage_analytics", PostgresUsageAnalytics)
