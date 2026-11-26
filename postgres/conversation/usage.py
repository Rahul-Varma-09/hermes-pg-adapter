"""``UsageRepository`` on Postgres: token and cost accounting.

The session's counters, costs, stored billing route and prompt-cache clock are its own ``sessions`` columns; each
per-call attribution row is a ``session_model_usage`` row keyed by (session, model, provider, base URL, mode, task),
with ``''`` for an unset provider, base URL, mode or (main-loop) task. A queued delta is applied at once in its own
transaction under the session row's lock, so every read barrier is already met and ``flush_usage`` returns True; an
apply error is logged and the delta dropped, never raised into a turn.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Mapping, Optional

from psycopg import Connection

from persistence import SessionId
from persistence.conversation.usage import (
    AuxiliaryTaskUsage, BillingRoute, PromptCacheClockRecord, RouteUsage, TokenCounts, UsageDelta, UsageTotals,
    UsageTotalsQuery,
)

from .._backend import PostgresStore, register_repository
from .._sql import execute, row as _row, rows as _rows, scalar
from . import _lineage
from .sessions import is_retired

logger = logging.getLogger(__name__)

_LINEAGE_HOP_CAP = 100
_TOKEN_COLUMNS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens")


def _has_usage(delta: UsageDelta) -> bool:
    """The attribution gate: a counter, a call or a truthy estimate (an actual cost alone is not usage)."""
    return delta.api_call_count > 0 or delta.tokens != TokenCounts() or bool(delta.estimated_cost_usd)


def _tokens(row: Mapping[str, Any]) -> TokenCounts:
    return TokenCounts(**{name: int(row[name] or 0) for name in _TOKEN_COLUMNS})


def _route(row: Mapping[str, Any]) -> BillingRoute:
    return BillingRoute(model=row["model"], billing_provider=row["billing_provider"] or None,
                        billing_base_url=row["billing_base_url"] or None, billing_mode=row["billing_mode"] or None)


class PostgresUsageRepository:
    """Usage accounting of the bound profile's schema, plus the suites' seeding seam."""

    def __init__(self, store_handle: PostgresStore) -> None:
        self._store = store_handle

    # ── the suites' seeding seam ──

    def seed_session(self, session_id: str, *, message_count: int = 0, parent_session_id: Optional[str] = None,
                     archived: bool = False) -> SessionId:
        with self._store.transaction() as conn:
            self._backfill_session_row(conn, session_id, "cli", None)
            execute(conn, "UPDATE sessions SET message_count = %s, parent_session_id = %s, archived = %s WHERE id = %s",
                    (message_count, parent_session_id, archived, session_id))
        return SessionId(session_id)

    def set_session_archived(self, session_id: str, archived: bool) -> None:
        with self._store.transaction() as conn:
            execute(conn, "UPDATE sessions SET archived = %s WHERE id = %s", (archived, session_id))

    # ── writes ──

    def record_auxiliary_usage(self, session_id: SessionId, task: str, *, model: Optional[str] = None,
                               billing_provider: Optional[str] = None, billing_base_url: Optional[str] = None,
                               tokens: TokenCounts = TokenCounts(), estimated_cost_usd: Optional[float] = None,
                               api_call_count: int = 1) -> None:
        if not session_id or not task:
            return
        with self._store.transaction() as conn:
            self._backfill_session_row(conn, session_id, None, None)
            # Auxiliary rows never inherit the main-loop route: missing fields stay unknown or unset.
            route = BillingRoute(model=model or "unknown", billing_provider=billing_provider or None,
                                 billing_base_url=billing_base_url or None)
            calls = 1 if api_call_count is None else int(api_call_count)
            self._attribute(conn, session_id, task, UsageDelta(tokens=tokens, api_call_count=calls,
                                                               estimated_cost_usd=estimated_cost_usd, route=route),
                            route, gated=False)

    def set_billing_route(self, session_id: SessionId, *, provider: str, base_url: str,
                          billing_mode: Optional[str] = None) -> None:
        with self._store.transaction() as conn:
            execute(conn, "UPDATE sessions SET billing_provider = %s, billing_base_url = %s, "
                          "billing_mode = coalesce(%s, billing_mode) WHERE id = %s",
                    (provider, base_url, billing_mode, session_id))

    def queue_usage(self, session_id: SessionId, delta: UsageDelta) -> None:
        try:
            with self._store.transaction() as conn:
                self._apply_delta(conn, session_id, delta)
        except Exception as exc:
            # The writer logs an apply failure and drops the delta; accounting loss never reaches the turn.
            logger.warning("async token accounting: apply failed (session=%s): %s", session_id, exc)

    def flush_usage(self, timeout: float = 5.0) -> bool:
        return True

    # ── reads ──

    def recent_route(self, session_id: SessionId) -> Optional[RouteUsage]:
        with self._store.transaction() as conn:
            found = _row(conn, "SELECT model, billing_provider, billing_base_url, billing_mode, api_call_count "
                               "FROM session_model_usage WHERE session_id = %s AND task = '' AND model <> 'unknown' "
                               "AND billing_provider <> '' ORDER BY last_seen DESC, seq DESC LIMIT 1", (session_id,))
        return None if found is None else RouteUsage(route=_route(found), api_call_count=int(found["api_call_count"]))

    def usage_totals(self, query: UsageTotalsQuery = UsageTotalsQuery()) -> UsageTotals:
        with self._store.transaction() as conn:
            found = _row(conn, """
                SELECT coalesce(sum(input_tokens + output_tokens), 0) AS tokens,
                       coalesce(sum(coalesce(actual_cost_usd, estimated_cost_usd, 0)), 0) AS cost
                FROM sessions WHERE parent_session_id IS NULL AND message_count >= %s AND (%s OR NOT archived)""",
                         (query.min_message_count, query.include_archived))
        return UsageTotals(tokens=int(found["tokens"]), cost_usd=float(found["cost"]))

    def auxiliary_usage_by_task(self, session_id: SessionId) -> Mapping[str, AuxiliaryTaskUsage]:
        if not session_id:
            return {}
        with self._store.transaction() as conn:
            chain = self._lineage_chain(conn, session_id)
            found = _rows(conn, f"""
                SELECT task, sum(api_call_count) AS calls, {', '.join(f'sum({c}) AS {c}' for c in _TOKEN_COLUMNS)},
                       sum(estimated_cost_usd) AS cost
                FROM session_model_usage WHERE session_id = ANY(%s) AND task <> ''
                GROUP BY task ORDER BY task COLLATE "C" """, (chain,))
        return {item["task"]: AuxiliaryTaskUsage(task=item["task"], api_call_count=int(item["calls"]),
                                                 tokens=_tokens(item), estimated_cost_usd=float(item["cost"] or 0.0))
                for item in found}

    def prompt_cache_clock(self, session_id: SessionId) -> Optional[PromptCacheClockRecord]:
        with self._store.transaction() as conn:
            row = _row(conn, "SELECT prompt_cache_clock, cache_read_tokens, cache_write_tokens FROM sessions WHERE id = %s",
                       (session_id,))
            if row is None:
                return None
            last_assistant_at = None
            if (row["cache_read_tokens"] or 0) + (row["cache_write_tokens"] or 0) > 0:
                last_assistant_at = scalar(conn, "SELECT max(timestamp) FROM messages WHERE session_id = %s AND active "
                                                 "AND role = 'assistant'", (session_id,))
            generation = self._generation(conn, session_id)
        return PromptCacheClockRecord(clock=row["prompt_cache_clock"], last_assistant_at=last_assistant_at,
                                      generation=generation)

    def prompt_cache_generation(self, session_id: SessionId) -> int:
        with self._store.transaction() as conn:
            return self._generation(conn, session_id)

    @staticmethod
    def _generation(conn: Connection, session_id: str) -> int:
        """The durable context generation: the session's compaction-archived rows."""
        return int(scalar(conn, "SELECT count(*) FROM messages WHERE session_id = %s AND NOT active AND compacted",
                          (session_id,)))

    # ── internals ──

    def _backfill_session_row(self, conn: Connection, session_id: str, source: Optional[str],
                              model: Optional[str]) -> dict[str, Any]:
        """The row guard: a missing row is created with ``source`` (``unknown`` without one) and the store owner's
        profile; an ``unknown`` placeholder takes a real source; an unset profile or model is filled. A deleted
        session is not recreated."""
        row = _row(conn, "SELECT * FROM sessions WHERE id = %s FOR UPDATE", (session_id,))
        if row is None:
            if is_retired(conn, session_id):
                from persistence import RetiredSessionError

                raise RetiredSessionError(session_id, "not_found")
            execute(conn, "INSERT INTO sessions (id, source, created_source, profile_name, model, started_at) "
                          "VALUES (%s, %s, %s, %s, %s, %s)",
                    (session_id, source or "unknown", source or "unknown", self._store.owner_profile, model, time.time()))
        else:
            execute(conn, "UPDATE sessions SET source = CASE WHEN source = 'unknown' AND %s::text IS NOT NULL THEN %s "
                          "ELSE source END, profile_name = coalesce(profile_name, %s), model = coalesce(model, %s) "
                          "WHERE id = %s", (source, source, self._store.owner_profile, model, session_id))
        return _row(conn, "SELECT * FROM sessions WHERE id = %s", (session_id,))

    def _apply_delta(self, conn: Connection, session_id: str, delta: UsageDelta) -> None:
        """The row guard, then the counters, the route fill and the attribution row as one write."""
        session = self._backfill_session_row(conn, session_id, delta.source, delta.route.model if delta.route else None)
        prior_calls = int(session["api_call_count"] or 0)
        values: dict[str, Any] = {name: int(session[name] or 0) + getattr(delta.tokens, name) for name in _TOKEN_COLUMNS}
        values["api_call_count"] = prior_calls + delta.api_call_count
        values["estimated_cost_usd"] = (session["estimated_cost_usd"] or 0.0) + (delta.estimated_cost_usd or 0.0)
        if delta.actual_cost_usd is not None:
            values["actual_cost_usd"] = (session["actual_cost_usd"] or 0.0) + delta.actual_cost_usd
        for name in ("cost_status", "cost_source", "pricing_version"):
            if getattr(delta, name) is not None:
                values[name] = getattr(delta, name)
        if delta.prompt_cache_clock is not None:
            values["prompt_cache_clock"] = delta.prompt_cache_clock
        model = session["model"]
        stored = {name: session[name] for name in ("billing_provider", "billing_base_url", "billing_mode")}
        route = delta.route
        accounted = _has_usage(delta) or bool(delta.actual_cost_usd)
        if accounted and route is not None:
            first_accounted = (not delta.task and prior_calls == 0 and bool(route.model) and bool(route.billing_provider)
                               and (model != route.model or stored["billing_provider"] != route.billing_provider))
            if first_accounted:
                model = route.model
                stored = {"billing_provider": route.billing_provider, "billing_base_url": route.billing_base_url,
                          "billing_mode": route.billing_mode}
            else:
                stored = {name: stored[name] if stored[name] is not None else getattr(route, name) for name in stored}
        values.update(stored, model=model)
        assignments = ", ".join(f"{name} = %s" for name in values)
        execute(conn, f"UPDATE sessions SET {assignments} WHERE id = %s", (*values.values(), session_id))
        # Missing main-loop fields come from the session row; a task row keeps them unknown or unset.
        incoming = delta.route
        fallback = {"billing_provider": None, "billing_base_url": None, "billing_mode": None} if delta.task else stored
        attribution = BillingRoute(
            model=(incoming.model if incoming else None) or (None if delta.task else model) or "unknown",
            billing_provider=(incoming.billing_provider if incoming else None) or fallback["billing_provider"] or None,
            billing_base_url=(incoming.billing_base_url if incoming else None) or fallback["billing_base_url"] or None,
            billing_mode=(incoming.billing_mode if incoming else None) or fallback["billing_mode"] or None)
        self._attribute(conn, session_id, delta.task or None, delta, attribution)

    @staticmethod
    def _attribute(conn: Connection, session_id: str, task: Optional[str], delta: UsageDelta,
                   route: Optional[BillingRoute], *, gated: bool = True) -> None:
        if route is None or (gated and not _has_usage(delta)):
            return
        now = time.time()
        tokens = [getattr(delta.tokens, name) for name in _TOKEN_COLUMNS]
        execute(conn, f"""
            INSERT INTO session_model_usage (session_id, model, billing_provider, billing_base_url, billing_mode, task,
                api_call_count, {', '.join(_TOKEN_COLUMNS)}, estimated_cost_usd, actual_cost_usd, cost_status,
                cost_source, first_seen, last_seen)
            VALUES (%s, %s, %s, %s, %s, %s, %s, {', '.join(['%s'] * len(_TOKEN_COLUMNS))}, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (session_id, model, billing_provider, billing_base_url, billing_mode, task) DO UPDATE SET
                api_call_count = session_model_usage.api_call_count + excluded.api_call_count,
                {', '.join(f'{c} = session_model_usage.{c} + excluded.{c}' for c in _TOKEN_COLUMNS)},
                estimated_cost_usd = session_model_usage.estimated_cost_usd + excluded.estimated_cost_usd,
                actual_cost_usd = session_model_usage.actual_cost_usd + excluded.actual_cost_usd,
                cost_status = coalesce(excluded.cost_status, session_model_usage.cost_status),
                cost_source = coalesce(excluded.cost_source, session_model_usage.cost_source),
                last_seen = excluded.last_seen""",
                (session_id, route.model, route.billing_provider or "", route.billing_base_url or "",
                 route.billing_mode or "", task or "", delta.api_call_count, *tokens,
                 float(delta.estimated_cost_usd or 0.0), float(delta.actual_cost_usd or 0.0), delta.cost_status,
                 delta.cost_source, now, now))

    def _lineage_chain(self, conn: Connection, session_id: str) -> list[str]:
        """The session and its ancestors over parent edges (any lineage kind, capped), cut at the nearest explicit
        branch node inclusive."""
        chain: list[str] = []
        current: Optional[str] = session_id
        while current and current not in chain and len(chain) < _LINEAGE_HOP_CAP:
            chain.append(current)
            row = _row(conn, "SELECT parent_session_id, model_config FROM sessions WHERE id = %s", (current,))
            if row is None:
                break
            config = _lineage.config_of(row)
            kind = next((marker for marker in ("_delegate_from", "_branched_from", "_reset_from")
                         if isinstance(config.get(marker), str) and config[marker].strip()), None)
            if kind == "_branched_from":
                break
            current = row["parent_session_id"]
        return chain


register_repository("usage", PostgresUsageRepository)
