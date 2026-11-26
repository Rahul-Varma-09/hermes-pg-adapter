"""The worker's usage operations: the session counter update and the model-usage attribution.

The session's counters, costs, stored route and prompt-cache clock are its ``sessions`` columns; attribution rows are
the usage repository's ``session_model_usage`` rows (``PostgresUsageRepository._attribute``). Both run inside the
receipt transaction.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional

from psycopg import Connection

from persistence.conversation.usage import BillingRoute, TokenCounts, UsageDelta
from session_rules.surfaces import created_surface

from .._sql import execute, row as _row
from ..conversation import _lineage
from ..conversation.usage import PostgresUsageRepository

TOKEN_COUNTERS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens")
_UNSET_CREATED = (None, "", "unknown", "local")
_ROUTE = ("billing_provider", "billing_base_url", "billing_mode")


def _tokens(payload: Mapping[str, Any]) -> TokenCounts:
    return TokenCounts(**{name: payload.get(name, 0) for name in TOKEN_COUNTERS})


def _add_cost(stored: Optional[float], delta: Optional[float]) -> Optional[float]:
    return stored if delta is None else (stored or 0.0) + delta


def stamp_cache_generation(conn: Connection, session_id: str, clock: str) -> str:
    """The worker's prompt-cache event with the owner's durable context generation set; text that is not a JSON object
    is kept as sent."""
    try:
        event = json.loads(clock)
    except ValueError:
        return clock
    if not isinstance(event, dict):
        return clock
    event["generation"] = PostgresUsageRepository._generation(conn, session_id)
    return json.dumps(event, separators=(",", ":"))


def apply_main_usage(conn: Connection, session_id: str, payload: Mapping[str, Any]) -> None:
    """The main-loop usage report on the worker's session: counters added (set when ``absolute``), the first accounted
    route stored, and the attribution row unless ``absolute``."""
    session = _row(conn, "SELECT * FROM sessions WHERE id = %s FOR UPDATE", (session_id,))
    if session is None:
        return
    model, provider, task = payload.get("model"), payload.get("billing_provider"), payload.get("task") or ""
    tokens = _tokens(payload)
    calls, estimated, actual = payload.get("api_call_count", 0), payload.get("estimated_cost_usd"), payload.get("actual_cost_usd")
    has_usage = bool(tokens != TokenCounts() or calls or estimated)
    accounted = has_usage or bool(actual)
    absolute = bool(payload.get("absolute"))
    route_model = session["model"]
    route = {name: session[name] for name in _ROUTE}
    if (not task and int(session["api_call_count"] or 0) == 0 and accounted and model and provider
            and (route_model != model or route["billing_provider"] != provider)):
        route_model = model
        route = {"billing_provider": provider, "billing_base_url": payload.get("billing_base_url"),
                 "billing_mode": payload.get("billing_mode")}
    if absolute:
        values: dict[str, Any] = {name: getattr(tokens, name) for name in TOKEN_COUNTERS}
        values.update(api_call_count=calls, estimated_cost_usd=float(estimated or 0),
                      actual_cost_usd=session["actual_cost_usd"] if actual is None else actual)
    else:
        values = {name: int(session[name] or 0) + getattr(tokens, name) for name in TOKEN_COUNTERS}
        values.update(api_call_count=int(session["api_call_count"] or 0) + calls,
                      estimated_cost_usd=(session["estimated_cost_usd"] or 0.0) + (estimated or 0.0),
                      actual_cost_usd=_add_cost(session["actual_cost_usd"], actual))
    for name in ("cost_status", "cost_source", "pricing_version"):
        values[name] = payload.get(name) if payload.get(name) is not None else session[name]
    if payload.get("prompt_cache_clock") is not None:
        values["prompt_cache_clock"] = stamp_cache_generation(conn, session_id, payload["prompt_cache_clock"])
    if accounted:
        route = {name: route[name] if route[name] is not None else payload.get(name) for name in _ROUTE}
        route_model = route_model if route_model is not None else model
    values.update(route, model=route_model)
    execute(conn, f"UPDATE sessions SET {', '.join(f'{name} = %s' for name in values)} WHERE id = %s",
            (*values.values(), session_id))
    if not absolute and has_usage:
        attribution = BillingRoute(
            model=model or (None if task else route_model) or "unknown",
            billing_provider=provider or (None if task else route["billing_provider"]) or None,
            billing_base_url=payload.get("billing_base_url") or (None if task else route["billing_base_url"]) or None,
            billing_mode=payload.get("billing_mode") or (None if task else route["billing_mode"]) or None)
        delta = UsageDelta(tokens=tokens, api_call_count=calls, estimated_cost_usd=estimated, route=attribution,
                           actual_cost_usd=actual, cost_status=payload.get("cost_status"),
                           cost_source=payload.get("cost_source"))
        PostgresUsageRepository._attribute(conn, session_id, task or None, delta, attribution)


def apply_auxiliary_usage(conn: Connection, session_id: str, payload: Mapping[str, Any]) -> None:
    """The task's attribution row only; missing route fields stay unknown or unset."""
    route = BillingRoute(model=payload.get("model") or "unknown", billing_provider=payload.get("billing_provider") or None,
                         billing_base_url=payload.get("billing_base_url") or None)
    delta = UsageDelta(tokens=_tokens(payload), api_call_count=payload.get("api_call_count", 1),
                       estimated_cost_usd=payload.get("estimated_cost_usd"), route=route,
                       actual_cost_usd=payload.get("actual_cost_usd"), cost_status=payload.get("cost_status"),
                       cost_source=payload.get("cost_source"))
    PostgresUsageRepository._attribute(conn, session_id, payload["task"], delta, route, gated=False)


def stamp_created_source(conn: Connection, session_id: str, source: Any) -> None:
    """The worker platform as ``created_source`` of an unset row, then on its compression ancestors."""
    surfaced = created_surface(source)
    found = _row(conn, "SELECT created_source FROM sessions WHERE id = %s", (session_id,))
    if not surfaced or found is None:
        return
    if found["created_source"] in _UNSET_CREATED:
        execute(conn, "UPDATE sessions SET created_source = %s WHERE id = %s", (surfaced, session_id))
    _lineage.stamp_ancestors(conn, session_id)
