"""The deletion fence: ledger refusals and the retirement of sessions with their receipts, files and routes.

Every delete path (a user delete, a bulk delete, the empty sweep, an unadmitted discard of a real session) runs
``retire`` inside the transaction that removes the rows, so the archives, the retirement markers and the row removal
commit together. Refusals raise before anything changes.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Iterable, Optional, Sequence

from psycopg import Connection
from psycopg.types.json import Jsonb

from persistence import AuthorityRefusal, PermanentStoreError, UnknownExecution
from session_rules.media import admission_media_references, hosted_document_references, retirement_candidates_in_text

from .._sql import execute, rows as _rows
from ..authority import _meta

_API_SETTINGS_PREFIX = "gateway.api.settings.v1."
#: authority_meta key prefixes of the media and spillover retirement candidates.
MEDIA_PREFIX = "gateway.media.retired.v1."
SPILLOVER_PREFIX = "gateway.spillover.retired.v1."
_VIEWER_CLOSE_PREFIX = "gateway.viewer_close.v1:"


def live_ledger_states(conn: Connection, session_id: str) -> set[str]:
    """The non-terminal admission and worker statuses addressing one session."""
    found = _rows(conn, "SELECT status FROM session_admissions WHERE target_session_id = %s AND status <> 'terminal' "
                        "UNION SELECT status FROM worker_executions WHERE session_id = %s AND status <> 'terminal'",
                  (session_id, session_id))
    return {item["status"] for item in found}


def require_ledger_idle(conn: Connection, session_ids: Sequence[str]) -> None:
    """Refuse deleting a target with live ledger work; unknown wins over busy per target."""
    for sid in session_ids:
        live = live_ledger_states(conn, sid)
        if "unknown" in live:
            raise UnknownExecution("session", sid)
        if live:
            raise AuthorityRefusal("session_busy")


def _retire_spillover(conn: Connection, doomed: Sequence[str]) -> None:
    """Every archive the doomed sessions' rows name in their content or API content becomes a candidate."""
    for message in _rows(conn, "SELECT coalesce(content, content_json::text) AS content, api_content FROM messages "
                                  "WHERE session_id = ANY(%s)", (list(doomed),)):
        for name in retirement_candidates_in_text(message["content"] or "", message["api_content"]):
            _meta.put_new(conn, SPILLOVER_PREFIX + name, json.dumps(name))


def _retire_admissions(conn: Connection, doomed: Sequence[str], media_root: Optional[str], now: float) -> None:
    for admission in _rows(conn, "SELECT * FROM session_admissions WHERE target_session_id = ANY(%s) ORDER BY seq",
                           (list(doomed),)):
        payload = admission["payload"] if isinstance(admission["payload"], dict) else {}
        hosted = (hosted_document_references(admission["request_id"], payload, media_root)
                  if media_root is not None else [])
        for reference in [*admission_media_references(payload), *hosted,
                          *payload.get("api_turn_v1", {}).get("media", [])]:
            _meta.put_new(conn, MEDIA_PREFIX + hashlib.sha256(reference["path"].encode()).hexdigest(), json.dumps(reference))
        # Keep the digest for exact retries, not another copy of user input or history.
        kept = {**admission, "payload": {}, "lineage": []}
        execute(conn, "INSERT INTO admission_terminals (admission_id, principal_id, target_session_id, request_id, row, "
                      "retired_at) VALUES (%s, %s, %s, %s, %s, %s)",
                (admission["admission_id"], admission["principal_id"], admission["target_session_id"],
                 admission["request_id"], Jsonb(kept), now))
    execute(conn, "DELETE FROM session_admissions WHERE target_session_id = ANY(%s)", (list(doomed),))


def _retire_workers(conn: Connection, doomed: Sequence[str], now: float) -> None:
    for worker in _rows(conn, "SELECT * FROM worker_executions WHERE session_id = ANY(%s)", (list(doomed),)):
        receipts = _rows(conn, "SELECT sequence, payload_digest, result_json FROM worker_receipts "
                               "WHERE execution_id = %s ORDER BY sequence", (worker["execution_id"],))
        # Digests stay so a late duplicate is still recognised; every result but the last is user data that goes.
        for receipt in receipts[:-1]:
            receipt["result_json"] = None
            receipt["scrubbed"] = True
        if not execute(conn, "INSERT INTO worker_terminals (execution_id, session_id, row, retired_at) "
                             "VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
                       (worker["execution_id"], worker["session_id"], Jsonb({**worker, "receipts": receipts}), now)):
            raise PermanentStoreError(f"terminal worker receipt already exists: {worker['execution_id']}")
        execute(conn, "DELETE FROM worker_receipts WHERE execution_id = %s", (worker["execution_id"],))
    execute(conn, "DELETE FROM worker_executions WHERE session_id = ANY(%s)", (list(doomed),))


def mark_retired(conn: Connection, session_ids: Iterable[str], *, expiring: Iterable[str] = ()) -> None:
    """Publish the permanent retirement markers; a sweep's (``expiring``) ages out and never shortens a permanent
    one, and re-retiring an id is a fact, never a conflict."""
    expiring = set(expiring)
    now = time.time()
    for sid in session_ids:
        if sid in expiring:
            execute(conn, "INSERT INTO retired_sessions (session_id, retired_at, expires) VALUES (%s, %s, true) "
                          "ON CONFLICT (session_id) DO UPDATE SET retired_at = excluded.retired_at "
                          "WHERE retired_sessions.expires", (sid, now))
        else:
            execute(conn, "INSERT INTO retired_sessions (session_id, retired_at, expires) VALUES (%s, %s, false) "
                          "ON CONFLICT (session_id) DO UPDATE SET retired_at = excluded.retired_at, expires = false",
                    (sid, now))


def retire(conn: Connection, doomed: Sequence[str], *, media_root: Optional[str], expiring: Iterable[str] = ()) -> None:
    """Retire ``doomed`` and remove their rows, in the caller's transaction: archive their admissions and workers,
    record the files they name, drop their messages, usage, settings and routes, detach surviving children and
    publish the retirement markers. The caller has already checked the ledger."""
    doomed = list(dict.fromkeys(doomed))
    if not doomed:
        return
    now = time.time()
    _retire_spillover(conn, doomed)
    _retire_admissions(conn, doomed, media_root, now)
    _retire_workers(conn, doomed, now)
    execute(conn, "DELETE FROM authority_meta WHERE key = ANY(%s)",
            ([prefix + sid for sid in doomed for prefix in (_API_SETTINGS_PREFIX, _VIEWER_CLOSE_PREFIX)],))
    execute(conn, "DELETE FROM gateway_routing WHERE session_id = ANY(%s)", (doomed,))
    execute(conn, "DELETE FROM messages WHERE session_id = ANY(%s)", (doomed,))
    execute(conn, "UPDATE sessions SET parent_session_id = NULL WHERE parent_session_id = ANY(%s) "
                  "AND NOT id = ANY(%s)", (doomed, doomed))
    execute(conn, "DELETE FROM sessions WHERE id = ANY(%s)", (doomed,))
    mark_retired(conn, doomed, expiring=expiring)
