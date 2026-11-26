"""The one row shape of ``async_delegations``: the coordination ledger and the worker delegation verbs write it."""

from __future__ import annotations

import time
from typing import Any, Optional


def dispatch_row(delegation_id: str, origin_session: str, origin_session_id: Optional[str], dispatched_at: float) -> dict[str, Any]:
    """A freshly dispatched delegation's row."""
    return {
        "delegation_id": delegation_id, "origin_session": origin_session,
        "origin_ui_session_id": "", "parent_session_id": None,
        "origin_session_id": origin_session_id or "", "owner_execution_id": None,
        "owner_pid": None, "owner_birth": None, "task": None,
        "state": "running", "dispatched_at": dispatched_at, "completed_at": None, "updated_at": time.time(),
        "event": None, "result": None, "children": None, "delivery_state": "pending", "delivery_attempts": 0,
        "delivery_claim": None, "delivery_claimed_at": None, "delivered_at": None,
    }
