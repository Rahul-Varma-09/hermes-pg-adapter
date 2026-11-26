"""PostgreSQL persistence provider for Hermes Agent.

Selected by ``persistence.provider: postgres``; see ``_settings`` for the rest of the section. Each repository module
registers itself on import, so every one is imported here explicitly rather than left to the plugin loader's
unordered sibling scan.
"""

from __future__ import annotations

from . import _backend, services  # noqa: F401  (services registers ``health``)
from .authority import admissions, bindings, coordination, mutation, runtime  # noqa: F401
from .gateway import (  # noqa: F401
    hosted_room_attachments, hosted_room_driver, hosted_room_peers, hosted_room_policy, hosted_room_replicas, hosted_rooms,
    routing,
)
from .housekeeping import maintenance, profile_identity, state_meta  # noqa: F401
from .conversation import lineage, messages, search, sessions, usage, usage_analytics  # noqa: F401
from .workers import executions  # noqa: F401
from ._provider import PostgresProvider


def register(ctx) -> None:
    ctx.register_persistence_provider(PostgresProvider())
