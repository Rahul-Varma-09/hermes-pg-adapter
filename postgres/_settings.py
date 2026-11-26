"""The ``persistence`` config section as the Postgres provider reads it.

    persistence:
      provider: postgres
      dsn_env: HERMES_PG_DSN      # env var holding the connection string (the default)
      dsn: postgresql://...       # or the string itself; dsn_env wins when both are set
      schema_prefix: hermes_      # every profile's schema starts with it
      allow_incomplete: false     # open even while some repositories are not ported yet

A password belongs in ``~/.pgpass`` (``%APPDATA%\\postgresql\\pgpass.conf`` on Windows), never in config.yaml.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Mapping

from persistence import StoreUnavailable

DEFAULT_DSN_ENV = "HERMES_PG_DSN"
DEFAULT_SCHEMA_PREFIX = "hermes_"

_PREFIX_RE = re.compile(r"^[a-z_][a-z0-9_]{0,30}$")


@dataclass(frozen=True, slots=True)
class PostgresSettings:
    dsn: str
    schema_prefix: str
    allow_incomplete: bool


def read_settings(section: Mapping[str, Any]) -> PostgresSettings:
    """``section`` resolved; a missing connection string or a bad prefix raises ``StoreUnavailable``."""
    dsn_env = str(section.get("dsn_env") or DEFAULT_DSN_ENV)
    dsn = os.environ.get(dsn_env) or str(section.get("dsn") or "")
    if not dsn:
        raise StoreUnavailable(f"postgres persistence: set {dsn_env} or persistence.dsn to a connection string")
    prefix = str(section.get("schema_prefix") or DEFAULT_SCHEMA_PREFIX)
    if not _PREFIX_RE.fullmatch(prefix):
        raise StoreUnavailable(f"postgres persistence: schema_prefix {prefix!r} must match {_PREFIX_RE.pattern}")
    allow_incomplete = bool(section.get("allow_incomplete")) or os.environ.get("HERMES_PG_ALLOW_INCOMPLETE") == "1"
    return PostgresSettings(dsn, prefix, allow_incomplete)
