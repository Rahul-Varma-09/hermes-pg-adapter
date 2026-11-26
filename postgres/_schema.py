"""One Postgres schema per profile store, and the migrations that build it.

A store is named by its ``hermes_home_key`` alone (``ProfileScope.name`` is a label that never selects storage), so the
schema is ``<prefix><label>_<digest>``: the label is the owning profile (``default`` or the profile's name) for a
human reading ``\\dn``, the digest of the home key keeps two homes apart. ``COMMENT ON SCHEMA`` records the home key.

Migrations are ``migrations/NNNN_<name>.sql``, run in order inside one transaction under a per-schema advisory lock with
``search_path`` set to the schema, so their SQL names tables unqualified. ``schema_migrations`` records each version.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from psycopg import Connection, sql

_MIGRATIONS_DIR = Path(__file__).parent / "migrations"
_MIGRATION_RE = re.compile(r"^(\d{4})_[a-z0-9_]+\.sql$")


def schema_name(prefix: str, home_key: str) -> str:
    """The schema serving the store of ``home_key``."""
    from persistence_assembly import store_owner_name

    owner = store_owner_name(Path(home_key)) or Path(home_key).name
    label = re.sub(r"[^a-z0-9_]+", "_", owner.lower()).strip("_")[:20] or "home"
    digest = hashlib.sha256(home_key.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}{label}_{digest}"


def migrations() -> list[tuple[int, str]]:
    """``(version, sql)`` for every migration file, in version order."""
    found = []
    for path in sorted(_MIGRATIONS_DIR.glob("*.sql")):
        match = _MIGRATION_RE.fullmatch(path.name)
        if match is None:
            raise RuntimeError(f"postgres persistence: badly named migration {path.name}")
        found.append((int(match.group(1)), path.read_text(encoding="utf-8")))
    return found


def schema_exists(conn: Connection, schema: str) -> bool:
    return conn.execute("SELECT 1 FROM pg_namespace WHERE nspname = %s", (schema,)).fetchone() is not None


def migrate(conn: Connection, schema: str, home_key: str) -> int:
    """Create ``schema`` when missing and apply every pending migration; returns how many ran."""
    ident = sql.Identifier(schema)
    with conn.transaction():
        conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (schema,))
        conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(ident))
        conn.execute(sql.SQL("COMMENT ON SCHEMA {} IS {}").format(ident, sql.Literal(f"hermes store: {home_key}")))
        conn.execute(sql.SQL("SET LOCAL search_path TO {}").format(ident))
        conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations ("
                     " version integer PRIMARY KEY,"
                     " applied_at timestamptz NOT NULL DEFAULT now())")
        applied = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}
        ran = 0
        for version, text in migrations():
            if version in applied:
                continue
            conn.execute(text)
            conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (version,))
            ran += 1
        conn.execute("INSERT INTO store_identity (home_key) VALUES (%s) ON CONFLICT DO NOTHING", (home_key,))
        return ran
