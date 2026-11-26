"""Copy one Hermes profile's SQLite ``state.db`` into that profile's Postgres schema.

Run it with Hermes' own Python, from the Hermes checkout (``source ./activate``), with the profile's Postgres
connection string in ``HERMES_PG_DSN``::

    python ~/Documents/Projects/hermes-adapter/migrate_sqlite_to_postgres.py --home ~/.hermes            # dry run
    python ~/Documents/Projects/hermes-adapter/migrate_sqlite_to_postgres.py --home ~/.hermes --commit   # migrate

A dry run copies into a scratch schema (``<profile schema>_dryrun``), compares it with the source and drops it. With
``--commit`` it copies into the profile's own schema, which must hold no sessions yet. Either way the whole copy is one
transaction: a failure leaves the target as it was.

What is copied: sessions, the stored prompts and tool pins, every transcript row (ids kept, read through Hermes' own
SQLite repository so every content encoding and sidecar arrives decoded), admissions, model usage, gateway routing and
heartbeats, the runtime epoch, the local creation receipts, settled admission results, retirement markers, and the
``state_meta`` keys whose meaning is the same in both stores. Search indexes are rebuilt by the Postgres store; SQLite
storage bookkeeping (FTS version, file generation, backfill marks) is not carried. A table or ``state_meta`` key kind
this tool does not know refuses the whole run and is named, so nothing is dropped silently.

The SQLite file is opened read-only and never changed. Stop the gateway before ``--commit`` so nothing is written to
``state.db`` after it has been copied.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

# Tables copied (in this order), and tables that hold nothing the Postgres store keeps.
_COPIED = ("system_prompts", "sessions", "messages", "session_admissions", "session_model_usage", "gateway_routing",
           "gateway_heartbeats", "runtime_epoch", "state_meta", "conversation_generations")
_DERIVED = ("schema_version",)
_DERIVED_PREFIXES = ("messages_fts",)
# state_meta keys of SQLite's own storage bookkeeping.
_SQLITE_BOOKKEEPING = {"fts_storage_version", "db_file_generation", "message_uid_backfill", "legacy_refusal_repair"}
# state_meta keys and prefixes that mean the same in the Postgres store's state_meta.
_STATE_META_SAME = {"store_instance_id", "store_created_at_utc", "telegram_dm_topic_schema_version"}
_STATE_META_SAME_PREFIXES = ("gateway.hosted.owner.v1:", "gateway.hosted.transport.v1:", "gateway.hosted.member_homes.v1:",
                             "goal:", "heartbeat:", "loop:")
_BOOLEANS = ("expiry_finalized", "archived", "auto_archived", "pinned", "hidden")


def _authority_key(key: str, value: str) -> Optional[tuple[Any, Any]]:
    """The Postgres ``authority_meta`` key and value of one SQLite ``state_meta`` row, or None when it is not one."""
    mapping: tuple[tuple[str, Callable[[str], Any], Callable[[str], Any]], ...] = (
        ("gateway.local_policy.v1:", lambda rest: "binding:" + rest, json.loads),
        ("gateway.admission.result.v1.", lambda rest: ("admission_result", rest), json.loads),
        ("gateway.local_legacy.v1:", lambda rest: "gateway.local_legacy.v1:" + rest, json.loads),
        ("gateway.history_control.v1.", lambda rest: "history_control:" + rest, json.loads),
        ("gateway.api.binding.v1.", lambda rest: "gateway.api.binding.v1." + rest, json.loads),
        ("gateway.api.conversation.v1.", lambda rest: "gateway.api.conversation.v1." + rest, lambda text: text),
        ("gateway.mutation.v1.", lambda rest: "gateway.mutation.v1." + rest, lambda text: text),
        ("gateway.viewer_close.v1:", lambda rest: "gateway.viewer_close.v1:" + rest, lambda text: text),
        ("gateway.deferred_pick.v1.", lambda rest: "gateway.deferred_pick.v1." + rest, lambda text: text),
    )
    for prefix, new_key, new_value in mapping:
        if key.startswith(prefix):
            return new_key(key[len(prefix):]), new_value(value)
    return None


def _plan(src: sqlite3.Connection) -> list[str]:
    """Refusals: every non-empty table and ``state_meta`` key kind this tool does not carry."""
    problems = []
    for (name,) in src.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"):
        if name in _COPIED or name in _DERIVED or name.startswith(_DERIVED_PREFIXES):
            continue
        if src.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0]:
            problems.append(f"table {name} has rows this tool does not copy")
    for key, value in src.execute("SELECT key, value FROM state_meta"):
        known = (key in _SQLITE_BOOKKEEPING or key in _STATE_META_SAME or key.startswith(_STATE_META_SAME_PREFIXES)
                 or key.startswith("gateway.retired_session.v1.") or _authority_key(key, value or "null") is not None)
        if not known:
            problems.append(f"state_meta key {key!r} is of a kind this tool does not copy")
    return problems


def _gateway_running(home: Path) -> Optional[int]:
    try:
        pid = int(json.loads((home / "gateway.pid").read_text())["pid"])
        os.kill(pid, 0)
        return pid
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _copy(src: sqlite3.Connection, conn: Any, sqlite_store: Any, plugin: Any) -> dict[str, int]:
    """Copy every carried table into ``conn``'s schema (search_path set); returns rows copied per table."""
    from psycopg.types.json import Jsonb
    from persistence.conversation.messages import NewMessage, TranscriptRead, TranscriptView
    from persistence.core.conventions import RowVisibility

    store, meta = plugin.conversation._transcript, plugin.authority._meta
    counts: dict[str, int] = {}

    def pg_columns(table: str) -> set[str]:
        return {row[0] for row in conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_schema = current_schema() AND table_name = %s",
            (table,)).fetchall()}

    for hash_, prompt in src.execute("SELECT hash, prompt FROM system_prompts"):
        conn.execute("INSERT INTO system_prompts (hash, prompt) VALUES (%s, %s) ON CONFLICT DO NOTHING", (hash_, prompt))
    counts["system_prompts"] = src.execute("SELECT count(*) FROM system_prompts").fetchone()[0]

    # Sessions: the shared columns, flags as booleans, the config as JSON; parents linked once every row exists.
    src.row_factory = sqlite3.Row
    shared = [c for c in (row[1] for row in src.execute("PRAGMA table_info(sessions)")) if c in pg_columns("sessions")]
    parents = []
    for row in src.execute("SELECT * FROM sessions ORDER BY started_at, id"):
        values = {c: row[c] for c in shared}
        for flag in _BOOLEANS:
            if flag in values:
                values[flag] = bool(values[flag] or 0)
        if values.get("model_config"):
            values["model_config"] = Jsonb(json.loads(values["model_config"]))
        if row["system_prompt"] is not None and not values.get("system_prompt_hash"):
            values["system_prompt_hash"] = plugin._sql.store_text(conn, row["system_prompt"])
        if values.get("parent_session_id"):
            parents.append((row["id"], values["parent_session_id"]))
        values["parent_session_id"] = None
        conn.execute(f"INSERT INTO sessions ({', '.join(values)}) VALUES ({', '.join(['%s'] * len(values))})", tuple(values.values()))
    for sid, parent in parents:
        conn.execute("UPDATE sessions SET parent_session_id = %s WHERE id = %s", (parent, sid))
    counts["sessions"] = src.execute("SELECT count(*) FROM sessions").fetchone()[0]

    # Transcripts, through the SQLite repository's own decoding, ids kept.
    def sidecar(text: Any) -> Any:
        return json.loads(text) if isinstance(text, str) else text

    copied = 0
    for (sid,) in src.execute("SELECT id FROM sessions"):
        for row in sqlite_store.messages.read_transcript(sid, TranscriptRead(view=TranscriptView.ALL)):
            message = store.payload(NewMessage(
                role=row.role, content=row.content, timestamp=row.timestamp,
                tool_calls=tuple(row.tool_calls) if isinstance(row.tool_calls, list) else (),
                tool_name=row.tool_name, tool_call_id=row.tool_call_id, display_kind=row.display_kind,
                compressed_summary=row.compressed_summary, api_content=row.api_content,
                display_metadata=row.display_metadata, platform_message_id=row.platform_message_id, observed=row.observed,
                effect_disposition=row.effect_disposition, finish_reason=row.finish_reason, reasoning=row.reasoning,
                reasoning_content=row.reasoning_content, reasoning_details=sidecar(row.reasoning_details),
                codex_reasoning_items=sidecar(row.codex_reasoning_items), codex_message_items=sidecar(row.codex_message_items),
                message_uid=row.message_uid or uuid.uuid4().hex,
                absorbed_message_uids=tuple(sidecar(row.absorbed_message_uids) or ()),
                tool_call_uids=sidecar(row.tool_call_uids), tool_call_uid=row.tool_call_uid, token_count=row.token_count))
            active, compacted = row.visibility is RowVisibility.ACTIVE, row.visibility is RowVisibility.COMPACTION_ARCHIVED
            names = ", ".join(store.PAYLOAD_COLUMNS)
            conn.execute(f"INSERT INTO messages (id, session_id, {names}, active, compacted) OVERRIDING SYSTEM VALUE "
                         f"VALUES (%s, %s, {', '.join(['%s'] * len(store.PAYLOAD_COLUMNS))}, %s, %s)",
                         (int(row.id), sid, *store.columns(message), active, compacted))
            copied += 1
    conn.execute("SELECT setval(pg_get_serial_sequence('messages', 'id'), greatest(coalesce(max(id), 0), 1), max(id) IS NOT NULL) FROM messages")
    counts["messages"] = copied

    for row in src.execute("SELECT * FROM session_admissions ORDER BY seq"):
        conn.execute("INSERT INTO session_admissions (seq, admission_id, request_id, principal_id, target_session_id, lineage, "
                     "payload, digest, intent, status, outcome, owner_epoch, generation) "
                     "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                     (row["seq"], row["admission_id"], row["request_id"], row["principal_id"], row["target_session_id"],
                      Jsonb(json.loads(row["lineage_json"] or "[]")), Jsonb(json.loads(row["payload_json"] or "{}")),
                      row["payload_digest"], row["intent"], row["status"], row["outcome"], row["owner_epoch"], row["generation"]))
    conn.execute("SELECT setval(pg_get_serial_sequence('session_admissions', 'seq'), greatest(coalesce(max(seq), 0), 1), "
                 "max(seq) IS NOT NULL) FROM session_admissions")
    counts["session_admissions"] = src.execute("SELECT count(*) FROM session_admissions").fetchone()[0]

    usage_cols = [c for c in (row[1] for row in src.execute("PRAGMA table_info(session_model_usage)")) if c in pg_columns("session_model_usage")]
    for row in src.execute("SELECT * FROM session_model_usage ORDER BY last_seen"):
        values = {c: row[c] for c in usage_cols}
        for column in ("billing_provider", "billing_base_url", "billing_mode", "task"):
            values[column] = values.get(column) or ""
        for column in ("estimated_cost_usd", "actual_cost_usd"):
            values[column] = values.get(column) or 0.0
        conn.execute(f"INSERT INTO session_model_usage ({', '.join(values)}) VALUES ({', '.join(['%s'] * len(values))})",
                     tuple(values.values()))
    counts["session_model_usage"] = src.execute("SELECT count(*) FROM session_model_usage").fetchone()[0]

    for row in src.execute("SELECT scope, session_key, entry_json, updated_at FROM gateway_routing"):
        try:
            target = json.loads(row["entry_json"]).get("session_id")
        except (ValueError, AttributeError):
            target = None
        conn.execute("INSERT INTO gateway_routing (scope, session_key, entry_json, session_id, updated_at) VALUES (%s, %s, %s, %s, %s)",
                     (row["scope"], row["session_key"], row["entry_json"], target if isinstance(target, str) else None, row["updated_at"]))
    counts["gateway_routing"] = src.execute("SELECT count(*) FROM gateway_routing").fetchone()[0]

    for row in src.execute("SELECT * FROM gateway_heartbeats"):
        conn.execute("INSERT INTO gateway_heartbeats (backend_id, pid, started_at, last_heartbeat, profile, host) VALUES (%s, %s, %s, %s, %s, %s)",
                     (row["backend_id"], row["pid"], row["started_at"], row["last_heartbeat"], row["profile"] or "", row["host"] or ""))
    counts["gateway_heartbeats"] = src.execute("SELECT count(*) FROM gateway_heartbeats").fetchone()[0]

    for row in src.execute("SELECT epoch, instance_id FROM runtime_epoch"):
        conn.execute("INSERT INTO runtime_epoch (singleton, epoch, instance_id) VALUES (true, %s, %s) "
                     "ON CONFLICT (singleton) DO UPDATE SET epoch = excluded.epoch, instance_id = excluded.instance_id",
                     (row["epoch"], row["instance_id"]))

    if src.execute("SELECT 1 FROM sqlite_master WHERE name = 'conversation_generations'").fetchone():
        for row in src.execute("SELECT source, session_key, generation FROM conversation_generations"):
            conn.execute("INSERT INTO conversation_generations (source, session_key, generation) VALUES (%s, %s, %s)", tuple(row))

    carried = 0
    for key, value in src.execute("SELECT key, value FROM state_meta"):
        if key in _SQLITE_BOOKKEEPING:
            continue
        if key in _STATE_META_SAME or key.startswith(_STATE_META_SAME_PREFIXES):
            conn.execute("INSERT INTO state_meta (key, value) VALUES (%s, %s)", (key, value))
        elif key.startswith("gateway.retired_session.v1."):
            marker = json.loads(value)
            conn.execute("INSERT INTO retired_sessions (session_id, retired_at, expires) VALUES (%s, %s, %s)",
                         (key[len("gateway.retired_session.v1."):], float(marker.get("retired_at") or 0), bool(marker.get("expires"))))
        else:
            new_key, new_value = _authority_key(key, value)  # type: ignore[misc]
            meta.put(conn, new_key, new_value)
        carried += 1
    counts["state_meta"] = carried
    return counts


def _compare(src: sqlite3.Connection, sqlite_store: Any, pg_store: Any) -> list[str]:
    """Differences between the source and the copy, read back through both stores' public verbs."""
    from persistence.conversation.messages import TranscriptRead, TranscriptView

    problems = []
    for (sid,) in src.execute("SELECT id FROM sessions"):
        before, after = sqlite_store.sessions.get(sid), pg_store.sessions.get(sid)
        if after is None:
            problems.append(f"session {sid} is missing")
            continue
        for field in ("source", "title", "model", "message_count", "started_at", "ended_at", "end_reason", "session_key", "chat_id"):
            if getattr(before, field) != getattr(after, field):
                problems.append(f"session {sid}: {field} {getattr(before, field)!r} != {getattr(after, field)!r}")
        read = TranscriptRead(view=TranscriptView.ALL)
        old = [r.as_wire_row() for r in sqlite_store.messages.read_transcript(sid, read)]
        new = [r.as_wire_row() for r in pg_store.messages.read_transcript(sid, read)]
        if old != new:
            problems.append(f"session {sid}: transcript differs ({len(old)} rows before, {len(new)} after)")
    for (key,) in src.execute("SELECT key FROM state_meta WHERE key LIKE 'gateway.local_policy.v1:%'"):
        owner = key[len("gateway.local_policy.v1:"):]
        if sqlite_store.bindings.receipt(owner) != pg_store.bindings.receipt(owner):
            problems.append(f"local receipt of {owner} differs")
    for (admission_id,) in src.execute("SELECT admission_id FROM session_admissions"):
        if sqlite_store.admissions.get(admission_id=admission_id) != pg_store.admissions.get(admission_id=admission_id):
            problems.append(f"admission {admission_id} differs")
        if (sqlite_store.admissions.settled_result(admission_id=admission_id)
                != pg_store.admissions.settled_result(admission_id=admission_id)):
            problems.append(f"settled result of admission {admission_id} differs")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--home", type=Path, default=Path("~/.hermes"), help="the profile's Hermes home (holds state.db)")
    parser.add_argument("--commit", action="store_true", help="copy into the profile's own schema (default: dry run)")
    args = parser.parse_args()

    from persistence import ProfileScope
    from persistence.sqlite import SQLiteBackend
    from plugins.persistence_providers import load_persistence_provider

    home = args.home.expanduser().resolve()
    state_db = home / "state.db"
    if not state_db.is_file():
        print(f"no state.db under {home}")
        return 2
    scope = ProfileScope(profile_dir=home, name="default")
    src = sqlite3.connect(f"file:{state_db}?mode=ro", uri=True)
    problems = _plan(src)
    if problems:
        print("refusing to migrate; these would not be carried:\n  " + "\n  ".join(problems))
        return 1
    if args.commit and (pid := _gateway_running(home)) is not None:
        print(f"refusing to migrate: the Hermes gateway (pid {pid}) is still running on {home}; stop it first "
              "so nothing is written to state.db after it has been copied")
        return 1

    provider = load_persistence_provider("postgres")
    plugin = sys.modules[type(provider).__module__.rsplit(".", 1)[0]]
    section = {"provider": "postgres"}
    backend = provider.assemble({scope.hermes_home_key: state_db}, {scope.hermes_home_key: section})
    profile_schema = backend.schema_for(scope)
    target = profile_schema if args.commit else profile_schema + "_dryrun"
    backend._schemas[scope.hermes_home_key] = target
    pg_store = backend.open(scope)
    sqlite_backend = SQLiteBackend({scope.hermes_home_key: state_db}, read_only=True)
    sqlite_store = sqlite_backend.open(scope)
    try:
        with pg_store.transaction() as conn:
            existing = conn.execute("SELECT count(*) FROM sessions").fetchone()[0]
            if existing:
                print(f"refusing to migrate: schema {target} already holds {existing} sessions")
                return 1
            counts = _copy(src, conn, sqlite_store, plugin)
        problems = _compare(src, sqlite_store, pg_store)
        if args.commit and not problems:
            from persistence_assembly import pin_provider

            pin_provider(home, "postgres")
        print(f"{'migrated into' if args.commit else 'dry run in'} schema {target}:")
        for table, count in counts.items():
            print(f"  {table:22} {count}")
        print("verification: " + ("transcripts and sessions read back identically" if not problems else "\n  " + "\n  ".join(problems)))
        return 0 if not problems else 1
    finally:
        if not args.commit:
            from psycopg import sql

            with backend.pool().connection() as conn:
                conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(target)))
        sqlite_backend.close()
        backend.close()


if __name__ == "__main__":
    sys.exit(main())
