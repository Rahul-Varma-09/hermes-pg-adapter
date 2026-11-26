-- Conversation core and the runtime ledger it guards, ported from persistence/sqlite/_schema_ddl.py (schema v33).
--
-- Instants stay epoch-second double precision: the contract carries every instant as a time.time() float and compares
-- it exactly, which timestamptz (microsecond precision) cannot round-trip. Flags are boolean, internal JSON is jsonb,
-- and JSON the contract hands back verbatim (origin_json, payloads whose digest is part of an identity) stays text.

CREATE TABLE system_prompts (
    hash   text PRIMARY KEY,
    prompt text NOT NULL
);

CREATE TABLE sessions (
    id                                 text PRIMARY KEY,
    runtime_revision                   bigint NOT NULL DEFAULT 0,
    runtime_generation                 bigint NOT NULL DEFAULT 0,
    source                             text NOT NULL,
    created_source                     text,
    user_id                            text,
    session_key                        text,
    chat_id                            text,
    chat_type                          text,
    thread_id                          text,
    display_name                       text,
    origin_json                        text,
    expiry_finalized                   boolean NOT NULL DEFAULT false,
    model                              text,
    model_config                       jsonb,
    -- The prompt and the tools[] pin are stored once in system_prompts and named by their sha256 here.
    system_prompt_hash                 text REFERENCES system_prompts (hash),
    parent_session_id                  text REFERENCES sessions (id),
    started_at                         double precision NOT NULL,
    ended_at                           double precision,
    end_reason                         text,
    message_count                      bigint NOT NULL DEFAULT 0,
    tool_call_count                    bigint NOT NULL DEFAULT 0,
    input_tokens                       bigint NOT NULL DEFAULT 0,
    output_tokens                      bigint NOT NULL DEFAULT 0,
    cache_read_tokens                  bigint NOT NULL DEFAULT 0,
    cache_write_tokens                 bigint NOT NULL DEFAULT 0,
    reasoning_tokens                   bigint NOT NULL DEFAULT 0,
    cwd                                text,
    git_branch                         text,
    git_repo_root                      text,
    git_metadata_generation            bigint NOT NULL DEFAULT 0,
    billing_provider                   text,
    billing_base_url                   text,
    billing_mode                       text,
    estimated_cost_usd                 double precision,
    actual_cost_usd                    double precision,
    cost_status                        text,
    cost_source                        text,
    pricing_version                    text,
    title                              text,
    title_source                       text,
    last_activity_at                   double precision,
    last_activity_description          text,
    last_activity_provenance           text,
    api_call_count                     bigint NOT NULL DEFAULT 0,
    handoff_state                      text,
    handoff_platform                   text,
    handoff_error                      text,
    compression_failure_cooldown_until double precision,
    compression_failure_error          text,
    compression_fallback_streak        integer NOT NULL DEFAULT 0,
    compression_ineffective_count      integer NOT NULL DEFAULT 0,
    compression_recovery_deadline      double precision,
    compression_overload_streak        integer NOT NULL DEFAULT 0,
    profile_name                       text,
    transport_profile                  text,
    rewind_count                       integer NOT NULL DEFAULT 0,
    archived                           boolean NOT NULL DEFAULT false,
    auto_archived                      boolean NOT NULL DEFAULT false,
    pinned                             boolean NOT NULL DEFAULT false,
    hidden                             boolean NOT NULL DEFAULT false,
    last_read_at                       double precision,
    tool_names                         text,
    transcript_revision                bigint NOT NULL DEFAULT 0,
    prompt_cache_clock                 text
);

CREATE INDEX idx_sessions_source ON sessions (source, id);
CREATE INDEX idx_sessions_parent ON sessions (parent_session_id);
CREATE INDEX idx_sessions_started ON sessions (started_at DESC);
CREATE INDEX idx_sessions_session_key ON sessions (session_key, started_at DESC);
CREATE INDEX idx_sessions_gateway_peer ON sessions (source, user_id, chat_id, chat_type, thread_id, started_at DESC);
CREATE INDEX idx_sessions_handoff_state ON sessions (handoff_state, started_at);
CREATE INDEX idx_sessions_system_prompt_hash ON sessions (system_prompt_hash);
CREATE INDEX idx_sessions_tool_names ON sessions (tool_names);
CREATE INDEX idx_sessions_effective_activity ON sessions ((coalesce(last_activity_at, started_at)) DESC, started_at DESC);
CREATE INDEX idx_sessions_local_created_started ON sessions (created_source, started_at DESC) WHERE source = 'local';
CREATE UNIQUE INDEX idx_sessions_title_unique ON sessions (title) WHERE title IS NOT NULL;

-- A deleted id is never creatable again (SQLite keeps these as gateway.retired_session.v1.* state_meta keys). A
-- sweep's marker ``expires`` at the retry horizon; a user delete's never does.
CREATE TABLE retired_sessions (
    session_id text PRIMARY KEY,
    retired_at double precision NOT NULL,
    expires    boolean NOT NULL DEFAULT false
);

-- A row's content is in at most one of three columns: text, structured (multimodal) content as jsonb, or text that
-- holds a NUL byte (Postgres text cannot) as its UTF-8 bytes. Visibility: active; compaction-archived (NOT active AND
-- compacted); rewound (neither). JSON payloads are json, not jsonb: their text (key order included) is part of
-- a row's identity. Display order is computed at read time; display_slot is the slot an in-place prune kept.
CREATE TABLE messages (
    id                    bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    session_id            text NOT NULL REFERENCES sessions (id),
    role                  text NOT NULL,
    content               text,
    content_json          json,
    content_bytes         bytea,
    tool_call_id          text,
    tool_calls            json,
    tool_name             text,
    effect_disposition    text,
    timestamp             double precision NOT NULL,
    token_count           bigint,
    finish_reason         text,
    reasoning             text,
    reasoning_content     text,
    reasoning_details     json,
    codex_reasoning_items json,
    codex_message_items   json,
    platform_message_id   text,
    observed              boolean NOT NULL DEFAULT false,
    compressed_summary    boolean NOT NULL DEFAULT false,
    active                boolean NOT NULL DEFAULT true,
    compacted             boolean NOT NULL DEFAULT false,
    api_content           text,
    display_kind          text,
    display_metadata      json,
    display_slot          bigint,
    message_uid           text NOT NULL,
    absorbed_message_uids json,
    tool_call_uids        json,
    tool_call_uid         text,
    CHECK (num_nonnulls(content, content_json, content_bytes) <= 1)
);

CREATE INDEX idx_messages_session ON messages (session_id, timestamp);
CREATE INDEX idx_messages_session_id ON messages (session_id, id);
CREATE INDEX idx_messages_session_active ON messages (session_id, active, timestamp);
CREATE INDEX idx_messages_session_uid ON messages (session_id, message_uid) WHERE active;
CREATE INDEX idx_messages_assistant_calls_by_session ON messages (session_id)
    WHERE role = 'assistant' AND tool_calls IS NOT NULL;

CREATE TABLE session_model_usage (
    session_id         text NOT NULL REFERENCES sessions (id) ON DELETE CASCADE,
    model              text NOT NULL,
    billing_provider   text NOT NULL DEFAULT '',
    billing_base_url   text NOT NULL DEFAULT '',
    billing_mode       text NOT NULL DEFAULT '',
    task               text NOT NULL DEFAULT '',
    api_call_count     bigint NOT NULL DEFAULT 0,
    input_tokens       bigint NOT NULL DEFAULT 0,
    output_tokens      bigint NOT NULL DEFAULT 0,
    cache_read_tokens  bigint NOT NULL DEFAULT 0,
    cache_write_tokens bigint NOT NULL DEFAULT 0,
    reasoning_tokens   bigint NOT NULL DEFAULT 0,
    estimated_cost_usd double precision NOT NULL DEFAULT 0,
    actual_cost_usd    double precision NOT NULL DEFAULT 0,
    cost_status        text,
    cost_source        text,
    first_seen         double precision,
    last_seen          double precision,
    -- Insertion order: of two routes written at the same instant, the later-inserted one is the recent route.
    seq                bigint GENERATED ALWAYS AS IDENTITY UNIQUE,
    PRIMARY KEY (session_id, model, billing_provider, billing_base_url, billing_mode, task)
);
CREATE INDEX idx_session_model_usage_model ON session_model_usage (model);

-- Monotonic per-peer conversation generation; never garbage-collected (see the SQLite DDL for why).
CREATE TABLE conversation_generations (
    source      text NOT NULL,
    session_key text NOT NULL,
    generation  bigint NOT NULL DEFAULT 0,
    PRIMARY KEY (source, session_key)
);

-- Prompt refresh epochs: '' keys the profile-wide epoch, a session id that session's own.
CREATE TABLE prompt_refresh_epochs (
    session_id text PRIMARY KEY,
    epoch      bigint NOT NULL
);

CREATE TABLE state_meta (
    key   text PRIMARY KEY,
    value text
);

CREATE TABLE runtime_epoch (
    singleton   boolean PRIMARY KEY DEFAULT true CHECK (singleton),
    epoch       bigint NOT NULL CHECK (epoch > 0),
    instance_id text NOT NULL
);

CREATE TABLE session_admissions (
    seq               bigint GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    admission_id      text NOT NULL UNIQUE,
    request_id        text NOT NULL,
    principal_id      text NOT NULL,
    target_session_id text NOT NULL REFERENCES sessions (id) ON DELETE RESTRICT,
    lineage           jsonb NOT NULL,
    payload           jsonb NOT NULL,
    digest            text NOT NULL,
    intent            text NOT NULL CHECK (intent IN ('queue', 'steer', 'redirect')),
    status            text NOT NULL CHECK (status IN ('queued', 'started', 'unknown', 'terminal')),
    outcome           text,
    owner_epoch       bigint,
    generation        bigint,
    UNIQUE (principal_id, target_session_id, request_id),
    CHECK (status <> 'started' OR (owner_epoch IS NOT NULL AND generation IS NOT NULL))
);
CREATE INDEX session_admissions_pending ON session_admissions (target_session_id, status, seq);

CREATE TABLE worker_executions (
    execution_id    text PRIMARY KEY,
    session_id      text NOT NULL REFERENCES sessions (id) ON DELETE RESTRICT,
    kind            text NOT NULL CHECK (kind IN ('cron', 'child', 'compute', 'kanban')),
    owner_epoch     bigint NOT NULL,
    generation      bigint NOT NULL,
    status          text NOT NULL CHECK (status IN ('registered', 'running', 'unknown', 'terminal')),
    adoption_digest text NOT NULL,
    last_sequence   bigint NOT NULL DEFAULT 0 CHECK (last_sequence >= 0)
);
CREATE INDEX idx_worker_executions_session_status ON worker_executions (session_id, status);

CREATE TABLE worker_receipts (
    execution_id   text NOT NULL REFERENCES worker_executions (execution_id) ON DELETE RESTRICT,
    sequence       bigint NOT NULL CHECK (sequence > 0),
    payload_digest text NOT NULL,
    result_json    text NOT NULL,
    PRIMARY KEY (execution_id, sequence)
);
