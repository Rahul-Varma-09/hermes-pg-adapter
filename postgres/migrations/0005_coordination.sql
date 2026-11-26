-- Coordination ledgers: async delegations, delivery obligations, resumable subagent records and the gateway's
-- hygiene failure streaks. Payloads (task, event, result, children, document) are jsonb.

CREATE TABLE async_delegations (
    delegation_id        text PRIMARY KEY,
    origin_session       text NOT NULL,
    origin_ui_session_id text NOT NULL DEFAULT '',
    parent_session_id    text,
    origin_session_id    text NOT NULL DEFAULT '',
    owner_execution_id   text,
    owner_pid            bigint,
    owner_birth          bigint,
    task                 jsonb,
    state                text NOT NULL,
    dispatched_at        double precision NOT NULL,
    completed_at         double precision,
    updated_at           double precision NOT NULL,
    event                jsonb,
    result               jsonb,
    children             jsonb,
    delivery_state       text NOT NULL DEFAULT 'pending',
    delivery_attempts    integer NOT NULL DEFAULT 0,
    delivery_claim       text,
    delivery_claimed_at  double precision,
    delivered_at         double precision
);
CREATE INDEX idx_async_delegations_delivery ON async_delegations (delivery_state, completed_at);

CREATE TABLE delivery_obligations (
    obligation_id    text PRIMARY KEY,
    session_key      text NOT NULL,
    platform         text NOT NULL,
    chat_id          text NOT NULL,
    thread_id        text,
    content          text NOT NULL,
    state            text NOT NULL,
    attempts         integer NOT NULL DEFAULT 0,
    created_at       double precision NOT NULL,
    updated_at       double precision NOT NULL,
    owner_pid        bigint,
    owner_started_at bigint,
    last_error       text,
    adapter_profile  text
);

CREATE TABLE resumable_subagents (
    subagent_id      text PRIMARY KEY,
    state            text NOT NULL,
    pinned           boolean NOT NULL DEFAULT false,
    stop_cause       text,
    owner_pid        bigint,
    owner_started_at bigint,
    document         jsonb NOT NULL,
    created_at       double precision NOT NULL,
    updated_at       double precision NOT NULL,
    expires_at       double precision
);

CREATE TABLE gateway_hygiene_state (
    session_key    text PRIMARY KEY,
    failure_streak integer NOT NULL DEFAULT 0
);
