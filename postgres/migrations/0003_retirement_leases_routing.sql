-- What a session delete retires in its transaction, the leases that guard it, and the routing entries it drops.

-- A retired admission keeps its identity and digest (an exact retry reads its terminal outcome), never the payload.
CREATE TABLE admission_terminals (
    admission_id      text PRIMARY KEY,
    principal_id      text NOT NULL,
    target_session_id text NOT NULL,
    request_id        text NOT NULL,
    row               jsonb NOT NULL,
    retired_at        double precision NOT NULL,
    UNIQUE (principal_id, target_session_id, request_id)
);

-- A retired worker execution with its receipts; every result but the last is scrubbed, digests stay.
CREATE TABLE worker_terminals (
    execution_id text PRIMARY KEY,
    session_id   text NOT NULL,
    row          jsonb NOT NULL,
    retired_at   double precision NOT NULL
);

-- A compaction lease on a session and a turn lease on a conversation (the compression root of its sessions).
CREATE TABLE compression_locks (
    session_id  text PRIMARY KEY,
    holder      text NOT NULL,
    acquired_at double precision NOT NULL,
    expires_at  double precision NOT NULL
);
CREATE INDEX idx_compression_locks_expires ON compression_locks (expires_at);

CREATE TABLE session_turn_leases (
    conversation_id text PRIMARY KEY,
    holder          text NOT NULL,
    acquired_at     double precision NOT NULL,
    expires_at      double precision NOT NULL
);
CREATE INDEX idx_session_turn_leases_expires ON session_turn_leases (expires_at);

-- Gateway routing entries: a peer's key -> the session it routes to. The entry is kept as the exact JSON text its
-- writer produced (the contract hands it back verbatim); ``session_id`` is the entry's target, when it names one as
-- a string, for the per-session sweeps.
CREATE TABLE gateway_routing (
    scope       text NOT NULL DEFAULT '',
    session_key text NOT NULL,
    entry_json  text NOT NULL,
    session_id  text,
    updated_at  double precision NOT NULL,
    PRIMARY KEY (scope, session_key)
);
CREATE INDEX idx_gateway_routing_session ON gateway_routing (session_id);
