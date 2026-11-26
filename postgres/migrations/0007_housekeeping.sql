-- Housekeeping: the state_meta read order and the gateway backend heartbeats.

-- ``list_prefix`` reads in insertion order (a ``set`` on a stored key keeps its place), as SQLite's table scan does.
ALTER TABLE state_meta ADD COLUMN seq bigint GENERATED ALWAYS AS IDENTITY;
CREATE INDEX idx_state_meta_seq ON state_meta (seq);

CREATE TABLE gateway_heartbeats (
    backend_id     text PRIMARY KEY,
    pid            bigint NOT NULL,
    started_at     double precision NOT NULL,
    last_heartbeat double precision NOT NULL,
    profile        text NOT NULL DEFAULT '',
    host           text NOT NULL DEFAULT ''
);
