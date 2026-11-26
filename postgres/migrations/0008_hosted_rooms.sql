-- Hosted rooms (Group Chats): every hosted-room table of one placement, as rows of one keyed table, and the
-- attachment blob bytes. ``tbl`` names the logical table; ``k1`` is the row's JSON-encoded key, or a per-room table's
-- room id; ``k2`` is a per-room table's JSON-encoded inner key ('' for a table keyed once). The same DDL builds the
-- install's shared placement schema, so it is idempotent.

CREATE TABLE IF NOT EXISTS hosted_room_rows (
    tbl text  NOT NULL,
    k1  text  NOT NULL,
    k2  text  NOT NULL DEFAULT '',
    row jsonb NOT NULL,
    PRIMARY KEY (tbl, k1, k2)
);
CREATE INDEX IF NOT EXISTS idx_hosted_room_rows_event_id ON hosted_room_rows (tbl, k1, (row ->> 'event_id'));

CREATE TABLE IF NOT EXISTS hosted_room_blob_files (
    name text  PRIMARY KEY,
    data bytea NOT NULL
);
