-- The authority family's durable key/value space: mutation receipts, local creation receipts (``binding:`` keys),
-- deferred picks, viewer closes and terminal evidence. A tuple-shaped key is stored as its JSON array text.
CREATE TABLE authority_meta (
    key   text PRIMARY KEY,
    value jsonb NOT NULL
);
CREATE INDEX idx_authority_meta_prefix ON authority_meta (key text_pattern_ops);
