-- The store's own identity: which Hermes home it serves. The repository tables arrive in later migrations, one
-- family at a time, ported from persistence/sqlite/_schema_ddl.py.
CREATE TABLE store_identity (
    singleton  boolean PRIMARY KEY DEFAULT true CHECK (singleton),
    home_key   text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
