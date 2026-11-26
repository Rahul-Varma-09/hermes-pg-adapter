-- The store's incarnation: a new token whenever the store is replaced under its open handles (restored over, or
-- dropped and created again), so a handle can tell the store it opened from the one now at its name.
ALTER TABLE store_identity ADD COLUMN incarnation text NOT NULL DEFAULT md5(random()::text || clock_timestamp()::text);
