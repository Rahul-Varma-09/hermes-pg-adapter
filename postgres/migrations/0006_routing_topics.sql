-- Telegram DM topic mode and topic bindings (SQLite's v3 telegram_dm_topic_* tables). A binding goes with its
-- session row.

CREATE TABLE telegram_dm_topic_mode (
    profile_name                  text NOT NULL DEFAULT 'default',
    chat_id                       text NOT NULL,
    user_id                       text NOT NULL,
    enabled                       boolean NOT NULL DEFAULT true,
    activated_at                  double precision NOT NULL,
    updated_at                    double precision NOT NULL,
    has_topics_enabled            boolean,
    allows_users_to_create_topics boolean,
    capability_checked_at         double precision,
    intro_message_id              text,
    pinned_message_id             text,
    PRIMARY KEY (profile_name, chat_id)
);

CREATE TABLE telegram_dm_topic_bindings (
    profile_name text NOT NULL DEFAULT 'default',
    chat_id      text NOT NULL,
    thread_id    text NOT NULL,
    user_id      text NOT NULL,
    session_key  text NOT NULL,
    session_id   text NOT NULL REFERENCES sessions (id) ON DELETE CASCADE,
    managed_mode text NOT NULL DEFAULT 'auto',
    linked_at    double precision NOT NULL,
    updated_at   double precision NOT NULL,
    PRIMARY KEY (profile_name, chat_id, thread_id)
);
CREATE UNIQUE INDEX idx_telegram_dm_topic_bindings_session ON telegram_dm_topic_bindings (session_id);
CREATE INDEX idx_telegram_dm_topic_bindings_user ON telegram_dm_topic_bindings (profile_name, user_id, chat_id);
