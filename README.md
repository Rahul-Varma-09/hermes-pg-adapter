# hermes-adapter

A PostgreSQL persistence provider for [Hermes Agent](https://github.com/NousResearch/hermes-agent). Each Hermes
profile's sessions, messages and runtime state live in their own schema of one Postgres database instead of
`state.db`.

The plugin is `postgres/`. It needs the Hermes branch `feature/postgres-persistence`, which adds the
`persistence.provider` plugin hook (`persistence/core/provider.py`, `plugins/persistence_providers/`,
`persistence_assembly.py`).

## Status

Phase 3 of the plan: the provider, connection pool, per-profile schemas, migrations, health probe and error
mapping are in place. Only the `health` repository is ported, so Hermes refuses to use the provider unless
`allow_incomplete: true` is set. Don't switch a real profile to it yet.

## Install

macOS:

```bash
ln -s ~/Documents/Projects/hermes-adapter/postgres ~/.hermes/plugins/postgres
cd ~/Development/hermes-agent && source ./activate && hermes plugins enable postgres
```

Windows (PowerShell; a junction needs no admin rights):

```powershell
New-Item -ItemType Junction -Path "$env:LOCALAPPDATA\hermes\plugins\postgres" -Target "<repo>\hermes-adapter\postgres"
hermes plugins enable postgres
```

`hermes plugins enable` lets `pm` install `psycopg` and `psycopg-pool` into Hermes's environment. Don't `pip install`
them yourself: the next sync would remove them.

## Configure

Put the password in `~/.pgpass` (Windows: `%APPDATA%\postgresql\pgpass.conf`) and the connection string in
`HERMES_PG_DSN`, e.g. `postgresql://hermes@localhost:5432/hermes`. Then in `~/.hermes/config.yaml`:

```yaml
persistence:
  provider: postgres
  dsn_env: HERMES_PG_DSN      # default
  schema_prefix: hermes_      # default
```

A profile's schema is `<prefix><profile>_<hash of its home path>`, e.g. `hermes_default_3f9a1c2b7d`; its comment
records the home path (`\dn+` in psql).

## Test

From `~/Development/hermes-agent` with `source ./activate`:

```bash
export HERMES_PG_TEST_DSN=postgresql://hermes@localhost:5432/hermes
export HERMES_PG_PLUGIN_DIR=~/Documents/Projects/hermes-adapter/postgres
"$__HERMES_TEST_PYTHON" -m pytest tests/contracts -q -k postgres -p no:randomly
```

Each test uses fresh `hermes_ct_<random>_*` schemas and drops them afterwards. Baseline at the end of Phase 3:
15 passed, 1729 failed, 144 errors. The failures are the repositories still to port.
