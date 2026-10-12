---
type: concept
title: Database connections and environments
summary: How code obtains DB sessions (async_session_maker), NullPool in Lambda, pool sizing and the import-time engine, plus the test/local/dev/prod databases and their env vars.
sources:
  - packages/odds-core/odds_core/database.py::async_session_maker
  - packages/odds-core/odds_core/config.py::DatabaseConfig
  - packages/odds-lambda/odds_lambda/jobs/agent_run.py::_build_agent_subprocess_env
  - tests/conftest.py
  - tests/migration/conftest.py
  - docs/DATABASE.md#database-environments
  - docs/DATABASE.md#connection-pooling
  - docs/DEVELOPMENT.md#database-settings-databaseconfig
  - docs/DEPLOYMENT.md#operations
verified_at: 7887039
---

# Database connections and environments

All code gets sessions from one module-level factory: `packages/odds-core/odds_core/database.py::async_session_maker`, used as `async with async_session_maker() as session`. There is no wrapper layer; do not add one. Never share a session across async tasks. Everything is async (`asyncpg` URLs, `AsyncSession`).

## Engine creation

`database.py` builds the engine at import time from `get_settings().database`:

- If `AWS_LAMBDA_FUNCTION_NAME` is set: `NullPool`. Each handler call runs `asyncio.run`, which makes a new event loop, and pooled asyncpg connections bound to an old loop fail. The switch keys on that env var, not on the scheduler backend.
- Otherwise: `pool_size=DatabaseConfig.pool_size` (default 5, env `DATABASE_POOL_SIZE`) with hardcoded `max_overflow=10`.
- Sessions use `expire_on_commit=False`.

`DatabaseConfig` (`packages/odds-core/odds_core/config.py::DatabaseConfig`) uses prefix `DATABASE_`: `DATABASE_URL` (required, no default) and `DATABASE_POOL_SIZE`. See [Configuration](configuration.md).

## Environments

| Environment | Database | Notes |
|---|---|---|
| Test | in-process PostgreSQL (py-pglite) on TCP port 5434 | `tests/conftest.py` creates tables once per session and truncates all tables before each test. Used by `tests/unit` and `tests/integration`. |
| Local | `odds` in Docker Compose `postgres:15`, host port 5433 | `docker compose up -d postgres`; `postgres/postgres` (`docker-compose.yml`) |
| Dev | remote managed Postgres | CI/CD only, via `secrets.DATABASE_URL` |
| Production | remote managed Postgres | production deployments only |

`.env.example` defines `LOCAL_DATABASE_URL`, `DEV_DATABASE_URL`, `PROD_DATABASE_URL` and sets `DATABASE_URL=${DEV_DATABASE_URL}`; override it to `${LOCAL_DATABASE_URL}` for local work. Never point `DATABASE_URL` at production locally. Lambda gets `DATABASE_URL` from Terraform. The betting agent subprocess swaps in `AGENT_DATABASE_URL` (a read-mostly role) when set (`_build_agent_subprocess_env`); when unset it inherits the write-capable DSN and only logs a warning ([Betting agent](betting-agent.md)).

`tests/migration` is different: it connects to whatever `DATABASE_URL` points at and is run only by the deploy workflows ([Database migration](howto-database-migration.md)). Its conftest replaces `odds_core.database.engine` and `async_session_maker` with a NullPool pair in `pytest_configure`.

Backups and point-in-time recovery are the managed provider's; that is not verifiable from the repo. Data gaps are filled with [backfill](howto-historical-backfill.md).

## Hazards

- The engine is created on import, so importing any module that touches `odds_core.database` requires `DATABASE_URL` to be set, and Lambda/pool choice is fixed at that moment. Replacing the engine later (as `tests/migration/conftest.py` does) does not affect modules that already did `from odds_core.database import async_session_maker`.
- Docs say "pool of 5"; the effective ceiling is `pool_size + 10` per process.
- No `pool_pre_ping` is set; stale connections in a long-lived local process are not probed.
- `.env.example` defaults to the dev database, not local, so a fresh copy of it writes to dev.

## Related

- [Odds writer](odds-writer.md)
- [AWS infrastructure](aws-infrastructure.md)
- [Running tests](howto-run-tests.md)
- [Local setup](howto-local-setup.md)
