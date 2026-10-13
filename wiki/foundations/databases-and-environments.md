---
type: reference
title: Databases and environments
description: The session factory and engine (NullPool in Lambda, pool sizes), the test/local/dev/prod databases and which variable points where, and the --db override.
sources:
  - odds_core.database
  - odds_core.database.async_session_maker
  - odds_core.config.DatabaseConfig
  - odds_lambda.jobs.agent_run._build_agent_subprocess_env
  - odds_cli.db_override.override_database_url
  - tests/conftest.py
  - docker-compose.yml
  - deployment/aws/terraform/terraform.tfvars.example
  - .env.example
  - .github/workflows/deploy-prod.yml
status: stable
---

# Databases and environments

All code gets sessions from one module-level factory: `odds_core.database.async_session_maker`, used as `async with async_session_maker() as session`. There is no wrapper layer. A session is never shared across async tasks. Everything is async (`asyncpg` URLs, `AsyncSession`).

## Engine creation

`odds_core.database` builds the engine at import time from `get_settings().database`, so the Lambda-versus-pool choice is fixed at that moment. Replacing the engine later (as `tests/migration/conftest.py` does) does not affect modules that already did `from odds_core.database import async_session_maker`.

- If `AWS_LAMBDA_FUNCTION_NAME` is set: `NullPool`. Each handler call runs `asyncio.run`, which makes a new event loop, and pooled asyncpg connections bound to an old loop fail. The switch keys on that env var, not on the scheduler backend.
- Otherwise: `pool_size=DatabaseConfig.pool_size` (default 5, env `DATABASE_POOL_SIZE`) with hardcoded `max_overflow=10`, so a process opens up to `DATABASE_POOL_SIZE + 10` connections. No `pool_pre_ping` is set, so stale connections in a long-lived local process are not probed.
- Sessions use `expire_on_commit=False`.

`odds_core.config.DatabaseConfig` uses prefix `DATABASE_`: `DATABASE_URL` (required, no default) and `DATABASE_POOL_SIZE` ([Configuration](configuration.md)). Importing any `odds_core` module therefore needs `DATABASE_URL` set ([Workspace packages](workspace-packages.md#odds-core)).

## Environments

| Environment | Database | Notes |
|---|---|---|
| Test | in-process PostgreSQL (py-pglite) on TCP port 5434 | `tests/conftest.py` creates tables once per session and truncates all tables before each test. Used by `tests/unit` and `tests/integration` ([Test suite](test-suite.md)). |
| Local | `odds` in Docker Compose `postgres:15`, host port 5433 | `docker compose up -d postgres`; `postgres/postgres` (`docker-compose.yml`) |
| Dev | remote managed Postgres, a Neon dev branch | CI/CD only, via `secrets.DATABASE_URL`; reset freely (inferred) |
| Production | remote managed Postgres, the Neon main branch | production deployments only |

`.env.example` defines `LOCAL_DATABASE_URL`, `DEV_DATABASE_URL`, `PROD_DATABASE_URL` and sets `DATABASE_URL=${DEV_DATABASE_URL}`, so a fresh copy of it points at the shared dev database; override it to `${LOCAL_DATABASE_URL}` for local work (#429, [local setup](howto-develop-locally.md)). Lambda gets `DATABASE_URL` from Terraform, which takes the dev or production Neon connection string per environment (`deployment/aws/terraform/terraform.tfvars.example`; the provider is not verifiable from the repo beyond that template).

> [!WARNING]
> Never point `DATABASE_URL` at production from a local shell. Production is migrated and written only by `deploy-prod.yml`; inspect it only with the read-only `scripts/check_*.py`.

- Production migrates only inside `.github/workflows/deploy-prod.yml` ([migrations](howto-database-migration.md)). To inspect production when debugging, use the read-only `scripts/check_*.py` scripts ([Debug the AWS self-scheduling jobs](../scheduling/howto-debug-aws-scheduler.md)).
- Backups and point-in-time recovery are the managed provider's; that is not verifiable from the repo. Data gaps are filled with [historical loads](../collection/odds-api-backfill.md).
- `tests/migration` connects to whatever `DATABASE_URL` points at and is run only by the deploy workflows ([Test suite](test-suite.md#migration-tests)).

### The agent database role

The betting agent subprocess swaps in `AGENT_DATABASE_URL` when set (`odds_lambda.jobs.agent_run._build_agent_subprocess_env`). That `odds_agent` role has SELECT on everything, INSERT/UPDATE on `paper_trades`, `match_briefs` and `agent_wakeups`, and write access to the APScheduler tables (needed by `odds_mcp.server.refresh_scrape`). When the variable is unset the agent inherits the write-capable DSN and only logs a warning ([Database role](../agent/agent-run-job.md#database-role)).

### The `--db` flag

`odds scheduler start --db NAME` and `odds agent run --db NAME` swap the dbname in `DATABASE_URL` through `odds_cli.db_override.override_database_url`. The engine already built in this process keeps the original URL, so in-process `async_session_maker()` calls still hit the original database; only subprocesses and later `odds_core.config.get_settings()` calls see the swap ([The odds CLI](odds-cli.md#shared-conventions)).

## Related

- [Create and apply a database migration](howto-database-migration.md)
- [Copy production data locally](howto-copy-production-data.md)
- [Set up and run locally](howto-develop-locally.md)
- [Storage layer](../storage/storage-layer.md)
- [AWS infrastructure](../operations/aws-infrastructure.md)
