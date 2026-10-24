---
type: reference
title: Databases and environments
description: The session factory and engine (NullPool in Lambda, pool sizes), the test, local, dev and production databases and the Neon archive, which variable points where, and the --db override.
sources:
  - odds_core.database
  - odds_core.database.async_session_maker
  - odds_core.config.DatabaseConfig
  - odds_lambda.jobs.agent_run._build_agent_subprocess_env
  - odds_cli.db_override.override_database_url
  - tests/conftest.py
  - docker-compose.yml
  - deployment/aws/terraform/terraform.tfvars.example
  - odds_cli.commands.copy_from_prod.copy_from_production
  - odds_lambda.copy_from_production
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
| Test | in-process PostgreSQL (py-pglite) on TCP port 5434 | `tests/conftest.py` creates tables once per session, truncates all tables before each test and holds the shared fixtures. Used by `tests/unit` and `tests/integration` ([Test suite](test-suite.md)). |
| Local | `odds` in Docker Compose `postgres:15`, host port 5433 | `docker compose up -d postgres`; `postgres/postgres` (`docker-compose.yml`) |
| Dev | remote managed Postgres, a Neon dev branch | CI/CD only, via `secrets.DATABASE_URL` |
| Production | `odds` in the production host's Docker Compose `postgres:15`, host port 5433 | written by the production host's scheduler and by the agent as `odds_agent` ([ADR-0008](../../docs/adr/0008-mini-pc-is-production.md)) |
| Neon archive | remote managed Postgres, the Neon main branch | frozen; the database the AWS Lambdas wrote to. Its EPL and MLB data was merged into production on 2026-10-17 (#483). |

Production is the Compose Postgres of a single local host, which its own processes reach through a local `DATABASE_URL`. A development machine never points `DATABASE_URL` at it ([AGENTS.md](../../AGENTS.md#constraints)).

`.env.example` defines `LOCAL_DATABASE_URL`, `DEV_DATABASE_URL`, `PROD_DATABASE_URL` and sets `DATABASE_URL=${DEV_DATABASE_URL}`, so a fresh copy of it points at the shared dev database (#429). On existing machines `PROD_DATABASE_URL` names the Neon archive, not the production database. Lambda gets `DATABASE_URL` from Terraform, which takes the dev or production Neon connection string per environment (`deployment/aws/terraform/terraform.tfvars.example`; the provider is not verifiable from the repo beyond that template).

- Production migrates only through [deploy.sh](../scheduling/howto-run-scheduler-service.md#redeploy) on the production host, which runs `alembic upgrade head` before restarting the scheduler ([migrations](howto-database-migration.md)). `.github/workflows/deploy-prod.yml` migrates the Neon archive and must not be run ([Where each job runs](../scheduling/job-deployment-matrix.md#dormant-aws-rules)).
- No backup of the production database is configured in the repo. Data gaps are filled with [historical loads](../collection/odds-api-backfill.md).
- `tests/migration` connects to whatever `DATABASE_URL` points at and is run only by the deploy workflows ([Test suite](test-suite.md#migration-tests)).

### The agent database role

The betting agent subprocess swaps in `AGENT_DATABASE_URL` when set (`odds_lambda.jobs.agent_run._build_agent_subprocess_env`). That `odds_agent` role has SELECT on everything, INSERT/UPDATE on `paper_trades`, `match_briefs` and `agent_wakeups`, and write access to the APScheduler tables (needed by the MCP tool `odds_mcp.server.refresh_scrape`). When the variable is unset the agent inherits the write-capable DSN and only logs a warning ([Database role](../agent/agent-run-job.md#database-role)).

### The `--db` flag

`odds scheduler start --db NAME` and `odds agent run --db NAME` swap the dbname in `DATABASE_URL` through `odds_cli.db_override.override_database_url`. The engine already built in this process keeps the original URL, so in-process `odds_core.database.async_session_maker()` calls still hit the original database; only subprocesses and later `odds_core.config.get_settings()` calls see the swap ([The odds CLI](odds-cli.md#shared-conventions)).

## Copying production data

`odds copy from-prod START END` ([how-to](howto-copy-production-data.md)). `START` and `END` are positional `YYYY-MM-DD` dates, where `odds backfill` and `odds quality` use `--start`/`--end`; `END` is inclusive through 23:59:59 UTC. Options (`odds_cli.commands.copy_from_prod.copy_from_production`): `--sport/-s` (default `soccer_epl`), `--dry-run`, `--skip-existing/--overwrite` (default skip existing), `--prod-url`, `--local-url`.

- Source URL: `--prod-url`, else `PROD_DATABASE_URL`; with neither, the command fails. Target URL: `--local-url`, else `LOCAL_DATABASE_URL` (required). On existing machines `PROD_DATABASE_URL` names the Neon archive, so without `--prod-url` the copy reads frozen data, not the production database.
- Events come from `odds_lambda.storage.readers.OddsReader.get_events_by_date_range` for the sport and window, in kick-off order. With `--skip-existing` (the default), an event already present locally is skipped untouched.
- `--overwrite` replaces an existing event. Each event is read from production in its own session and replaced in one local transaction, so a failed production query fails only that event: its local predictions, odds rows and snapshots are deleted, the event row is merged with every column (status, scores and timestamps included), and production's snapshots (every column, so `api_request_id`, `fetch_tier` and `created_at` survive), production's odds rows (copied, not regenerated from `raw_data`) and production's predictions are inserted. Each prediction is re-pointed at the local id of its production snapshot, taken from the inserts themselves; a prediction whose snapshot does not belong to its event raises `ValueError`. After the run the summary counts local matches stored as several events (`odds_lambda.copy_from_production.CopyStats.local_twin_matches`); the copier never merges them ([Event merge](../storage/event-merge.md)).
- An event whose copy raises is rolled back whole, listed by id in the summary, counted in `events_failed` and never in `events_copied`; the run continues and the command exits 1 if any event failed.
- Running `odds_lambda.copy_from_production` as a script bypasses the CLI: its `__main__` block reads only the two dates and `--dry-run`, so every other option is ignored.
- `--dry-run` counts events and snapshots it would copy and writes nothing; its odds and prediction counts stay 0.

## Related

- [Create and apply a database migration](howto-database-migration.md)
- [Copy production data locally](howto-copy-production-data.md)
- [Set up and run locally](howto-develop-locally.md)
- [Storage layer](../storage/storage-layer.md)
- [AWS infrastructure](../operations/aws-infrastructure.md)
