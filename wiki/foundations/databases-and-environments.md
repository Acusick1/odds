---
type: component
title: Databases, environments and copying production data
description: How code gets sessions, NullPool in Lambda, the test/local/dev/prod databases and which env var points where, how to copy production data locally, and the fallbacks that can silently hit the wrong database.
sources:
  - odds_core.database.async_session_maker
  - odds_core.config.DatabaseConfig
  - odds_lambda.jobs.agent_run._build_agent_subprocess_env
  - odds_lambda.copy_from_production.copy_from_prod
  - odds_lambda.copy_from_production.ProductionDataCopier
  - odds_cli.commands.copy_from_prod.copy_from_production
  - odds_cli.db_override.override_database_url
  - tests/conftest.py
  - tests/migration/conftest.py
  - docker-compose.yml
  - deployment/aws/terraform/terraform.tfvars.example
  - .env.example
status: stable
---

# Databases, environments and copying production data

All code gets sessions from one module-level factory: `odds_core.database.async_session_maker`, used as `async with async_session_maker() as session`. There is no wrapper layer; do not add one. Never share a session across async tasks. Everything is async (`asyncpg` URLs, `AsyncSession`).

## Engine creation

`odds_core.database` builds the engine at import time from `get_settings().database`:

- If `AWS_LAMBDA_FUNCTION_NAME` is set: `NullPool`. Each handler call runs `asyncio.run`, which makes a new event loop, and pooled asyncpg connections bound to an old loop fail. The switch keys on that env var, not on the scheduler backend.
- Otherwise: `pool_size=DatabaseConfig.pool_size` (default 5, env `DATABASE_POOL_SIZE`) with hardcoded `max_overflow=10`.
- Sessions use `expire_on_commit=False`.

`odds_core.config.DatabaseConfig` (`odds_core.config.DatabaseConfig`) uses prefix `DATABASE_`: `DATABASE_URL` (required, no default) and `DATABASE_POOL_SIZE` ([Configuration](configuration.md)). Importing any `odds_core` module therefore needs `DATABASE_URL` set.

## Environments

| Environment | Database | Notes |
|---|---|---|
| Test | in-process PostgreSQL (py-pglite) on TCP port 5434 | `tests/conftest.py` creates tables once per session and truncates all tables before each test. Used by `tests/unit` and `tests/integration`. |
| Local | `odds` in Docker Compose `postgres:15`, host port 5433 | `docker compose up -d postgres`; `postgres/postgres` (`docker-compose.yml`) |
| Dev | remote managed Postgres, a Neon dev branch | CI/CD only, via `secrets.DATABASE_URL`; reset freely (inferred) |
| Production | remote managed Postgres, the Neon main branch | production deployments only |

`.env.example` defines `LOCAL_DATABASE_URL`, `DEV_DATABASE_URL`, `PROD_DATABASE_URL` and sets `DATABASE_URL=${DEV_DATABASE_URL}`; override it to `${LOCAL_DATABASE_URL}` for local work ([local setup](howto-develop-locally.md)). Never point `DATABASE_URL` at production locally. Lambda gets `DATABASE_URL` from Terraform, which takes the dev or production Neon connection string per environment (`deployment/aws/terraform/terraform.tfvars.example`; the provider is not verifiable from the repo beyond that template).

Rules for production:

- Never run migrations or `tests/migration` against it by hand; production migrates only inside `deploy-prod.yml` ([migrations](howto-database-migration.md)).
- To inspect production when debugging, use only the read-only `scripts/check_*.py` scripts ([Debug the AWS self-scheduling jobs](../scheduling/howto-debug-aws-scheduler.md)).
- Backups and point-in-time recovery are the managed provider's; that is not verifiable from the repo. Data gaps are filled with [historical loads](../collection/howto-load-historical-odds.md).

`tests/migration` connects to whatever `DATABASE_URL` points at and is run only by the deploy workflows; bare `uv run pytest` also collects it because `testpaths` is `tests`. Its conftest replaces `odds_core.database.engine` and `async_session_maker` with a NullPool pair in `pytest_configure` ([Run the tests](howto-develop-locally.md#run-the-tests)).

### The agent database role

The betting agent subprocess swaps in `AGENT_DATABASE_URL` when set (`odds_lambda.jobs.agent_run._build_agent_subprocess_env`). That `odds_agent` role has SELECT on everything, INSERT/UPDATE on `paper_trades`, `match_briefs` and `agent_wakeups`, and write access to the APScheduler tables (needed by `odds_mcp.server.refresh_scrape`). When the variable is unset the agent inherits the write-capable DSN and only logs a warning ([Betting agent](../agent/betting-agent.md#database-role)).

### The `--db` flag

`odds scheduler start --db NAME` and `odds agent run --db NAME` swap the dbname in `DATABASE_URL` through `odds_cli.db_override.override_database_url`, but the engine already built in this process keeps the original URL; only subprocesses see the swap ([Configuration](configuration.md#hazards)).

## Copy production data locally

`odds copy from-prod START END` copies events, odds snapshots and predictions for one sport and date range from the production database into the local database, so local work runs on real data. The CLI wrapper is `odds_cli.commands.copy_from_prod`; the logic is in `odds_lambda.copy_from_production` (`odds_lambda.copy_from_production.copy_from_prod`, `odds_lambda.copy_from_production.ProductionDataCopier`).

```bash
odds copy from-prod 2025-08-01 2026-04-10 --sport soccer_epl --dry-run
```

`START` and `END` are positional `YYYY-MM-DD` dates; `END` is inclusive through 23:59:59 UTC. Options (`odds_cli.commands.copy_from_prod.copy_from_production`): `--sport/-s` (default `soccer_epl`), `--dry-run`, `--skip-existing/--overwrite` (default skip existing), `--prod-url`, `--local-url`.

- Source URL: `--prod-url`, else `PROD_DATABASE_URL`, else `DATABASE_URL`. Target URL: `--local-url`, else `LOCAL_DATABASE_URL` (required).
- Events come from `odds_lambda.storage.readers.OddsReader.get_events_by_date_range` for the sport and window. With `--skip-existing`, an event already present locally is skipped.
- A copied event is upserted with `odds_lambda.storage.writers.OddsWriter.upsert_event`; each snapshot not already present locally (within a 1 minute tolerance) is re-stored with `odds_lambda.storage.writers.OddsWriter.store_odds_snapshot`, which recomputes the normalized odds and tier ([Odds snapshots and OddsWriter](../storage/odds-storage.md)).
- Predictions are copied one by one; duplicates against `uq_prediction_event_snap_model` are skipped.
- `--dry-run` counts events and snapshots it would copy and writes nothing; its prediction count stays 0.
- A per-event failure is logged and counted, and the run continues.

## Hazards

- **Production fallback.** With `PROD_DATABASE_URL` unset, `DATABASE_URL` becomes the production source. In the standard local setup that is the local database, so the command copies it into itself while printing "Production". Set `PROD_DATABASE_URL` explicitly. Neither it nor `LOCAL_DATABASE_URL` is read from `.env`: they are `os.getenv` lookups and must be exported in the shell.
- **`.env.example` defaults to dev.** A fresh copy of it writes to the shared dev database, not local.
- Copied predictions keep production's `snapshot_id`, while snapshots are re-created locally with new ids, so a prediction's `snapshot_id` can point at a different local snapshot or fail the foreign key (read from the code, not run).
- A duplicate prediction triggers `local_session.rollback()` on the session that also holds that event's uncommitted upsert and snapshot writes, so those writes can be discarded when `--overwrite` meets existing predictions (read from the code, not run).
- The engine is created on import, so Lambda versus pool choice is fixed at that moment. Replacing the engine later (as `tests/migration/conftest.py` does) does not affect modules that already did `from odds_core.database import async_session_maker`.
- Docs say "pool of 5"; the effective ceiling is `pool_size + 10` per process. No `pool_pre_ping` is set, so stale connections in a long-lived local process are not probed.
- The copier module lives in odds-lambda but is a Rich-console CLI tool, and its `__main__` block ignores most options ([Packages and the odds CLI](packages-and-cli.md#odds-lambda)).

## Related

- [Create and apply a database migration](howto-database-migration.md)
- [Set up, run and test locally](howto-develop-locally.md)
- [OddsReader and the storage layer](../storage/odds-reader-and-storage-layer.md)
- [AWS infrastructure](../operations/aws-infrastructure.md)
