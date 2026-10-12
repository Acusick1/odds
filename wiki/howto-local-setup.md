---
type: howto
title: Set up a local development environment
summary: "uv sync, a working .env, Docker Postgres on port 5433, migrations, and the first commands to run."
sources:
  - migrations/env.py
  - docker-compose.yml
  - packages/odds-core/odds_core/config.py::DataCollectionConfig
  - packages/odds-cli/odds_cli/commands/status.py::show_quota
  - docs/DATABASE.md#local-database-odds
  - docs/DATABASE.md#environment-variables
verified_at: 7887039
---

# Set up a local development environment

Prerequisites: Python 3.12+, [uv](https://github.com/astral-sh/uv), Docker, and a The Odds API key ([README](../README.md#quick-start)).

1. **Install.** `uv sync`. This installs the five workspace packages and the OddsHarvester fork from git ([Code conventions](code-conventions.md)).
2. **Create `.env`.** Start from `.env.example` but do not copy it unchanged ([Configuration](configuration.md#hazards)):
   - `ODDS_API_KEY=<your key>`
   - `DATABASE_URL=${LOCAL_DATABASE_URL}` and `LOCAL_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/odds`. The template's default `DATABASE_URL=${DEV_DATABASE_URL}` points at the shared dev database.
   - Write list values as JSON arrays: `SPORTS=["soccer_epl"]`, and the same for `BOOKMAKERS`, `MARKETS`, `REGIONS` if set.
3. **Start Postgres.** `docker compose up -d postgres` runs `postgres:15` (database `odds`, `postgres`/`postgres`) on host port 5433, chosen to avoid a system Postgres on 5432 (`docker-compose.yml`).
4. **Migrate.** `uv run alembic upgrade head`. `migrations/env.py` overwrites `sqlalchemy.url` with `get_settings().database.url`, so the 5432 URL in `alembic.ini` is never used. Details: [Create and apply a database migration](howto-database-migration.md).
5. **Check.** `uv run odds status show`, then `uv run odds status quota` (free tier: 500 units per month per key, [Odds API quota](odds-api-quota.md)).
6. **First fetch.** `uv run odds fetch current --sport soccer_epl`. Pass `--sport`: the default is `basketball_nba` ([odds CLI](cli-overview.md#hazards)). Each fetch costs units.
7. **Run the scheduler** (optional). `uv run odds scheduler start` ([Local scheduler](local-scheduler.md)).

Never point `DATABASE_URL` at production ([Database connections](database-connections.md)). Tests need none of this database: see [Run the test suite](howto-run-tests.md).

## Hazards

- Any import of `odds_core` builds settings, so a missing `ODDS_API_KEY` or `DATABASE_URL` fails every command, not just those that use the API ([odds-core](package-odds-core.md)).
- The `cp .env.example .env` step in the README fails at import because of the comma-separated list values in the template.
- Variables read with `os.getenv` (`LOCAL_DATABASE_URL`, `PROD_DATABASE_URL`, `AGENT_DATABASE_URL`) must be exported in the shell; `.env` does not supply them to `odds copy from-prod` or the agent.

## Related

- [Project overview](project-overview.md)
- [Copy from production](copy-from-production.md) — seed local data
