---
type: howto
title: Set up, run and test locally
description: How to install, write a working .env, start Docker Postgres, migrate, make a first fetch, and run unit and integration tests without touching the dev or production database.
sources:
  - migrations/env.py
  - docker-compose.yml
  - pyproject.toml
  - packages/odds-core/odds_core/config.py::DataCollectionConfig
  - packages/odds-core/odds_core/config.py::SchedulerConfig
  - packages/odds-cli/odds_cli/commands/status.py::show_quota
  - tests/conftest.py::pglite_config
  - tests/conftest.py::pglite_async_session
  - tests/migration/conftest.py::pytest_configure
  - tests/integration/conftest.py
  - .github/workflows/test.yml
  - .github/workflows/deploy-dev.yml
  - .github/workflows/deploy-prod.yml
status: stable
---

# Set up, run and test locally

Prerequisites: Python 3.12+, [uv](https://github.com/astral-sh/uv), Docker, and a The Odds API key ([README](../../README.md#quick-start)).

## Set up

1. **Install.** `uv sync`. This installs the five workspace packages and the OddsHarvester fork from git ([Packages and the odds CLI](packages-and-cli.md)).
2. **Create `.env`.** Start from `.env.example` but do not copy it unchanged ([Configuration](configuration.md#hazards)):
   - `ODDS_API_KEY=<your key>`
   - `DATABASE_URL=${LOCAL_DATABASE_URL}` and `LOCAL_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/odds`. The template's `DATABASE_URL=${DEV_DATABASE_URL}` points at the shared dev database, so a fresh copy writes to dev.
   - Write list values as JSON arrays: `SPORTS=["soccer_epl"]`, and the same for `BOOKMAKERS`, `MARKETS`, `REGIONS` if set. The template's comma-separated lists fail JSON parsing and `SPORTS=basketball_nba` is NBA.
   - Export `LOCAL_DATABASE_URL`, `PROD_DATABASE_URL` and `AGENT_DATABASE_URL` in the shell if you use them: they are read with `os.getenv`, and `.env` is never exported to the process environment.
3. **Start Postgres.** `docker compose up -d postgres` runs `postgres:15` (database `odds`, `postgres`/`postgres`) on host port 5433, chosen to avoid a system Postgres on 5432 (`docker-compose.yml`).
4. **Migrate.** `uv run alembic upgrade head`. `migrations/env.py` overwrites the `alembic.ini` URL with `DATABASE_URL`, so the 5432 URL in the ini is never used. Details: [Create and apply a database migration](howto-database-migration.md).
5. **Check.** `uv run odds status show`, then `uv run odds status quota`. The free tier is 500 units per month per key, and each `fetch current` costs 1 unit per region per market (3 by default) ([Odds API client and quota](../collection/odds-api-client-and-quota.md)).
6. **First fetch.** `uv run odds fetch current --sport soccer_epl`. Pass `--sport`: the default is `basketball_nba` ([odds CLI](packages-and-cli.md#hazards)). Each fetch costs units.
7. **Run the scheduler** (optional). `uv run odds scheduler start` ([Local scheduler](../scheduling/local-scheduler.md)). It bootstraps only the default jobs; `agent-run` and `daily-digest` are opt-in and must be added to `SCHEDULER_BOOTSTRAP_JOBS` (`packages/odds-core/odds_core/config.py::SchedulerConfig`, [Jobs](../scheduling/jobs-and-deployment-matrix.md#where-each-job-runs)).

Never point `DATABASE_URL` at production ([Databases](databases-and-environments.md)). To seed local data from production see [copying production data](databases-and-environments.md#copy-production-data-locally). Tests need none of this database.

## Run the tests

```bash
uv run pytest tests/unit tests/integration
uv run pytest tests/unit/test_paper_trading.py -v
```

Run unit and integration tests together; leave `tests/migration` to the deploy workflows ([AGENTS.md](../../AGENTS.md#tests)).

### What the tests run against

- `tests/unit` and `tests/integration` use an in-process PostgreSQL (py-pglite) on TCP port 5434. `tests/conftest.py::pglite_config` forces TCP; the session fixture creates all SQLModel tables once; `pglite_async_session` truncates every table (`RESTART IDENTITY CASCADE`) before each test and yields an `AsyncSession` with `expire_on_commit=False`. Aliases `test_engine` and `test_session` exist for older tests.
- `tests/migration` connects to whatever `DATABASE_URL` points at and replaces `odds_core.database.engine` with a `NullPool` engine in `pytest_configure`. Only the deploy workflows run it: dev runs all of it, prod runs `test_schema_validation.py` and `test_cli_readonly.py` ([CI/CD workflows](../operations/ci-cd-workflows.md)).
- Markers: `integration` (live external APIs, may be skipped offline) and `live_api` (third-party HTTP, run only by the scheduled live-API workflow). CI runs integration with `-m "not live_api"` (`.github/workflows/test.yml`).

### Layout and conventions

```
tests/unit/         ~90 flat files across all five packages, no conftest of its own
tests/integration/  shared fixtures in conftest.py (e.g. events with opening and closing snapshots)
tests/migration/    schema and CLI checks against a real database
tests/fixtures/     sample Odds API JSON
```

- Class-based tests: pytest is configured with `python_classes = Test*`, `asyncio_mode = "auto"`, `testpaths = ["tests"]` and `pythonpath = ["scripts"]` (root `pyproject.toml`).
- Shared fixtures go in `conftest.py`; unit tests have none, so fixtures are repeated or imported.

## Hazards

- Any import of `odds_core` builds settings, so a missing `ODDS_API_KEY` or `DATABASE_URL` fails every command, not just those that use the API ([Packages and the odds CLI](packages-and-cli.md#odds-core)). Tests too: `tests/conftest.py` defaults only `ODDS_API_KEY`; CI supplies `DATABASE_URL` from a secret.
- `cp .env.example .env` fails at import because of the comma-separated list values in the template.
- `uv run pytest` with no arguments collects `tests/migration` too, because `testpaths` is `tests`. With a production-looking `DATABASE_URL` that includes write tests against production (`pyproject.toml`, `tests/migration/conftest.py::pytest_configure`).
- `.github/workflows/test.yml` sets up Python 3.11 while packages require `>=3.12`.

## Related

- [Databases, environments and copying production data](databases-and-environments.md)
- [Conventions](conventions.md)
- [Project overview and pivots](project-overview.md)
