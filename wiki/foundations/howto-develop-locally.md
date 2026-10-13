---
type: howto
title: Set up and run locally
description: Install, write a working .env, start Postgres, migrate, make a first fetch, run the offline tests, and add a setting.
sources:
  - migrations/env.py
  - docker-compose.yml
  - pyproject.toml
  - odds_core.config.DataCollectionConfig
  - odds_core.config.SchedulerConfig
  - odds_cli.commands.status.show_quota
  - tests/conftest.py
  - deployment/aws/terraform/lambda.tf
  - deployment/aws/terraform/scraper.tf
  - deployment/aws/terraform/variables.tf
status: stable
---

# Set up and run locally

Prerequisites: Python 3.12+, [uv](https://github.com/astral-sh/uv), Docker, and a The Odds API key ([README](../../README.md#quick-start)).

## Set up

1. **Install.** `uv sync`. This installs the five workspace packages and the OddsHarvester fork from git ([Workspace packages](workspace-packages.md)).
2. **Create `.env`.** Start from `.env.example` but do not copy it unchanged: `odds_core.config.get_settings` raises on its comma-separated lists ([List fields and the template](configuration.md#list-fields-and-the-template)):
   - `ODDS_API_KEY=<your key>`
   - `DATABASE_URL=${LOCAL_DATABASE_URL}` and `LOCAL_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/odds`. The template's `DATABASE_URL=${DEV_DATABASE_URL}` points at the shared dev database, so a fresh copy writes to dev (#429).
   - Write list values as JSON arrays: `SPORTS=["soccer_epl"]`, and the same for `BOOKMAKERS`, `MARKETS`, `REGIONS` if set. `SPORTS=basketball_nba` in the template is NBA.
   - Export `LOCAL_DATABASE_URL`, `PROD_DATABASE_URL` and `AGENT_DATABASE_URL` in the shell if you use them: they are read with `os.getenv`, and `.env` is never exported to the process environment ([variables read outside Settings](configuration.md#variables-read-outside-settings)).
3. **Start Postgres.** `docker compose up -d postgres` runs `postgres:15` (database `odds`, `postgres`/`postgres`) on host port 5433, chosen to avoid a system Postgres on 5432 (`docker-compose.yml`).
4. **Migrate.** `uv run alembic upgrade head`. `migrations/env.py` overwrites the `alembic.ini` URL with `DATABASE_URL`, so the 5432 URL in the ini is never used. Details: [Create and apply a database migration](howto-database-migration.md).
5. **Check.** `uv run odds status show`, then `uv run odds status quota`. The free tier is 500 units per month per key, and each `fetch current` costs 1 unit per region per market (3 by default) ([Odds API client](../collection/odds-api-client.md)).
6. **First fetch.** `uv run odds fetch current --sport soccer_epl`. Pass `--sport`: the default is `basketball_nba` ([Sport defaults](odds-cli.md#sport-defaults)). Each fetch costs units.
7. **Run the scheduler** (optional). `uv run odds scheduler start` ([Local scheduler](../scheduling/local-scheduler.md)). It bootstraps only the default jobs; `agent-run` and `daily-digest` are opt-in and must be added to `SCHEDULER_BOOTSTRAP_JOBS` (`odds_core.config.SchedulerConfig`, [Where each job runs](../scheduling/job-deployment-matrix.md#where-each-job-runs)).

Any import of `odds_core` builds settings, so a missing `ODDS_API_KEY` or `DATABASE_URL` fails every command ([Workspace packages](workspace-packages.md#odds-core)). Never point `DATABASE_URL` at production ([Databases and environments](databases-and-environments.md#environments)). To seed local data from production see [Copy production data locally](howto-copy-production-data.md). Tests need none of this database.

## Run the tests

> [!WARNING]
> `uv run pytest` with no path also runs `tests/migration`, whose write tests use whatever `DATABASE_URL` points at, production included (#429). Pass `tests/unit tests/integration`.

```bash
uv run pytest tests/unit tests/integration
uv run pytest tests/unit/test_paper_trading.py -v
```

Run unit and integration tests together; leave `tests/migration` to the deploy workflows ([AGENTS.md](../../AGENTS.md#tests)). What the suites run against, the markers and the layout are in [Test suite](test-suite.md).

## Add a setting

1. **Add the field** to the right class in `odds_core.config` with a type and a default, so existing environments keep working `(inferred)`. Deprecate a field rather than removing it `(inferred)`.
2. **Add it to `.env.example`.** Write list fields as JSON arrays; the template's other lists are the pattern to avoid.
3. **Wire Terraform if a Lambda needs it.** Declare a variable in `deployment/aws/terraform/variables.tf` and pass it in the `environment` block of `deployment/aws/terraform/lambda.tf` (scheduler) or `deployment/aws/terraform/scraper.tf` (scraper). Each block sets its own subset of variables, so a setting not listed there runs on its code default ([AWS infrastructure](../operations/aws-infrastructure.md)).
4. **Read it through Settings.** A variable read with `os.getenv` bypasses `.env` ([Configuration](configuration.md#variables-read-outside-settings)).

## Related

- [Databases and environments](databases-and-environments.md)
- [Code conventions](code-conventions.md)
- [About the system and its pivots](project-overview.md)
