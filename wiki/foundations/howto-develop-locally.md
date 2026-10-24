---
type: howto
title: Set up and run locally
description: Install, write a working .env, start Postgres, migrate, make a first fetch, enable the pre-commit hooks and issue claiming, and run the offline tests.
sources:
  - migrations/env.py
  - docker-compose.yml
  - pyproject.toml
  - odds_core.config.DataCollectionConfig
  - odds_core.config.SchedulerConfig.bootstrap_jobs
  - odds_cli.commands.status.show_quota
  - tests/conftest.py
  - .pre-commit-config.yaml
  - .githooks/claim-issue.sh
status: stable
---

# Set up and run locally

Prerequisites: Python 3.12+, [uv](https://github.com/astral-sh/uv), Docker, and a The Odds API key ([README](../../README.md#quick-start)).

## Set up

1. **Install.** `uv sync`. This installs the five workspace packages ([Workspace packages](workspace-packages.md)).
2. **Create `.env`.** Start from `.env.example`, but do not copy it unchanged: `odds_core.config.get_settings` raises on its comma-separated lists ([List fields and the template](configuration.md#list-fields-and-the-template)):
   - `ODDS_API_KEY=<your key>`
   - `DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/odds`, written literally. `${LOCAL_DATABASE_URL}` would expand to an empty string, because the template defines `LOCAL_DATABASE_URL` below `DATABASE_URL`. The template's `DATABASE_URL=${DEV_DATABASE_URL}` points at the shared dev database, so a fresh copy writes to dev (#429).
   - Write list values as JSON arrays: `SPORTS=["soccer_epl"]`, and the same for `MARKETS`, `REGIONS` if set. Leave `BOOKMAKERS` unset unless you need a JSON object of sport to bookmakers ([List fields](configuration.md#list-fields-and-the-template)). `SPORTS=basketball_nba` in the template is NBA.
   - Leave `ALERT_ENABLED` false locally; alerts post to Discord only when it is true ([Alerts](../operations/alerts.md#delivery)).
   - Export `LOCAL_DATABASE_URL`, `PROD_DATABASE_URL` and `AGENT_DATABASE_URL` in the shell if you use them: they are read with `os.getenv`, and `.env` is never exported to the process environment ([variables read outside Settings](configuration.md#variables-read-outside-settings)).
3. **Start Postgres.** `docker compose up -d postgres` runs `postgres:15` (database `odds`, `postgres`/`postgres`) on host port 5433, chosen to avoid a system Postgres on 5432 (`docker-compose.yml`).
4. **Migrate.** `uv run alembic upgrade head`. `migrations/env.py` overwrites the `alembic.ini` URL with `DATABASE_URL`, so the 5432 URL in the ini is never used. Details: [Create and apply a database migration](howto-database-migration.md).
5. **Check.** `uv run odds status show`, then `uv run odds status quota`. The free tier is 500 units per month per key, and each `fetch current` costs 1 unit per region per market (3 by default) ([Odds API client](../collection/odds-api-client.md)).
6. **First fetch.** `uv run odds fetch current --sport soccer_epl`. Pass `--sport`: the default is `basketball_nba` ([Sport defaults](odds-cli.md#sport-defaults)). Each fetch costs units.
7. **Enable the hooks** (before committing). `uv run pre-commit install` runs the lint hooks on every commit; `uv run pre-commit run --all-files` runs them everywhere ([Code conventions](code-conventions.md#mechanical-gates)).
8. **Enable issue claiming** (before creating an issue branch). `ln -s ../../.githooks/claim-issue.sh "$(git rev-parse --git-common-dir)/hooks/post-checkout"` installs the `claim-issue` git hook, which assigns issue `<n>` to you when you create a branch named `<n>-<slug>` or `issue-<n>-<slug>` and never blocks a checkout (`.githooks/claim-issue.sh`). Install it once per clone; worktrees share it and run the main checkout's copy of the script.
9. **Run the scheduler** (optional). Set `SCHEDULER_BOOTSTRAP_JOBS` to the jobs you need, then `uv run odds scheduler start` ([Local scheduler](../scheduling/local-scheduler.md)). The default in `odds_core.config.SchedulerConfig.bootstrap_jobs` is the production job set, including `fetch-odds` and `fetch-scores`, which spend Odds API quota (`fetch-polymarket-odds` spends none; [Where each job runs](../scheduling/job-deployment-matrix.md#where-each-job-runs)).

Any import of `odds_core` builds settings, so a missing `ODDS_API_KEY` or `DATABASE_URL` fails every command ([Workspace packages](workspace-packages.md#odds-core)). To seed local data from production see [Copy production data locally](howto-copy-production-data.md). Tests need none of this database.

> [!WARNING]
> Never point `DATABASE_URL` at production from a development machine. Production is a single host's local database, migrated only by [deploy.sh](../scheduling/howto-run-scheduler-service.md#redeploy) on that host ([Databases and environments](databases-and-environments.md#environments)).

## Run the tests

> [!WARNING]
> `uv run pytest` with no path also runs `tests/migration`, whose write tests hit whatever `DATABASE_URL` points at (#429). Pass `tests/unit tests/integration`.

```bash
uv run pytest tests/unit tests/integration
uv run pytest tests/unit/test_paper_trading.py -v
```

Run unit and integration tests together; leave `tests/migration` to the deploy workflows ([AGENTS.md](../../AGENTS.md#tests)). What the suites run against, the shared fixtures in `tests/conftest.py`, the markers and the layout are in [Test suite](test-suite.md). A webhook exported in your shell or `.env` is safe: tests never post to it.

## Related

- [Databases and environments](databases-and-environments.md)
- [Add a setting](howto-add-setting.md)
- [Code conventions](code-conventions.md)
- [About the system and its pivots](project-overview.md)
