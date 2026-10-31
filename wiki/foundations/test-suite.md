---
type: reference
title: Test suite
description: The test directories, pytest configuration, fixtures (py-pglite), markers, and which suites CI and the deploy workflows run.
sources:
  - pyproject.toml
  - tests/conftest.py
  - tests/migration/conftest.py
  - tests/integration/conftest.py
  - tests/test_helpers.py
  - .github/workflows/test.yml
  - .github/workflows/deploy-dev.yml
  - .github/workflows/deploy-prod.yml
  - .github/workflows/live-api.yml
status: stable
---

# Test suite

How to run the tests is in [Set up and run locally](howto-develop-locally.md#run-the-tests).

## Layout

```
tests/unit/         ~110 flat files across all five packages; `conftest.py` holds the OddsPortal feed fixtures
tests/integration/  shared fixtures in conftest.py (e.g. an event with child rows (a snapshot with price rows, a paper trade and a match brief), an EPL event with a Betfair `1x2` snapshot without prices, `mixed_source_event` (an EPL event kicking off 2026-04-18 with an Odds API snapshot 3 h out and a later Polymarket one carrying mid, ask and bid price rows), and `job_session`, the test session handed to the OddsPortal and Betfair jobs as their session factory)
tests/migration/    schema and CLI checks against a real database
tests/fixtures/     sample Odds API JSON
```

## Configuration

- Class-based tests: pytest is configured with `python_classes = Test*`, `asyncio_mode = "auto"`, `testpaths = ["tests"]` and `pythonpath = ["scripts"]` (root `pyproject.toml`).
- Shared fixtures go in `conftest.py`; `tests/unit/conftest.py` provides `oddsportal_fixtures` (the directory of captured OddsPortal pages under `tests/fixtures/oddsportal/`), `encrypt_payload` (encrypts JSON as the feed does, for synthetic responses) and `make_ml_config` (a small `odds_analytics.training.config.MLTrainingConfig` with a delta target; it carries a tuning section unless called with `tuning=False`). Other unit tests repeat or import their fixtures.
- `tests/test_helpers.py` builds tagged snapshots: `make_tagged_snapshot` returns a `TaggedSnapshot` dataclass (an envelope's bookmakers, a source tag and, for FDUK, a timing basis), `make_quotes` gives a unit test the `odds_core.quotes.Quote` rows the writers would store for one, and `store_tagged` stores it with its price rows through `odds_lambda.storage.shaped_snapshots.store_shaped_snapshot` in an integration test, returning the stored snapshot of its first market. It also holds the gamePk helpers: `with_game_pk` marks an event to hold a gamePk, `hold_game_pks` records it as the event's `mlb_stats_api` external id, and `game_pk_of` reads it back through `odds_lambda.storage.game_pks`. A snapshot added to the session alone has no price rows, so the readers of `odds_core.quotes.quotes` see nothing.
- Markers: `integration` (live external APIs, may be skipped offline) and `live_api` (third-party HTTP, run only by the scheduled live-API workflow). CI runs integration with `-m "not live_api"` and, before the tests, a `deptry` check per package for undeclared imports (`.github/workflows/test.yml`, [Dependency checks](code-conventions.md#dependency-checks)).
- Any `odds_core` import builds settings (`tests/conftest.py`'s `mock_settings` fixture builds its own, with per-sport bookmakers for NBA only), so tests need a `DATABASE_URL`: `tests/conftest.py` defaults only `ODDS_API_KEY`, and CI supplies `DATABASE_URL` from a secret ([Workspace packages](workspace-packages.md#odds-core)). It overwrites `DISCORD_WEBHOOK_URL` with an empty value and `ALERT_ENABLED` with `false`, so a webhook in the developer's shell or `.env` never receives test alerts.

## Unit and integration databases

`tests/unit` and `tests/integration` use an in-process PostgreSQL (py-pglite) on TCP port 5434. `tests/conftest.py` forces TCP; the session fixture creates all SQLModel tables once; `pglite_async_session` truncates every table (`RESTART IDENTITY CASCADE`) before each test and restarts the events identity at 1,000,001, so a test can give an event a small literal id, and yields an `AsyncSession` with `expire_on_commit=False`. Aliases `test_engine` and `test_session` exist for older tests, and `gamma_event` and `gamma_nba_event` build a Gamma EPL and NBA match event for Polymarket tests. Run one pytest process at a time, since the port is fixed ([AGENTS.md](../../AGENTS.md#tests)). Fixtures for Odds API payloads live under `tests/fixtures/`.

## Migration tests

`tests/migration` connects to whatever `DATABASE_URL` points at and replaces `odds_core.database.engine` and `odds_core.database.async_session_maker` with a `NullPool` pair in `pytest_configure`. Only the deploy workflows run it, each after `terraform apply`: dev runs all of it, prod runs `test_schema_validation.py` and `test_cli_readonly.py` ([CI/CD workflows](../operations/ci-cd-workflows.md)). Because `testpaths` is `tests`, a bare `uv run pytest` also collects it ([Databases and environments](databases-and-environments.md#environments)).

## Related

- [Set up and run locally](howto-develop-locally.md)
- [Code conventions](code-conventions.md)
- [Create and apply a database migration](howto-database-migration.md)
