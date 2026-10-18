---
type: reference
title: Test suite
description: The test directories, pytest configuration, fixtures (py-pglite), markers, and which suites CI and the deploy workflows run.
sources:
  - pyproject.toml
  - tests/conftest.py
  - tests/migration/conftest.py
  - tests/integration/conftest.py
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
tests/unit/         ~90 flat files across all five packages; `conftest.py` holds the OddsPortal feed fixtures
tests/integration/  shared fixtures in conftest.py (e.g. events with opening and closing snapshots, an `op_live_` event with child rows)
tests/migration/    schema and CLI checks against a real database
tests/fixtures/     sample Odds API JSON
```

## Configuration

- Class-based tests: pytest is configured with `python_classes = Test*`, `asyncio_mode = "auto"`, `testpaths = ["tests"]` and `pythonpath = ["scripts"]` (root `pyproject.toml`).
- Shared fixtures go in `conftest.py`; `tests/unit/conftest.py` provides `oddsportal_fixtures` (the directory of captured OddsPortal pages under `tests/fixtures/oddsportal/`), `encrypt_payload` (encrypts JSON as the feed does, for synthetic responses) and `make_ml_config` (a small `odds_analytics.training.config.MLTrainingConfig` with a tuning section). Other unit tests repeat or import their fixtures.
- Markers: `integration` (live external APIs, may be skipped offline) and `live_api` (third-party HTTP, run only by the scheduled live-API workflow). CI runs integration with `-m "not live_api"` (`.github/workflows/test.yml`).
- Any `odds_core` import builds settings, so tests need a `DATABASE_URL`: `tests/conftest.py` defaults only `ODDS_API_KEY`, and CI supplies `DATABASE_URL` from a secret ([Workspace packages](workspace-packages.md#odds-core)).

## Unit and integration databases

`tests/unit` and `tests/integration` use an in-process PostgreSQL (py-pglite) on TCP port 5434. `tests/conftest.py` forces TCP; the session fixture creates all SQLModel tables once; `pglite_async_session` truncates every table (`RESTART IDENTITY CASCADE`) before each test and yields an `AsyncSession` with `expire_on_commit=False`. Aliases `test_engine` and `test_session` exist for older tests. Run one pytest process at a time, since the port is fixed ([AGENTS.md](../../AGENTS.md#tests)). Fixtures for Odds API payloads live under `tests/fixtures/`.

## Migration tests

`tests/migration` connects to whatever `DATABASE_URL` points at and replaces `odds_core.database.engine` and `odds_core.database.async_session_maker` with a `NullPool` pair in `pytest_configure`. Only the deploy workflows run it: dev runs all of it, prod runs `test_schema_validation.py` and `test_cli_readonly.py` ([CI/CD workflows](../operations/ci-cd-workflows.md)). Because `testpaths` is `tests`, a bare `uv run pytest` also collects it ([Databases and environments](databases-and-environments.md#environments)).

## Related

- [Set up and run locally](howto-develop-locally.md)
- [Code conventions](code-conventions.md)
- [Create and apply a database migration](howto-database-migration.md)
