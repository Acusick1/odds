---
type: howto
title: Run the test suite
summary: "Unit and integration tests on in-process py-pglite, migration tests left to the deploy workflows, and the test layout and conventions."
sources:
  - tests/conftest.py::pglite_config
  - tests/conftest.py::pglite_async_session
  - tests/migration/conftest.py::pytest_configure
  - tests/integration/conftest.py
  - .github/workflows/test.yml
  - .github/workflows/deploy-dev.yml
  - .github/workflows/deploy-prod.yml
  - docs/DEVELOPMENT.md#testing
  - docs/DATABASE.md#test-database
verified_at: 7887039
---

# Run the test suite

```bash
uv run pytest tests/unit tests/integration
uv run pytest tests/unit/test_paper_trading.py -v
```

Run unit and integration tests together; leave `tests/migration` to the deploy workflows ([AGENTS.md](../AGENTS.md#key-commands)).

## What the tests run against

- `tests/unit` and `tests/integration` use an in-process PostgreSQL (py-pglite) on TCP port 5434. `tests/conftest.py::pglite_config` forces TCP; the session fixture creates all SQLModel tables once; `pglite_async_session` truncates every table (`RESTART IDENTITY CASCADE`) before each test and yields an `AsyncSession` with `expire_on_commit=False`. Aliases `test_engine` and `test_session` exist for older tests ([Database connections](database-connections.md)).
- `tests/migration` connects to whatever `DATABASE_URL` points at and replaces `odds_core.database.engine` with a `NullPool` engine in `pytest_configure`. Only the deploy workflows run it: dev runs all of it, prod runs `test_schema_validation.py` and `test_cli_readonly.py` ([CI/CD workflows](ci-cd-workflows.md)).
- Markers: `integration` (live external APIs, may be skipped offline) and `live_api` (third-party HTTP, run only by the scheduled live-API workflow). CI runs integration with `-m "not live_api"` (`.github/workflows/test.yml`).

## Layout and conventions

```
tests/unit/         ~90 flat files across all five packages, no conftest of its own
tests/integration/  shared fixtures in conftest.py (e.g. events with opening and closing snapshots)
tests/migration/    schema and CLI checks against a real database
tests/fixtures/     sample Odds API JSON
```

- Class-based tests: pytest is configured with `python_classes = Test*`, `asyncio_mode = "auto"`, `testpaths = ["tests"]` and `pythonpath = ["scripts"]` (root `pyproject.toml`).
- Shared fixtures go in `conftest.py`; unit tests have none, so fixtures are repeated or imported.
- `uv run pytest` with no arguments collects `tests/migration` too, because `testpaths` is `tests`.

## Hazards

- `docs/DEVELOPMENT.md` lists `uv run pytest` as "All tests". With a production-looking `DATABASE_URL` that includes the migration tests ([Database connections](database-connections.md)).
- Importing `odds_core` builds settings, so `DATABASE_URL` must still resolve to some value even though py-pglite serves the data. `tests/conftest.py` defaults only `ODDS_API_KEY`; CI supplies `DATABASE_URL` from a secret ([Configuration](configuration.md#hazards)).
- `.github/workflows/test.yml` sets up Python 3.11 while packages require `>=3.12`.

## Related

- [Set up a local development environment](howto-local-setup.md)
- [Code conventions](code-conventions.md)
