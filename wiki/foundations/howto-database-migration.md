---
type: howto
title: Create and apply a database migration
description: How to change a model, autogenerate and review an Alembic migration, and promote it local, then CI dev, then prod without breaking the running release.
sources:
  - migrations/env.py
  - tests/migration/conftest.py
  - .github/workflows/deploy-dev.yml
  - .github/workflows/deploy-prod.yml
  - deploy.sh
  - odds-scheduler.service
status: stable
---

# Create and apply a database migration

Every change to a model under `packages/odds-core/odds_core/` needs an Alembic migration. Migrations live in `migrations/versions/`; `migrations/env.py` is the async environment and `alembic.ini` points `script_location` at `migrations`. The order is local, then CI dev, then prod; do not skip a stage.

## Steps

1. **Point at the local database.** Set `DATABASE_URL=${LOCAL_DATABASE_URL}` and start it with `docker compose up -d postgres`. `.env.example` defaults `DATABASE_URL` to the dev database, so check this before anything else. `migrations/env.py` overrides the `alembic.ini` URL with `get_settings().database.url`, so whatever `DATABASE_URL` holds is the target. `alembic.ini` holds a placeholder `sqlalchemy.url` (`localhost:5432`, not the compose port 5433); `env.py` overrides it, so editing the ini has no effect. Never run migrations against production ([Databases and environments](databases-and-environments.md#environments)).
2. **Register new model modules.** `migrations/env.py` imports each model module explicitly so `SQLModel.metadata` sees it. A new table in a new module must be added there or autogenerate emits nothing for it.
3. **Generate.** `uv run alembic revision --autogenerate -m "description"`.
4. **Review the generated file.** Autogenerate misses renames and custom types (Alembic's autogenerate limits; see the Alembic documentation). Data fixes are hand-written migrations in the same directory. Keep migrations backward-compatible with the previous release's code (additive columns, no drops in the same release) `(inferred)`, because the deploy workflows run `terraform apply` before `alembic upgrade head`, so new code can briefly run against the old schema (#443).

   > [!WARNING]
   > A migration that drops or rewrites data runs unchanged on production when `deploy-prod.yml` applies it. Production data cannot be recreated from the repo.
5. **Apply locally.** `uv run alembic upgrade head`. Roll back one with `uv run alembic downgrade -1`; inspect with `alembic history` and `alembic current`.
6. **Update affected queries** in the storage layer ([Storage layer](../storage/storage-layer.md)) and run the tests ([Run the tests](howto-develop-locally.md#run-the-tests)).
7. **Run migration tests locally.** `uv run pytest tests/migration/ -v`. These connect to `DATABASE_URL` and include write tests, so only do this against local ([Test suite](test-suite.md#migration-tests)).
8. **Merge to main.** CI runs tests, then `.github/workflows/deploy-dev.yml` applies `alembic upgrade head` to the dev database and runs all of `tests/migration`. See [CI/CD workflows](../operations/ci-cd-workflows.md). The Terraform apply in the same workflow ([AWS infrastructure](../operations/aws-infrastructure.md)) runs before the migration.
9. **Production.** `.github/workflows/deploy-prod.yml` is a manual dispatch or a published release, gated by the `production` environment approval; whether that gate is configured is GitHub configuration, so check Settings, Environments. It runs `alembic upgrade head`, then only `test_schema_validation.py` and `test_cli_readonly.py`. Production migrates only inside this workflow; do not run migrations or `tests/migration` against it by hand.

## Always-on host

On an always-on scheduler host, `deploy.sh` migrates before it restarts the scheduler, so the old scheduler briefly sees the new schema. That is safe for additive migrations. For a destructive one, stop the service first (`sudo systemctl stop odds-scheduler`), migrate, then redeploy ([Redeploy](../scheduling/howto-run-scheduler-service.md#redeploy)).

## Related

- [Databases and environments](databases-and-environments.md)
- [Events table](../storage/events.md) and [Odds snapshot and odds tables](../storage/odds-tables.md) — examples of what models map to
