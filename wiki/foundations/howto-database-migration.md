---
type: howto
title: Create and apply a database migration
description: How to change a model, autogenerate and review an Alembic migration, and promote it local, then CI dev, then production without breaking the running release.
sources:
  - migrations/env.py
  - tests/migration/conftest.py
  - .github/workflows/deploy-dev.yml
  - .github/workflows/deploy-prod.yml
  - odds-scheduler.service
status: stable
---

# Create and apply a database migration

Every change to a model under `packages/odds-core/odds_core/` needs an Alembic migration. Migrations live in `migrations/versions/`; `migrations/env.py` is the async environment and `alembic.ini` points `script_location` at `migrations`. The order is local, then CI dev, then prod; do not skip a stage.

## Steps

1. **Point at the local database.** Set `DATABASE_URL=${LOCAL_DATABASE_URL}` and start it with `docker compose up -d postgres`. `.env.example` sets `DATABASE_URL=${DEV_DATABASE_URL}`, the dev database, so check this before anything else ([Databases and environments](databases-and-environments.md#environments)). `migrations/env.py` overrides the `alembic.ini` URL with `get_settings().database.url`, so whatever `DATABASE_URL` holds is the migration target. `alembic.ini` holds a placeholder `sqlalchemy.url` (`localhost:5432`, not the compose port 5433); `env.py` overrides it, so editing the ini has no effect. Never run migrations against production ([Databases and environments](databases-and-environments.md#environments)).
2. **Register new model modules.** `migrations/env.py` imports each model module explicitly so `SQLModel.metadata` sees it. A new table in a new module must be added there, or autogenerate emits nothing for it. A table kept with no model (`legacy_event_ids`, [Events table](../storage/events.md#legacy-string-ids)) is listed in `UNMODELLED_TABLES` in the same file, so autogenerate does not drop it.
3. **Generate.** `uv run alembic revision --autogenerate -m "description"`.
4. **Review the generated file.** Autogenerate misses renames and custom types (Alembic's autogenerate limits; see the Alembic documentation). Data fixes are hand-written migrations in the same directory. Keep migrations backward-compatible with the previous release's code (additive columns, no drops in the same release) `(inferred)`, because the deploy workflows run `terraform apply` before `alembic upgrade head`, so new code can briefly run against the old schema (#443).

   > [!WARNING]
   > A migration that drops or rewrites data runs unchanged on production when [deploy.sh](../scheduling/howto-run-scheduler-service.md#redeploy) applies it. Production data cannot be recreated from the repo. deploy.sh dumps the database before migrating, and restoring that dump is the only way back ([Back up and restore](howto-back-up-production-database.md)).
5. **Apply locally.** `uv run alembic upgrade head`. Roll back one with `uv run alembic downgrade -1`; inspect with `alembic history` and `alembic current`.
6. **Update affected queries** in the storage layer ([Storage layer](../storage/storage-layer.md)) and run the tests ([Run the tests](howto-develop-locally.md#run-the-tests)).
7. **Run migration tests locally.** `uv run pytest tests/migration/ -v`. These connect to `DATABASE_URL` and include write tests, so only do this against local ([Test suite](test-suite.md#migration-tests)).
8. **Merge to main.** CI runs tests, then `.github/workflows/deploy-dev.yml` applies `alembic upgrade head` to the dev database and runs all of `tests/migration`. See [CI/CD workflows](../operations/ci-cd-workflows.md). The workflow builds the Lambda image and applies Terraform ([AWS infrastructure](../operations/aws-infrastructure.md)) before the migration.
9. **Production.** Run [./deploy.sh](../scheduling/howto-run-scheduler-service.md#redeploy) on the production host once the change is on main. When the database is not at head it stops the scheduler first, migrates, smoke-tests and restarts it, so no migration runs under the old code; a failure after the stop leaves the scheduler down and says so. The hand-run agent's MCP server is not managed by the script: do not deploy during an agent session. After a migration that rewrites a large table, run `VACUUM ANALYZE` on it. Do not run migrations outside deploy.sh, or `tests/migration` at all, against production. `.github/workflows/deploy-prod.yml` migrates the Neon archive, not production, and must not be run, including through a published release ([Where each job runs](../scheduling/job-deployment-matrix.md#dormant-aws-rules)).

## Related

- [Databases and environments](databases-and-environments.md)
- [Events table](../storage/events.md) and [Odds snapshots table](../storage/odds-tables.md) — examples of what models map to
