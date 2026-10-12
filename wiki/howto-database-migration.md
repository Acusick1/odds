---
type: howto
title: Create and apply a database migration
summary: Change a model, autogenerate and review an Alembic migration, then apply it local, then CI dev, then prod, following the safety rules.
sources:
  - migrations/env.py
  - tests/migration/conftest.py
  - .github/workflows/deploy-dev.yml
  - .github/workflows/deploy-prod.yml
  - docs/DATABASE.md#database-migrations
  - docs/DATABASE.md#migration-testing-workflow
  - docs/DATABASE.md#safety-rules
  - docs/DEVELOPMENT.md#modifying-database-schema
verified_at: 7887039
---

# Create and apply a database migration

Every change to a model under `packages/odds-core/odds_core/` needs an Alembic migration. Migrations live in `migrations/versions/`; `migrations/env.py` is the async environment and `alembic.ini` points `script_location` at `migrations`.

## Steps

1. **Point at the local database.** Set `DATABASE_URL=${LOCAL_DATABASE_URL}` and start it with `docker compose up -d postgres` ([Database connections](database-connections.md)). `migrations/env.py` overrides the `alembic.ini` URL with `get_settings().database.url`, so whatever `DATABASE_URL` holds is the target. Never run this against production.
2. **Register new model modules.** `migrations/env.py` imports each model module explicitly so `SQLModel.metadata` sees it. A new table in a new module must be added there or autogenerate emits nothing for it.
3. **Generate.** `uv run alembic revision --autogenerate -m "description"`.
4. **Review the generated file.** Autogenerate misses renames and custom types (per the docs). Data fixes are hand-written migrations in the same directory.
5. **Apply locally.** `uv run alembic upgrade head`. Roll back one with `uv run alembic downgrade -1`; inspect with `alembic history` and `alembic current`.
6. **Update affected queries** in the storage layer ([Storage layer](storage-layer.md)) and run the tests ([Running tests](howto-run-tests.md)).
7. **Run migration tests locally.** `uv run pytest tests/migration/ -v`. These connect to `DATABASE_URL` and include write tests, so only do this against local.
8. **Merge to `main`.** CI runs tests, then `.github/workflows/deploy-dev.yml` applies `alembic upgrade head` to the dev database and runs all of `tests/migration`. See [CI/CD workflows](ci-cd-workflows.md).
9. **Production.** `.github/workflows/deploy-prod.yml` is a manual dispatch or a published release, gated by the `production` environment approval. It runs `alembic upgrade head`, then only `test_schema_validation.py` and `test_cli_readonly.py`. Do not run migrations against production by hand.

## Order

local, then CI dev, then prod. Do not skip a stage.

## Hazards

- In both deploy workflows the Terraform apply (new Lambda image) runs before `alembic upgrade head`. New code can be live briefly against the old schema, so make migrations backward-compatible with the previous release's code (additive columns, no drops in the same release).
- The local database can be reset; dev is ephemeral; production is not. A migration that is destructive on local is destructive on prod.
- `alembic.ini` holds a placeholder `sqlalchemy.url` (`localhost:5432`, not the compose port 5433); `env.py` overrides it, so editing the ini has no effect.
- Whether the production approval gate is configured in GitHub cannot be verified from the repo.

## Related

- [Database connections](database-connections.md)
- [CI/CD workflows](ci-cd-workflows.md)
- [Events table](table-events.md) and [Odds snapshots table](table-odds-snapshots.md) (examples of what models map to)
