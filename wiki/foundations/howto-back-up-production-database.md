---
type: howto
title: Back up and restore the production database
description: How to dump the production database before a migration or any other write to it, where the dumps go, and how to restore one.
sources:
  - scripts/backup-prod-db.sh
  - deploy.sh
  - docker-compose.yml
status: draft
---

# Back up and restore the production database

Production is the Compose Postgres on the production host ([Databases and environments](databases-and-environments.md#environments)), and no scheduled backup of it exists. Its data cannot be recreated from the repo, so every write that is not routine collection is preceded by a dump. [deploy.sh](../scheduling/howto-run-scheduler-service.md#redeploy) takes one itself before a pending migration; anything else is dumped by hand.

## Back up

1. **Run the script on the production host before the write.** From the repo root: `./scripts/backup-prod-db.sh <label>`, with a label naming what is about to write, such as `pre-712` for a PR's data fix.
   - It runs `pg_dump -Fc` inside the compose `postgres` service and writes `odds-<label>-<UTC timestamp>.dump` to `$ODDS_BACKUP_DIR`, default `~/backups` (`scripts/backup-prod-db.sh`).
   - It reads the archive back with `pg_restore --list` before giving it its final name, so a listed `.dump` is readable; a failed run leaves no file and exits non-zero.
   - It takes about 30 s and the dump is about 100 MB at the time of writing; nothing prunes old dumps, so delete them by hand.
2. **Check the printed path,** then make the write.

`deploy.sh` runs the script with the label `pre-deploy-<short sha of the new HEAD>` after stopping the scheduler and before `alembic upgrade head`, so the dump is the exact state the migration starts from. A failed dump aborts the deploy before the migration, with the scheduler stopped. A deploy with no pending migration takes no dump.

## Restore

The dump holds the schema of the code that was running when it was taken: restore it under that code, not the new code.

1. **Stop everything that writes.** `sudo systemctl stop odds-scheduler`, and end any agent session, whose MCP server holds connections to the database.
2. **Restore over the database.** From the repo root:

   ```
   docker compose exec -T postgres pg_restore -U postgres -d odds --clean --if-exists --single-transaction < ~/backups/<file>.dump
   ```

   `--clean --if-exists` drops each object before recreating it, and `--single-transaction` leaves the database untouched if the restore fails.

   > [!WARNING]
   > This replaces the production data with the dump. Everything written since the dump is lost, so take a fresh dump first if any of it matters.
3. **Check out the matching code.** For a `pre-deploy-*` dump that is the commit before the deploy, which a failed `deploy.sh` prints: `git checkout <previous head> && uv sync`. Check `uv run alembic current` names the revision the dump was taken at.
4. **Start the scheduler.** `sudo systemctl start odds-scheduler`, then `uv run odds scheduler list-jobs`.

## Related

- [Create and apply a database migration](howto-database-migration.md)
- [Run the scheduler as a systemd service](../scheduling/howto-run-scheduler-service.md)
