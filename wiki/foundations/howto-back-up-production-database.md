---
type: howto
title: Back up and restore the production database
description: How to set up the nightly off-host copy, dump the production database before a write, check that a dump restores, and restore one from the host or the off-host copy.
sources:
  - scripts/backup-prod-db.sh
  - scripts/restore-check.sh
  - deploy.sh
  - docker-compose.yml
mentions:
  - odds_core.config.BackupConfig
status: draft
---

# Back up and restore the production database

Production is the Compose Postgres on the production host ([Databases and environments](databases-and-environments.md#environments)). Its data cannot be recreated from the repo. The [nightly backup job](database-backup-job.md) dumps it every night and copies the dump to an object store off the host, once its remote is set up. Every write that is not routine collection is also preceded by a dump on the host: [deploy.sh](../scheduling/howto-run-scheduler-service.md#redeploy) takes one itself before a pending migration, and anything else is dumped by hand.

## Set up the off-host copy

Once, on the production host. Until these steps are done the nightly job does nothing.

1. **Create the bucket and a token.** In Cloudflare R2 or Backblaze B2, create a private bucket and an API token with read and write access to that bucket only.
2. **Install rclone** where the scheduler service finds it: `sudo -v ; curl https://rclone.org/install.sh | sudo bash` installs `/usr/bin/rclone` ([rclone install](https://rclone.org/install/)).
3. **Add the remote** as the service's user, `rclone config`, with the provider's S3 settings ([R2](https://rclone.org/s3/#cloudflare-r2), [B2](https://rclone.org/s3/#backblaze-b2)). The credentials stay in `~/.config/rclone/rclone.conf`. Check it with `rclone lsf <remote>:<bucket>`.
4. **Point the job at it.** Add `BACKUP_REMOTE=<remote>:<bucket>` to `.env` ([add a setting](howto-add-setting.md)), then `sudo systemctl restart odds-scheduler`.
5. **Run it once.** `uv run odds scheduler run-once backup-database`, then `rclone lsf <remote>:<bucket>` lists the `odds-nightly-*.dump` it uploaded.

## Back up

1. **Run the script on the production host before the write.** From the repo root: `./scripts/backup-prod-db.sh <label>`, with a label naming what is about to write, such as `pre-712` for a PR's data fix.
   - It runs `pg_dump -Fc` inside the compose `postgres` service and writes `odds-<label>-<UTC timestamp>.dump` to `$ODDS_BACKUP_DIR`, default `~/backups` (`scripts/backup-prod-db.sh`).
   - It reads the archive back with `pg_restore --list` before giving it its final name, so a listed `.dump` is readable; a failed run leaves no file and exits non-zero.
   - It takes about 30 s and the dump is about 100 MB at the time of writing. The nightly job prunes only its own `nightly` dumps, so delete labelled dumps by hand.
2. **Check the printed path,** then make the write.

`deploy.sh` runs the script with the label `pre-deploy-<short sha of the new HEAD>` after stopping the scheduler and the web UI and before `alembic upgrade head`, so the dump is the exact state the migration starts from. A failed dump aborts the deploy before the migration, with the scheduler and the web UI stopped. A deploy with no pending migration takes no dump.

## Check a dump restores

`./scripts/restore-check.sh <file>.dump` restores the dump into a throwaway Postgres container of the compose `postgres` image without owners or grants (the scratch server has no `odds_agent` role) and prints its row counts of `events`, `odds_snapshots`, `prices`, `candidate_decisions` and `paper_trades` beside production's. It publishes no port, removes the container on exit and only reads production. Production keeps collecting after a dump is taken, so its counts exceed the dump's by the rows written since; a nightly dump checked the next morning differs by those rows only.

## Restore from the off-host copy

When the host's dumps are lost, fetch one first, on the production host or a replacement with the repo and rclone set up as in [Set up the off-host copy](#set-up-the-off-host-copy):

1. **List the dumps.** `rclone lsf <remote>:<bucket>`; the newest `odds-nightly-*.dump` is last night's.
2. **Copy one down.** `rclone copyto <remote>:<bucket>/<file>.dump ~/backups/<file>.dump`.
3. **Check it,** with `./scripts/restore-check.sh ~/backups/<file>.dump` ([Check a dump restores](#check-a-dump-restores)), then restore it as below. On a new host, start the database (`docker compose up -d postgres`) and restore into the empty `odds` database.

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

- [Nightly database backup job](database-backup-job.md)
- [Create and apply a database migration](howto-database-migration.md)
- [Run the scheduler as a systemd service](../scheduling/howto-run-scheduler-service.md)
