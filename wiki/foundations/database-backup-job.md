---
type: reference
title: Nightly database backup job
description: What the backup-database job does each night, its settings, which dumps its retention keeps and prunes, and how a failed or missing backup is reported.
sources:
  - odds_lambda.jobs.backup_database
  - odds_core.config.BackupConfig
  - scripts/backup-prod-db.sh
mentions:
  - odds_core.alerts.job_alert_context
  - odds_lambda.health_monitor.HealthMonitor.check_job_heartbeats
status: draft
---

# Nightly database backup job

`backup-database` (`odds_lambda.jobs.backup_database.main`) dumps the production database every night and copies the dump off the production host with [rclone](https://rclone.org/docs/), to an S3-compatible object store. It is a global job with the cron `0 5 * * *` in `odds_core.job_specs.JOB_SPECS`, and it is in the default `odds_core.config.SchedulerConfig.bootstrap_jobs`, so the production scheduler runs it ([Where each job runs](../scheduling/job-deployment-matrix.md#where-each-job-runs)). Restoring a dump is on [Back up and restore](howto-back-up-production-database.md#restore-from-the-off-host-copy).

## Flow

1. **Gate.** With `BACKUP_REMOTE` unset the job logs `backup_remote_not_configured` and returns, writing no heartbeat and sending no alert. Under `SCHEDULER_DRY_RUN` (the deploy smoke) it logs `backup_dry_run_skipped` and returns.
2. **Dump.** It runs `scripts/backup-prod-db.sh nightly` with `ODDS_BACKUP_DIR` set to the job's directory. The script writes and checks `odds-nightly-<UTC timestamp>.dump` and prints its path on stdout ([Back up](howto-back-up-production-database.md#back-up)).
3. **Upload.** `rclone copyto` copies the dump to `<remote>/<file name>`, where `<remote>` is `BACKUP_REMOTE`.
4. **Prune.** It lists the remote with `rclone lsf --files-only` and deletes, with `rclone deletefile`, every nightly dump the retention does not keep, then does the same in the local directory.

Steps 2 to 4 run inside `odds_core.alerts.job_alert_context("backup-database")`. A failed dump, upload or prune raises with the command's stderr, which sends the critical `job_failure:backup-database` and writes no heartbeat ([Alerts](../operations/alerts.md#delivery)). A run cancelled by the run timeout first sends the command's process group SIGTERM, then SIGKILL after `odds_lambda.jobs.backup_database.STOP_GRACE`, so no host-side dump or upload outlives it; the `pg_dump` inside the container ends when it next writes to the closed pipe. A clean run writes `heartbeat:backup-database`, so a heartbeat means a dump reached the remote. Nothing is pruned before the upload succeeds. The run must finish within `odds_lambda.scheduling.runner.JOB_TIMEOUT` (20 min) ([Local scheduler](../scheduling/local-scheduler.md#scheduling-primitives)).

The job shells out to `rclone` on the scheduler's `PATH` and to `docker compose`, so it runs only on the production host, as the user of `odds-scheduler.service` ([scheduler service](../scheduling/howto-run-scheduler-service.md)). rclone reads its remotes from that user's `~/.config/rclone/rclone.conf`; no bucket or credential is in the repo or in `.env`.

## Settings

`odds_core.config.BackupConfig`, the `backup` section of `odds_core.config.Settings`:

| Env name | Field | Default | Meaning |
|---|---|---|---|
| `BACKUP_REMOTE` | `remote` | unset | rclone destination, such as `r2:odds-backups`; a remote name ending in `:` takes the file at its root |
| `ODDS_BACKUP_DIR` | `dir` | `~/backups` | local dump directory, shared with `scripts/backup-prod-db.sh` |
| `BACKUP_KEEP_DAILY` | `keep_daily` | 14 | days whose newest nightly dump is kept |
| `BACKUP_KEEP_WEEKLY` | `keep_weekly` | 8 | ISO weeks whose newest nightly dump is kept |

## Retention

`odds_lambda.jobs.backup_database.select_kept` keeps the newest dump of each of the `keep_daily` most recent UTC days that have a dump, and the newest dump of each of the `keep_weekly` most recent ISO weeks that have one; a dump kept by either rule stays. Periods are counted among those that hold a dump, so nights the job failed never push the last good dumps out. The same rule runs on the remote and in the local directory.

Only files named `odds-nightly-<YYYYmmddTHHMMZ>.dump` (`odds_lambda.jobs.backup_database.parse_nightly`) are candidates. Dumps with any other label, such as `pre-deploy-*` from `deploy.sh` (taken before a pending migration) or a hand-run `pre-<issue>`, are never pruned, and nothing else copies them off the host.

## Missing backups

`backup-database` has a 26 h entry in `odds_core.config.AlertConfig.heartbeat_expectations`, so the hourly health check sends `missing_heartbeat:backup-database` when no run has reached the remote for a day plus slack (`odds_lambda.health_monitor.HealthMonitor.check_job_heartbeats`, [Heartbeats](../operations/health-monitor.md#heartbeats)). Like every heartbeat expectation it stays silent until the job's first clean run, so a host where `BACKUP_REMOTE` was never set raises nothing.

## Related

- [Back up and restore the production database](howto-back-up-production-database.md)
- [Databases and environments](databases-and-environments.md)
- [Health monitor](../operations/health-monitor.md)
