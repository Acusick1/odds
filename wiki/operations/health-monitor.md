---
type: reference
title: Health monitor, check-health job and odds status
description: HealthMonitor, the check-health job's five checks, heartbeat purge, its own alert path, and what odds status show, quota and events print and read.
sources:
  - odds_lambda.health_monitor.HealthMonitor
  - odds_lambda.health_monitor.HealthMonitor.purge_old_heartbeats
  - odds_lambda.health_monitor.HealthMonitor.check_job_heartbeats
  - odds_lambda.health_monitor.HealthMonitor.check_consecutive_failures
  - odds_lambda.health_monitor.HealthMonitor.check_api_quota
  - odds_lambda.health_monitor.check_system_health
  - odds_lambda.health_monitor.HealthMetrics
  - odds_lambda.health_monitor.HealthStatus
  - odds_lambda.jobs.check_health.main
  - odds_core.config.AlertConfig
  - odds_core.config.SchedulerConfig.bootstrap_jobs
  - odds_core.models.FetchLog
  - odds_cli.commands.status
  - odds_cli.commands.status._show_quota
  - odds_cli.commands.scheduler.health
  - odds_lambda.storage.readers.OddsReader.get_database_stats
  - odds_lambda.storage.readers.OddsReader.get_fetch_logs
status: stable
---

# Health monitor, check-health job and odds status

`odds_lambda.jobs.check_health.main` runs `odds_lambda.health_monitor.check_system_health` inside `job_alert_context("check-health")` and then books itself 60 minutes ahead through a raw backend call (no pre-schedule; see [self-scheduling](../scheduling/self-scheduling.md)). It is global and in the default production `odds_core.config.SchedulerConfig.bootstrap_jobs`, so it runs on the mini-PC against the production database, where every job's heartbeat lands ([job-deployment-matrix](../scheduling/job-deployment-matrix.md#where-each-job-runs)). Nothing alerts when the host itself stops (#489). `odds_lambda.health_monitor.check_system_health` collects `odds_lambda.health_monitor.HealthMetrics`, then runs, in order:

1. `odds_lambda.health_monitor.HealthMonitor.check_stale_data`: hours since `max(OddsSnapshot.snapshot_time)` over all sources versus `stale_data_hours` (2). An empty table is unhealthy.
2. `odds_lambda.health_monitor.HealthMonitor.check_consecutive_failures`: leading run of `success=False` among the latest 10 `odds_core.models.FetchLog` rows versus `consecutive_failures_threshold` (3). It reads rows of any sport or source; API outages write no row (#472) and `odds_lambda.jobs.fetch_betfair_exchange` writes `success=False` when a run stores no snapshots, so those runs count as failures ([fetch logs](../collection/odds-ingestion-service.md#fetch-logs)).
3. `odds_lambda.health_monitor.HealthMonitor.check_api_quota`: `api_quota_remaining` of the latest row that has one, divided by `settings.api.quota`; below 0.1 alert `quota_critical`, below 0.2 `quota_low`. This is the same fraction fetch-odds applies, but rate-limited ([odds-api-quota](../collection/odds-api-quota.md)).
4. `odds_lambda.health_monitor.HealthMonitor.check_data_quality`: `error` plus `critical` `odds_core.models.DataQualityLog` rows in 24 h (each query capped at 1000) versus `data_quality_error_threshold` (10). The validator itself only emits `warning` and `error`, so `critical` never occurs ([odds-validator](../storage/odds-validator.md)).
5. `odds_lambda.health_monitor.HealthMonitor.check_job_heartbeats`: compares the newest `heartbeat:<job>` row per key in `odds_core.config.AlertConfig.heartbeat_expectations` to its allowed hours. The default keys and allowed hours are in [Heartbeats](#heartbeats). A job with no history is not alerted (#384).

Then `odds_lambda.health_monitor.HealthMonitor.purge_old_heartbeats` runs on every health check: it deletes `heartbeat:%` rows older than `heartbeat_retention_days` (7) and is the only retention sweep in the system.

## Heartbeats

A job inside `odds_core.alerts.job_alert_context` inserts `heartbeat:<name>` on clean exit ([alerts](alerts.md#alert-history)). `odds_core.config.AlertConfig.heartbeat_expectations` holds the allowed hours per key; the table lists every job module under `odds_lambda.jobs` and whether it writes a heartbeat. How the mechanisms fit together is on [alerting-and-heartbeats](alerting-and-heartbeats.md).

| Job | Writes a heartbeat | Default expectation |
|---|---|---|
| fetch-oddsportal (per sport) | yes | 11 h EPL, 12 h MLB (overnight window plus polling interval) |
| fetch-oddsportal-results (per sport) | yes | 26 h (EPL, MLB) |
| daily-digest (EPL) | yes | none |
| check-health | yes | 2 h |
| update-status | yes | none |
| fetch-betfair-exchange | yes | none |
| fetch-polymarket-odds (per sport) | yes | none until the job is bootstrapped |
| score-predictions | yes | none |
| fetch-mlb-probables | yes | none |
| fetch-espn-fixtures | yes | none |
| fetch-odds, fetch-scores, fetch-polymarket, backfill-polymarket, agent-run | no: not wrapped in `odds_core.alerts.job_alert_context` (#438) | none |

## Failure handling

- If any step raises, the monitor sends `health_check_failure` (critical) and returns an unhealthy `odds_lambda.health_monitor.HealthStatus` instead of raising. `check_health` therefore writes a heartbeat after a failed run, and it schedules after the work, so an exception that did escape would end the chain (#438).
- `odds_lambda.health_monitor.HealthMetrics` is returned in `odds_lambda.health_monitor.HealthStatus` and logged but never sent to Discord; only issue strings are alerted (#384).

## Alert path

`odds_lambda.health_monitor.HealthMonitor._send_alert` checks `alert_enabled`, then `odds_lambda.health_monitor.HealthMonitor._should_send_alert` (a count over `alert_history` using `odds_core.config.AlertConfig.alert_rate_limit_minutes`), calls `odds_core.alerts.AlertManager.alert`, then `odds_lambda.health_monitor.HealthMonitor._record_alert` with the injected `AsyncSession`. This duplicates `odds_core.alerts.check_rate_limit` and `odds_core.alerts.record_to_alert_history` because the monitor carries its own session and settings ([alerts](alerts.md#alert-history)).

## odds status

`odds_cli.commands.status` is the on-demand view of the same tables. It prints Rich tables only; there is no machine-readable output.

- `odds status show [--verbose]`: latest fetch (minutes ago, tick or cross), event counts by status, odds and snapshot totals, quota and 24 h success rate from `odds_lambda.storage.readers.OddsReader.get_database_stats`; `--verbose` adds data-quality rows from the last 24 h (limit 10).
- `odds status quota`: latest quota plus the last 10 `odds_core.models.FetchLog` rows (`odds_cli.commands.status._show_quota`). It reads the newest 10 rows of any source through `odds_lambda.storage.readers.OddsReader.get_fetch_logs` and shows nothing when the newest row has a null quota, as Betfair rows do (#441).
- `odds scheduler health` (not part of `odds status`): prints a green or red overall status, the backend name, the passed and failed checks and any details from `odds_lambda.scheduling.backends.base.SchedulerBackend.health_check`, and exits 1 when unhealthy or on error.
- `odds status events --days N --team T` (default 7 days): recent events with status and scores. It has no sport filter.
- `status show` takes the latest fetch row from all sources and computes the 24 h success rate over all of them, so Betfair rows can distort both. Its quota is the newest non-null `api_quota_remaining`, unlike `status quota` ([odds-api-quota](../collection/odds-api-quota.md#where-the-views-disagree), [fetch logs](../collection/odds-ingestion-service.md#fetch-logs)).

## Related

- [Alerts and the alert_history table](alerts.md)
- [How failures are detected](alerting-and-heartbeats.md)
- [Reporting surfaces](reporting-surfaces.md)
- [Add a scheduler job](../scheduling/howto-add-scheduler-job.md)
