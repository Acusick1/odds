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
  - odds_core.models.FetchLog
  - odds_cli.commands.status
  - odds_cli.commands.status._show_quota
  - odds_lambda.storage.readers.OddsReader.get_database_stats
  - odds_lambda.storage.readers.OddsReader.get_fetch_logs
status: stable
---

# Health monitor, check-health job and odds status

`odds_lambda.jobs.check_health.main` runs `odds_lambda.health_monitor.check_system_health` inside `job_alert_context("check-health")` and then books itself 60 minutes ahead through a raw backend call (no pre-schedule; see [self-scheduling](../scheduling/self-scheduling.md)). It is global, created by Terraform on AWS, and not in the default local `bootstrap_jobs` ([job-deployment-matrix](../scheduling/job-deployment-matrix.md#where-each-job-runs)). `odds_lambda.health_monitor.check_system_health` collects `odds_lambda.health_monitor.HealthMetrics`, then runs, in order:

1. `check_stale_data`: hours since `max(OddsSnapshot.snapshot_time)` over all sources versus `stale_data_hours` (2). An empty table is unhealthy.
2. `check_consecutive_failures`: leading run of `success=False` among the latest 10 `odds_core.models.FetchLog` rows versus `consecutive_failures_threshold` (3). It reads rows of any sport or source; API outages write no row (#472) and Betfair zero-snapshot runs count as failures (#441; [fetch logs](../collection/odds-ingestion-service.md#fetch-logs)).
3. `check_api_quota`: `api_quota_remaining` of the latest row that has one, divided by `settings.api.quota`; below 0.1 alert `quota_critical`, below 0.2 `quota_low`. This is the same fraction fetch-odds applies, but rate-limited ([odds-api-quota](../collection/odds-api-quota.md)).
4. `check_data_quality`: `error` plus `critical` `odds_core.models.DataQualityLog` rows in 24 h (each query capped at 1000) versus `data_quality_error_threshold` (10). The validator itself only emits `warning` and `error`, so `critical` never occurs ([odds-validator](../storage/odds-validator.md)).
5. `odds_lambda.health_monitor.HealthMonitor.check_job_heartbeats`: compares the newest `heartbeat:<job>` row per key in `odds_core.config.AlertConfig.heartbeat_expectations` to its allowed hours. Default keys: `fetch-oddsportal-epl` and `-mlb` (2 h), `fetch-oddsportal-results-epl` and `-mlb` (26 h), `daily-digest-epl` (26 h), `check-health` (2 h). A job with no history is not alerted (#384).

Then `odds_lambda.health_monitor.HealthMonitor.purge_old_heartbeats` runs on every health check: it deletes `heartbeat:%` rows older than `heartbeat_retention_days` (7) and is the only retention sweep in the system.

## Failure handling

- If any step raises, the monitor sends `health_check_failure` (critical) and returns an unhealthy `odds_lambda.health_monitor.HealthStatus` instead of raising. `check_health` therefore writes a heartbeat after a failed run, and it schedules after the work, so an exception that did escape would end the chain (#438).
- `odds_lambda.health_monitor.HealthMetrics` is returned in `odds_lambda.health_monitor.HealthStatus` and logged but never sent to Discord; only issue strings are alerted (#384).

## Alert path

`_send_alert` checks `alert_enabled`, then `odds_lambda.health_monitor.HealthMonitor._should_send_alert` (a count over `alert_history` using `odds_core.config.AlertConfig.alert_rate_limit_minutes`), calls `alert_manager.alert`, then `odds_lambda.health_monitor.HealthMonitor._record_alert` with the injected `AsyncSession`. This duplicates `odds_core.alerts.check_rate_limit` and `odds_core.alerts.record_to_alert_history` because the monitor carries its own session and settings ([alerts](alerts.md#alert-history)).

## odds status

`odds_cli.commands.status` is the on-demand view of the same tables. It prints Rich tables only; there is no machine-readable output.

- `odds status show [--verbose]`: latest fetch (minutes ago, tick or cross), event counts by status, odds and snapshot totals, quota and 24 h success rate from `odds_lambda.storage.readers.OddsReader.get_database_stats`; `--verbose` adds data-quality rows from the last 24 h (limit 10).
- `odds status quota`: latest quota plus the last 10 `odds_core.models.FetchLog` rows (`odds_cli.commands.status._show_quota`). It reads the newest 10 rows of any source through `odds_lambda.storage.readers.OddsReader.get_fetch_logs` and shows nothing when the newest row has a null quota, as Betfair rows do (#441).
- `odds status events --days N --team T` (default 7 days): recent events with status and scores. It has no sport filter.
- `status show` takes the latest fetch row from all sources and computes the 24 h success rate over all of them, so Betfair rows can distort both. Its quota is the newest non-null `api_quota_remaining`, unlike `status quota` ([odds-api-quota](../collection/odds-api-quota.md#where-the-views-disagree), [fetch logs](../collection/odds-ingestion-service.md#fetch-logs)).

## Related

- [Alerts and the alert_history table](alerts.md)
- [How failures are detected](alerting-and-heartbeats.md)
- [Reporting surfaces](reporting-surfaces.md)
- [Add a scheduler job](../scheduling/howto-add-scheduler-job.md)
