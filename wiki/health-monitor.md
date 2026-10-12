---
type: component
title: Health monitor and check-health job
summary: The five health checks (stale data, failure streaks, quota, quality, heartbeats), their thresholds, and the duplicated alert plumbing.
sources:
  - docs/REPORTING.md#3-health-monitoring--packagesodds-lambdaodds_lambdahealth_monitorpy
  - docs/REPORTING.md#9-known-limitations
  - docs/ARCHITECTURE.md#health-check-job
  - docs/DEVELOPMENT.md#alerts-alertconfig
  - packages/odds-lambda/odds_lambda/health_monitor.py::HealthMonitor
  - packages/odds-lambda/odds_lambda/jobs/check_health.py::main
verified_at: 7887039
---

# Health monitor and check-health job

`packages/odds-lambda/odds_lambda/jobs/check_health.py::main` runs `check_system_health()` inside `job_alert_context("check-health")` and then books itself 60 minutes ahead through a raw backend call (no pre-schedule; see [self-scheduling](self-scheduling.md)). It is global, created by Terraform on AWS, and not in the default local `bootstrap_jobs` ([deployment matrix](job-deployment-matrix.md)). The logic is `packages/odds-lambda/odds_lambda/health_monitor.py::HealthMonitor(session, settings)`.

## Checks

`check_system_health` collects `HealthMetrics`, then runs, in order:

1. `check_stale_data`: hours since `max(OddsSnapshot.snapshot_time)` over all sources versus `stale_data_hours` (2). An empty table is unhealthy.
2. `check_consecutive_failures`: leading run of `success=False` among the latest 10 `FetchLog` rows (any sport or source) versus `consecutive_failures_threshold` (3) ([table fetch_logs](table-fetch-logs.md)).
3. `check_api_quota`: `api_quota_remaining` of the latest row that has one, divided by `settings.api.quota`; below `quota_critical_threshold` (0.1) alert `quota_critical`, below `quota_warning_threshold` (0.2) alert `quota_low` ([odds API quota](odds-api-quota.md)).
4. `check_data_quality`: `error` plus `critical` `DataQualityLog` rows in 24 h (each query capped at 1000) versus `data_quality_error_threshold` (10) ([data quality validation](data-quality-validation.md)).
5. `check_job_heartbeats`: for each `heartbeat_expectations` entry, newest `heartbeat:<job>` row older than the allowed hours fires `missing_heartbeat:<job>` ([table alert_history](table-alert-history.md)).

Then `purge_old_heartbeats()` runs. Alert types and severities are in [table alert_history](table-alert-history.md). If any step raises, the monitor sends `health_check_failure` (critical) and returns an unhealthy `HealthStatus` instead of raising.

## Plumbing

`_send_alert` checks `alert_enabled`, then `_should_send_alert` (a count over `alert_history` using `alert_rate_limit_minutes`), calls `alert_manager.alert`, then `_record_alert` with the injected `AsyncSession`. This duplicates `check_rate_limit` and `record_to_alert_history` in [alerts module](alerts-module.md) because the monitor carries its own session and settings.

## Hazards

- Two rate-limit implementations with different config: the monitor honours `ALERT_RATE_LIMIT_MINUTES`; `job_alert_context` does not.
- `HealthMetrics` is returned in `HealthStatus` and logged by the job but never sent to Discord; only issue strings are alerted.
- Because `check_system_health` swallows its own failures, `check_health` still writes a heartbeat after a failed health run.
- If `check_system_health` itself raised, `check_health` would not reschedule (post-work scheduling), ending the chain.
- `FetchLog` is written only after a successful API response, so API outages do not produce failure rows; the failure streak counts only per-event ingest errors and Betfair zero-snapshot runs ([job-fetch-odds](job-fetch-odds.md), [Betfair job](job-fetch-betfair-exchange.md)).
- Heartbeat expectations only cover jobs wrapped in `job_alert_context`; `fetch-odds` and `fetch-scores` are not covered.
- A job with no heartbeat history never alerts, including after a rename.

## Related

- [Reporting surfaces](reporting-surfaces.md).
