---
type: component
title: Alerts, heartbeats and the health monitor
description: How alerts reach Discord, the alert-type taxonomy and severity-blind rate limiting in alert_history, heartbeats and their expectations, the check-health job's five checks, and which jobs have no heartbeat at all.
sources:
  - packages/odds-core/odds_core/alerts.py::AlertManager
  - packages/odds-core/odds_core/alerts.py::DiscordAlert
  - packages/odds-core/odds_core/alerts.py::job_alert_context
  - packages/odds-core/odds_core/alerts.py::check_rate_limit
  - packages/odds-core/odds_core/alerts.py::record_to_alert_history
  - packages/odds-core/odds_core/config.py::AlertConfig
  - packages/odds-core/odds_core/models.py::AlertHistory
  - packages/odds-lambda/odds_lambda/health_monitor.py::HealthMonitor
  - packages/odds-lambda/odds_lambda/health_monitor.py::HealthMonitor.purge_old_heartbeats
  - packages/odds-lambda/odds_lambda/jobs/check_health.py::main
status: stable
---

# Alerts, heartbeats and the health monitor

`packages/odds-core/odds_core/alerts.py` is the only path to Discord. Jobs own their embed formatting; this module owns delivery, rate limiting and heartbeats. `packages/odds-lambda/odds_lambda/health_monitor.py::HealthMonitor` polls the database for stalls and quota problems. Both write to the `alert_history` table.

## Delivery

- `packages/odds-core/odds_core/alerts.py::AlertBase` is the channel interface (`send`, `send_embed`). `packages/odds-core/odds_core/alerts.py::DiscordAlert` posts to a webhook; severity colours are info blue, warning yellow, error red, critical dark red. A non-204 response or exception is logged (`discord_alert_failed` / `_error`) and swallowed.
- `packages/odds-core/odds_core/alerts.py::AlertManager` builds channels from `settings.alerts` at construction (Discord only, when `discord_webhook_url` is set). `alert(message, severity)` and `send_embed(embed)` no-op with a debug log if `alert_enabled` is false and warn if there are no channels.
- `alert_manager = AlertManager()` is a module-level singleton created at import, so it snapshots `ALERT_ENABLED` and the webhook then.

Helpers:

- `packages/odds-core/odds_core/alerts.py::send_info`, `packages/odds-core/odds_core/alerts.py::send_warning`, `packages/odds-core/odds_core/alerts.py::send_error`, `packages/odds-core/odds_core/alerts.py::send_critical`: one-shot, not rate limited, not recorded.
- `check_rate_limit(alert_type, rate_limit_minutes=30)`: true if no `alert_history` row of that type is newer than the window. The default is the constant `_DEFAULT_RATE_LIMIT_MINUTES`.
- `record_to_alert_history(alert_type, severity, message)`: inserts a row.
- `job_alert_context(job_name)`: async context manager. On exception: if the manager is enabled and `job_failure:<job_name>` is not rate limited, send a critical and record it; always re-raise. On clean exit: record `heartbeat:<job_name>` (silent; failures to record are logged only).
- `send_job_warning(alert_type, message)`: rate-limited warning for soft failures (empty scrape); returns whether it sent. Used by the [OddsPortal](../collection/oddsportal.md) and [Betfair](../collection/betfair-exchange.md) jobs.

`packages/odds-core/odds_core/config.py::AlertConfig` (`packages/odds-core/odds_core/config.py`; no env prefix): `DISCORD_WEBHOOK_URL`, `ALERT_ENABLED` (default false), quota thresholds 0.2 and 0.1, `CONSECUTIVE_FAILURES_THRESHOLD` 3, `STALE_DATA_HOURS` 2, `ALERT_RATE_LIMIT_MINUTES` 30, `DATA_QUALITY_ERROR_THRESHOLD` 10, `heartbeat_expectations`, `heartbeat_retention_days` 7 ([configuration](../foundations/configuration.md)).

## The alert_history table

`packages/odds-core/odds_core/models.py::AlertHistory` (`alert_history`) holds `alert_type`, `severity`, `message`, `sent_at`, nullable JSON `context`, with composite index `ix_alert_type_sent_at` that serves both rate-limit and heartbeat lookups. It is a dedup ledger and a heartbeat log in one table. Writers: `packages/odds-core/odds_core/alerts.py::record_to_alert_history` and `packages/odds-lambda/odds_lambda/health_monitor.py::HealthMonitor._record_alert`. Readers: `packages/odds-core/odds_core/alerts.py::check_rate_limit`, `packages/odds-lambda/odds_lambda/health_monitor.py::HealthMonitor._should_send_alert`, `packages/odds-lambda/odds_lambda/health_monitor.py::HealthMonitor.check_job_heartbeats`, `purge_old_heartbeats`.

| `alert_type` (the rate-limit key) | Source | Severity |
|---|---|---|
| `job_failure:<job>` | `packages/odds-core/odds_core/alerts.py::job_alert_context` | critical |
| `heartbeat:<job>` | `packages/odds-core/odds_core/alerts.py::job_alert_context` on success | info, never sent to Discord |
| `scrape_empty:<job>`, `job_empty:<job>` | `packages/odds-core/odds_core/alerts.py::send_job_warning` in the scrape and Betfair jobs | warning |
| `stale_data` | `packages/odds-lambda/odds_lambda/health_monitor.py::HealthMonitor` | warning |
| `consecutive_failures` | `packages/odds-lambda/odds_lambda/health_monitor.py::HealthMonitor` | error |
| `quota_low`, `quota_critical` | `packages/odds-lambda/odds_lambda/health_monitor.py::HealthMonitor` | warning, critical |
| `data_quality_errors` | `packages/odds-lambda/odds_lambda/health_monitor.py::HealthMonitor` | warning |
| `missing_heartbeat:<job>` | `packages/odds-lambda/odds_lambda/health_monitor.py::HealthMonitor` | warning |
| `health_check_failure` | `packages/odds-lambda/odds_lambda/health_monitor.py::HealthMonitor` | critical |

The default window is 30 minutes (`packages/odds-core/odds_core/alerts.py::check_rate_limit`'s argument for jobs, `packages/odds-core/odds_core/config.py::AlertConfig.alert_rate_limit_minutes` for the monitor).

## Heartbeats

On clean exit a job inside `job_alert_context(name)` inserts `heartbeat:<name>`. `packages/odds-lambda/odds_lambda/health_monitor.py::HealthMonitor.check_job_heartbeats` compares the newest row per key in `packages/odds-core/odds_core/config.py::AlertConfig.heartbeat_expectations` to its allowed hours. Default keys: `fetch-oddsportal-epl` and `-mlb` (2 h), `fetch-oddsportal-results-epl` and `-mlb` (26 h), `daily-digest-epl` (26 h), `check-health` (2 h). A job with no history is not alerted. `packages/odds-lambda/odds_lambda/health_monitor.py::HealthMonitor.purge_old_heartbeats` deletes `heartbeat:%` rows older than `heartbeat_retention_days` (7), on every health check; it is the only retention sweep in the system.

Jobs with no heartbeat: `fetch-odds`, `fetch-scores`, `fetch-polymarket` and `backfill-polymarket` do not use `packages/odds-core/odds_core/alerts.py::job_alert_context`, and `update-status` has no default expectation, so only stale-data checks catch a stalled chain; `fetch-scores` writes none. They send hand-rolled criticals instead ([Odds API ingestion](../collection/odds-api-ingestion.md), [Polymarket](../collection/polymarket.md)). Fetch-odds quota alerts use `packages/odds-core/odds_core/alerts.py::send_warning`/`packages/odds-core/odds_core/alerts.py::send_critical` without `alert_type`, so they bypass `alert_history` entirely.

## The check-health job

`packages/odds-lambda/odds_lambda/jobs/check_health.py::main` runs `packages/odds-lambda/odds_lambda/health_monitor.py::check_system_health` inside `job_alert_context("check-health")` and then books itself 60 minutes ahead through a raw backend call (no pre-schedule; see [self-scheduling](../scheduling/self-scheduling.md)). It is global, created by Terraform on AWS, and not in the default local `bootstrap_jobs` ([jobs and deployment matrix](../scheduling/jobs-and-deployment-matrix.md)). `packages/odds-lambda/odds_lambda/health_monitor.py::check_system_health` collects `packages/odds-lambda/odds_lambda/health_monitor.py::HealthMetrics`, then runs, in order:

1. `check_stale_data`: hours since `max(OddsSnapshot.snapshot_time)` over all sources versus `stale_data_hours` (2). An empty table is unhealthy.
2. `check_consecutive_failures`: leading run of `success=False` among the latest 10 `packages/odds-core/odds_core/models.py::FetchLog` rows (any sport or source, Betfair included) versus `consecutive_failures_threshold` (3).
3. `check_api_quota`: `api_quota_remaining` of the latest row that has one, divided by `settings.api.quota`; below 0.1 alert `quota_critical`, below 0.2 `quota_low`. This is the same fraction fetch-odds applies, but rate-limited ([quota](../collection/odds-api-client-and-quota.md)).
4. `check_data_quality`: `error` plus `critical` `packages/odds-core/odds_core/models.py::DataQualityLog` rows in 24 h (each query capped at 1000) versus `data_quality_error_threshold` (10). The validator itself only emits `warning` and `error` ([data validation](../storage/data-validation.md)).
5. `packages/odds-lambda/odds_lambda/health_monitor.py::HealthMonitor.check_job_heartbeats`: as above.

Then `packages/odds-lambda/odds_lambda/health_monitor.py::HealthMonitor.purge_old_heartbeats` runs. If any step raises, the monitor sends `health_check_failure` (critical) and returns an unhealthy `packages/odds-lambda/odds_lambda/health_monitor.py::HealthStatus` instead of raising. `_send_alert` checks `alert_enabled`, then `packages/odds-lambda/odds_lambda/health_monitor.py::HealthMonitor._should_send_alert` (a count over `alert_history` using `packages/odds-core/odds_core/config.py::AlertConfig.alert_rate_limit_minutes`), calls `alert_manager.alert`, then `packages/odds-lambda/odds_lambda/health_monitor.py::HealthMonitor._record_alert` with the injected `AsyncSession`. This duplicates `packages/odds-core/odds_core/alerts.py::check_rate_limit` and `packages/odds-core/odds_core/alerts.py::record_to_alert_history` because the monitor carries its own session and settings.

## Hazards

- **Two rate-limit implementations.** `packages/odds-core/odds_core/config.py::AlertConfig.alert_rate_limit_minutes` governs only `packages/odds-lambda/odds_lambda/health_monitor.py::HealthMonitor`; `packages/odds-core/odds_core/alerts.py::check_rate_limit` uses its own hard-coded 30, so changing the setting does not retune job failure alerts.
- **Severity is not in the rate-limit key.** A warning suppresses a critical of the same `alert_type` for the window. The quota pair works only because the types differ.
- **Silent delivery loss.** Discord-only: a rotated webhook silently disables alerting. `packages/odds-core/odds_core/alerts.py::record_to_alert_history` runs whether or not the POST succeeded, so a failed delivery still suppresses retries for the window. The digest's `sent` stat is likewise set even when delivery failed ([reporting](reporting-and-daily-digest.md)).
- **No lock.** The rate-limit read and the insert are separate sessions; two concurrent jobs of one type can both send.
- **Import-time snapshot.** `packages/odds-core/odds_core/alerts.py::AlertManager` reads settings at import; flipping `ALERT_ENABLED` later needs a new instance, and tests patch `alert_manager`.
- **Heartbeat key matching is by convention.** Heartbeat names must equal the `packages/odds-core/odds_core/alerts.py::job_alert_context` argument and the `packages/odds-core/odds_core/config.py::AlertConfig.heartbeat_expectations` key. A rename leaves a stale key, and since a missing history is not alerted, the check passes silently.
- **String-prefix heartbeats.** Heartbeats are distinguished from sent alerts only by the `heartbeat:` prefix, which `packages/odds-lambda/odds_lambda/health_monitor.py::HealthMonitor.purge_old_heartbeats` also filters on (`startswith`); nothing enforces "heartbeat rows are silent, alert rows were sent".
- **Handler double alert.** The Lambda handler adds an unconditional `packages/odds-core/odds_core/alerts.py::send_critical` on failure on top of the job's own `packages/odds-core/odds_core/alerts.py::job_alert_context` critical ([scheduler backends and handler](../scheduling/scheduler-backends.md)).
- **Delivery depends on deployed env.** The scraper function's Terraform sets neither `ALERT_ENABLED` nor `DISCORD_WEBHOOK_URL` ([AWS infrastructure](aws-infrastructure.md)).
- **FetchLog blind spots.** `packages/odds-core/odds_core/models.py::FetchLog` is written only after a successful API response, so API outages produce no failure rows; the streak counts only per-event ingest errors and Betfair zero-snapshot runs (`success = snapshots_stored > 0`), and Betfair rows have null quota ([Odds API ingestion](../collection/odds-api-ingestion.md), [Betfair](../collection/betfair-exchange.md)).
- **Metrics stay in the log.** `packages/odds-lambda/odds_lambda/health_monitor.py::HealthMetrics` is returned in `packages/odds-lambda/odds_lambda/health_monitor.py::HealthStatus` and logged but never sent to Discord; only issue strings are alerted.
- **Health failures hide.** `packages/odds-lambda/odds_lambda/health_monitor.py::check_system_health` swallows its own failures, so `check_health` still writes a heartbeat after a failed run; if it did raise, `check_health` would not reschedule (post-work scheduling), ending the chain.
- **Misleading docstring.** `packages/odds-core/odds_core/config.py::AlertConfig`'s docstring says "infrastructure for future use"; it drives deployed alerting.

## Related

- [Reporting and the daily digest](reporting-and-daily-digest.md), [Add a scheduler job](../scheduling/howto-add-scheduler-job.md), [AWS infrastructure](aws-infrastructure.md)
