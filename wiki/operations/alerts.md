---
type: reference
title: Alerts and the alert_history table
description: Discord delivery, the alert helpers and job_alert_context, AlertConfig, rate limiting and the alert_history taxonomy.
sources:
  - odds_core.alerts.AlertManager
  - odds_core.alerts.AlertBase
  - odds_core.alerts.DiscordAlert
  - odds_core.alerts.job_alert_context
  - odds_core.alerts.send_job_warning
  - odds_core.alerts.check_rate_limit
  - odds_core.alerts.record_to_alert_history
  - odds_core.config.AlertConfig
  - odds_core.models.AlertHistory
  - odds_lambda.health_monitor.HealthMonitor
status: stable
---

# Alerts and the alert_history table

`odds_core.alerts` is the only path to Discord. Jobs own their embed formatting; this module owns delivery, rate limiting and heartbeats. [health-monitor](health-monitor.md) polls the database and writes to the same `alert_history` table. How the pieces fit together is on [alerting-and-heartbeats](alerting-and-heartbeats.md).

## Delivery

- `odds_core.alerts.AlertBase` is the channel interface (`odds_core.alerts.AlertBase.send`, `odds_core.alerts.AlertBase.send_embed`). `odds_core.alerts.DiscordAlert` posts to a webhook; severity colours are info blue, warning yellow, error red, critical dark red. Discord is the only channel. A non-204 response or exception is logged (`discord_alert_failed` / `_error`) and swallowed, so a rotated webhook silently disables alerting (#457, #384).
- `odds_core.alerts.AlertManager` builds channels from `settings.alerts` at construction (Discord only, when `discord_webhook_url` is set). `odds_core.alerts.AlertManager.alert` and `odds_core.alerts.AlertManager.send_embed` no-op with a debug log if `alert_enabled` is false and warn if there are no channels.
- `odds_core.alerts.alert_manager` is a module-level singleton created at import, so it snapshots `ALERT_ENABLED` and the webhook then; flipping `ALERT_ENABLED` later needs a new instance, and tests patch `odds_core.alerts.alert_manager` (#457).

Helpers:

- `odds_core.alerts.send_info`, `odds_core.alerts.send_warning`, `odds_core.alerts.send_error`, `odds_core.alerts.send_critical`: one-shot, not rate limited, not recorded.
- `check_rate_limit(alert_type, rate_limit_minutes=30)`: true if no `alert_history` row of that type is newer than the window. The default is the constant `_DEFAULT_RATE_LIMIT_MINUTES`, a hard-coded 30 minutes; `ALERT_RATE_LIMIT_MINUTES` governs only the health monitor (#384).
- `record_to_alert_history(alert_type, severity, message)`: inserts a row. It runs whether or not the POST succeeded, so a failed delivery still suppresses retries for the window (#457, #384). The rate-limit read and the insert are separate sessions with no lock, so two concurrent jobs of one type can both send (#457).
- `job_alert_context(job_name)`: async context manager. On exception: if the manager is enabled and `job_failure:<job_name>` is not rate limited, send a critical and record it; always re-raise. On clean exit: record `heartbeat:<job_name>` (silent; failures to record are logged only).
- `send_job_warning(alert_type, message)`: rate-limited warning for soft failures (empty scrape); returns whether it sent. Used by the [OddsPortal](../collection/fetch-oddsportal-job.md) and [Betfair](../collection/betfair-exchange.md) jobs.

On AWS, the Lambda handler also sends its own critical on any exception ([lambda-handler](../scheduling/scheduler-backends.md#lambda-handler)).

`odds_core.config.AlertConfig` (no env prefix; its docstring says "infrastructure for future use", yet it drives deployed alerting): `DISCORD_WEBHOOK_URL`, `ALERT_ENABLED` (default false), quota thresholds 0.2 and 0.1, `CONSECUTIVE_FAILURES_THRESHOLD` 3, `STALE_DATA_HOURS` 2, `ALERT_RATE_LIMIT_MINUTES` 30, `DATA_QUALITY_ERROR_THRESHOLD` 10, `heartbeat_expectations`, `heartbeat_retention_days` 7 ([configuration](../foundations/configuration.md)).

## Alert history

`odds_core.models.AlertHistory` (`alert_history`) holds `alert_type`, `severity`, `message`, `sent_at`, nullable JSON `context`, with composite index `ix_alert_type_sent_at` that serves both rate-limit and heartbeat lookups. It is a dedup ledger and a heartbeat log in one table. Writers: `odds_core.alerts.record_to_alert_history` and `odds_lambda.health_monitor.HealthMonitor._record_alert`. Readers: `odds_core.alerts.check_rate_limit`, `odds_lambda.health_monitor.HealthMonitor._should_send_alert`, `odds_lambda.health_monitor.HealthMonitor.check_job_heartbeats`, `odds_lambda.health_monitor.HealthMonitor.purge_old_heartbeats`.

| `alert_type` (the rate-limit key) | Source | Severity |
|---|---|---|
| `job_failure:<job>` | `odds_core.alerts.job_alert_context` | critical |
| `heartbeat:<job>` | `odds_core.alerts.job_alert_context` on success | info, never sent to Discord |
| `scrape_empty:<job>`, `job_empty:<job>` | `odds_core.alerts.send_job_warning` in the scrape and Betfair jobs | warning |
| `stale_data` | `odds_lambda.health_monitor.HealthMonitor` | warning |
| `consecutive_failures` | `odds_lambda.health_monitor.HealthMonitor` | error |
| `quota_low`, `quota_critical` | `odds_lambda.health_monitor.HealthMonitor` | warning, critical |
| `data_quality_errors` | `odds_lambda.health_monitor.HealthMonitor` | warning |
| `missing_heartbeat:<job>` | `odds_lambda.health_monitor.HealthMonitor` | warning |
| `health_check_failure` | `odds_lambda.health_monitor.HealthMonitor` | critical |

The default window is 30 minutes (`odds_core.alerts.check_rate_limit`'s argument for jobs, `odds_core.config.AlertConfig.alert_rate_limit_minutes` for the monitor). The key is `alert_type` alone, so a warning suppresses a critical of the same type for the window; the quota pair works only because the types differ (#384).

Heartbeats are distinguished from sent alerts only by the `heartbeat:` prefix, which `odds_lambda.health_monitor.HealthMonitor.purge_old_heartbeats` also filters on (`startswith`); nothing enforces that heartbeat rows are silent and alert rows were sent (#384).

## Related

- [Health monitor and check-health job](health-monitor.md)
- [How failures are detected](alerting-and-heartbeats.md)
- [Reporting surfaces](reporting-surfaces.md)
