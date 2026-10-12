---
type: component
title: Alerts module
summary: "alerts.py: AlertManager and DiscordAlert, the send_* helpers, job_alert_context, send_job_warning, configuration, and what happens when delivery fails."
sources:
  - docs/REPORTING.md#2-alert-primitives--packagesodds-coreodds_corealertspy
  - docs/REPORTING.md#9-known-limitations
  - docs/ARCHITECTURE.md#discord-alerts
  - docs/DEVELOPMENT.md#alerts-alertconfig
  - packages/odds-core/odds_core/alerts.py::AlertManager
  - packages/odds-core/odds_core/alerts.py::DiscordAlert
  - packages/odds-core/odds_core/alerts.py::job_alert_context
  - packages/odds-core/odds_core/config.py::AlertConfig
verified_at: 7887039
---

# Alerts module

`packages/odds-core/odds_core/alerts.py` is the only path to Discord. Jobs own their embed formatting; this module owns delivery, rate limiting and heartbeats.

## Delivery

- `AlertBase` is the channel interface (`send`, `send_embed`). `DiscordAlert` posts to a webhook; severity colours are info blue, warning yellow, error red, critical dark red. A non-204 response or exception is logged (`discord_alert_failed` / `_error`) and swallowed.
- `AlertManager` builds channels from `settings.alerts` at construction (Discord only, when `discord_webhook_url` is set). `alert(message, severity)` and `send_embed(embed)` no-op with a debug log if `alert_enabled` is false and warn if there are no channels.
- `alert_manager = AlertManager()` is a module-level singleton created at import.

## Helpers

- `send_info`, `send_warning`, `send_error`, `send_critical`: one-shot, not rate limited, not recorded.
- `check_rate_limit(alert_type, rate_limit_minutes=30)`: true if no `alert_history` row of that type is newer than the window ([table alert_history](table-alert-history.md)). The default is the constant `_DEFAULT_RATE_LIMIT_MINUTES`.
- `record_to_alert_history(alert_type, severity, message)`: inserts a row.
- `job_alert_context(job_name)`: async context manager. On exception: if the manager is enabled and `job_failure:<job_name>` is not rate limited, send a critical and record it; always re-raise. On clean exit: record `heartbeat:<job_name>` (silent; failures to record are logged only).
- `send_job_warning(alert_type, message)`: rate-limited warning for soft failures (empty scrape); returns whether it sent. Used by the [OddsPortal](job-fetch-oddsportal.md) and [Betfair](job-fetch-betfair-exchange.md) jobs.

## Configuration

`AlertConfig` (`packages/odds-core/odds_core/config.py`; no env prefix): `DISCORD_WEBHOOK_URL`, `ALERT_ENABLED` (default false), quota thresholds 0.2 and 0.1, `CONSECUTIVE_FAILURES_THRESHOLD` 3, `STALE_DATA_HOURS` 2, `ALERT_RATE_LIMIT_MINUTES` 30, `DATA_QUALITY_ERROR_THRESHOLD` 10, `heartbeat_expectations`, `heartbeat_retention_days` 7. Consumers: [health monitor](health-monitor.md) and [configuration](configuration.md).

## Hazards

- `alert_rate_limit_minutes` governs only `HealthMonitor`; `check_rate_limit` uses its own hard-coded 30. Changing the setting does not retune job failure alerts.
- Discord-only: a rotated webhook silently disables alerting. `record_to_alert_history` runs whether or not the POST succeeded, so a failed delivery still suppresses retries for the window.
- The rate-limit read and the insert are separate sessions without a lock; two concurrent jobs of one type can both send.
- `AlertManager` reads settings at import; flipping `ALERT_ENABLED` later needs a new instance, and tests patch `alert_manager`.
- Heartbeat names must equal the `job_alert_context` argument and the `heartbeat_expectations` key by convention only. `fetch-odds`, `fetch-scores`, `fetch-polymarket` and `backfill-polymarket` do not use the context, so they have no heartbeat and send hand-rolled criticals ([job-fetch-odds](job-fetch-odds.md)).
- `AlertConfig`'s docstring says "infrastructure for future use"; it drives deployed alerting.
- The [Lambda handler](lambda-handler.md) adds an unconditional critical on top of `job_alert_context`.
- Delivery depends on the deployed env: the scraper function's Terraform sets neither `ALERT_ENABLED` nor `DISCORD_WEBHOOK_URL` ([AWS infrastructure](aws-infrastructure.md)).

## Related

- [Reporting surfaces](reporting-surfaces.md), [add a scheduler job](howto-add-scheduler-job.md).
