---
type: reference
title: Alerts and the alert_history table
description: Discord delivery, the alert helpers and job_alert_context, AlertConfig, rate limiting and the alert_history taxonomy.
sources:
  - odds_core.alerts.AlertManager
  - odds_core.alerts.AlertBase
  - odds_core.alerts.DiscordAlert
  - odds_core.alerts.job_alert_context
  - odds_core.alerts.send_job_failure
  - odds_core.alerts.send_job_warning
  - odds_core.alerts.send_recorded_alert
  - odds_core.alerts.send_outage_warning
  - odds_core.alerts.send_scheduling_failure
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
- `odds_core.alerts.AlertManager` builds channels from `settings.alerts` at construction (Discord only, when `discord_webhook_url` is set). `odds_core.alerts.AlertManager.alert` and `odds_core.alerts.AlertManager.send_embed` no-op with a debug log if `odds_core.config.Settings.alerts_deliverable` is false (`alert_enabled` off, or `SCHEDULER_DRY_RUN` on, so the deploy smoke posts nothing) and warn if there are no channels.
- `odds_core.alerts.alert_manager` is a module-level singleton created at import, so it snapshots `ALERT_ENABLED` and the webhook then; flipping `ALERT_ENABLED` later needs a new instance, and tests patch `odds_core.alerts.alert_manager` (#457).

Helpers:

- `odds_core.alerts.send_critical`: one-shot, not rate limited, not recorded; only the Lambda handler uses it ([lambda-handler](../scheduling/scheduler-backends.md#lambda-handler)).
- `odds_core.alerts.send_recorded_alert(alert_type, severity, message, *, every, reset_by=None, context=None, session=None, now=None)`: the one dedup rule for every alert that repeats while its condition holds. It returns False when the manager is undeliverable; otherwise it reads the newest `alert_history` row of `alert_type` (ignored when a newer `reset_by` row exists, which starts a new condition), and sends and records only if there is none or it is older than `every`. The send is bounded by `odds_core.alerts.ALERT_SEND_TIMEOUT_SECONDS`, and a failure is logged, rolls back a caller-supplied `session` and returns False. The record uses `session` and `context` when given (the health monitor passes both). The read and the insert have no lock, so two concurrent senders of one type can both send (#457).
- `odds_core.alerts.record_to_alert_history(alert_type, severity, message)`: inserts a row, whether or not the POST succeeded, so a failed delivery still suppresses retries for the interval (#457, #384); heartbeats use it directly.
- `odds_core.alerts.send_job_failure(job_name, error)` and `odds_core.alerts.send_outage_warning` call `odds_core.alerts.send_recorded_alert` with `reset_by=heartbeat:<job_name>` and `every` of `odds_core.config.AlertConfig.stale_reminder_minutes` (360). An outage runs from a job's first failure to its next `heartbeat:<job_name>`, whatever the job's cadence; `odds_core.alerts.job_outage_alerted_at` is the single definition of an open outage, shared with the health monitor. Used by `odds_core.alerts.job_alert_context` and by `odds_lambda.scheduling.runner.run_chain_job` for a run cancelled by its timeout.
- `odds_core.alerts.send_scheduling_failure(job_name, error, final=...)`: `odds_core.alerts.send_recorded_alert` with `every` of `odds_core.config.AlertConfig.stale_reminder_minutes`, no `reset_by`. `final` is true when no other booking survives, so the chain may end, and the alert is then a critical; otherwise it is a warning. A skipped run's pre-schedule is its only booking (final). For a run that executes, the pre-schedule is not final because the post-run booking follows, and the post-run booking is final only if the pre-schedule failed. A failed `fetch-oddsportal-results` booking is final unless the run raised, because `odds_lambda.scheduling.runner.run_chain_job` then re-books. The two severities are keyed apart so a warning never suppresses a critical. Used by every job that books its own next run.
- `job_alert_context(job_name)`: async context manager. On exception: `odds_core.alerts.send_job_failure`, the send is bounded by `odds_core.alerts.ALERT_SEND_TIMEOUT_SECONDS`, and a failing or timed-out send is logged (`job_failure_alert_failed`) and never masks the job's exception, which is always re-raised unchanged. On clean exit: record `heartbeat:<job_name>` (silent; under `SCHEDULER_DRY_RUN` `odds_core.alerts.record_to_alert_history` writes nothing, so no heartbeat row; failures to record are logged only).
- `odds_core.alerts.send_job_warning(alert_type, message)`: warning for soft failures (empty scrape) through `odds_core.alerts.send_recorded_alert` at most every 30 minutes (`odds_core.alerts.JOB_WARNING_INTERVAL`); returns whether it sent. Used by the [OddsPortal](../collection/fetch-oddsportal-job.md) and [Betfair](../collection/betfair-exchange.md) jobs.

On AWS, the Lambda handler also sends its own critical on any exception ([lambda-handler](../scheduling/scheduler-backends.md#lambda-handler)).

`odds_core.config.AlertConfig` (no env prefix; its docstring says "infrastructure for future use", yet it drives deployed alerting): `DISCORD_WEBHOOK_URL`, `ALERT_ENABLED` (default false), quota thresholds 0.2 and 0.1, `STALE_DATA_HOURS` 2, `DATA_QUALITY_ERROR_THRESHOLD` 10, `heartbeat_expectations` (hours per compound job name such as `fetch-polymarket-odds-epl`, `fetch-odds-epl` or `fetch-espn-fixtures`, 2 to 49 by default), `heartbeat_retention_days` 7 ([configuration](../foundations/configuration.md)).

## Alert history

`odds_core.models.AlertHistory` (`alert_history`) holds `alert_type`, `severity`, `message`, `sent_at`, nullable JSON `context`, with composite index `ix_alert_type_sent_at` that serves both dedup and heartbeat lookups. It is one table serving as dedup ledger and heartbeat log. Writer: `odds_core.alerts.record_to_alert_history`. Readers: `odds_core.alerts.latest_sent_at` (behind `odds_core.alerts.send_recorded_alert`), `odds_lambda.health_monitor.HealthMonitor.check_job_heartbeats`, `odds_lambda.health_monitor.HealthMonitor.purge_old_heartbeats`.

| `alert_type` (the dedup key) | Source | Severity |
|---|---|---|
| `job_failure:<job>` | `odds_core.alerts.job_alert_context`; `odds_lambda.scheduling.runner.run_chain_job` for a run that exceeds its timeout | critical |
| `job_chain_rebooked:<job>` (a failed local run re-booked 15 minutes on) | `odds_lambda.scheduling.runner.run_chain_job` through `odds_core.alerts.send_outage_warning` (once per outage) | warning |
| `scheduling_failure:<job>` (no booking survives for the next run) | `odds_core.alerts.send_scheduling_failure` | critical |
| `scheduling_warning:<job>` (a booking failed but another one survives; fetch-odds, fetch-scores, update-status, fetch-oddsportal-results) | `odds_core.alerts.send_scheduling_failure` | warning |
| `heartbeat:<job>` | `odds_core.alerts.job_alert_context` on success | info, never sent to Discord |
| `job_empty:<job>` | `odds_core.alerts.send_job_warning` in the Polymarket odds job | warning |
| `stale_data:<source>:<sport>` (same rule as every monitor alert) | `odds_lambda.health_monitor.HealthMonitor.check_stale_sources` | critical |
| `missing_betfair_close` (sent after kick-off as a coverage record; one alert per run listing events not already alerted in the lookback, so it ignores the reminder interval) | `odds_lambda.health_monitor.HealthMonitor.check_missing_betfair_closes` | critical |
| `quota_low`, `quota_critical` | `odds_lambda.health_monitor.HealthMonitor.check_api_quota` | warning, critical |
| `data_quality_errors` | `odds_lambda.health_monitor.HealthMonitor.check_data_quality` | warning |
| `duplicate_events:<sport>`, `repeated_fixtures:<sport>` | `odds_lambda.health_monitor.HealthMonitor.check_event_identity` | warning |
| `unsettled_events:<sport>` (league sports; current-season events pending over 24 h after kick-off; reminds once a day, not every `stale_reminder_minutes`) | `odds_lambda.health_monitor.HealthMonitor.check_unsettled_events` | warning |
| `results_stalled:<sport>` (an OddsPortal results run settled nothing while events that kicked off after its last settling run are pending over 24 h) | `odds_lambda.jobs.fetch_oddsportal_results.record_run_outcome` | warning |
| `missing_heartbeat:<job>` | `odds_lambda.health_monitor.HealthMonitor.check_job_heartbeats` | warning |
| `dead_schedule_chain:<job>` (a bootstrapped self-scheduling job with no run booked or one overdue) | `odds_lambda.health_monitor.HealthMonitor.check_schedule_chains` | critical |
| `health_check_failure` (the monitor itself raised) | `odds_lambda.health_monitor.HealthMonitor.check_system_health` | critical |

A monitor alert goes through `odds_lambda.health_monitor.HealthMonitor._send_alert`, which calls `odds_core.alerts.send_recorded_alert` with `every` of `odds_core.config.AlertConfig.stale_reminder_minutes` (360), however often `check-health` runs. A health-check condition has no heartbeat to close it as a job outage has, so it is "new" when no alert of its type falls inside the interval: one that clears and returns within 6 h stays muted until the interval ends. Nothing is sent when a condition clears. The key is `alert_type` alone, so a warning suppresses a critical of the same type; the quota pair works only because the types differ. API quota is raised only by the monitor (`quota_low`, `quota_critical`); `fetch-odds` sends none (#587).

Heartbeats are distinguished from sent alerts only by the `heartbeat:` prefix, which `odds_lambda.health_monitor.HealthMonitor.purge_old_heartbeats` also filters on (`startswith`); nothing enforces that heartbeat rows are silent and alert rows were sent (#384).

## Related

- [Health monitor and check-health job](health-monitor.md)
- [How failures are detected](alerting-and-heartbeats.md)
- [Reporting surfaces](reporting-surfaces.md)
