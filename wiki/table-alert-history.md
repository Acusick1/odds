---
type: component
title: alert_history table, alert types and heartbeats
summary: The alert-type taxonomy, severity-blind rate limiting, and the heartbeat rows that share the table, with their retention.
sources:
  - docs/REPORTING.md#alert-type-taxonomy
  - docs/REPORTING.md#8-storage-backing-reports
  - docs/REPORTING.md#9-known-limitations
  - docs/DATABASE.md#core-tables
  - packages/odds-core/odds_core/models.py::AlertHistory
  - packages/odds-core/odds_core/alerts.py::check_rate_limit
  - packages/odds-core/odds_core/alerts.py::record_to_alert_history
  - packages/odds-lambda/odds_lambda/health_monitor.py::purge_old_heartbeats
verified_at: 7887039
---

# alert_history table, alert types and heartbeats

`packages/odds-core/odds_core/models.py::AlertHistory` (`alert_history`) holds `alert_type`, `severity`, `message`, `sent_at`, nullable JSON `context`, with composite index `ix_alert_type_sent_at` that serves both rate-limit and heartbeat lookups. It is a dedup ledger and a heartbeat log in one table.

## Writers and readers

- Writers: `record_to_alert_history` ([alerts module](alerts-module.md)) and `HealthMonitor._record_alert` ([health monitor](health-monitor.md)).
- Readers: `check_rate_limit`, `HealthMonitor._should_send_alert`, `HealthMonitor.check_job_heartbeats`, `purge_old_heartbeats`.

## Alert types (the rate-limit key)

| `alert_type` | Source | Severity |
|---|---|---|
| `job_failure:<job>` | `job_alert_context` | critical |
| `heartbeat:<job>` | `job_alert_context` on success | info, never sent to Discord |
| `scrape_empty:<job>`, `job_empty:<job>` | `send_job_warning` in the scrape and Betfair jobs | warning |
| `stale_data` | `HealthMonitor` | warning |
| `consecutive_failures` | `HealthMonitor` | error |
| `quota_low`, `quota_critical` | `HealthMonitor` | warning, critical |
| `data_quality_errors` | `HealthMonitor` | warning |
| `missing_heartbeat:<job>` | `HealthMonitor` | warning |
| `health_check_failure` | `HealthMonitor` | critical |

The default window is 30 minutes (`check_rate_limit`'s argument for jobs, `AlertConfig.alert_rate_limit_minutes` for the monitor).

## Heartbeats

On clean exit a job inside `job_alert_context(name)` inserts `heartbeat:<name>`. `check_job_heartbeats` compares the newest row per key in `AlertConfig.heartbeat_expectations` to its allowed hours. A job with no history is not alerted. `purge_old_heartbeats` deletes `heartbeat:%` rows older than `heartbeat_retention_days` (7), on every health check. It is the only retention sweep in the system.

## Hazards

- Severity is not in the rate-limit key: a warning suppresses a critical of the same `alert_type` for the window. The quota pair works only because the types differ.
- `fetch-odds` quota alerts are sent without an `alert_type`, so they are never recorded here; only `HealthMonitor` writes `quota_low` and `quota_critical` rows ([job-fetch-odds](job-fetch-odds.md)). The `docs/REPORTING.md` taxonomy lists `fetch_odds` as a source of those types.
- Heartbeats are distinguished from sent alerts only by the string prefix `heartbeat:`, which `purge_old_heartbeats` also filters on (`startswith`). Nothing enforces "heartbeat rows are silent, alert rows were sent".
- `heartbeat_expectations` keys must match the compound job names passed to `job_alert_context`. A rename leaves a stale key, and since a missing history is not alerted, the check passes silently. Default keys: `fetch-oddsportal-epl` and `-mlb` (2 h), `fetch-oddsportal-results-epl` and `-mlb` (26 h), `daily-digest-epl` (26 h), `check-health` (2 h).
- A row is recorded even when the Discord POST failed.

## Related

- [Reporting surfaces](reporting-surfaces.md), [table fetch_logs](table-fetch-logs.md).
