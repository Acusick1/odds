---
type: explanation
title: How failures are detected
description: How event-driven alerts, heartbeats and the polling health monitor fit together, which jobs each covers, and the blind spots.
sources:
  - odds_core.alerts.job_alert_context
  - odds_core.alerts.send_job_warning
  - odds_core.config.AlertConfig.heartbeat_expectations
  - odds_lambda.health_monitor.HealthMonitor
  - odds_lambda.jobs.check_health.main
status: stable
---

# How failures are detected

Three mechanisms watch the system, and each sees a different failure.

| Mode | Mechanism | Sees |
|---|---|---|
| Event-driven | `odds_core.alerts.job_alert_context` sends `job_failure:<job>` and `odds_core.alerts.send_job_warning` sends soft-failure warnings ([alerts](alerts.md)) | a job that raised or came back empty |
| Heartbeat | `heartbeat:<job>` rows written on clean exit, compared to the per-job allowed hours in `odds_core.config.AlertConfig.heartbeat_expectations` | a job that stopped running |
| Poll-driven | `odds_lambda.health_monitor.HealthMonitor` ([health-monitor](health-monitor.md)) | stale snapshots, fetch failures, low quota, data-quality errors |

Event-driven alerts fire only if the job runs far enough to raise. A job that never starts, or whose chain of self-scheduled runs ended, raises nothing, which is what heartbeats and the poll cover. The poll covers data symptoms (no new snapshots within the stale-data threshold) that do not depend on a single job writing a row. (Inferred: the split exists because self-scheduling ends a chain silently when a run dies; see [self-scheduling](../scheduling/self-scheduling.md#failure-policy-by-job).)

## Heartbeats

On clean exit a job inside `job_alert_context(name)` inserts `heartbeat:<name>`. The check-health job compares the newest row per expectation key to its allowed hours and purges old heartbeat rows; the defaults and mechanics are on [health-monitor](health-monitor.md#heartbeats). Nothing checks that a job's `odds_core.alerts.job_alert_context` name matches a `heartbeat_expectations` key, and a key with no history is not alerted, so a rename leaves a stale key and the check passes silently (#384).

Which jobs write a heartbeat, and their default expectations, are in [health-monitor](health-monitor.md#heartbeats).

fetch-odds, fetch-scores and the NBA Polymarket jobs send hand-rolled criticals on failure instead of a heartbeat, so only the stale-data check catches a stalled chain ([fetch-odds-job](../collection/fetch-odds-job.md), [results-jobs](../collection/results-jobs.md), [polymarket](../collection/polymarket.md)).

## Blind spots

- All three mechanisms run on the mini-PC, the only production host. If the host or its WSL guest stops, nothing runs to raise an alert; #489 tracks an external dead-man's switch ([ADR-0008](../../docs/adr/0008-mini-pc-is-production.md)).
- Delivery depends on the environment that runs the job: on the dormant AWS stack, the scraper function's Terraform sets neither `ALERT_ENABLED` nor `DISCORD_WEBHOOK_URL`, so its alerts are off (#443; [aws-infrastructure](aws-infrastructure.md)).
- The fetch-failure streak reads `fetch_logs`, which records no failed API call ([health-monitor](health-monitor.md)).
- Discord is the only channel and a failed post is still recorded, so a failed delivery suppresses retries for the window ([alerts](alerts.md#delivery)).
- The Lambda handler adds a second critical on top of the job's own ([lambda-handler](../scheduling/scheduler-backends.md#lambda-handler)).

## Related

- [Alerts and the alert_history table](alerts.md)
- [Health monitor and check-health job](health-monitor.md)
- [Reporting surfaces](reporting-surfaces.md)
