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
| Heartbeat | `heartbeat:<job>` rows written on clean exit, compared to `odds_core.config.AlertConfig.heartbeat_expectations` | a job that stopped running |
| Poll-driven | `odds_lambda.health_monitor.HealthMonitor` ([health-monitor](health-monitor.md)) | stale snapshots, fetch failures, low quota, data-quality errors |

Event-driven alerts fire only if the job runs far enough to raise. A job that never starts, or whose chain of self-scheduled runs ended, raises nothing, which is what heartbeats and the poll cover. The poll covers data symptoms (no new snapshots for 2 hours) that do not depend on a single job writing a row. (Inferred: the split exists because self-scheduling ends a chain silently when a run dies; see [self-scheduling](../scheduling/self-scheduling.md#failure-policy-by-job).)

## Heartbeats

On clean exit a job inside `job_alert_context(name)` inserts `heartbeat:<name>`. The check-health job compares the newest row per expectation key to its allowed hours and purges heartbeat rows after 7 days; the defaults and mechanics are on [health-monitor](health-monitor.md). Nothing checks that a job's `odds_core.alerts.job_alert_context` name matches a `heartbeat_expectations` key, and a key with no history is not alerted, so a rename leaves a stale key and the check passes silently (#384).

Which jobs are covered:

| Job | Writes a heartbeat | Has a default expectation |
|---|---|---|
| fetch-oddsportal (per sport) | yes | 2 h |
| fetch-oddsportal-results (per sport) | yes | 26 h |
| daily-digest (EPL) | yes | 26 h |
| check-health | yes | 2 h |
| update-status | yes | no |
| fetch-betfair-exchange | yes | no |
| fetch-odds, fetch-scores, fetch-polymarket, backfill-polymarket | no: not wrapped in `odds_core.alerts.job_alert_context` (#438) | no |

The four jobs without a heartbeat send hand-rolled criticals on failure instead, so only the stale-data check catches a stalled chain ([fetch-odds-job](../collection/fetch-odds-job.md), [results-jobs](../collection/results-jobs.md), [polymarket](../collection/polymarket.md)).

## Blind spots

- Delivery depends on deployed environment: the scraper function's Terraform sets neither `ALERT_ENABLED` nor `DISCORD_WEBHOOK_URL`, so its alerts are off (#443; [aws-infrastructure](aws-infrastructure.md)). Check the deployed function with `aws lambda get-function-configuration --function-name odds-scheduler-scraper`.
- The fetch-failure streak reads `fetch_logs`, which records no failed API call ([health-monitor](health-monitor.md)).
- Discord is the only channel and a failed post is still recorded, so a failed delivery suppresses retries for the window ([alerts](alerts.md#delivery)).
- The Lambda handler adds a second critical on top of the job's own ([lambda-handler](../scheduling/scheduler-backends.md#lambda-handler)).

## Related

- [Alerts and the alert_history table](alerts.md)
- [Health monitor and check-health job](health-monitor.md)
- [Reporting surfaces](reporting-surfaces.md)
