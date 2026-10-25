---
type: explanation
title: How failures are detected
description: How event-driven alerts, heartbeats and the polling health monitor fit together, which jobs each covers, and the blind spots.
sources:
  - odds_core.alerts.job_alert_context
  - odds_core.alerts.send_job_warning
  - odds_core.config.AlertConfig.heartbeat_expectations
  - odds_lambda.health_monitor.HealthMonitor
  - odds_lambda.collection_calendar.require_rows_for_fixtures
  - odds_lambda.collection_calendar.EmptyCollectionError
  - odds_lambda.jobs.check_health.main
status: stable
---

# How failures are detected

Three mechanisms watch the system, and each sees a different failure.

| Mode | Mechanism | Sees |
|---|---|---|
| Event-driven | `odds_core.alerts.job_alert_context` sends `job_failure:<job>` once per outage (first failure since the last heartbeat, then each reminder interval) through `odds_core.alerts.send_job_failure` ([a failing send never masks the job's exception](alerts.md#delivery)) `odds_core.alerts.send_job_warning` sends soft-failure warnings, and `odds_core.alerts.send_scheduling_failure` reports a job that could not book its next run, reminded rather than per run ([alerts](alerts.md)) | a job that raised or wrote nothing while fixtures were scheduled |
| Heartbeat | `heartbeat:<job>` rows written on clean exit, compared to the per-job allowed hours (overnight pause or daily cadence, plus slack) in `odds_core.config.AlertConfig.heartbeat_expectations` | a job that stopped running |
| Poll-driven | the checks of `odds_lambda.health_monitor.HealthMonitor` (stale sources, quota, data quality, heartbeats, event identity, unsettled events, schedule chains), run hourly by `check-health` ([health-monitor](health-monitor.md)) | a source that stopped writing snapshots while its job is not itself failing, an MLB schedule job that stopped completing (a missing heartbeat), EPL events whose live Betfair close was not collected (reported after kick-off), low quota, data-quality errors, duplicate events, MLB events without a `gamePk` that are not marked as having none, EPL and MLB events left pending long after kick-off, results runs that settle nothing, self-scheduling jobs with no run booked or an overdue one |

A persisting poll-driven condition alerts when it starts and then at most every `odds_core.config.AlertConfig.stale_reminder_minutes`, the interval a job outage also reminds at; both go through `odds_core.alerts.send_recorded_alert` and neither sends a message when it clears; a failed send is not recorded ([alerts](alerts.md#alert-history)).

Event-driven alerts fire only if the job runs far enough to raise. A job that never starts, or whose chain of self-scheduled runs ended, raises nothing, which is what heartbeats and the poll cover; the poll's [schedule chain check](health-monitor.md#schedule-chain-check) sees an ended chain directly. None of these alerts is sent, or recorded, under `SCHEDULER_DRY_RUN` (`odds_core.config.Settings.alerts_deliverable`). The poll covers data symptoms (no new snapshots from a source within its cadence) that do not depend on a single job writing a row. (Inferred: the split exists because self-scheduling ends a chain silently when a run dies; see [self-scheduling](../scheduling/self-scheduling.md#failure-policy-by-job).)

## Heartbeats

On clean exit a job inside `job_alert_context(name)` inserts `heartbeat:<name>`. The check-health job compares the newest row per expectation key to its allowed hours and purges old heartbeat rows; the defaults and mechanics are on [health-monitor](health-monitor.md#heartbeats). Nothing checks that a job's `odds_core.alerts.job_alert_context` name matches a `heartbeat_expectations` key, and a key with no history is not alerted, so a rename leaves a stale key and the check passes silently (#384).

Which jobs write a heartbeat (the EPL Polymarket job included), and their default expectations, are in [health-monitor](health-monitor.md#heartbeats).

The NBA Polymarket job (`fetch-polymarket`) is inside `odds_core.alerts.job_alert_context` but is not in the default bootstrap list, so it has no heartbeat expectation ([polymarket](../collection/polymarket.md)).

## Empty collection

A collection job that has run cleanly can still have collected nothing. `odds_lambda.collection_calendar.require_rows_for_fixtures` closes that gap for the OddsPortal, Betfair Exchange and Odds API jobs: when a sport got zero snapshots in a run while the database has a SCHEDULED fixture of that sport inside the job's fixture horizon (`odds_lambda.collection_calendar.CollectionSource`), it raises `odds_lambda.collection_calendar.EmptyCollectionError`. OddsPortal and Betfair raise inside `odds_core.alerts.job_alert_context`, so the run sends the critical `job_failure:<job>` and writes no heartbeat; fetch-odds raises inside its own handler, which sends a critical. A run with no fixtures in the horizon (international break, off-season) passes and writes its heartbeat. The Betfair job passes the keys of fixtures it identified but rightly did not price (a market in play, past its Betfair start, or suspended with no prices; `odds_lambda.collection_calendar.require_rows_for_fixtures` `accounted_by_sport`) and the calendar leaves them out, so a run whose only fixtures are in play passes while another fixture with no rows still raises. The OddsPortal job also fails when a league raised, instead of writing a heartbeat after logging the error. The Polymarket odds job keeps its own listing check ([polymarket-odds-snapshots](../collection/polymarket-odds-snapshots.md#the-fetch-polymarket-odds-job)), and the results jobs (`fetch-oddsportal-results`, `fetch-mlb-results`) fail when updating an event in the database raised ([results-jobs](../collection/results-jobs.md#run-alerts-and-booking)). A failing Betfair or OddsPortal run puts its own errors in the message. The poll-side counterpart is the [per-source stale-data check](health-monitor.md#stale-data-check).

## Blind spots

- All three mechanisms run on the production host, and nothing else runs scheduled jobs. If the host or the scheduler stops, nothing runs to raise an alert; #489 tracks an external dead-man's switch ([ADR-0008](../../docs/adr/0008-mini-pc-is-production.md)).
- Delivery depends on the environment that runs the job: on the dormant AWS stack, see the alert variables per function on [aws-infrastructure](aws-infrastructure.md).
- `fetch_logs` records no failed API call, and no check reads it for outages: a fetch-odds failure reaches Discord only through the job's own critical and the stale-data check ([fetch logs](../collection/odds-ingestion-service.md#fetch-logs)).
- Discord is the only channel and a failed post is still recorded, so a failed delivery suppresses retries for the window ([alerts](alerts.md#delivery)).
- The Lambda handler adds a second critical on top of the job's own ([lambda-handler](../scheduling/scheduler-backends.md#lambda-handler)).

## Related

- [Alerts and the alert_history table](alerts.md)
- [Health monitor and check-health job](health-monitor.md)
- [Reporting surfaces](reporting-surfaces.md)
