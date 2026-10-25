---
type: reference
title: The fetch-odds job
description: Event sync, gate, pre/post-schedule and log events of odds_lambda.jobs.fetch_odds.
sources:
  - odds_lambda.jobs.fetch_odds.main
  - odds_lambda.jobs.fetch_odds.CADENCE
  - odds_lambda.jobs.fetch_odds.SOURCE
  - odds_lambda.event_sync.EventSyncService
  - odds_lambda.scheduling.helpers.get_next_kickoff
  - odds_lambda.storage.writers.OddsWriter.store_odds_snapshot
  - odds_lambda.scheduling.backends.aws.AWSEventBridgeBackend
  - odds_core.config.SchedulerConfig
  - odds_lambda.scheduling.jobs._JOB_SPORTS
  - deployment/aws/terraform/eventbridge.tf
status: stable
---

# The fetch-odds job

`odds_lambda.jobs.fetch_odds.main(ctx)` is the scheduled caller of [the ingestion service](odds-ingestion-service.md). It is in the default local scheduler `bootstrap_jobs` and runs on the production host as `fetch-odds-epl`; its `odds_lambda.scheduling.jobs._JOB_SPORTS` allowlist limits it to EPL to save Odds API quota (the other bootstrapped collectors, including `fetch-polymarket-odds`, and `fetch-mlb-results` spend none; [job-deployment-matrix](../scheduling/job-deployment-matrix.md#where-each-job-runs), [odds-api-quota](odds-api-quota.md)). Its AWS rule is dormant.

## Flow

It requires `ctx.sport` (a missing sport raises `ValueError` before any work) and runs for that one sport:

1. **Event sync**: `odds_lambda.event_sync.EventSyncService.sync_sports` calls the free `/events` endpoint and bulk-upserts events, then merges every `op_`, `op_live_` or `fduk_` event for the same match into each synced event (`odds_lambda.storage.event_merge.EventMerger.adopt_twins`, [Event merge](../storage/event-merge.md)); a failure is logged (`event_sync_failed`) and ignored. It runs on every invocation, before the gate, so `odds_lambda.scheduling.helpers.get_next_kickoff` sees fixtures. `odds_lambda.storage.writers.OddsWriter.bulk_upsert_events` leaves the surrogate key, status, scores, MLB `gamePk` and `no_game_pk_reason` of existing events alone ([events](../storage/events.md#upsert-verbs)).
2. **Gate**: `decide_forward_resilient(sports, CADENCE, lookahead_days=settings.scheduler.lookahead_days)`. The cadence is `odds_lambda.jobs.fetch_odds.CADENCE` ([decision-engine](../scheduling/decision-engine.md#cadence)); it polls every 2 h within 3 h of a kickoff and every 3 h from 3 to 12 h to fit the free-tier quota ([odds-api-quota](odds-api-quota.md#scheduled-spend)). fetch-odds is the only job that takes `lookahead_days` from `odds_core.config.SchedulerConfig` (the settings that also name the scheduler backend, dry-run flag and bootstrap jobs), and it has no overnight window.
3. **Pre-schedule** under the compound name; a failed booking sends a `scheduling_warning:<job>` warning when another booking is still to come or has succeeded, and a `scheduling_failure:<job>` critical when none survives (a skipped run's only booking, or a post-fetch booking after a failed pre-schedule; [alerts](../operations/alerts.md#delivery)). The rest of the run is inside [`odds_core.alerts.job_alert_context`](../operations/alerts.md#delivery) under that name, so a gated run also writes a heartbeat.
4. **Skip** if the gate says so; otherwise **ingest**: `odds_lambda.ingestion.OddsIngestionService.ingest_sports`.
5. **Quota alerts**: none; the health monitor is the only writer ([odds-api-quota](odds-api-quota.md)).
6. **Empty-run check**: `odds_lambda.collection_calendar.require_rows_for_fixtures` raises `odds_lambda.collection_calendar.EmptyCollectionError` for an EPL run that stored no snapshot while a SCHEDULED fixture lies within 48 hours (`odds_lambda.jobs.fetch_odds.SOURCE`; sports it does not cover are skipped, and it passes no `accounted_by_sport`); the failure handler below sends the critical.
7. **Post-schedule** after re-querying the next kickoff.

## Failure behaviour

- A raised run sends the `job_failure:fetch-odds-epl` critical once per outage (the first failure since its last heartbeat, then a reminder every `odds_core.config.AlertConfig.stale_reminder_minutes`, 360 min) through `odds_core.alerts.job_alert_context` and re-raises; a clean or gated run writes `heartbeat:fetch-odds-epl`, which `odds_core.config.AlertConfig.heartbeat_expectations` expects every 49 h (the 48 h opening-tier interval plus 1 h; the job does not pause overnight). The stale-source check leaves a source alone while that failure is recent ([health-monitor](../operations/health-monitor.md#stale-data-check)). On AWS the Lambda handler sends a second critical (#442; [lambda-handler](../scheduling/scheduler-backends.md#lambda-handler)).
- On the local scheduler a run past `odds_lambda.scheduling.runner.JOB_TIMEOUT` is cancelled and alerted. A failed run is re-booked 15 minutes later only when it booked nothing, and the pre-schedule has already booked the next tier interval (30 minutes at the closing tier), so a failing API call is retried at the normal cadence, not every 15 minutes ([Local scheduler](../scheduling/local-scheduler.md#scheduling-primitives)).
- A run without a sport raises `ValueError`, and `odds scheduler run-once fetch-odds` fails the same way; use `fetch-odds-epl`.

## Log events

`fetch_odds_job_started`, `fetch_odds_executing`, `fetch_odds_next_scheduled`, `fetch_odds_scheduling_failed` (`odds_lambda.jobs.fetch_odds.main`), `odds_snapshot_stored` (`odds_lambda.storage.writers.OddsWriter.store_odds_snapshot`, one per event, with its odds-row count and fetch tier) and `eventbridge_rule_updated` (`odds_lambda.scheduling.backends.aws.AWSEventBridgeBackend._schedule_with_retry`).

## Related

- [Odds API ingestion service](odds-ingestion-service.md)
- [Self-scheduling](../scheduling/self-scheduling.md)
- [Debug the AWS self-scheduling jobs](../scheduling/howto-debug-aws-scheduler.md)
