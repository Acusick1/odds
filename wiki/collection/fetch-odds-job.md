---
type: reference
title: The fetch-odds job
description: Event sync, gate, pre/post-schedule, quota alerts and log events of odds_lambda.jobs.fetch_odds.
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

`odds_lambda.jobs.fetch_odds.main(ctx)` is the scheduled caller of [the ingestion service](odds-ingestion-service.md). It is in the default local scheduler `bootstrap_jobs` and runs on the mini-PC as `fetch-odds-epl`: its `odds_lambda.scheduling.jobs._JOB_SPORTS` allowlist limits it to EPL to save Odds API quota (the other bootstrapped collectors, including `fetch-polymarket-odds`, spend none; [job-deployment-matrix](../scheduling/job-deployment-matrix.md#where-each-job-runs), [odds-api-quota](odds-api-quota.md)). Its AWS rule is dormant.

## Flow

It runs `sports` as `[ctx.sport]` or `data_collection.sports`:

1. **Event sync**: `odds_lambda.event_sync.EventSyncService.sync_sports` calls the free `/events` endpoint and bulk-upserts events, then merges every `op_`, `op_live_` or `fduk_` event for the same match into each synced event (`odds_lambda.storage.event_merge.EventMerger.adopt_twins`, [Event merge](../storage/event-merge.md)); a failure is logged (`event_sync_failed`) and ignored. It runs on every invocation, before the gate, so `odds_lambda.scheduling.helpers.get_next_kickoff` sees fixtures. `odds_lambda.storage.writers.OddsWriter.bulk_upsert_events` leaves the status and scores of existing events alone ([events](../storage/events.md#upsert-verbs)).
2. **Gate**: `decide_forward_resilient(sports, CADENCE, lookahead_days=settings.scheduler.lookahead_days)`. The cadence is `odds_lambda.jobs.fetch_odds.CADENCE` ([decision-engine](../scheduling/decision-engine.md#cadence)). fetch-odds is the only job that takes `lookahead_days` from `odds_core.config.SchedulerConfig`, and it has no overnight window.
3. **Pre-schedule** under the compound name, then return if the gate says skip.
4. **Ingest**: `odds_lambda.ingestion.OddsIngestionService.ingest_sports`.
5. **Quota alerts**: per sport, `quota_remaining / settings.api.quota` below `quota_critical_threshold` sends `odds_core.alerts.send_critical`; below `quota_warning_threshold` sends `odds_core.alerts.send_warning` ([odds-api-quota](odds-api-quota.md)). These calls use no `alert_type`, so they repeat on every run below threshold and are not recorded in `alert_history` (#442).
6. **Empty-run check**: `odds_lambda.collection_calendar.require_rows_for_fixtures` raises `odds_lambda.collection_calendar.EmptyCollectionError` for an EPL run that stored no snapshot while a SCHEDULED fixture lies within 48 hours (`odds_lambda.jobs.fetch_odds.SOURCE`; sports it does not cover are skipped); the failure handler below sends the critical.
7. **Post-schedule** after re-querying the next kickoff.

## Failure behaviour

- The job is not wrapped in `odds_core.alerts.job_alert_context`, so it writes no heartbeat and `heartbeat_expectations` has no `fetch-odds` entry (#438; [alerting-and-heartbeats](../operations/alerting-and-heartbeats.md#heartbeats)).
- On failure it sends an unrate-limited `odds_core.alerts.send_critical` and re-raises; on AWS the Lambda handler sends a second critical (#442; [lambda-handler](../scheduling/scheduler-backends.md#lambda-handler)).
- Invoked without a `sport` (for example `odds scheduler run-once fetch-odds`), the job runs all configured sports, MLB included, and reschedules under the bare `fetch-odds` name, which matches no bootstrapped schedule or EventBridge rule (#470).

## Log events

`fetch_odds_job_started`, `fetch_odds_executing` and `fetch_odds_next_scheduled` (`odds_lambda.jobs.fetch_odds.main`), `odds_snapshot_stored` (`odds_lambda.storage.writers.OddsWriter.store_odds_snapshot`) and `eventbridge_rule_updated` (`odds_lambda.scheduling.backends.aws.AWSEventBridgeBackend._schedule_with_retry`).

## Related

- [Odds API ingestion service](odds-ingestion-service.md)
- [Self-scheduling](../scheduling/self-scheduling.md)
- [Debug the AWS self-scheduling jobs](../scheduling/howto-debug-aws-scheduler.md)
