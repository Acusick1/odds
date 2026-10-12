---
type: component
title: fetch-odds job
summary: "The per-sport Odds API job: event sync from the free /events endpoint, game-aware gate, ingest, tier-based reschedule and quota alerts."
sources:
  - docs/ARCHITECTURE.md#event-sync-embedded-in-fetch-odds
  - docs/ARCHITECTURE.md#fetch-odds-job
  - docs/DEBUGGING_SCHEDULER.md#5-read-the-fetch-odds-logs
  - docs/DEPLOYMENT.md#job-deployment-matrix
  - packages/odds-lambda/odds_lambda/jobs/fetch_odds.py::main
  - packages/odds-lambda/odds_lambda/event_sync.py::EventSyncService
  - packages/odds-lambda/odds_lambda/ingestion.py::OddsIngestionService
verified_at: 7887039
---

# fetch-odds job

`packages/odds-lambda/odds_lambda/jobs/fetch_odds.py::main(ctx)` is the Odds API collector. On AWS it is a per-sport rule on the scheduler Lambda (`fetch-odds-epl`); it is not in the default local `bootstrap_jobs` ([deployment matrix](job-deployment-matrix.md)). `sports` is `[ctx.sport]` or `data_collection.sports`.

## Flow

1. **Event sync**: `EventSyncService.sync_sports` (`packages/odds-lambda/odds_lambda/event_sync.py::EventSyncService`) calls the free `/events` endpoint and bulk-upserts events; a failure is logged (`event_sync_failed`) and ignored. It runs on every invocation, before the gate, so the scheduler knows about fixtures ([odds API client](odds-api-client.md), [event lifecycle](event-lifecycle.md)).
2. **Gate**: `decide_forward_resilient(sports, CADENCE, lookahead_days=settings.scheduler.lookahead_days)` ([decision engine](scheduling-decision-engine.md)). `CADENCE` is 0.5/3/12/24/48 h by tier and 24 h with no game.
3. **Pre-schedule** under the compound name, then return if the gate says skip ([self-scheduling](self-scheduling.md)).
4. **Ingest**: `OddsIngestionService.ingest_sports` fetches `/odds` per sport, writes raw and normalised rows, and one `FetchLog` row ([odds ingestion flow](odds-ingestion-flow.md), [table fetch_logs](table-fetch-logs.md)).
5. **Quota alerts**: per sport, `quota_remaining / settings.api.quota` below `quota_critical_threshold` sends `send_critical`; below `quota_warning_threshold` sends `send_warning` ([odds API quota](odds-api-quota.md)).
6. **Post-schedule** after re-querying the next kickoff.

Log events to grep (from `docs/DEBUGGING_SCHEDULER.md#5-read-the-fetch-odds-logs`): `fetch_odds_job_started`, `fetch_odds_executing`, `odds_snapshot_stored`, `fetch_odds_next_scheduled`, `eventbridge_rule_updated`.

## Hazards

- The job is not wrapped in `job_alert_context`: it writes no heartbeat, and `heartbeat_expectations` has no `fetch-odds` entry, so a stalled chain is caught only by stale-data checks ([health monitor](health-monitor.md)). On failure it sends an unrate-limited `send_critical` and re-raises, and the [Lambda handler](lambda-handler.md) then sends a second one.
- Quota alerts bypass `alert_history`: they fire on every run while below threshold, up to every 30 minutes in the closing tier ([table alert_history](table-alert-history.md)).
- Without a `sport`, the job runs all configured sports and reschedules under the bare `fetch-odds` name, which matches no EventBridge rule ([lambda handler](lambda-handler.md)).
- `FetchLog.success` is false only when some event failed to ingest; an API error raises before any row is written.
- The tiers written onto snapshots come from the writer, not from the gate ([fetch tier](fetch-tier.md)).

## Related

- [Job registry](job-registry.md), [debug the AWS scheduler](howto-debug-aws-scheduler.md).
