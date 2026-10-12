---
type: process
title: Odds API ingestion and the fetch-odds job
description: "How one Odds API fetch becomes stored rows through OddsIngestionService, what the fetch-odds job adds (event sync, gate, scheduling, quota alerts), and what fetch_logs records and omits."
sources:
  - packages/odds-lambda/odds_lambda/ingestion.py::OddsIngestionService
  - packages/odds-lambda/odds_lambda/jobs/fetch_odds.py::main
  - packages/odds-lambda/odds_lambda/event_sync.py::EventSyncService
  - packages/odds-lambda/odds_lambda/jobs/fetch_scores.py::_fetch_and_update_scores
  - packages/odds-cli/odds_cli/commands/fetch.py::_fetch_current
  - packages/odds-lambda/odds_lambda/storage/writers.py::OddsWriter
  - packages/odds-core/odds_core/models.py::FetchLog
  - docs/ARCHITECTURE.md#data-collection-pipeline
  - docs/ARCHITECTURE.md#event-sync-embedded-in-fetch-odds
  - docs/ARCHITECTURE.md#fetch-odds-job
  - docs/CLI.md#fetch-current-odds
  - docs/DATABASE.md#core-tables
  - docs/DEBUGGING_SCHEDULER.md#5-read-the-fetch-odds-logs
  - docs/DEPLOYMENT.md#job-deployment-matrix
  - docs/REPORTING.md#8-storage-backing-reports
verified: "7887039"
status: stable
---

# Odds API ingestion and the fetch-odds job

`OddsIngestionService` (`packages/odds-lambda/odds_lambda/ingestion.py::OddsIngestionService`) is the single path from [the Odds API client](odds-api-client-and-quota.md) to the database. It has two callers: the fetch-odds job (`packages/odds-lambda/odds_lambda/jobs/fetch_odds.py::main`) and `odds fetch current` (`packages/odds-cli/odds_cli/commands/fetch.py::_fetch_current`). On AWS the job is a per-sport rule on the scheduler Lambda (`fetch-odds-epl`); it is not in the default local `bootstrap_jobs`. Verify with `odds scheduler list-jobs --backend aws` ([jobs-and-deployment-matrix](../scheduling/jobs-and-deployment-matrix.md)).

## Steps per sport (`ingest_sport`)

1. `client.get_odds(...)` with regions, markets, bookmakers from `DataCollectionConfig` ([sources-and-keys](sources-and-keys.md)).
2. For each event, in its own session and commit: `OddsWriter.upsert_event`, then `store_odds_snapshot` ([odds-storage](../storage/odds-storage.md)). That stores the raw JSON `OddsSnapshot` and normalized `Odds` rows, with `fetch_tier` and `hours_until_commence` computed from `Event.commence_time` ([fetch-tier](../storage/fetch-tier.md)).
3. Validation runs inside `store_odds_snapshot` when `data_quality.enable_validation` (or the `validate` argument) is set; warnings become `DataQualityLog` rows ([data-validation](../storage/data-validation.md)).
4. One `FetchLog` row via `OddsWriter.log_fetch`, then a `SportIngestionResult` ([fetch_logs](#fetch_logs)).

`ingest_sports` loops sports sequentially. `OddsIngestionCallbacks` (`on_fetch_complete`, `on_event_processed`, `on_event_failed`) let callers instrument without subclassing; the CLI uses them for progress output.

## The fetch-odds job

`fetch_odds.main(ctx)` runs `sports` as `[ctx.sport]` or `data_collection.sports`:

1. **Event sync**: `EventSyncService.sync_sports` (`packages/odds-lambda/odds_lambda/event_sync.py::EventSyncService`) calls the free `/events` endpoint and bulk-upserts events; a failure is logged (`event_sync_failed`) and ignored. It runs on every invocation, before the gate, so `get_next_kickoff` sees fixtures. `bulk_upsert_events` overwrites status and scores of existing events ([events](../storage/events.md#hazards), [event-lifecycle](event-lifecycle.md)).
2. **Gate**: `decide_forward_resilient(sports, CADENCE, lookahead_days=settings.scheduler.lookahead_days)`. `CADENCE` is 0.5/3/12/24/48 h by tier and 24 h with no game ([fetch-tier](../storage/fetch-tier.md#where-tiers-are-written-and-read)). fetch-odds is the only job that takes `lookahead_days` from `SchedulerConfig`, and it has no overnight window ([self-scheduling](../scheduling/self-scheduling.md)).
3. **Pre-schedule** under the compound name, then return if the gate says skip.
4. **Ingest**: `OddsIngestionService.ingest_sports`.
5. **Quota alerts**: per sport, `quota_remaining / settings.api.quota` below `quota_critical_threshold` sends `send_critical`; below `quota_warning_threshold` sends `send_warning` ([quota](odds-api-client-and-quota.md#where-usage-is-recorded-and-alerted)).
6. **Post-schedule** after re-querying the next kickoff.

Log events to grep (from `docs/DEBUGGING_SCHEDULER.md#5-read-the-fetch-odds-logs`): `fetch_odds_job_started`, `fetch_odds_executing`, `odds_snapshot_stored`, `fetch_odds_next_scheduled`, `eventbridge_rule_updated`.

### What the CLI does instead

`odds fetch current` is only the service call and a summary. `--sport` defaults to `basketball_nba` (not a configured sport), and there is no event sync, scheduling or alert. It exits 1 on any exception.

## fetch_logs

`packages/odds-core/odds_core/models.py::FetchLog` (`fetch_logs`) is one row per fetch operation: `fetch_time` (indexed, UTC), `sport_key`, `events_count`, `bookmakers_count`, `success`, `error_message`, `api_quota_remaining`, `response_time_ms`.

**Writers.**
- `OddsIngestionService.ingest_sport` adds one row per sport after the events are processed: `success` is true when no event failed ingest, `bookmakers_count` is the configured bookmaker count, and quota and response time come from the API response.
- `fetch-betfair-exchange` adds a sport-level row when snapshots were stored or events seen: `success = snapshots_stored > 0`, `bookmakers_count=1`, no quota ([betfair-exchange](betfair-exchange.md)).
- Scores, events, OddsPortal and Polymarket jobs write nothing here; Polymarket has its own fetch-log table ([polymarket](polymarket.md)).

**Readers.**
- `odds status show` takes the latest row; `odds status quota` the latest 10 ([reporting-and-daily-digest](../operations/reporting-and-daily-digest.md)).
- `OddsReader.get_database_stats`: 24 h success rate and the newest non-null `api_quota_remaining`.
- `HealthMonitor.check_consecutive_failures` reads the latest 10 `FetchLog` rows of any source, Betfair included, and `check_api_quota` ([alerts-and-health](../operations/alerts-and-health.md)).

## Hazards

- **Failed fetches leave no `FetchLog`.** `log_fetch` runs only after `get_odds` returns, so an API error raises before any row exists. `FetchLog.error_message` is never set by the service; `success` is false only when per-event failures occurred, so an outage leaves no failure record.
- **Mixed sources in the log.** Betfair rows carry no quota and mix into the "latest 10" and the 24 h success rate; the quota reader skips null values by design.
- **`bookmakers_count` is the configured count**, `len(settings.data_collection.bookmakers)`, not the number in the response.
- **Per-event failures are swallowed** into `SportIngestionResult.failures`; the job logs them but only raises if the whole fetch fails.
- **No heartbeat.** The job is not wrapped in `job_alert_context`, so it writes no heartbeat and `heartbeat_expectations` has no `fetch-odds` entry; only stale-data checks catch a stalled chain ([alerts-and-health](../operations/alerts-and-health.md#heartbeats)). On failure it sends an unrate-limited `send_critical` and re-raises, and the [Lambda handler](../scheduling/scheduler-backends.md) then sends a second unconditional critical.
- **Quota alerts bypass `alert_history`**: they fire on every run while below threshold, up to every 30 minutes in the closing tier.
- **Bare-name reschedule.** Invoked without a `sport`, the job runs all configured sports and reschedules under the bare `fetch-odds` name, which matches no EventBridge rule.
- **Docs misattribute validation.** `docs/REPORTING.md#8-storage-backing-reports` attributes `data_quality_logs` writes to `OddsIngestionService`; they come from `OddsWriter.store_odds_snapshot`.
- **CLI duplication.** `odds fetch scores` does not share a service with its job: `_fetch_scores` duplicates the update loop in `packages/odds-lambda/odds_lambda/jobs/fetch_scores.py::_fetch_and_update_scores`. The job loops sports and alerts; the CLI takes one sport ([event-lifecycle](event-lifecycle.md)).
- Deployed fetch schedule is not verifiable from the repo: `odds scheduler list-jobs --backend aws`.

## Related

- [Odds API client, key rotation and quota](odds-api-client-and-quota.md)
- [Self-scheduling](../scheduling/self-scheduling.md)
- [Debug the AWS self-scheduling jobs](../scheduling/howto-debug-aws-scheduler.md)
- [Event lifecycle and results jobs](event-lifecycle.md)
