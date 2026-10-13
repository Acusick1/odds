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
  - deployment/aws/terraform/eventbridge.tf
  - packages/odds-core/odds_core/config.py::SchedulerConfig
  - packages/odds-lambda/odds_lambda/scheduling/backends/aws.py::AWSEventBridgeBackend
status: stable
---

# Odds API ingestion and the fetch-odds job

`packages/odds-lambda/odds_lambda/ingestion.py::OddsIngestionService` is the single path from [the Odds API client](odds-api-client-and-quota.md) to the database. It has two callers: the fetch-odds job (`packages/odds-lambda/odds_lambda/jobs/fetch_odds.py::main`) and `odds fetch current` (`packages/odds-cli/odds_cli/commands/fetch.py::_fetch_current`). On AWS the job is a per-sport rule on the scheduler Lambda (`fetch-odds-epl`); it is not in the default local `bootstrap_jobs`. Verify with `odds scheduler list-jobs --backend aws` ([jobs-and-deployment-matrix](../scheduling/jobs-and-deployment-matrix.md)).

## Steps per sport (`packages/odds-lambda/odds_lambda/ingestion.py::OddsIngestionService.ingest_sport`)

1. `client.get_odds(...)` with regions, markets, bookmakers from `packages/odds-core/odds_core/config.py::DataCollectionConfig` ([sources-and-keys](sources-and-keys.md)).
2. For each event, in its own session and commit: `packages/odds-lambda/odds_lambda/storage/writers.py::OddsWriter.upsert_event`, then `store_odds_snapshot` ([odds-storage](../storage/odds-storage.md)). That stores the raw JSON `packages/odds-core/odds_core/models.py::OddsSnapshot` and normalized `packages/odds-core/odds_core/models.py::Odds` rows, with `fetch_tier` and `hours_until_commence` computed from `packages/odds-core/odds_core/models.py::Event.commence_time` ([fetch-tier](../storage/fetch-tier.md)).
3. Validation runs inside `store_odds_snapshot` when `data_quality.enable_validation` (or the `validate` argument) is set; warnings become `packages/odds-core/odds_core/models.py::DataQualityLog` rows ([data-validation](../storage/data-validation.md)).
4. One `packages/odds-core/odds_core/models.py::FetchLog` row via `packages/odds-lambda/odds_lambda/storage/writers.py::OddsWriter.log_fetch`, then a `packages/odds-lambda/odds_lambda/ingestion.py::SportIngestionResult` ([fetch_logs](#fetch_logs)).

`ingest_sports` loops sports sequentially. `packages/odds-lambda/odds_lambda/ingestion.py::OddsIngestionCallbacks` (`on_fetch_complete`, `on_event_processed`, `on_event_failed`) let callers instrument without subclassing; the CLI uses them for progress output.

## The fetch-odds job

`fetch_odds.main(ctx)` runs `sports` as `[ctx.sport]` or `data_collection.sports`:

1. **Event sync**: `packages/odds-lambda/odds_lambda/event_sync.py::EventSyncService.sync_sports` (`packages/odds-lambda/odds_lambda/event_sync.py::EventSyncService`) calls the free `/events` endpoint and bulk-upserts events; a failure is logged (`event_sync_failed`) and ignored. It runs on every invocation, before the gate, so `packages/odds-lambda/odds_lambda/scheduling/helpers.py::get_next_kickoff` sees fixtures. `bulk_upsert_events` overwrites status and scores of existing events ([events](../storage/events.md#hazards), [event-lifecycle](event-lifecycle.md)).
2. **Gate**: `decide_forward_resilient(sports, CADENCE, lookahead_days=settings.scheduler.lookahead_days)`. `packages/odds-lambda/odds_lambda/jobs/fetch_odds.py::CADENCE` is 0.5/3/12/24/48 h by tier and 24 h with no game ([fetch-tier](../storage/fetch-tier.md#where-tiers-are-written-and-read)). fetch-odds is the only job that takes `lookahead_days` from `packages/odds-core/odds_core/config.py::SchedulerConfig`, and it has no overnight window ([self-scheduling](../scheduling/self-scheduling.md)).
3. **Pre-schedule** under the compound name, then return if the gate says skip.
4. **Ingest**: `packages/odds-lambda/odds_lambda/ingestion.py::OddsIngestionService.ingest_sports`.
5. **Quota alerts**: per sport, `quota_remaining / settings.api.quota` below `quota_critical_threshold` sends `packages/odds-core/odds_core/alerts.py::send_critical`; below `quota_warning_threshold` sends `packages/odds-core/odds_core/alerts.py::send_warning` ([quota](odds-api-client-and-quota.md#where-usage-is-recorded-and-alerted)).
6. **Post-schedule** after re-querying the next kickoff.

Log events to grep: `fetch_odds_job_started`, `fetch_odds_executing` and `fetch_odds_next_scheduled` (`packages/odds-lambda/odds_lambda/jobs/fetch_odds.py::main`), `odds_snapshot_stored` (`packages/odds-lambda/odds_lambda/storage/writers.py::OddsWriter.store_odds_snapshot`) and `eventbridge_rule_updated` (`packages/odds-lambda/odds_lambda/scheduling/backends/aws.py::AWSEventBridgeBackend._schedule_with_retry`).

### What the CLI does instead

`odds fetch current` is only the service call and a summary. `--sport` defaults to `basketball_nba` (not a configured sport), and there is no event sync, scheduling or alert. It exits 1 on any exception.

## fetch_logs

`packages/odds-core/odds_core/models.py::FetchLog` (`fetch_logs`) is one row per fetch operation: `fetch_time` (indexed, UTC), `sport_key`, `events_count`, `bookmakers_count`, `success`, `error_message`, `api_quota_remaining`, `response_time_ms`.

**Writers.**
- `packages/odds-lambda/odds_lambda/ingestion.py::OddsIngestionService.ingest_sport` adds one row per sport after the events are processed: `success` is true when no event failed ingest, `bookmakers_count` is the configured bookmaker count, and quota and response time come from the API response.
- `fetch-betfair-exchange` adds a sport-level row when snapshots were stored or events seen: `success = snapshots_stored > 0`, `bookmakers_count=1`, no quota ([betfair-exchange](betfair-exchange.md)).
- Scores, events, OddsPortal and Polymarket jobs write nothing here; Polymarket has its own fetch-log table ([polymarket](polymarket.md)).

**Readers.**
- `odds status show` takes the latest row; `odds status quota` the latest 10 ([reporting-and-daily-digest](../operations/reporting-and-daily-digest.md)).
- `packages/odds-lambda/odds_lambda/storage/readers.py::OddsReader.get_database_stats`: 24 h success rate and the newest non-null `api_quota_remaining`.
- `packages/odds-lambda/odds_lambda/health_monitor.py::HealthMonitor.check_consecutive_failures` reads the latest 10 `packages/odds-core/odds_core/models.py::FetchLog` rows of any source, Betfair included, and `check_api_quota` ([alerts-and-health](../operations/alerts-and-health.md)).

## Manual event discovery

`odds discover upcoming [--sport]` (`packages/odds-cli/odds_cli/commands/discover.py::upcoming`) runs the same free `/events` sync as step 1 without fetching odds; it costs 0 quota units. `--sport` defaults to `basketball_nba`, so pass `--sport soccer_epl` for EPL.

## Hazards

- **Failed fetches leave no `packages/odds-core/odds_core/models.py::FetchLog`.** `log_fetch` runs only after `get_odds` returns, so an API error raises before any row exists. `packages/odds-core/odds_core/models.py::FetchLog.error_message` is never set by the service; `success` is false only when per-event failures occurred, so an outage leaves no failure record.
- **Mixed sources in the log.** Betfair rows carry no quota and mix into the "latest 10" and the 24 h success rate; the quota reader skips null values by design.
- **`bookmakers_count` is the configured count**, `len(settings.data_collection.bookmakers)`, not the number in the response.
- **Per-event failures are swallowed** into `packages/odds-lambda/odds_lambda/ingestion.py::SportIngestionResult.failures`; the job logs them but only raises if the whole fetch fails.
- **No heartbeat.** The job is not wrapped in `packages/odds-core/odds_core/alerts.py::job_alert_context`, so it writes no heartbeat and `heartbeat_expectations` has no `fetch-odds` entry; only stale-data checks catch a stalled chain ([alerts-and-health](../operations/alerts-and-health.md#heartbeats)). On failure it sends an unrate-limited `packages/odds-core/odds_core/alerts.py::send_critical` and re-raises, and the [Lambda handler](../scheduling/scheduler-backends.md) then sends a second unconditional critical.
- **Quota alerts bypass `alert_history`**: they fire on every run while below threshold, up to every 30 minutes in the closing tier.
- **Bare-name reschedule.** Invoked without a `sport`, the job runs all configured sports and reschedules under the bare `fetch-odds` name, which matches no EventBridge rule.
- **CLI duplication.** `odds fetch scores` does not share a service with its job: `packages/odds-cli/odds_cli/commands/fetch.py::_fetch_scores` duplicates the update loop in `packages/odds-lambda/odds_lambda/jobs/fetch_scores.py::_fetch_and_update_scores`. The job loops sports and alerts; the CLI takes one sport ([event-lifecycle](event-lifecycle.md)).
- Deployed fetch schedule is not verifiable from the repo: `odds scheduler list-jobs --backend aws`.

## Related

- [Odds API client, key rotation and quota](odds-api-client-and-quota.md)
- [Self-scheduling](../scheduling/self-scheduling.md)
- [Debug the AWS self-scheduling jobs](../scheduling/howto-debug-aws-scheduler.md)
- [Event lifecycle and results jobs](event-lifecycle.md)
