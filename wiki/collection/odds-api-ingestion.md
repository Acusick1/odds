---
type: process
title: Odds API ingestion and the fetch-odds job
description: "How one Odds API fetch becomes stored rows through OddsIngestionService, what the fetch-odds job adds (event sync, gate, scheduling, quota alerts), and what fetch_logs records and omits."
sources:
  - odds_lambda.ingestion.OddsIngestionService
  - odds_lambda.jobs.fetch_odds.main
  - odds_lambda.event_sync.EventSyncService
  - odds_lambda.jobs.fetch_scores._fetch_and_update_scores
  - odds_cli.commands.fetch._fetch_current
  - odds_lambda.storage.writers.OddsWriter
  - odds_core.models.FetchLog
  - deployment/aws/terraform/eventbridge.tf
  - odds_core.config.SchedulerConfig
  - odds_lambda.scheduling.backends.aws.AWSEventBridgeBackend
status: stable
---

# Odds API ingestion and the fetch-odds job

`odds_lambda.ingestion.OddsIngestionService` is the single path from [the Odds API client](odds-api-client-and-quota.md) to the database. It has two callers: the fetch-odds job (`odds_lambda.jobs.fetch_odds.main`) and `odds fetch current` (`odds_cli.commands.fetch._fetch_current`). On AWS the job is a per-sport rule on the scheduler Lambda (`fetch-odds-epl`); it is not in the default local `bootstrap_jobs`. Verify with `odds scheduler list-jobs --backend aws` ([jobs-and-deployment-matrix](../scheduling/jobs-and-deployment-matrix.md)).

## Steps per sport (`odds_lambda.ingestion.OddsIngestionService.ingest_sport`)

1. `client.get_odds(...)` with regions, markets, bookmakers from `odds_core.config.DataCollectionConfig` ([sources-and-keys](sources-and-keys.md)).
2. For each event, in its own session and commit: `odds_lambda.storage.writers.OddsWriter.upsert_event`, then `store_odds_snapshot` ([odds-storage](../storage/odds-storage.md)). That stores the raw JSON `odds_core.models.OddsSnapshot` and normalized `odds_core.models.Odds` rows, with `fetch_tier` and `hours_until_commence` computed from `odds_core.models.Event.commence_time` ([fetch-tier](../storage/fetch-tier.md)).
3. Validation runs inside `store_odds_snapshot` when `data_quality.enable_validation` (or the `validate` argument) is set; warnings become `odds_core.models.DataQualityLog` rows ([data-validation](../storage/data-validation.md)).
4. One `odds_core.models.FetchLog` row via `odds_lambda.storage.writers.OddsWriter.log_fetch`, then a `odds_lambda.ingestion.SportIngestionResult` ([fetch_logs](#fetch_logs)).

`ingest_sports` loops sports sequentially. `odds_lambda.ingestion.OddsIngestionCallbacks` (`on_fetch_complete`, `on_event_processed`, `on_event_failed`) let callers instrument without subclassing; the CLI uses them for progress output.

## The fetch-odds job

`fetch_odds.main(ctx)` runs `sports` as `[ctx.sport]` or `data_collection.sports`:

1. **Event sync**: `odds_lambda.event_sync.EventSyncService.sync_sports` (`odds_lambda.event_sync.EventSyncService`) calls the free `/events` endpoint and bulk-upserts events; a failure is logged (`event_sync_failed`) and ignored. It runs on every invocation, before the gate, so `odds_lambda.scheduling.helpers.get_next_kickoff` sees fixtures. `bulk_upsert_events` overwrites status and scores of existing events ([events](../storage/events.md#hazards), [event-lifecycle](event-lifecycle.md)).
2. **Gate**: `decide_forward_resilient(sports, CADENCE, lookahead_days=settings.scheduler.lookahead_days)`. `odds_lambda.jobs.fetch_odds.CADENCE` is 0.5/3/12/24/48 h by tier and 24 h with no game ([fetch-tier](../storage/fetch-tier.md#where-tiers-are-written-and-read)). fetch-odds is the only job that takes `lookahead_days` from `odds_core.config.SchedulerConfig`, and it has no overnight window ([self-scheduling](../scheduling/self-scheduling.md)).
3. **Pre-schedule** under the compound name, then return if the gate says skip.
4. **Ingest**: `odds_lambda.ingestion.OddsIngestionService.ingest_sports`.
5. **Quota alerts**: per sport, `quota_remaining / settings.api.quota` below `quota_critical_threshold` sends `odds_core.alerts.send_critical`; below `quota_warning_threshold` sends `odds_core.alerts.send_warning` ([quota](odds-api-client-and-quota.md#where-usage-is-recorded-and-alerted)).
6. **Post-schedule** after re-querying the next kickoff.

Log events to grep: `fetch_odds_job_started`, `fetch_odds_executing` and `fetch_odds_next_scheduled` (`odds_lambda.jobs.fetch_odds.main`), `odds_snapshot_stored` (`odds_lambda.storage.writers.OddsWriter.store_odds_snapshot`) and `eventbridge_rule_updated` (`odds_lambda.scheduling.backends.aws.AWSEventBridgeBackend._schedule_with_retry`).

### What the CLI does instead

`odds fetch current` is only the service call and a summary. `--sport` defaults to `basketball_nba` (not a configured sport), and there is no event sync, scheduling or alert. It exits 1 on any exception.

## fetch_logs

`odds_core.models.FetchLog` (`fetch_logs`) is one row per fetch operation: `fetch_time` (indexed, UTC), `sport_key`, `events_count`, `bookmakers_count`, `success`, `error_message`, `api_quota_remaining`, `response_time_ms`.

**Writers.**
- `odds_lambda.ingestion.OddsIngestionService.ingest_sport` adds one row per sport after the events are processed: `success` is true when no event failed ingest, `bookmakers_count` is the configured bookmaker count, and quota and response time come from the API response.
- `fetch-betfair-exchange` adds a sport-level row when snapshots were stored or events seen: `success = snapshots_stored > 0`, `bookmakers_count=1`, no quota ([betfair-exchange](betfair-exchange.md)).
- Scores, events, OddsPortal and Polymarket jobs write nothing here; Polymarket has its own fetch-log table ([polymarket](polymarket.md)).

**Readers.**
- `odds status show` takes the latest row; `odds status quota` the latest 10 ([reporting-and-daily-digest](../operations/reporting-and-daily-digest.md)).
- `odds_lambda.storage.readers.OddsReader.get_database_stats`: 24 h success rate and the newest non-null `api_quota_remaining`.
- `odds_lambda.health_monitor.HealthMonitor.check_consecutive_failures` reads the latest 10 `odds_core.models.FetchLog` rows of any source, Betfair included, and `check_api_quota` ([alerts-and-health](../operations/alerts-and-health.md)).

## Manual event discovery

`odds discover upcoming [--sport]` (`odds_cli.commands.discover.upcoming`) runs the same free `/events` sync as step 1 without fetching odds; it costs 0 quota units. `--sport` defaults to `basketball_nba`, so pass `--sport soccer_epl` for EPL.

## Hazards

- **Failed fetches leave no `odds_core.models.FetchLog`.** `log_fetch` runs only after `get_odds` returns, so an API error raises before any row exists. `odds_core.models.FetchLog.error_message` is never set by the service; `success` is false only when per-event failures occurred, so an outage leaves no failure record.
- **Mixed sources in the log.** Betfair rows carry no quota and mix into the "latest 10" and the 24 h success rate; the quota reader skips null values by design.
- **`bookmakers_count` is the configured count**, `len(settings.data_collection.bookmakers)`, not the number in the response.
- **Per-event failures are swallowed** into `odds_lambda.ingestion.SportIngestionResult.failures`; the job logs them but only raises if the whole fetch fails.
- **No heartbeat.** The job is not wrapped in `odds_core.alerts.job_alert_context`, so it writes no heartbeat and `heartbeat_expectations` has no `fetch-odds` entry; only stale-data checks catch a stalled chain ([alerts-and-health](../operations/alerts-and-health.md#heartbeats)). On failure it sends an unrate-limited `odds_core.alerts.send_critical` and re-raises, and the [Lambda handler](../scheduling/scheduler-backends.md) then sends a second unconditional critical.
- **Quota alerts bypass `alert_history`**: they fire on every run while below threshold, up to every 30 minutes in the closing tier.
- **Bare-name reschedule.** Invoked without a `sport`, the job runs all configured sports and reschedules under the bare `fetch-odds` name, which matches no EventBridge rule.
- **CLI duplication.** `odds fetch scores` does not share a service with its job: `odds_cli.commands.fetch._fetch_scores` duplicates the update loop in `odds_lambda.jobs.fetch_scores._fetch_and_update_scores`. The job loops sports and alerts; the CLI takes one sport ([event-lifecycle](event-lifecycle.md)).
- Deployed fetch schedule is not verifiable from the repo: `odds scheduler list-jobs --backend aws`.

## Related

- [Odds API client, key rotation and quota](odds-api-client-and-quota.md)
- [Self-scheduling](../scheduling/self-scheduling.md)
- [Debug the AWS self-scheduling jobs](../scheduling/howto-debug-aws-scheduler.md)
- [Event lifecycle and results jobs](event-lifecycle.md)
