---
type: reference
title: Odds API ingestion service
description: OddsIngestionService steps and callbacks, the fetch_logs table, and odds fetch current / odds discover upcoming.
sources:
  - odds_lambda.ingestion.OddsIngestionService
  - odds_lambda.ingestion.OddsIngestionCallbacks
  - odds_lambda.ingestion.SportIngestionResult
  - odds_lambda.storage.writers.OddsWriter.upsert_event
  - odds_lambda.storage.writers.OddsWriter.store_odds_snapshot
  - odds_lambda.storage.writers.OddsWriter.log_fetch
  - odds_lambda.storage.readers.OddsReader.get_database_stats
  - odds_lambda.health_monitor.HealthMonitor.check_consecutive_failures
  - odds_core.models.FetchLog
  - odds_cli.commands.fetch._fetch_current
  - odds_cli.commands.discover.upcoming
status: stable
---

# Odds API ingestion service

`odds_lambda.ingestion.OddsIngestionService` is the single path from [the Odds API client](odds-api-client.md) to the database. It has two callers: the fetch-odds job ([fetch-odds-job](fetch-odds-job.md)) and `odds fetch current` (`odds_cli.commands.fetch._fetch_current`).

## Steps per sport

`odds_lambda.ingestion.OddsIngestionService.ingest_sport`:

1. `client.get_odds(...)` with regions, markets, bookmakers from `odds_core.config.DataCollectionConfig` ([sport-and-market-keys](sport-and-market-keys.md#odds-api-scope)).
2. For each event, in its own session and commit: `odds_lambda.storage.writers.OddsWriter.upsert_event`, then `odds_lambda.storage.writers.OddsWriter.store_odds_snapshot` ([odds-writer](../storage/odds-writer.md)). That stores the raw JSON `odds_core.models.OddsSnapshot` and normalized `odds_core.models.Odds` rows, with `fetch_tier` and `hours_until_commence` computed from `odds_core.models.Event.commence_time` ([fetch-tier](../storage/fetch-tier.md)).
3. Validation runs inside `odds_lambda.storage.writers.OddsWriter.store_odds_snapshot` when `data_quality.enable_validation` (or the `validate` argument) is set; warnings become `odds_core.models.DataQualityLog` rows ([odds-validator](../storage/odds-validator.md)).
4. One `odds_core.models.FetchLog` row via `odds_lambda.storage.writers.OddsWriter.log_fetch`, then a `odds_lambda.ingestion.SportIngestionResult` ([fetch logs](#fetch-logs)).

`odds_lambda.ingestion.OddsIngestionService.ingest_sports` loops sports sequentially. `odds_lambda.ingestion.OddsIngestionCallbacks` (`on_fetch_complete`, `on_event_processed`, `on_event_failed`) let callers instrument without subclassing; the CLI uses them for progress output. Per-event failures are collected into `odds_lambda.ingestion.SportIngestionResult.failures`, not raised; the job logs them and raises only if the whole fetch fails.

## What the CLI does instead

`odds fetch current` is only the service call and a summary. `--sport` defaults to `basketball_nba` (not a configured sport), and there is no event sync, scheduling or alert. It exits 1 on any exception.

## Fetch logs

`odds_core.models.FetchLog` (`fetch_logs`) is one row per fetch operation: `fetch_time` (indexed, UTC), `sport_key`, `events_count`, `bookmakers_count`, `success`, `error_message`, `api_quota_remaining`, `response_time_ms`.

**Writers.**
- `odds_lambda.ingestion.OddsIngestionService.ingest_sport` adds one row per sport after the events are processed. `success` is true when no event failed ingest, quota and response time come from the API response, and `bookmakers_count` is the configured bookmaker count, `len(settings.data_collection.bookmakers)`, not the number in the response (#472).
- The snapshots and `odds` rows it stores carry the soccer three-way market as `1x2`, relabelled when the client returns the response ([sport-and-market-keys](sport-and-market-keys.md#market-key-dialects)).
- `odds_lambda.storage.writers.OddsWriter.log_fetch` runs only after `odds_lambda.data_fetcher.TheOddsAPIClient.get_odds` returns, so an API error leaves no row, and `error_message` is never set (#472).
- `fetch-betfair-exchange` adds a sport-level row when snapshots were stored or events seen: `success = snapshots_stored > 0`, `bookmakers_count=1`, no quota ([betfair-exchange](betfair-exchange.md)). These rows mix into every reader below; see [odds-api-quota](odds-api-quota.md#where-the-views-disagree).
- Scores, events, OddsPortal and Polymarket jobs write nothing here; Polymarket has its own fetch-log table ([polymarket](polymarket.md)).

**Readers.**
- `odds status show` takes the latest row; `odds status quota` the latest 10 ([odds status](../operations/health-monitor.md#odds-status)).
- `odds_lambda.storage.readers.OddsReader.get_database_stats`: 24 h success rate and the newest non-null `api_quota_remaining`.
- `odds_lambda.health_monitor.HealthMonitor.check_consecutive_failures` reads the latest 10 rows of any source, Betfair included, and `odds_lambda.health_monitor.HealthMonitor.check_api_quota` reads the quota ([health-monitor](../operations/health-monitor.md)).

## Manual event discovery

`odds discover upcoming [--sport]` (`odds_cli.commands.discover.upcoming`) runs the same free `/events` sync as the job's first step without fetching odds; it costs 0 quota units. `--sport` defaults to `basketball_nba`, so pass `--sport soccer_epl` for EPL.

## Related

- [Odds API client and key rotation](odds-api-client.md)
- [Odds API quota accounting](odds-api-quota.md)
- [Self-scheduling](../scheduling/self-scheduling.md)
