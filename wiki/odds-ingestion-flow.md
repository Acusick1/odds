---
type: process
title: Odds ingestion flow
summary: How one Odds API fetch becomes stored data - client, validator, OddsWriter (raw plus normalized), FetchLog - via the OddsIngestionService shared by the job and `odds fetch current`.
sources:
  - packages/odds-lambda/odds_lambda/ingestion.py::OddsIngestionService
  - packages/odds-lambda/odds_lambda/jobs/fetch_odds.py::main
  - packages/odds-cli/odds_cli/commands/fetch.py::_fetch_current
  - packages/odds-lambda/odds_lambda/storage/writers.py::OddsWriter
  - docs/ARCHITECTURE.md#data-collection-pipeline
  - docs/CLI.md#fetch-current-odds
  - docs/REPORTING.md#8-storage-backing-reports
verified_at: 7887039
---

# Odds ingestion flow

`OddsIngestionService` (`packages/odds-lambda/odds_lambda/ingestion.py`) is the single path from [The Odds API client](odds-api-client.md) to the database. Two callers: the [fetch-odds job](job-fetch-odds.md) (`packages/odds-lambda/odds_lambda/jobs/fetch_odds.py::main`) and `odds fetch current` (`packages/odds-cli/odds_cli/commands/fetch.py::_fetch_current`).

## Steps per sport (`ingest_sport`)

1. `client.get_odds(...)` with regions, markets, bookmakers from `DataCollectionConfig` ([Data sources](data-sources.md)).
2. For each event, in its own session and commit: `OddsWriter.upsert_event`, then `store_odds_snapshot` ([Odds writer](odds-writer.md)). That stores the raw JSONB `OddsSnapshot` and normalized `Odds` rows, with `fetch_tier` and `hours_until_commence` computed from `Event.commence_time` ([Fetch tier](fetch-tier.md), [odds_snapshots](table-odds-snapshots.md), [Events](table-events.md)).
3. Validation runs inside `store_odds_snapshot` when `data_quality.enable_validation` (or the `validate` argument) is set; warnings become `DataQualityLog` rows ([Data quality validation](data-quality-validation.md)).
4. One `FetchLog` row via `OddsWriter.log_fetch`, then a `SportIngestionResult` ([fetch_logs](table-fetch-logs.md)).

`ingest_sports` loops sports sequentially. `OddsIngestionCallbacks` (`on_fetch_complete`, `on_event_processed`, `on_event_failed`) let callers instrument without subclassing; the CLI uses them for progress output.

## What each caller adds

- **Job**: first syncs events from the free `/events` endpoint (`EventSyncService`, failures non-fatal), gates on `decide_forward_resilient`, pre-schedules the next run before fetching, re-schedules afterwards, and sends quota alerts ([Odds API quota](odds-api-quota.md), [Self-scheduling](self-scheduling.md)).
- **CLI**: only the service call and a summary; `--sport` defaults to `basketball_nba` (not a configured sport), and no event sync, scheduling or alerts. Exits 1 on any exception.

## Hazards

- **Failed fetches leave no `FetchLog`.** `log_fetch` runs only after `get_odds` returns, so an API error raises before any row exists. `FetchLog.error_message` is never set by the service; `success` is false only when per-event failures occurred.
- **`bookmakers_count` is the configured count**, `len(settings.data_collection.bookmakers)`, not the number in the response.
- **Per-event failures are swallowed** into `SportIngestionResult.failures`; the job logs them but only raises if the whole fetch fails.
- `REPORTING.md` attributes `data_quality_logs` writes to `OddsIngestionService`; they come from `OddsWriter.store_odds_snapshot`.
- `odds fetch scores` does not share a service with its job: `_fetch_scores` duplicates the update loop in `packages/odds-lambda/odds_lambda/jobs/fetch_scores.py::_fetch_and_update_scores`. The job loops sports and alerts; the CLI takes one sport.
- Deployed fetch schedule is not verifiable from the repo: `odds scheduler list-jobs --backend aws`.

## Related

- [Job registry](job-registry.md)
- [Job: fetch-scores](job-fetch-scores.md)
- [Event lifecycle](event-lifecycle.md)
