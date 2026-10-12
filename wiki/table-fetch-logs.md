---
type: component
title: fetch_logs table
summary: Success, quota and timing for each fetch, written by ingestion and read by the status CLI and health checks.
sources:
  - docs/DATABASE.md#core-tables
  - docs/REPORTING.md#8-storage-backing-reports
  - packages/odds-core/odds_core/models.py::FetchLog
  - packages/odds-lambda/odds_lambda/ingestion.py::OddsIngestionService
verified_at: 7887039
---

# fetch_logs table

`packages/odds-core/odds_core/models.py::FetchLog` (`fetch_logs`) is one row per fetch operation: `fetch_time` (indexed, UTC), `sport_key`, `events_count`, `bookmakers_count`, `success`, `error_message`, `api_quota_remaining`, `response_time_ms`.

## Writers

- `OddsIngestionService.ingest_sport` (`packages/odds-lambda/odds_lambda/ingestion.py`) adds one row per sport after the events are processed: `success` is true when no event failed ingest, `bookmakers_count` is the configured bookmaker count, and quota and response time come from the API response. It persists through `OddsWriter.log_fetch` ([odds ingestion flow](odds-ingestion-flow.md), [odds writer](odds-writer.md)).
- `fetch-betfair-exchange` adds a sport-level row when snapshots were stored or events seen: `success = snapshots_stored > 0`, `bookmakers_count=1`, no quota ([Betfair job](job-fetch-betfair-exchange.md)).
- Scores, events, OddsPortal and Polymarket jobs write nothing here; Polymarket has its own fetch-log table.

## Readers

- `odds status show` takes the latest row; `odds status quota` the latest 10 ([reporting surfaces](reporting-surfaces.md), [odds API quota](odds-api-quota.md)).
- `OddsReader.get_database_stats`: 24 h success rate (`success` count over total) and the newest non-null `api_quota_remaining`.
- `HealthMonitor.check_consecutive_failures` and `check_api_quota` ([health monitor](health-monitor.md)).

## Hazards

- No row is written if the Odds API call raises, so an outage leaves no failure record; `success=False` means only that some event failed ingest.
- Rows from different sources mix in the "latest 10" and in the 24 h rate. Betfair rows carry no quota, and the quota reader skips null values by design.
- `bookmakers_count` records the configured bookmaker list length, not the number of bookmakers in the response.

## Related

- [Table alert_history](table-alert-history.md), [job-fetch-odds](job-fetch-odds.md).
