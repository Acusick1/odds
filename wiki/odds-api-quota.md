---
type: concept
title: Odds API quota and cost
summary: Unit cost per Odds API endpoint, free and paid tiers, how key rotation multiplies quota, and where usage is logged, alerted and checked.
sources:
  - packages/odds-lambda/odds_lambda/data_fetcher.py::TheOddsAPIClient
  - packages/odds-core/odds_core/models.py::FetchLog
  - packages/odds-core/odds_core/config.py::APIConfig
  - packages/odds-core/odds_core/config.py::AlertConfig
  - packages/odds-cli/odds_cli/commands/status.py::_show_quota
  - packages/odds-analytics/odds_analytics/gap_backfill_planner.py::GapBackfillPlanner
  - ./AGENTS.md#api-cost
  - docs/DEPLOYMENT.md#the-odds-api
  - docs/CLI.md#check-api-quota
verified_at: 7887039
---

# Odds API quota and cost

Quota is spent only by [The Odds API client](odds-api-client.md). The repo's cost table (`AGENTS.md#api-cost`) is documentation; the API's billing is not verifiable from code.

## Cost per call

| Endpoint | Client method | Units |
|---|---|---|
| `/odds` | `get_odds` | 1 per region per market, regardless of game count (default 1 region x 3 markets = 3) |
| `/events` | `get_events` | free |
| `/scores` | `get_scores` | 1 without `daysFrom`, 2 with |
| `/historical/odds` | `get_historical_odds` | 10 per region per market |
| `/historical/events` | `get_historical_events` | 1 (per its docstring) |

Bookmakers are a server-side filter and do not change cost. Tiers (`docs/DEPLOYMENT.md#the-odds-api`): free 500 requests/key/month with no historical endpoints; Basic $25/month, 20,000. Multiple keys (`ODDS_API_KEYS`) multiply the effective quota; the active one is tracked in SSM.

## Where usage is recorded

- Every `OddsIngestionService.ingest_sport` writes a `FetchLog` row with `api_quota_remaining` (see [Odds ingestion flow](odds-ingestion-flow.md), [fetch_logs](table-fetch-logs.md)). The Betfair job also writes `FetchLog` rows, with a null quota.
- `fetch-odds` compares remaining quota to `APIConfig.quota` (`ODDS_API_QUOTA`, default 500) and sends a warning below `AlertConfig.quota_warning_threshold` (0.2) and critical below `quota_critical_threshold` (0.1). `HealthMonitor.check_api_quota` applies the same fraction ([Health monitor](health-monitor.md)).
- Check manually: `uv run odds status quota` (`_show_quota`) prints the latest and last 10 `FetchLog` quotas ([CLI overview](cli-overview.md), [Reporting surfaces](reporting-surfaces.md)).

## Hazards

- **`/scores` always costs 2.** `get_scores` always sends `daysFrom` (default 1), so the 1-unit case is unreachable through the client.
- **Quota is per key, thresholds assume one.** `x-requests-remaining` is the active key's balance; percentages divide by `APIConfig.quota`. On a 20,000-unit paid key with the default 500, `status quota` and alerts show nonsense until `ODDS_API_QUOTA` is set.
- **Historical cost is a hardcoded 30** (10 x 1 region x 3 markets) in `GapBackfillPlanner.QUOTA_PER_SNAPSHOT`, and again as literals in `backfill_executor.py` and `game_selector.py`; none reads the configured regions or markets, so backfill estimates drift if those change.
- **`status quota` reads the newest 10 rows unfiltered** (`OddsReader.get_fetch_logs`) and shows nothing when the newest row has a null quota, as Betfair rows do. `HealthMonitor.check_api_quota` and `status show` use the latest non-null quota instead (`OddsReader.get_database_stats`).
- Failed `/odds` calls write no `FetchLog`, so they leave no quota trace.

## Related

- [Historical backfill how-to](howto-historical-backfill.md)
- [Job: fetch-scores](job-fetch-scores.md)
