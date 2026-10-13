---
type: reference
title: OddsReader
description: Every OddsReader query, which read normalized rows and which re-parse raw_data, and the time-window semantics.
sources:
  - odds_lambda.storage.readers.OddsReader
  - odds_lambda.storage.readers.OddsReader.get_odds_at_time
  - odds_lambda.storage.readers.OddsReader.get_best_odds
  - odds_lambda.storage.readers.OddsReader.get_sharp_prices
  - odds_lambda.storage.readers.OddsReader.get_latest_book_prices
  - odds_lambda.storage.readers.OddsReader.get_database_stats
  - odds_lambda.storage.readers.OddsReader.get_games_by_date
  - odds_lambda.storage.readers.BookPriceResult
  - odds_lambda.storage.readers.BookPriceEntry
  - odds_lambda.storage.readers.BookPriceMeta
  - odds_core.snapshot_utils.extract_odds_from_snapshot
  - odds_core.utils.raw_data_has_market
  - odds_core.match_brief_models.SharpPriceResult
  - odds_analytics.backtesting.services
status: stable
---

# OddsReader

`odds_lambda.storage.readers.OddsReader` is the read side for odds: one class, 20 async methods, built on an `AsyncSession`. It also defines `odds_lambda.storage.readers.BookPriceResult`, `odds_lambda.storage.readers.BookPriceEntry` and `odds_lambda.storage.readers.BookPriceMeta` dataclasses. Module map and commit rules are on [Storage layer](storage-layer.md).

## Which path each method uses

**Normalized `odds` rows** ([Odds snapshot and odds tables](odds-tables.md)): `get_odds_at_time`, `get_line_movement`, `get_best_odds`.

**Re-parsed `raw_data`**, via `odds_core.snapshot_utils.extract_odds_from_snapshot` and `odds_core.utils.raw_data_has_market`: `get_sharp_prices`, `get_latest_book_prices`, `get_latest_snapshot(market=...)`. These return transient `odds_core.models.Odds` objects with `id=None`. Prices pass through unchanged, unlike the writer's `int(...)` coercion ([How odds are stored](odds-storage-model.md#two-parsers-two-behaviours)).

**Snapshot and event accessors**: `get_snapshots_by_tier`, `get_first_snapshot_in_tier`, `get_last_snapshot_in_tier`, `get_tier_coverage_for_event`, `get_snapshot_count_for_event`, `get_snapshots_for_event`, `snapshot_exists`, `get_events_by_date_range` (optional `event_id_pattern` LIKE and `min_snapshots`), `get_events_by_team`, `get_event_by_id`.

**Operational**: `get_fetch_logs`, `get_data_quality_logs`, `get_database_stats`, `get_games_by_date`.

## Odds at a point in time

`get_odds_at_time(event_id, timestamp, tolerance_minutes=5)` returns the `odds_timestamp` closest to the target within ± `tolerance_minutes`, so a snapshot after the target can win (#435); it returns all rows at that timestamp, or an empty list if none. The backtester calls it with `tolerance_minutes=30` around a 1 h decision time (`odds_analytics.backtesting.services`), so post-decision odds can win ([Look-ahead bias](../modelling/look-ahead-bias.md#where-the-rule-breaks)). Events from historical OddsPortal/FDUK ingests have no normalized rows, so it returns nothing for them.

## Other behaviour

- `get_best_odds` orders by `price DESC`; with `timestamp` it filters to a plus/minus 5 minute window and does not pin one snapshot; without it, uses the event's latest `odds_timestamp`.
- `get_sharp_prices(event_id, market, sharp_bookmakers, lookback_hours=2.0, now=None)` walks snapshots newest-first and takes the first sharp bookmaker in priority order per outcome; returns `odds_core.match_brief_models.SharpPriceResult` with age metadata ([Sharp benchmark](../modelling/sharp-benchmark.md)). Resolution is per outcome, so one response can mix Pinnacle and Betfair.
- `get_latest_book_prices` keeps the newest price per `(bookmaker_key, outcome_name, point)` across the window, so partial-book snapshots (e.g. Betfair-only) do not hide retail books. Its ages are relative to the anchor snapshot, not wall clock. Used by the MCP server ([MCP odds, prediction and retail-edge tools](../agent/mcp-market-tools.md)).
- `get_database_stats()` returns `events_by_status`, `total_events`, `total_odds_records`, `total_snapshots`, `fetch_success_rate_24h`, `api_quota_remaining`; used by `odds status` and the [health monitor](../operations/health-monitor.md). `api_quota_remaining` is the newest non-null quota, whereas `get_fetch_logs` does not skip nulls ([Odds API quota](../collection/odds-api-quota.md)).
- `get_games_by_date(target_date, status=FINAL)` uses a noon-to-noon UTC window and has no sport filter, so it drives `validate daily` for every sport ([Coverage and completeness checks](coverage-checks.md)).

## Related

- [Storage layer](storage-layer.md)
- [How odds are stored](odds-storage-model.md)
- [Event matching across sources](event-matching.md)
- [CLV and the training target](../modelling/clv-target.md) and [Backtest strategies](../evaluation/backtest-strategies.md) consume these queries
