---
type: component
title: OddsReader
summary: Time-aware odds queries (odds at time, line movement, best odds, sharp prices, database stats), and which ones read normalized rows and which re-parse raw JSON.
sources:
  - packages/odds-lambda/odds_lambda/storage/readers.py::OddsReader
  - packages/odds-core/odds_core/snapshot_utils.py::extract_odds_from_snapshot
  - docs/DATABASE.md#oddsreader-methods
verified_at: 7887039
---

# OddsReader

`packages/odds-lambda/odds_lambda/storage/readers.py::OddsReader` is the read side of the [storage layer](storage-layer.md). One class, 20 async methods, built on an `AsyncSession`. Also defines `BookPriceResult`, `BookPriceEntry` and `BookPriceMeta` dataclasses.

## Which path each method uses

**Normalized `odds` rows** ([table-odds](table-odds.md)): `get_odds_at_time`, `get_line_movement`, `get_best_odds`.

**Re-parsed `raw_data`** ([odds_snapshots](table-odds-snapshots.md)), via `packages/odds-core/odds_core/snapshot_utils.py::extract_odds_from_snapshot` and `raw_data_has_market`: `get_sharp_prices`, `get_latest_book_prices`, `get_latest_snapshot(market=...)`. These return transient `Odds` objects with `id=None`. Prices pass through unchanged, unlike the writer's `int(...)` coercion ([OddsWriter](odds-writer.md#hazards)).

**Snapshot and event accessors**: `get_snapshots_by_tier`, `get_first_snapshot_in_tier`, `get_last_snapshot_in_tier`, `get_tier_coverage_for_event`, `get_snapshot_count_for_event`, `get_snapshots_for_event`, `snapshot_exists`, `get_events_by_date_range` (optional `event_id_pattern` LIKE and `min_snapshots`), `get_events_by_team`, `get_event_by_id`.

**Operational**: `get_fetch_logs`, `get_data_quality_logs` (see [validation](data-quality-validation.md)), `get_database_stats`, `get_games_by_date`.

## Behaviour to know

- `get_odds_at_time(event_id, timestamp, tolerance_minutes=5)` picks the distinct `odds_timestamp` closest to the target within plus/minus the tolerance and returns all rows at it. Empty list if none.
- `get_best_odds` orders by `price DESC`; with `timestamp` it filters to a plus/minus 5 minute window and does not pin one snapshot; without it, uses the event's latest `odds_timestamp`.
- `get_sharp_prices(event_id, market, sharp_bookmakers, lookback_hours=2.0, now=None)` walks snapshots newest-first and takes the first sharp bookmaker in priority order per outcome; returns `SharpPriceResult` with age metadata (see [sharp benchmark](sharp-benchmark.md)).
- `get_latest_book_prices` keeps the newest price per `(bookmaker_key, outcome_name, point)` across the window, so partial-book snapshots (e.g. Betfair-only) do not hide retail books. Used by the MCP server ([market-data tools](mcp-market-data-tools.md)).
- `get_database_stats()` returns `events_by_status`, `total_events`, `total_odds_records`, `total_snapshots`, `fetch_success_rate_24h`, `api_quota_remaining`; used by `odds status` and the [health monitor](health-monitor.md).
- `get_games_by_date(target_date, status=FINAL)` uses a noon-to-noon UTC window and has no sport filter.

## Hazards

- **Look-ahead.** `AGENTS.md` and `docs/DATABASE.md` call `get_odds_at_time` the guard against look-ahead bias, but the window is symmetric: a snapshot up to `tolerance_minutes` after `timestamp` can win if it is closer. The backtester calls it with `tolerance_minutes=30` around the decision time (`packages/odds-analytics/odds_analytics/backtesting/services.py`). See [look-ahead bias](look-ahead-bias.md).
- **Doc coverage.** `docs/DATABASE.md#oddsreader-methods` lists 4 of the methods; the rest above are undocumented there.
- **Naming.** `get_database_stats` is the only `*_stats` on this class; sibling readers use `get_pipeline_stats`. "Snapshot" means an `OddsSnapshot` row here, a list of `Odds` at a time in `get_odds_at_time`, and something else in Polymarket.
- **Sport scope.** `get_games_by_date`, and so `validate daily`, spans all sports ([completeness validation](completeness-validation.md)).

## Related

- [Storage layer](storage-layer.md)
- [CLV](clv.md) and [backtest strategies](backtest-strategies.md) consume these queries
