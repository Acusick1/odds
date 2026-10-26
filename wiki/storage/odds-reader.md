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
status: stable
---

# OddsReader

`odds_lambda.storage.readers.OddsReader` is the read side for odds: one class, 20 async methods, built on an `AsyncSession`. It also defines `odds_lambda.storage.readers.BookPriceResult`, `odds_lambda.storage.readers.BookPriceEntry` and `odds_lambda.storage.readers.BookPriceMeta` dataclasses. Module map and commit rules are on [Storage layer](storage-layer.md).

## Which path each method uses

**Normalized `odds` rows** ([Odds snapshot and odds tables](odds-tables.md)): `odds_lambda.storage.readers.OddsReader.get_odds_at_time`, `odds_lambda.storage.readers.OddsReader.get_line_movement`, `odds_lambda.storage.readers.OddsReader.get_best_odds`.

**Re-parsed `raw_data`**, via `odds_core.snapshot_utils.extract_odds_from_snapshot` and `odds_core.utils.raw_data_has_market`: `odds_lambda.storage.readers.OddsReader.get_sharp_prices`, `odds_lambda.storage.readers.OddsReader.get_latest_book_prices`, `odds_lambda.storage.readers.OddsReader.get_latest_snapshot(market=...)` (newest time, lowest id first among the snapshots of that time). These return transient `odds_core.models.Odds` objects with `id=None` and `snapshot_id` set to the snapshot they were parsed from, as a stored row carries it. Prices go through `odds_core.odds_math.outcome_price_decimal`, as in the writer ([How odds are stored](odds-storage-model.md#two-parsers-one-price-rule)).

**Snapshot and event accessors**: `odds_lambda.storage.readers.OddsReader.get_snapshots_by_tier`, `odds_lambda.storage.readers.OddsReader.get_first_snapshot_in_tier`, `odds_lambda.storage.readers.OddsReader.get_last_snapshot_in_tier`, `odds_lambda.storage.readers.OddsReader.get_tier_coverage_for_event`, `odds_lambda.storage.readers.OddsReader.get_snapshot_count_for_event` (both count distinct snapshot times per event, since a fetch stores one snapshot per market at one time), `odds_lambda.storage.readers.OddsReader.get_snapshots_for_event`, `odds_lambda.storage.readers.OddsReader.snapshot_exists`, `odds_lambda.storage.readers.OddsReader.get_events_by_date_range` (optional `event_id_pattern` LIKE and `min_snapshots`, counted in distinct snapshot times; ordered by `commence_time`, then event ID, so events with the same kick-off have a fixed order that does not depend on the query plan), `odds_lambda.storage.readers.OddsReader.get_events_by_team`, `odds_lambda.storage.readers.OddsReader.get_event_by_id`.

**Operational**: `odds_lambda.storage.readers.OddsReader.get_fetch_logs`, `odds_lambda.storage.readers.OddsReader.get_data_quality_logs`, `odds_lambda.storage.readers.OddsReader.get_database_stats`, `odds_lambda.storage.readers.OddsReader.get_games_by_date`.

## Odds at a point in time

`odds_lambda.storage.readers.OddsReader.get_odds_at_time(event_id, timestamp, tolerance_minutes=5)` returns the `odds_timestamp` closest to the target within ± `tolerance_minutes`, so a snapshot after the target can win (#435); it returns all rows at that timestamp, or an empty list if none. It can therefore return odds after the target ([Look-ahead bias](../modelling/look-ahead-bias.md#where-the-rule-breaks)). Events from historical OddsPortal/FDUK ingests have no normalized rows, so it returns nothing for them.

## Other behaviour

- `odds_lambda.storage.readers.OddsReader.get_best_odds` orders by `price_decimal DESC`; with `timestamp` it filters to a plus/minus 5 minute window and does not pin one snapshot; without it, uses the event's latest `odds_timestamp`.
- `odds_lambda.storage.readers.OddsReader.get_sharp_prices(event_id, market, sharp_bookmakers, lookback_hours=2.0, now=None)` walks snapshots newest-first and takes the first sharp bookmaker in priority order per outcome; returns `odds_core.match_brief_models.SharpPriceResult` with age metadata; each price entry carries `price` (derived American quote), `price_decimal` and `implied_prob` from the decimal ([Sharp benchmark](../modelling/sharp-benchmark.md)). Resolution is per outcome, so one response can mix Pinnacle and Betfair.
- `odds_lambda.storage.readers.OddsReader.get_latest_book_prices` keeps the newest price per `(bookmaker_key, outcome_name, point)` across the window, so partial-book snapshots (e.g. Betfair-only) do not hide retail books. Its ages are relative to the anchor snapshot, not wall clock.
- `odds_lambda.storage.readers.OddsReader.get_database_stats()` returns `events_by_status`, `total_events`, `total_odds_records`, `total_snapshots` (distinct event and snapshot-time pairs), `api_quota_remaining`; used by `odds status` and the [health monitor](../operations/health-monitor.md). `odds_lambda.storage.readers.OddsReader.get_fetch_logs` skips null quotas only with `with_quota=True` ([Odds API quota](../collection/odds-api-quota.md#where-quota-is-recorded-alerted-and-checked)).
- `odds_lambda.storage.readers.OddsReader.get_games_by_date(target_date, status=FINAL)` uses a noon-to-noon UTC window and has no sport filter, so it drives `validate daily` for every sport ([Coverage and completeness checks](coverage-checks.md)).

## Naming

- `odds_lambda.storage.readers.OddsReader.get_database_stats` is the only `*_stats` method on this class; the sibling readers (`odds_lambda.storage.polymarket_reader.PolymarketReader`, `odds_lambda.storage.injury_reader.InjuryReader`, `odds_lambda.storage.game_log_reader.GameLogReader`, `odds_lambda.storage.pbpstats_reader.PbpStatsReader`) name theirs `odds_lambda.storage.polymarket_reader.PolymarketReader.get_pipeline_stats` and its counterparts.
- "Snapshot" means an `odds_core.models.OddsSnapshot` ([columns and indexes](odds-tables.md#odds_snapshots)) row in most methods (no method reads its redesigned columns, such as `source`, `market`, `time_basis` or `market_status`, or the `prices` table), a list of `odds_core.models.Odds` (each with `price_decimal`) at one time in `odds_lambda.storage.readers.OddsReader.get_odds_at_time`, and something else again in the Polymarket tables.

## Related

- [Storage layer](storage-layer.md)
- [How odds are stored](odds-storage-model.md)
- [Event matching across sources](event-matching.md)
- [CLV and the training target](../modelling/clv-target.md) and [Replay backtest](../evaluation/candidate-backtest.md) consume these queries
