---
type: reference
title: OddsReader
description: Every OddsReader query and its time-window semantics.
sources:
  - odds_lambda.storage.readers.OddsReader
  - odds_lambda.storage.readers.OddsReader.get_database_stats
  - odds_lambda.storage.readers.OddsReader.get_games_by_date
  - odds_lambda.storage.readers.OddsReader.get_events_by_date_range
status: stable
---

# OddsReader

`odds_lambda.storage.readers.OddsReader` is the read side for events, snapshots and the operational logs: one class, 9 async methods, built on an `AsyncSession`. It reads no prices; price rows are read through `odds_core.quotes.quotes` ([Reading quotes](price-tables.md#reading-quotes)). Module map and commit rules are on [Storage layer](storage-layer.md).

## Methods

**Snapshot and event accessors**: `odds_lambda.storage.readers.OddsReader.get_snapshot_count_for_event` (counts distinct snapshot times per event, since a fetch stores one snapshot per market at one time), `odds_lambda.storage.readers.OddsReader.get_snapshots_for_event`, `odds_lambda.storage.readers.OddsReader.get_events_by_date_range` (optional `min_snapshots`, counted in distinct snapshot times; ordered by `commence_time`, then the integer event id, so events with the same kick-off have a fixed order that does not depend on the query plan), `odds_lambda.storage.readers.OddsReader.get_events_by_team`, `odds_lambda.storage.readers.OddsReader.get_event_by_id`. Each takes the event's integer id.

**Operational**: `odds_lambda.storage.readers.OddsReader.get_fetch_logs`, `odds_lambda.storage.readers.OddsReader.get_data_quality_logs`, `odds_lambda.storage.readers.OddsReader.get_database_stats`, `odds_lambda.storage.readers.OddsReader.get_latest_quota_by_key`, `odds_lambda.storage.readers.OddsReader.get_games_by_date`.

## Other behaviour

- `odds_lambda.storage.readers.OddsReader.get_database_stats()` returns `events_by_status`, `total_events`, `total_price_rows` (rows of the `prices` table), `total_snapshots` (distinct event and snapshot-time pairs), `api_quota_by_key` (the newest remaining and used quota per Odds API key, with the plan size and remaining fraction they give, from `odds_lambda.storage.readers.OddsReader.get_latest_quota_by_key`, which ignores rows over 31 days old or without a used count); used by `odds status` and the [health monitor](../operations/health-monitor.md). `odds_lambda.storage.readers.OddsReader.get_fetch_logs` skips null quotas only with `with_quota=True` ([Odds API quota](../collection/odds-api-quota.md#where-quota-is-recorded-alerted-and-checked)).
- `odds_lambda.storage.readers.OddsReader.get_games_by_date(target_date, status=FINAL)` uses a noon-to-noon UTC window and has no sport filter, so it drives `validate daily` for every sport ([Coverage and completeness checks](coverage-checks.md)).

## Naming

- `odds_lambda.storage.readers.OddsReader.get_database_stats` is the only `*_stats` method on this class; the sibling readers (`odds_lambda.storage.injury_reader.InjuryReader`, `odds_lambda.storage.game_log_reader.GameLogReader`, `odds_lambda.storage.pbpstats_reader.PbpStatsReader`) name theirs differently; the first two take the same integer event id.
- "Snapshot" means an `odds_core.models.OddsSnapshot` ([columns and indexes](odds-tables.md#odds_snapshots)) row in every method, and something else again in the Polymarket tables.

## Related

- [Storage layer](storage-layer.md)
- [How odds are stored](odds-storage-model.md)
- [Event matching across sources](event-matching.md)
- [CLV and the training target](../modelling/clv-target.md) and [Replay backtest](../evaluation/candidate-backtest.md) consume these queries
