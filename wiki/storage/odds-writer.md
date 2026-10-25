---
type: reference
title: OddsWriter
description: store_odds_snapshot, event upserts, find_or_create_event, logging helpers, and their edge cases.
sources:
  - odds_lambda.storage.writers.OddsWriter
  - odds_lambda.storage.writers.OddsWriter.store_odds_snapshot
  - odds_lambda.storage.writers.OddsWriter._parse_and_store_odds
  - odds_lambda.storage.writers.OddsWriter.find_or_create_event
  - odds_lambda.storage.writers.OddsWriter.find_unscheduled_mlb_event
  - odds_lambda.storage.event_merge.EVENT_MATCH_WINDOW
  - odds_lambda.storage.event_merge.find_match_id
  - odds_lambda.storage.writers.OddsWriter._build_event_id
  - odds_lambda.storage.writers.OddsWriter.log_fetch
  - odds_lambda.storage.writers.OddsWriter.log_data_quality_issue
  - odds_lambda.tier_utils.calculate_tier_from_timestamps
  - odds_lambda.ingestion.OddsIngestionService
  - odds_core.config.DataQualityConfig
  - odds_core.models.DataQualityLog
  - odds_core.models.FetchLog
status: stable
---

# OddsWriter

`odds_lambda.storage.writers.OddsWriter` owns an `AsyncSession` and an `odds_lambda.storage.validators.OddsValidator` (its vig floor skips zero-margin books, `odds_core.bookmakers.ZERO_MARGIN_BOOKMAKERS`). It never commits. It flushes each snapshot it stores, to give the snapshot's odds rows its id, and for MLB events in the savepoint below and when an event moving to its game's scheduled start has its snapshots re-timed; callers commit ([Commit contract](storage-layer.md#commit-contract)). The tables it writes are on [Odds snapshot and odds tables](odds-tables.md) and [Events table](events.md).

## store_odds_snapshot

`odds_lambda.storage.writers.OddsWriter.store_odds_snapshot(event_id, raw_data, snapshot_time=None, validate=True)` takes one event's dict; a list raises `AttributeError` because `raw_data.get("bookmakers")` runs before the `isinstance(raw_data, list)` check (#464). Steps:

1. Loads the event and computes `hours_until_commence` and the fetch tier from `snapshot_time` and the event's own `commence_time` (`odds_lambda.tier_utils.calculate_tier_from_timestamps`). The tier is per event, not per job run. If the event row is missing, both `fetch_tier` and `hours_until_commence` are stored as NULL ([Fetch tier](fetch-tier.md)).
2. If `validate`, runs the validator and writes one `odds_core.models.DataQualityLog` row for any warnings; storage proceeds regardless ([Odds validation](odds-validator.md)). `odds_lambda.ingestion.OddsIngestionService.ingest_sport` is the only caller that passes `validate`: the `validate` argument of that method or of `odds_lambda.ingestion.OddsIngestionService.ingest_sports` when given, else `odds_core.config.DataQualityConfig.enable_validation`. Every other caller of `odds_lambda.storage.writers.OddsWriter.store_odds_snapshot` uses the default `True`.
3. Adds an `odds_core.models.OddsSnapshot` ([columns and indexes](odds-tables.md#odds_snapshots)) with the full `raw_data` and `bookmaker_count`, and flushes it to get its id.
4. `odds_lambda.storage.writers.OddsWriter._parse_and_store_odds` adds one `odds_core.models.Odds` row per bookmaker/market/outcome, each with `snapshot_id` set to that snapshot. Bookmakers without a key are skipped; `last_update` falls back to `snapshot_time` if unparseable; `odds_timestamp` is `snapshot_time`. It stores the decimal price (it must exceed 1.0, `ck_odds_price_decimal_gt_one`) from `odds_core.odds_math.outcome_price_decimal`, which raises `KeyError` for an outcome with neither `price_decimal` nor `price`; the validator flags such an outcome as an error.

Returns `(OddsSnapshot, list[Odds])`.

## Event handling

Event upserts (`odds_lambda.storage.writers.OddsWriter.upsert_event`, `odds_lambda.storage.writers.OddsWriter.update_event_status`, `odds_lambda.storage.writers.OddsWriter.bulk_upsert_events`) are on [Events table](events.md#upsert-verbs); `odds_lambda.storage.writers.OddsWriter.update_event_status` raises `ValueError` for FINAL without both scores and for a move the stored status does not allow ([Event status lifecycle](../collection/event-status-lifecycle.md#allowed-moves)).

### find_or_create_event

`odds_lambda.storage.writers.OddsWriter.find_or_create_event(home, away, match_date, sport_key, sport_title, *, mlb_game=None)` normalizes team names and returns `(event_id, was_created)`. It never merges or deletes ([Event merge](../storage/event-merge.md)).

- **Other sports** look for same-sport events within `odds_lambda.storage.event_merge.EVENT_MATCH_WINDOW` (±2 hours) with `odds_lambda.storage.event_merge.find_match_id`. With several events in the window it returns the canonical one (`odds_core.event_source.canonical_event_id`: the Odds API event, else the best scraper source) and never creates another.
- **MLB** requires `mlb_game` (an `odds_lambda.mlb_game_resolver.ResolvedGame`) and raises `ValueError` without it. It returns the event holding the `gamePk`, else claims the nearest event within ±2 h that holds none, else creates the event with it; the event's `commence_time` follows the game's scheduled start (`odds_lambda.storage.event_kickoff.move_kickoff`, with the MLB Stats API as the [kick-off authority](kickoff-authority.md)). The claim rules, the id used on a collision and the timing are on [live MLB events](event-matching.md#live-mlb-events). The write runs in a savepoint (`begin_nested`), so when a concurrent run has taken the `gamePk` meanwhile the unique-index violation rolls back only this row, which then lands on the holder and logs `mlb_game_pk_duplicate`.

`odds_lambda.storage.writers.OddsWriter.find_unscheduled_mlb_event` is the lookup the live jobs use for MLB when the MLB schedule is unavailable; it creates nothing.

New ids are `op_live_{HOME}_{AWAY}_{YYYY-MM-DDTHHMM}` (`odds_lambda.storage.writers.OddsWriter._build_event_id`), so a rescheduled kickoff creates a new id unless the 2 h window (or, for MLB, the `gamePk`) matches an existing event. For other sports an existing event with that id is returned instead of created; for MLB that id goes to a new event with `_<gamePk>` appended. Creation raises `ValueError` if the match started more than `LIVE_EVENT_STALE_THRESHOLD` (1 hour) ago. Used by the [OddsPortal](../collection/oddsportal-scraper.md) and [Betfair](../collection/betfair-exchange.md) jobs; matcher comparison on [Event matching across sources](event-matching.md).

## Log helpers

`odds_lambda.storage.writers.OddsWriter.log_fetch` adds an `odds_core.models.FetchLog` ([fetch_logs](../collection/odds-ingestion-service.md#fetch-logs)) and `odds_lambda.storage.writers.OddsWriter.log_data_quality_issue` adds an `odds_core.models.DataQualityLog`.

## Related

- [Odds snapshot and odds tables](odds-tables.md)
- [How odds are stored](odds-storage-model.md)
- [Storage layer](storage-layer.md)
- [Odds API ingestion service](../collection/odds-ingestion-service.md)
