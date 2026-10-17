---
type: reference
title: OddsWriter
description: store_odds_snapshot, event upserts, find_or_create_event, logging helpers, and their edge cases.
sources:
  - odds_lambda.storage.writers.OddsWriter
  - odds_lambda.storage.writers.OddsWriter.store_odds_snapshot
  - odds_lambda.storage.writers.OddsWriter._parse_and_store_odds
  - odds_lambda.storage.writers.OddsWriter.bulk_insert_odds
  - odds_lambda.storage.writers.OddsWriter.find_or_create_event
  - odds_lambda.storage.writers.OddsWriter.adopt_live_twins
  - odds_lambda.storage.writers.OddsWriter.EVENT_MATCH_WINDOW
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

`odds_lambda.storage.writers.OddsWriter` owns an `AsyncSession` and an `odds_lambda.storage.validators.OddsValidator`. It never flushes or commits; callers do ([Commit contract](storage-layer.md#commit-contract)). The tables it writes are on [Odds snapshot and odds tables](odds-tables.md) and [Events table](events.md).

## store_odds_snapshot

`odds_lambda.storage.writers.OddsWriter.store_odds_snapshot(event_id, raw_data, snapshot_time=None, validate=True)` takes one event's dict; a list raises `AttributeError` because `raw_data.get("bookmakers")` runs before the `isinstance(raw_data, list)` check (#464). Steps:

1. Loads the event and computes `hours_until_commence` and the fetch tier from `snapshot_time` and the event's own `commence_time` (`odds_lambda.tier_utils.calculate_tier_from_timestamps`). The tier is per event, not per job run. If the event row is missing, both `fetch_tier` and `hours_until_commence` are stored as NULL ([Fetch tier](fetch-tier.md)).
2. If `validate`, runs the validator and writes one `odds_core.models.DataQualityLog` row for any warnings; storage proceeds regardless ([Odds validation](odds-validator.md)). Only `odds_lambda.ingestion.OddsIngestionService.ingest_sport` passes `validate`: the `validate` argument of that method or of `odds_lambda.ingestion.OddsIngestionService.ingest_sports` when given, else `odds_core.config.DataQualityConfig.enable_validation`. Every other caller uses the default `True`.
3. Adds an `odds_core.models.OddsSnapshot` with the full `raw_data` and `bookmaker_count`.
4. `odds_lambda.storage.writers.OddsWriter._parse_and_store_odds` adds one `odds_core.models.Odds` row per bookmaker/market/outcome. Bookmakers without a key are skipped; `last_update` falls back to `snapshot_time` if unparseable; `odds_timestamp` is `snapshot_time`. It stores `int(outcome.get("price", 0))`, so an absent price becomes 0 and the validator, which skips `None` prices, does not flag it (#464).

Returns `(OddsSnapshot, list[Odds])`.

## Event handling

Event upserts (`odds_lambda.storage.writers.OddsWriter.upsert_event`, `odds_lambda.storage.writers.OddsWriter.update_event_status`, `odds_lambda.storage.writers.OddsWriter.bulk_upsert_events`) are on [Events table](events.md#upsert-verbs).

### find_or_create_event

`odds_lambda.storage.writers.OddsWriter.find_or_create_event(home, away, match_date, sport_key, sport_title)` normalizes team names, looks for same-sport events within `odds_lambda.storage.writers.OddsWriter.EVENT_MATCH_WINDOW` (±2 hours), and returns `(event_id, was_created)`. With several events in the window it logs `ambiguous_event_match` and returns the lowest id among the non-`op_live_` candidates, or among all candidates if every one is `op_live_`, so an Odds API event wins over a scraper twin. New ids are `op_live_{HOME}_{AWAY}_{YYYY-MM-DDTHHMM}` (`odds_lambda.storage.writers.OddsWriter._build_event_id`), so a rescheduled kickoff creates a new id unless the 2 h window matches an existing event. Creation raises `ValueError` if the match started more than `LIVE_EVENT_STALE_THRESHOLD` (1 hour) ago. Used by the [OddsPortal](../collection/oddsportal-scraper.md) and [Betfair](../collection/betfair-exchange.md) jobs; matcher comparison on [Event matching across sources](event-matching.md).

### adopt_live_twins

`odds_lambda.storage.writers.OddsWriter.adopt_live_twins(events)` merges each `op_live_` event with the same sport, teams and a commence time within the match window into the given Odds API event: every `event_id` foreign key to `events.id` (found from `sqlmodel.SQLModel.metadata`, so new tables are covered) is re-pointed, the twin's status, scores and `completed_at` are copied when the API event has no scores, the twin is deleted, and `live_twin_adopted` is logged with the rows moved per table. It runs in the caller's transaction. `odds_lambda.event_sync.EventSyncService.sync_sport` calls it for every synced event, so twins created while only the scrapers ran are merged on the next fetch-odds run. The one unique key on `event_id`, `predictions (event_id, snapshot_id, model_name)`, cannot collide because the twin's snapshots move with its predictions.

## Bulk insert

`odds_lambda.storage.writers.OddsWriter.bulk_insert_odds` uses `ON CONFLICT DO NOTHING`, which has no conflict target to hit because `odds` has no unique constraint besides `id`, so it inserts duplicates; it has no caller outside tests (#448).

## Log helpers

`odds_lambda.storage.writers.OddsWriter.log_fetch` adds an `odds_core.models.FetchLog` ([fetch_logs](../collection/odds-ingestion-service.md#fetch-logs)) and `odds_lambda.storage.writers.OddsWriter.log_data_quality_issue` adds an `odds_core.models.DataQualityLog`.

## Related

- [Odds snapshot and odds tables](odds-tables.md)
- [How odds are stored](odds-storage-model.md)
- [Storage layer](storage-layer.md)
- [Odds API ingestion service](../collection/odds-ingestion-service.md)
