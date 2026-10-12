---
type: component
title: OddsWriter
summary: Writes OddsSnapshot and Odds rows, assigns the fetch tier per event, finds or creates events, and bulk-inserts for backfill.
sources:
  - packages/odds-lambda/odds_lambda/storage/writers.py::OddsWriter
  - packages/odds-lambda/odds_lambda/tier_utils.py::calculate_tier_from_timestamps
  - docs/DATABASE.md#storage-strategy-hybrid-raw--normalized
  - docs/DATABASE.md#batch-operations
  - docs/DEBUGGING_SCHEDULER.md#all-events-tagged-with-the-same-tier
verified_at: 7887039
---

# OddsWriter

`packages/odds-lambda/odds_lambda/storage/writers.py::OddsWriter` is the write path for odds data. It owns an `AsyncSession` and an [OddsValidator](data-quality-validation.md). It never flushes or commits; callers do ([storage layer](storage-layer.md#commit-contract)).

## store_odds_snapshot

`OddsWriter.store_odds_snapshot(event_id, raw_data, snapshot_time=None, validate=True)` stores each fetch twice ([hybrid storage](table-odds-snapshots.md)):

1. Loads the [event](table-events.md) and computes `hours_until_commence` and the [fetch tier](fetch-tier.md) from `snapshot_time` and the event's own `commence_time` (`packages/odds-lambda/odds_lambda/tier_utils.py::calculate_tier_from_timestamps`). The tier is per event, not per job run. If the event row is missing, both `fetch_tier` and `hours_until_commence` are stored as NULL.
2. If `validate`, runs the validator and writes one `DataQualityLog` row for any warnings; storage proceeds regardless.
3. Adds an `OddsSnapshot` with the full `raw_data` and `bookmaker_count`.
4. `_parse_and_store_odds` adds one [Odds](table-odds.md) row per bookmaker/market/outcome. Bookmakers without a key are skipped; `last_update` falls back to `snapshot_time` if unparseable; `odds_timestamp` is `snapshot_time`.

Returns `(OddsSnapshot, list[Odds])`.

## Events

- `upsert_event`: select-then-update. On an existing row it overwrites sport, teams, `commence_time` and `updated_at` only; status and scores are untouched.
- `find_or_create_event(home, away, match_date, sport_key, sport_title)`: normalizes team names, looks for same-sport events within plus/minus 2 hours, and returns `(event_id, was_created)`. New ids are `op_live_{HOME}_{AWAY}_{YYYY-MM-DDTHHMM}` (`OddsWriter._build_event_id`). Creation raises `ValueError` if the match started more than `LIVE_EVENT_STALE_THRESHOLD` (1 hour) ago. Used by the [OddsPortal](oddsportal-scraper.md) and [Betfair](betfair-exchange-client.md) jobs.
- `update_event_status`: sets scores and `completed_at` only when status is `FINAL` and both scores are given; scores passed with any other status are ignored.
- `bulk_upsert_events`: dedupes by id, then `ON CONFLICT (id) DO UPDATE` over every column except the key and `created_at`; returns approximate inserted/updated counts.

Also `log_fetch` (adds a `FetchLog`, see [fetch logs](table-fetch-logs.md)) and `log_data_quality_issue`.

## Hazards

- **Candidate tie-break.** With several matching events in the window, `find_or_create_event` logs `ambiguous_event_match` and returns the first row of an unordered query.
- **Dead list branch.** `store_odds_snapshot` calls `raw_data.get("bookmakers")` before its `isinstance(raw_data, list)` check, so a list input raises `AttributeError` before reaching it. Pass one event's dict.
- **Price 0.** A missing `price` key is stored as `0` via `int(outcome.get("price", 0))`; the validator skips `None` prices, so nothing flags it.
- **`bulk_insert_odds` does not dedupe.** It uses `ON CONFLICT DO NOTHING`, but `odds` has no unique constraint besides its autoincrement `id` (`packages/odds-core/odds_core/models.py::Odds`), so re-running a backfill inserts duplicates. It also has no non-test callers in the repo; `docs/DATABASE.md#batch-operations` recommends it for backfills.
- **Validate flag.** Only `OddsIngestionService` passes `validate` (from `DataQualityConfig.enable_validation`); every other caller uses the default `True`.
- **Tier boundaries.** `packages/odds-lambda/odds_lambda/tier_utils.py::calculate_tier` uses `<=` at 3/12/24/72 h; `packages/odds-lambda/odds_lambda/oddsportal_common.py::hours_to_tier` uses `<` instead, so snapshots at exactly those instants can get different tiers ([fetch tier](fetch-tier.md)). Migration `9ff81f073e6b` (`migrations/versions/9ff81f073e6b_recalculate_historical_fetch_tiers.py`) recalculated rows stored before per-event tiering.

## Related

- [Odds ingestion flow](odds-ingestion-flow.md)
- [Storage layer](storage-layer.md)
- [Event lifecycle](event-lifecycle.md)
