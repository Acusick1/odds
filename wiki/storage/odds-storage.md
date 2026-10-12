---
type: component
title: Odds snapshots, odds rows and OddsWriter
description: "The hybrid raw-plus-normalized odds storage, the raw_data format and provenance tags, how OddsWriter stores snapshots, assigns tiers and finds or creates events, and which sources lack normalized rows."
sources:
  - packages/odds-core/odds_core/models.py::OddsSnapshot
  - packages/odds-core/odds_core/models.py::Odds
  - packages/odds-core/odds_core/snapshot_utils.py::extract_odds_from_snapshot
  - packages/odds-lambda/odds_lambda/storage/writers.py::OddsWriter
  - packages/odds-lambda/odds_lambda/tier_utils.py::calculate_tier_from_timestamps
  - packages/odds-lambda/odds_lambda/tier_utils.py::calculate_tier
  - packages/odds-lambda/odds_lambda/oddsportal_common.py::build_raw_data
  - packages/odds-lambda/odds_lambda/oddsportal_common.py::hours_to_tier
  - packages/odds-lambda/odds_lambda/betfair/adapter.py
  - migrations/versions/9ff81f073e6b_recalculate_historical_fetch_tiers.py
  - docs/DATABASE.md#core-tables
  - docs/DATABASE.md#storage-strategy-hybrid-raw--normalized
  - docs/DATABASE.md#indexes
  - docs/DATABASE.md#batch-operations
  - docs/DEBUGGING_SCHEDULER.md#all-events-tagged-with-the-same-tier
verified: "7887039"
status: stable
---

# Odds snapshots, odds rows and OddsWriter

Odds are stored twice. `packages/odds-core/odds_core/models.py::OddsSnapshot` (table `odds_snapshots`) holds one row per fetch of one [event](events.md): the complete payload as JSON in `raw_data`. `packages/odds-core/odds_core/models.py::Odds` (table `odds`) holds the normalized half: one row per bookmaker, market and outcome for each snapshot. `packages/odds-lambda/odds_lambda/storage/writers.py::OddsWriter` writes both, per `docs/DATABASE.md#storage-strategy-hybrid-raw--normalized`: raw for debugging, auditing and exact preservation, normalized for SQL filtering and time series. Historical ingests write only the raw half, so `raw_data` is the one complete record ([howto-load-historical-odds](../collection/howto-load-historical-odds.md)).

## odds_snapshots

- `event_id` (FK), `snapshot_time` (tz-aware), `raw_data` (JSON), `bookmaker_count`, `created_at`.
- `fetch_tier` (string value of the [fetch tier](fetch-tier.md) enum: `in_play`, `closing`, `pregame`, `sharp`, `early`, `opening`) and `hours_until_commence`, both nullable.
- `api_request_id`: free-text provenance tag, see [Source tags](#source-tags).
- Indexes: single-column `event_id`, `snapshot_time`, `fetch_tier`; composite `ix_event_snapshot_time (event_id, snapshot_time)` and `ix_event_tier (event_id, fetch_tier)`.

### raw_data format

The Odds API per-event object: `bookmakers[]`, each with `key`, `title`, `last_update`, and `markets[]` of `{key, outcomes[{name, price, point?}]}` (American prices). Other sources adapt to this shape (`packages/odds-lambda/odds_lambda/oddsportal_common.py::build_raw_data`, `packages/odds-lambda/odds_lambda/betfair/adapter.py`) and add a `source` key. `packages/odds-core/odds_core/snapshot_utils.py::extract_odds_from_snapshot` turns it back into `Odds` objects, optionally filtered by market or outcome; it returns `[]` when `bookmakers` is absent and matches market keys exactly. Football is `h2h` (Odds API, FDUK) or `1x2` (OddsPortal, Betfair EPL), so a query with the wrong key finds nothing ([sources-and-keys](../collection/sources-and-keys.md)).

### Source tags

| `raw_data["source"]` | Producer |
|---|---|
| (absent) | The Odds API payloads |
| `oddsportal` | OddsPortal historical/closing builds (`build_raw_data`) |
| `oddsportal_live` | live-scrape adapter (`packages/odds-lambda/odds_lambda/oddsportal_adapter.py`) |
| `betfair_api` | Betfair Exchange job (also adds `betfair_event_id`) |
| `football_data_uk` | `scripts/ingest_football_data_uk.py` |

`api_request_id` is a second provenance channel, matched by training's `closing_source_priority` and `data_source` filters ([clv-target](../modelling/clv-target.md), [training-pipeline](../modelling/training-pipeline.md)): `oddsportal` (or `oddsportal_<market>`), `oddsportal_closing`, `football_data_uk`, and `betfair_api_<iso time>`; null for Odds API rows.

## odds

- `id` (autoincrement PK), `event_id` (FK `events.id`), `bookmaker_key`, `bookmaker_title`, `market_key` (`h2h`, `spreads`, `totals`, and `1x2` for football), `outcome_name` (team or Over/Under).
- `price` (int, American odds; convert with [odds-math-and-devigging](../foundations/odds-math-and-devigging.md) before comparing probabilities), `point` (float, null for moneyline).
- `odds_timestamp`: the snapshot time the row was valid for. `last_update`: the bookmaker's own timestamp, defaulting to the snapshot time. `created_at`.
- `is_valid` (default true) and `validation_notes` (nullable).
- Indexes: single-column `event_id`, `bookmaker_key`, `market_key`, `odds_timestamp`; composite `ix_event_bookmaker_market (event_id, bookmaker_key, market_key)` for lookups and line movement, and `ix_bookmaker_time (bookmaker_key, odds_timestamp)` for per-bookmaker analysis.

Written by `OddsWriter._parse_and_store_odds` and `OddsWriter.bulk_insert_odds`; read by `OddsReader.get_odds_at_time`, `get_line_movement` and `get_best_odds`. Many readers instead rebuild `Odds` objects from `raw_data` in memory and never persist them ([odds-reader-and-storage-layer](odds-reader-and-storage-layer.md)).

## OddsWriter

`OddsWriter` owns an `AsyncSession` and an `OddsValidator`. It never flushes or commits; callers do ([commit contract](odds-reader-and-storage-layer.md#commit-contract)).

### store_odds_snapshot

`OddsWriter.store_odds_snapshot(event_id, raw_data, snapshot_time=None, validate=True)`:

1. Loads the event and computes `hours_until_commence` and the fetch tier from `snapshot_time` and the event's own `commence_time` (`packages/odds-lambda/odds_lambda/tier_utils.py::calculate_tier_from_timestamps`). The tier is per event, not per job run. If the event row is missing, both `fetch_tier` and `hours_until_commence` are stored as NULL.
2. If `validate`, runs the validator and writes one `DataQualityLog` row for any warnings; storage proceeds regardless ([data-validation](data-validation.md)).
3. Adds an `OddsSnapshot` with the full `raw_data` and `bookmaker_count`.
4. `_parse_and_store_odds` adds one `Odds` row per bookmaker/market/outcome. Bookmakers without a key are skipped; `last_update` falls back to `snapshot_time` if unparseable; `odds_timestamp` is `snapshot_time`.

Returns `(OddsSnapshot, list[Odds])`.

### Events written by OddsWriter

- `upsert_event`: select-then-update. On an existing row it overwrites sport, teams, `commence_time` and `updated_at` only; status and scores are untouched.
- `find_or_create_event(home, away, match_date, sport_key, sport_title)`: normalizes team names, looks for same-sport events within ±2 hours, and returns `(event_id, was_created)`. New ids are `op_live_{HOME}_{AWAY}_{YYYY-MM-DDTHHMM}` (`OddsWriter._build_event_id`). Creation raises `ValueError` if the match started more than `LIVE_EVENT_STALE_THRESHOLD` (1 hour) ago. Used by the [OddsPortal](../collection/oddsportal.md) and [Betfair](../collection/betfair-exchange.md) jobs; matcher comparison on [events](events.md#matchers).
- `update_event_status`: sets scores and `completed_at` only when status is `FINAL` and both scores are given; scores passed with any other status are ignored.
- `bulk_upsert_events`: dedupes by id, then `ON CONFLICT (id) DO UPDATE` over every column except the key and `created_at`; returns approximate inserted/updated counts.

Also `log_fetch` (adds a `FetchLog`, see [odds-api-ingestion](../collection/odds-api-ingestion.md#fetch_logs)) and `log_data_quality_issue`.

## Hazards

- **Two provenance fields, no registry.** The vocabularies above are enumerated nowhere in code; each producer sets `source` and `api_request_id` independently.
- **Historical ingests skip `odds`.** `scripts/ingest_oddsportal.py` and `scripts/ingest_football_data_uk.py` construct only `Event` and `OddsSnapshot` rows, so `get_odds_at_time` and backtests return nothing for those events; use `raw_data` parsing ([backtesting](../evaluation/backtesting.md)).
- **`is_valid` and `validation_notes` are never set to anything but defaults.** The writer hard-codes `is_valid=True`; validation never blocks a write and its warnings go only to `data_quality_logs`. Filtering on `is_valid` filters nothing.
- **No uniqueness on `odds`.** There is no unique constraint on `(event_id, bookmaker_key, market_key, outcome_name, point, odds_timestamp)`, so re-ingesting a snapshot duplicates rows.
- **`bulk_insert_odds` does not dedupe.** It uses `ON CONFLICT DO NOTHING`, but `odds` has no unique constraint besides its autoincrement `id`, so re-running a backfill inserts duplicates. It has no non-test callers; `docs/DATABASE.md#batch-operations` recommends it for backfills.
- **Price 0 for absent price.** The writer stores `int(outcome.get("price", 0))`; the validator skips `None` prices, so nothing flags it. Raw-based readers see prices as the source gave them.
- **Tier is NULL when the event is missing** at write time. Tier boundaries: `packages/odds-lambda/odds_lambda/tier_utils.py::calculate_tier` uses `<=` at 3/12/24/72 h; `packages/odds-lambda/odds_lambda/oddsportal_common.py::hours_to_tier` (historical scripts) uses `<`, so snapshots at exactly those instants can get different tiers ([fetch-tier](fetch-tier.md#four-boundary-encodings)). Migration `9ff81f073e6b` (`migrations/versions/9ff81f073e6b_recalculate_historical_fetch_tiers.py`) recalculated rows stored before per-event tiering.
- **Candidate tie-break.** With several matching events in the window, `find_or_create_event` logs `ambiguous_event_match` and returns the first row of an unordered query.
- **Dead list branch.** `store_odds_snapshot` calls `raw_data.get("bookmakers")` before its `isinstance(raw_data, list)` check, so a list input raises `AttributeError`. Pass one event's dict.
- **Validate flag.** Only `OddsIngestionService` passes `validate` (from `DataQualityConfig.enable_validation`); every other caller uses the default `True`.
- **Extra keys.** Non-API sources can carry extra keys (`_snapshot_time` from `build_raw_data`); parsers must ignore unknown keys.

## Related

- [Odds API ingestion](../collection/odds-api-ingestion.md)
- [Events table and matching](events.md)
- [OddsReader and the storage layer](odds-reader-and-storage-layer.md)
- [Fetch tier](fetch-tier.md)
