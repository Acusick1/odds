---
type: reference
title: Odds snapshot and odds tables
description: The odds_snapshots and odds columns and indexes, the raw_data format, and the provenance tags.
sources:
  - odds_core.models.OddsSnapshot
  - odds_core.models.Odds
  - odds_core.snapshot_utils.extract_odds_from_snapshot
  - odds_lambda.storage.writers.OddsWriter._parse_and_store_odds
  - odds_lambda.storage.writers.OddsWriter.bulk_insert_odds
  - odds_lambda.storage.readers.OddsReader
  - odds_lambda.oddsportal_common.build_raw_data
  - odds_lambda.betfair.adapter.build_raw_data
  - odds_lambda.oddsportal_adapter
  - scripts/ingest_football_data_uk.py
status: stable
---

# Odds snapshot and odds tables

Odds are stored twice ([How odds are stored](odds-storage-model.md)). `odds_core.models.OddsSnapshot` (table `odds_snapshots`) holds one row per fetch of one [event](events.md): the complete payload as JSON in `raw_data`. `odds_core.models.Odds` (table `odds`) holds the normalized half: one row per bookmaker, market and outcome for each snapshot. Both are written by [OddsWriter](odds-writer.md).

## odds_snapshots

- `event_id` (FK), `snapshot_time` (tz-aware), `raw_data` (JSON), `bookmaker_count`, `created_at`.
- `fetch_tier` (string value of the [fetch tier](fetch-tier.md) enum: `in_play`, `closing`, `pregame`, `sharp`, `early`, `opening`) and `hours_until_commence`, both nullable.
- `api_request_id`: free-text provenance tag, see [Provenance](#provenance).
- Indexes: single-column `event_id`, `snapshot_time`, `fetch_tier`; composite `ix_event_snapshot_time (event_id, snapshot_time)` and `ix_event_tier (event_id, fetch_tier)`.

## raw_data format

The Odds API per-event object: `bookmakers[]`, each with `key`, `title`, `last_update`, and `markets[]` of `{key, outcomes[{name, price, point?}]}` (American prices). Other sources adapt to this shape (`odds_lambda.oddsportal_common.build_raw_data`, `odds_lambda.betfair.adapter.build_raw_data`) and add a `source` key; non-API sources can also carry extra keys (`_snapshot_time` from `odds_lambda.oddsportal_common.build_raw_data`), which parsers must ignore. `odds_core.snapshot_utils.extract_odds_from_snapshot` turns it back into `odds_core.models.Odds` objects, optionally filtered by market or outcome; it returns `[]` when `bookmakers` is absent and matches market keys exactly. Football three-way markets are `1x2` from every source (the Odds API's `h2h` is relabelled at fetch) and other sports use `h2h`, so a query with the wrong key finds nothing ([Market-key dialects](../collection/sport-and-market-keys.md#market-key-dialects)).

## Provenance

Two free-text fields tag where a snapshot came from. Each producer sets them as literals independently; no enum or registry defines the vocabularies (#468).

| `raw_data["source"]` | Producer |
|---|---|
| (absent) | The Odds API payloads |
| `oddsportal` | OddsPortal historical/closing builds (`odds_lambda.oddsportal_common.build_raw_data`) |
| `oddsportal_live` | live feed adapter (`odds_lambda.oddsportal_adapter.convert_market`) |
| `betfair_api` | Betfair Exchange job (also adds `betfair_event_id`) |
| `football_data_uk` | `scripts/ingest_football_data_uk.py`; market key `1x2`; snapshot times are approximations recorded in `raw_data.timing` ([Historical ingest scripts](../collection/historical-ingest-scripts.md#snapshot-shape)) |

`api_request_id` is matched by training's `closing_source_priority`; the `data_source` filter matches event-ID prefixes (`op_`, `fduk_`) instead (`odds_analytics.training.data_preparation.filter_events_by_date_range`, #468) ([CLV and the training target](../modelling/clv-target.md), [Training config](../modelling/training-config.md)): `oddsportal` (or `oddsportal_<market>`), `oddsportal_closing`, `football_data_uk`, and `betfair_api_<iso time>`; null for Odds API rows.

## odds

- `id` (autoincrement PK), `event_id` (FK `events.id`), `bookmaker_key`, `bookmaker_title`, `market_key` (`h2h`, `spreads`, `totals`, and `1x2` for football), `outcome_name` (team or Over/Under).
- `price` (int, American odds; convert with [Odds arithmetic helpers](../foundations/odds-math.md) before comparing probabilities), `point` (float, null for moneyline).
- `odds_timestamp`: the snapshot time the row was valid for. `last_update`: the bookmaker's own timestamp, defaulting to the snapshot time. `created_at`.
- `is_valid` (default true) and `validation_notes` (nullable): `is_valid` is always true and `validation_notes` always null, because the writer sets the defaults (#448).
- Indexes: single-column `event_id`, `bookmaker_key`, `market_key`, `odds_timestamp`; composite `ix_event_bookmaker_market (event_id, bookmaker_key, market_key)` for lookups and line movement, and `ix_bookmaker_time (bookmaker_key, odds_timestamp)` for per-bookmaker analysis.
- `odds` has no unique constraint beyond `id`, so storing a snapshot twice duplicates its rows (#448).

Written by `odds_lambda.storage.writers.OddsWriter._parse_and_store_odds` and `odds_lambda.storage.writers.OddsWriter.bulk_insert_odds`; read for prices by three `odds_lambda.storage.readers.OddsReader` methods, `odds_lambda.storage.readers.OddsReader.get_odds_at_time`, `odds_lambda.storage.readers.OddsReader.get_line_movement` and `odds_lambda.storage.readers.OddsReader.get_best_odds` ([OddsReader](odds-reader.md)).

## Related

- [OddsWriter](odds-writer.md)
- [How odds are stored](odds-storage-model.md)
- [Events table](events.md)
- [Fetch tier](fetch-tier.md)
