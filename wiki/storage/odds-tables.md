---
type: reference
title: Odds snapshot and odds tables
description: The odds_snapshots and odds columns and indexes, how an odds row links to its snapshot, the raw_data format, the provenance tags, and same-time snapshot pairs.
sources:
  - odds_core.models.OddsSnapshot
  - odds_core.models.Odds
  - odds_core.snapshot_utils.extract_odds_from_snapshot
  - odds_lambda.storage.writers.OddsWriter._parse_and_store_odds
  - odds_lambda.storage.same_time_pairs
  - migrations/versions/b6e3d9f2a4c7_add_odds_snapshot_id.py
  - tests/integration/test_odds_snapshot_link.py
  - odds_lambda.storage.readers.OddsReader
  - odds_lambda.oddsportal_common.build_raw_data
  - odds_lambda.betfair.adapter.build_raw_data
  - odds_lambda.oddsportal_adapter
  - scripts/ingest_football_data_uk.py
  - odds_lambda.polymarket_odds.build_raw_data
  - odds_core.snapshot_sources
status: stable
---

# Odds snapshot and odds tables

Odds are stored twice ([How odds are stored](odds-storage-model.md)). `odds_core.models.OddsSnapshot` (table `odds_snapshots`) holds one row per fetch of one [event](events.md): the complete payload as JSON in `raw_data`. `odds_core.models.Odds` (table `odds`) holds the normalized half: one row per bookmaker, market and outcome for each snapshot, linked to it by `snapshot_id`. Both are written by [OddsWriter](odds-writer.md).

## odds_snapshots

- `event_id` (FK), `snapshot_time` (tz-aware), `raw_data` (JSON), `bookmaker_count`, `created_at`.
- `fetch_tier` (string value of the [fetch tier](fetch-tier.md) enum: `in_play`, `closing`, `pregame`, `sharp`, `early`, `opening`) and `hours_until_commence`, both nullable.
- `api_request_id`: free-text provenance tag, see [Provenance](#provenance).
- Indexes: single-column `event_id`, `snapshot_time`, `fetch_tier`, `payload_id`; composite `ix_event_snapshot_time (event_id, snapshot_time)` and `ix_event_tier (event_id, fetch_tier)`.

The columns of the redesigned snapshot are nullable. Snapshots the OddsPortal live and Betfair jobs store have them filled ([Writing price rows](price-tables.md#writing-price-rows)); every other snapshot holds NULL in them:

- `source` (`odds_core.vocabulary.PRICE_SOURCES`), `market` (`odds_core.vocabulary.Market`) and `time_basis` (`odds_core.vocabulary.TimeBasis`), each a CHECKed closed set ([closed sets](price-tables.md#closed-sets)).
- Market metadata: `in_play`, `market_status` (`odds_core.vocabulary.MarketStatus`, a closed set), `total_matched` and `market_start_time`.
- `payload_id`: FK to `snapshot_payloads.id` ([Prices and snapshot payloads tables](price-tables.md#snapshot_payloads)).
- `uq_odds_snapshots_event_source_market_time` makes (`event_id`, `source`, `market`, `snapshot_time`) unique. PostgreSQL treats NULLs as distinct here, so snapshots with no `source` or `market` are never in conflict. How a writer handles a stored key is on [storing a snapshot under its key](price-tables.md#storing-a-snapshot-under-its-key).

## raw_data format

The Odds API per-event object: `bookmakers[]`, each with `key`, `title`, `last_update`, and `markets[]` of `{key, outcomes[{name, price | price_decimal, point?}]}`. The Odds API's outcomes carry an integer American `price`. Other sources adapt to this shape and carry the source decimal under `price_decimal` with no `price` key (`odds_lambda.oddsportal_common.build_raw_data`, `odds_lambda.betfair.adapter.build_raw_data`) and add a `source` key; non-API sources can also carry extra keys (`_snapshot_time` from `odds_lambda.oddsportal_common.build_raw_data`), which parsers must ignore. `odds_core.odds_math.outcome_price_decimal` reads either encoding ([Odds arithmetic helpers](../foundations/odds-math.md#price-encodings)). `odds_core.snapshot_utils.extract_odds_from_snapshot` turns it back into `odds_core.models.Odds` objects, optionally filtered by market or outcome; it returns `[]` when `bookmakers` is absent and matches market keys exactly. Football three-way markets are `1x2` from every source (the Odds API's `h2h` is relabelled at fetch) and other sports use `h2h`, so a query with the wrong key finds nothing ([Market-key dialects](../collection/sport-and-market-keys.md#market-key-dialects)).

## Provenance

Two free-text fields tag where a snapshot came from. Each producer sets them as literals independently (#468). `odds_core.snapshot_sources.SnapshotSource` lists the `raw_data["source"]` values for readers, and `odds_core.snapshot_sources.snapshot_kind` classifies a snapshot from them, splitting FDUK by `timing.basis` into opening and closing prices; it raises on a value it does not know.

| `raw_data["source"]` | Producer |
|---|---|
| (absent) | The Odds API payloads |
| `oddsportal` | OddsPortal historical/closing builds (`odds_lambda.oddsportal_common.build_raw_data`) |
| `oddsportal_live` | live feed adapter (`odds_lambda.oddsportal_adapter.convert_market`; also stores the feed's encoded event id as `oddsportal_event_id`) |
| `betfair_api` | Betfair Exchange job (also adds `betfair_event_id`) |
| `football_data_uk` | `scripts/ingest_football_data_uk.py` (attaches to an existing event of the same sport when one matches, fills its scores when it has none and replaces its own snapshots on a rerun); market key `1x2`; snapshot times are approximations recorded in `raw_data.timing` and as `time_basis` `upper_bound` (opening) or `close` ([Historical ingest scripts](../collection/historical-ingest-scripts.md#snapshot-shape)) |
| `polymarket` | `odds_lambda.polymarket_odds.build_raw_data` (price is the reciprocal of the Yes probability as `price_decimal`; exact probabilities in `polymarket_meta`, bid and ask only on live rows; its snapshots have `source` `polymarket`, market `1x2`, time basis `observed`), from the backfill and the fetch-polymarket-odds job ([Polymarket odds snapshots](../collection/polymarket-odds-snapshots.md)) |

Training's `data_source` filter matches event-ID prefixes (`op_`, `fduk_`; `odds_core.event_ids.SCRAPED_EVENT_ID_PREFIXES`), not `api_request_id` (`odds_analytics.training.data_preparation.filter_events_by_date_range`, #468) ([Training config](../modelling/training-config.md)). The `api_request_id` values are `oddsportal` (or `oddsportal_<market>`), `football_data_uk`, `betfair_api_<iso time>`, `polymarket_history` and `polymarket_live_<iso time>`; null for Odds API rows.

## odds

- `id` (autoincrement PK), `event_id` (FK `events.id`), `snapshot_id` (FK `odds_snapshots.id`, nullable, see [Snapshot link](#snapshot-link)), `bookmaker_key`, `bookmaker_title`, `market_key` (`h2h`, `spreads`, `totals`, and `1x2` for football), `outcome_name` (team or Over/Under).
- `price_decimal` (float, not null, constrained above 1.0 by `ck_odds_price_decimal_gt_one`; the source's decimal price). Derived read-only properties: `odds_core.models.Odds.implied_probability` and `odds_core.models.Odds.american_price` (nearest integer American quote, for display and bet records) ([Odds arithmetic helpers](../foundations/odds-math.md#price-encodings)). `point` (float, null for moneyline).
- `odds_timestamp`: the snapshot time the row was valid for. `last_update`: the bookmaker's own timestamp, defaulting to the snapshot time. `created_at`.
- `is_valid` (default true) and `validation_notes` (nullable): `is_valid` is always true and `validation_notes` always null, because the writer sets the defaults (#448).
- Indexes: single-column `event_id`, `snapshot_id`, `bookmaker_key`, `market_key`, `odds_timestamp`; composite `ix_event_bookmaker_market (event_id, bookmaker_key, market_key)` for lookups and line movement, and `ix_bookmaker_time (bookmaker_key, odds_timestamp)` for per-bookmaker analysis.
- `odds` has no unique constraint beyond `id`, so storing a snapshot twice duplicates its rows (#448). A writer passing a snapshot shape stores no odds rows for a snapshot whose key is already stored ([OddsWriter](odds-writer.md#store_odds_snapshot)).

Written by `odds_lambda.storage.writers.OddsWriter._parse_and_store_odds` and copied, with the snapshots and their price rows, by `odds_lambda.copy_from_production.ProductionDataCopier`; each linked to the snapshot of its own market; read for prices by exactly three `odds_lambda.storage.readers.OddsReader` methods: `odds_lambda.storage.readers.OddsReader.get_odds_at_time`, `odds_lambda.storage.readers.OddsReader.get_line_movement` and `odds_lambda.storage.readers.OddsReader.get_best_odds` ([OddsReader](odds-reader.md)).

## Snapshot link

`snapshot_id` names the snapshot a row was parsed from. Every writer sets it: `odds_lambda.storage.writers.OddsWriter._parse_and_store_odds` from the snapshot it has just flushed, and `odds_lambda.copy_from_production.ProductionDataCopier` from the local copy of the production snapshot. `odds_core.snapshot_utils.extract_odds_from_snapshot` sets it on the transient rows it builds too, so a stored row and a re-parsed one carry the same link.

Rows written before the column existed were linked by `migrations/versions/b6e3d9f2a4c7_add_odds_snapshot_id.py` to the one snapshot at their event and `odds_timestamp` whose `raw_data` quotes their bookmaker (trimmed of whitespace, as the writer stores it), market, outcome and point. Two kinds of historical row keep a NULL `snapshot_id`:

- **Shared copies:** several snapshots quote the key. Each is one copy of a [same-time pair](#same-time-pairs), and `odds events delete-same-time-pairs` links or deletes it.
- **Orphans:** no snapshot at that event and time quotes the key, because the snapshot the row was parsed from is missing. The largest group is EPL OddsPortal live rows from March to August 2026 whose market's snapshot is missing while the other market's snapshot at that time survived. Orphans are kept with a NULL `snapshot_id`; nothing resolves them.

The column is therefore nullable, with no database constraint, while every writer links each new row; `tests/integration/test_odds_snapshot_link.py` fails if code outside the known writers builds odds rows, or if `odds_lambda.storage.writers.OddsWriter.store_envelope` leaves one unlinked; it links each row to the snapshot of its own market. `ix_odds_snapshot_id` makes `SELECT count(*) FROM odds WHERE snapshot_id IS NULL` cheap, and the command prints that count split into orphans and shared copies.

## Same-time pairs

Two snapshots of one event form a pair when they share `snapshot_time` and `raw_data["source"]` and quote at least one bookmaker, market, outcome and point in common (`odds_lambda.storage.same_time_pairs`). Snapshots of one time with disjoint keys, such as OddsPortal's one snapshot per market, are not pairs.

| Pair | Meaning | `odds events delete-same-time-pairs` |
|---|---|---|
| identical | the same keys at the same prices: one fetch stored twice | keeps the lowest id, deletes the others |
| conflicting, OddsPortal live on an MLB event | a shared key has two prices: two games' prices collapsed onto one event before MLB events were keyed to the `gamePk` (#535, #612) | deletes both snapshots |
| any other conflicting | no established cause; those seen so far are OddsPortal history snapshots of NBA and EPL events | leaves them |

Deleted snapshots take their odds rows. A NULL row whose key only deleted snapshots quote goes. Of a key's NULL copies that one kept snapshot and a deleted one quote, the copy at the kept snapshot's price stays, linked to it, and the rest go; prices compare within float rounding because stored rows were converted to decimal apart from `raw_data`. Any other NULL row is left. A group that would delete a snapshot listed in a candidate decision's `input_snapshot_ids` is refused, so its [replay](../evaluation/candidate-parity.md) still finds its inputs. The steps are in [Delete same-time snapshot pairs](howto-delete-same-time-pairs.md); `odds_lambda.storage.same_time_pairs.event_same_time_pairs` also counts one event's pairs for [`odds events assign-game-pks`](event-merge.md#assigning-gamepks).

## Related

- [OddsWriter](odds-writer.md)
- [Prices and snapshot payloads tables](price-tables.md)
- [How odds are stored](odds-storage-model.md)
- [Events table](events.md)
- [Fetch tier](fetch-tier.md)
