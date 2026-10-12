---
type: component
title: odds table
summary: Normalized odds, one row per bookmaker/market/outcome, with American prices and the query indexes.
sources:
  - packages/odds-core/odds_core/models.py::Odds
  - packages/odds-lambda/odds_lambda/storage/writers.py::OddsWriter
  - docs/DATABASE.md#core-tables
  - docs/DATABASE.md#indexes
verified_at: 7887039
---

# odds table

`packages/odds-core/odds_core/models.py::Odds` (table `odds`) is the normalized half of [hybrid storage](table-odds-snapshots.md): one row per bookmaker, market and outcome for each snapshot of an [event](table-events.md). The raw JSON stays in `odds_snapshots`.

## Columns

- `id` (autoincrement PK), `event_id` (FK `events.id`).
- `bookmaker_key`, `bookmaker_title`, `market_key` (`h2h`, `spreads`, `totals`, and `1x2` for football), `outcome_name` (team or Over/Under).
- `price` (int, American odds), `point` (float, null for moneyline).
- `odds_timestamp`: the snapshot time the row was valid for. `last_update`: the bookmaker's own timestamp, defaulting to the snapshot time. `created_at`.
- `is_valid` (default true) and `validation_notes` (nullable).

## Indexes

Single-column: `event_id`, `bookmaker_key`, `market_key`, `odds_timestamp`. Composite (`__table_args__`):

- `ix_event_bookmaker_market` on `(event_id, bookmaker_key, market_key)`: odds lookups, [line movement](odds-reader.md).
- `ix_bookmaker_time` on `(bookmaker_key, odds_timestamp)`: per-bookmaker analysis.

## Writers and readers

Written by `OddsWriter._parse_and_store_odds` (live path, see [OddsWriter](odds-writer.md)) and `OddsWriter.bulk_insert_odds`. Read by `OddsReader.get_odds_at_time`, `get_line_movement` and `get_best_odds`. Many readers instead rebuild `Odds` objects from `raw_data` in memory (`packages/odds-core/odds_core/snapshot_utils.py::extract_odds_from_snapshot`); those are never persisted.

## Hazards

- **`is_valid` and `validation_notes` are never set to anything but defaults.** The writer hard-codes `is_valid=True`; [validation](data-quality-validation.md) results go to `data_quality_logs` only. Filtering on `is_valid` filters nothing.
- **No uniqueness.** There is no unique constraint on `(event_id, bookmaker_key, market_key, outcome_name, point, odds_timestamp)`, so re-ingesting a snapshot duplicates rows, and `ON CONFLICT DO NOTHING` in `bulk_insert_odds` never fires.
- **Price 0 for absent price.** The writer stores `int(outcome.get("price", 0))`.
- **Historical ingests skip this table.** `scripts/ingest_oddsportal.py` and `scripts/ingest_football_data_uk.py` construct only `Event` and `OddsSnapshot` rows ([historical odds ingestion](historical-odds-ingestion.md)), so `get_odds_at_time` and the other normalized-row readers return nothing for those events; use `raw_data` parsing.
- **Prices are American.** Convert via [odds math](odds-math.md) before comparing probabilities.

## Related

- [Storage layer](storage-layer.md)
- [Look-ahead bias](look-ahead-bias.md)
