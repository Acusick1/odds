---
type: component
title: odds_snapshots table and hybrid storage
summary: One raw JSON snapshot per fetch, tagged with fetch_tier. Covers the hybrid raw-plus-normalized design, the raw_data format and the source vocabulary.
sources:
  - packages/odds-core/odds_core/models.py::OddsSnapshot
  - packages/odds-core/odds_core/snapshot_utils.py::extract_odds_from_snapshot
  - packages/odds-lambda/odds_lambda/storage/writers.py::OddsWriter
  - packages/odds-lambda/odds_lambda/oddsportal_common.py::build_raw_data
  - packages/odds-lambda/odds_lambda/betfair/adapter.py
  - docs/DATABASE.md#core-tables
  - docs/DATABASE.md#storage-strategy-hybrid-raw--normalized
  - docs/DATABASE.md#indexes
verified_at: 7887039
---

# odds_snapshots table and hybrid storage

`packages/odds-core/odds_core/models.py::OddsSnapshot` (table `odds_snapshots`) holds one row per fetch of one [event](table-events.md): the complete payload as JSON in `raw_data`.

## Hybrid design

Every live fetch is stored twice by [OddsWriter](odds-writer.md): raw here (for debugging, auditing, schema flexibility, exact preservation) and normalized in [odds](table-odds.md) (for SQL filtering and time series), per `docs/DATABASE.md#storage-strategy-hybrid-raw--normalized`. Historical ingests write only the raw half ([historical odds ingestion](historical-odds-ingestion.md)), so `raw_data` is the one complete record.

## Columns

- `event_id` (FK), `snapshot_time` (tz-aware), `raw_data` (JSON), `bookmaker_count`, `created_at`.
- `fetch_tier` (string value of the [FetchTier](fetch-tier.md) enum: `in_play`, `closing`, `pregame`, `sharp`, `early`, `opening`) and `hours_until_commence`, both nullable. The live writer sets them from the event's commence time ([OddsWriter](odds-writer.md)).
- `api_request_id`: free-text provenance tag, see below.
- Indexes: single-column `event_id`, `snapshot_time`, `fetch_tier`; composite `ix_event_snapshot_time (event_id, snapshot_time)` for time series and `ix_event_tier (event_id, fetch_tier)` for tier analysis.

## raw_data format

The Odds API per-event object: `bookmakers[]`, each with `key`, `title`, `last_update`, and `markets[]` of `{key, outcomes[{name, price, point?}]}` (American prices). Other sources adapt to this shape (`packages/odds-lambda/odds_lambda/oddsportal_common.py::build_raw_data`, `packages/odds-lambda/odds_lambda/betfair/adapter.py`) and add a `source` key. `packages/odds-core/odds_core/snapshot_utils.py::extract_odds_from_snapshot` turns it back into `Odds` objects, optionally filtered by market or outcome; it returns `[]` when `bookmakers` is absent.

## `source` tags

| `raw_data["source"]` | Producer |
|---|---|
| (absent) | The Odds API payloads |
| `oddsportal` | OddsPortal historical/closing builds (`build_raw_data`) |
| `oddsportal_live` | live-scrape adapter (`packages/odds-lambda/odds_lambda/oddsportal_adapter.py`) |
| `betfair_api` | Betfair Exchange job (also adds `betfair_event_id`) |
| `football_data_uk` | `scripts/ingest_football_data_uk.py` |

`api_request_id` is a second provenance channel, matched by training's `closing_source_priority` and `data_source` filters: `oddsportal` (or `oddsportal_<market>`), `oddsportal_closing`, `football_data_uk`, and `betfair_api_<iso time>`; null for Odds API rows.

## Hazards

- **Two provenance fields, no registry.** The vocabularies above are enumerated nowhere in code; each producer sets `source` and `api_request_id` independently.
- **Tier is NULL when the event is missing** at write time, and scraped historical rows get tiers from `hours_to_tier` with `<` boundaries versus `<=` in the live path ([OddsWriter hazards](odds-writer.md#hazards)).
- **Non-API sources can carry extra keys** (`_snapshot_time` from `build_raw_data`); parsers must ignore unknown keys.
- **Re-parsing differs from stored `odds`.** Raw-based readers see prices as the source gave them; `Odds` rows use `int(...)` with default 0 ([OddsReader](odds-reader.md#which-path-each-method-uses)).

## Related

- [Storage layer](storage-layer.md)
- [OddsPortal scraper](oddsportal-scraper.md)
- [Data sources](data-sources.md)
