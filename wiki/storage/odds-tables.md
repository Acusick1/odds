---
type: reference
title: Odds snapshots table
description: The odds_snapshots columns, key and indexes, the envelope format writers build and archive as a snapshot's payload, and the tags producers put in it.
sources:
  - odds_core.models.OddsSnapshot
  - odds_lambda.storage.shaped_snapshots.store_shaped_snapshot
  - odds_lambda.oddsportal_common.build_envelope
  - odds_lambda.oddsportal_adapter
  - odds_lambda.polymarket_odds.build_envelope
  - scripts/ingest_football_data_uk.py
  - migrations/versions/a6f3c1e9d2b4_shape_orphan_odds_rows.py
  - migrations/versions/b8e2d5f7a1c3_drop_old_odds_representations.py
mentions:
  - odds_lambda.price_rows.expand_envelope
status: stable
---

# Odds snapshots table

`odds_core.models.OddsSnapshot` (table `odds_snapshots`) holds one source's prices for one market of an [event](events.md) at one time. Its prices are rows of the `prices` table, and the envelope they were read from is archived in `snapshot_payloads` ([Prices and snapshot payloads tables](price-tables.md)). Why odds are stored this way is on [How odds are stored](odds-storage-model.md). Every writer stores snapshots through `odds_lambda.storage.shaped_snapshots.store_shaped_snapshot` ([Writing price rows](price-tables.md#writing-price-rows)).

## odds_snapshots

| Column | Meaning |
|---|---|
| `id` | integer key |
| `event_id` | integer FK `events.id` |
| `snapshot_time` | when the prices held, by `time_basis` |
| `created_at` | when the row was stored; [as-of readers](price-tables.md#reading-quotes) read only rows stored by their `known_at` |
| `source` | the provider (`odds_core.vocabulary.PRICE_SOURCES`) |
| `market` | `odds_core.vocabulary.Market`; NULL only for a snapshot whose envelope quoted no market |
| `time_basis` | what `snapshot_time` means (`odds_core.vocabulary.TimeBasis`): `observed` (the fetch time), `source_stamped` (the source's own stamp), `upper_bound` (taken at or before it) or `close` (the closing price) |
| `in_play`, `market_status`, `total_matched`, `market_start_time` | the exchange market's state (Betfair only, `odds_core.vocabulary.MarketStatus` a closed set) |
| `payload_id` | FK `snapshot_payloads.id`, the archived envelope ([snapshot_payloads](price-tables.md#snapshot_payloads)) |

`source`, `market`, `time_basis` and `market_status` are CHECKed closed sets ([closed sets](price-tables.md#closed-sets)). `source` and `time_basis` are nullable in the schema, but every writer sets them, and `migrations/versions/b8e2d5f7a1c3_drop_old_odds_representations.py` refuses to run while a stored snapshot lacks either, or while its snapshot holds fewer matching price rows than an `odds` row's price needs (team outcomes counted as home and away together).

- `uq_odds_snapshots_event_source_market_time` makes (`event_id`, `source`, `market`, `snapshot_time`) unique. PostgreSQL treats NULLs as distinct here, so snapshots with no market never conflict. How a writer handles a stored key is on [storing a snapshot under its key](price-tables.md#storing-a-snapshot-under-its-key).
- Indexes: single-column `event_id`, `snapshot_time` and `payload_id`; composite `ix_event_snapshot_time` (`event_id`, `snapshot_time`).

No column holds a tier or the hours to kick-off: readers derive the tier from the snapshot's time, source, time basis and market ([Fetch tier](fetch-tier.md#the-tier-a-snapshot-covers)).

## Envelope format

Every writer builds one envelope per event in the Odds API's per-event shape: `bookmakers[]`, each with `key`, `title`, `last_update`, and `markets[]` of `{key, outcomes[{name, price | price_decimal, point?}]}`. The Odds API's outcomes carry an integer American `price`. Other sources carry the source's decimal under `price_decimal` and no `price` (`odds_lambda.oddsportal_common.build_envelope`, `odds_lambda.betfair.adapter.betfair_book_to_bookmaker_entry`, `odds_lambda.polymarket_odds.build_envelope`); `odds_core.odds_math.outcome_price_decimal` reads either ([Odds arithmetic helpers](../foundations/odds-math.md#price-encodings)). Football three-way markets are `1x2` from every source (the Odds API's `h2h` is relabelled at fetch) and other sports use `h2h` ([Market-key dialects](../collection/sport-and-market-keys.md#market-key-dialects)).

`odds_lambda.price_rows.expand_envelope` reads an envelope into price rows ([Reading an envelope](price-tables.md#reading-an-envelope)) and ignores keys it does not know. Producers add such keys, which reach only the archived payload:

| Producer | Extra keys |
|---|---|
| OddsPortal live (`odds_lambda.oddsportal_adapter.convert_market`) | `source: "oddsportal_live"`, `oddsportal_event_id`; per book, `betfair_matched` on the Betfair Exchange entry (in a ladder market, on each line's market element instead, [ladder markets](../collection/oddsportal-scraper.md#ladder-markets)) |
| OddsPortal history (`odds_lambda.oddsportal_common.build_envelope`) | `source: "oddsportal"`; `_snapshot_time`, which `scripts/ingest_oddsportal.py` removes and uses as the snapshot time (its basketball relink and load keep only the 30 NBA teams, `odds_lambda.polymarket_matching.canonical_nba_team`); a game with an ambiguous event match stores none |
| Betfair (`odds_lambda.jobs.fetch_betfair_exchange.ingest_sport`) | `betfair_event_id`; per book, `betfair_meta` with the runners and market state (a book with no back prices stores no snapshot) |
| Polymarket (`odds_lambda.polymarket_odds.build_envelope`) | `source: "polymarket"`; per book, `polymarket_meta` with the outcome probabilities (`yes_probabilities`) and, live, the bid and ask |
| football-data.co.uk (`scripts/ingest_football_data_uk.py`) | `source: "football_data_uk"`, `timing` (`basis` and whether the kick-off time was assumed), `match_stats`; a game with an ambiguous event match stores no envelope |

## Rebuilt payloads

Some EPL OddsPortal live snapshots from March to August 2026 hold the market of a scrape whose snapshot was lost while its prices survived as per-outcome rows in the `odds` table. `migrations/versions/a6f3c1e9d2b4_shape_orphan_odds_rows.py` stored them as `oddsportal` / `observed` snapshots with payloads built from those rows, which carry the key `rebuilt_from` (value `odds`) and no `oddsportal_event_id`. A copy of production taken 2026-10-25 holds 5,909 such snapshots with 192,369 prices (#677). Downgrading that migration removes only the rebuilt snapshots whose `odds` rows still exist, so once `odds` is dropped they are kept.

## Related

- [Prices and snapshot payloads tables](price-tables.md)
- [How odds are stored](odds-storage-model.md)
- [OddsWriter](odds-writer.md)
- [Fetch tier](fetch-tier.md)
