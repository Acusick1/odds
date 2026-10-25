---
type: reference
title: Prices and snapshot payloads tables
description: The prices and snapshot_payloads tables, the closed sets their text columns take, and the policy store_snapshot applies when a snapshot's key is already stored.
sources:
  - odds_core.models.Price
  - odds_core.models.SnapshotPayload
  - odds_core.vocabulary
  - odds_lambda.storage.snapshot_store
  - migrations/versions/9c822a7a3faf_expand_odds_data_model.py
status: stable
---

# Prices and snapshot payloads tables

`prices` holds one row per price in a snapshot, and `snapshot_payloads` holds what a source returned. They sit beside the `odds` table and `odds_snapshots.raw_data` ([Odds snapshot and odds tables](odds-tables.md)), which every reader and writer still uses. No writer fills them yet. The snapshot columns they rely on (`source`, `market`, `time_basis` and the market metadata) are on [odds_snapshots](odds-tables.md#odds_snapshots).

## prices

`odds_core.models.Price`, one row per snapshot × bookmaker × outcome × line × side:

| Column | Type | Meaning |
|---|---|---|
| `id` | bigint identity | key |
| `snapshot_id` | integer FK `odds_snapshots.id`, `ON DELETE CASCADE` | the snapshot the price came from; deleting the snapshot deletes its prices |
| `bookmaker` | text | bookmaker key |
| `outcome` | text, closed set | `odds_core.vocabulary.PriceOutcome`: home, draw, away, over, under |
| `line` | numeric, nullable | the totals or spreads line |
| `side` | text, closed set | `odds_core.vocabulary.Side`: back, lay, mid, traded |
| `price` | numeric | decimal odds, above 1.0 (`ck_prices_price_gt_one`) |
| `size` | numeric, nullable | amount available at that price and side |
| `matched` | numeric, nullable | volume traded on the outcome |
| `changed_at` | timestamptz, nullable | the source's last-change time for this price |

`uq_prices_snapshot_book_outcome_line_side` makes (`snapshot_id`, `bookmaker`, `outcome`, `line`, `side`) unique with `NULLS NOT DISTINCT`, so two rows with no line for the same book, outcome and side collide.

## snapshot_payloads

`odds_core.models.SnapshotPayload`: `id` (bigint identity), `payload` (JSONB) and `received_at`. A snapshot points at its payload through `odds_snapshots.payload_id`, and several snapshots can share one. `odds_core.models.OddsSnapshot` maps no relationship to it, so code that has a snapshot reaches a payload only by querying the table.

## Closed sets

The closed sets are `StrEnum`s in `odds_core.vocabulary`, stored as their values in TEXT columns (`odds_core.vocabulary.StrEnumText`) with a CHECK constraint listing the allowed values (`odds_core.vocabulary.closed_set_check`). A new value needs a migration that replaces the CHECK.

| Set | Values | Columns |
|---|---|---|
| `odds_core.vocabulary.Source` | odds_api, oddsportal, betfair, football_data_uk, polymarket, mlb_stats_api, espn | `odds_snapshots.source` (only `odds_core.vocabulary.PRICE_SOURCES`), `event_external_ids.source` (only `odds_core.vocabulary.EXTERNAL_ID_SOURCES`) |
| `odds_core.vocabulary.Market` | 1x2, h2h, spreads, totals | `odds_snapshots.market` |
| `odds_core.vocabulary.TimeBasis` | observed, source_stamped, upper_bound, close | `odds_snapshots.time_basis` |
| `odds_core.vocabulary.PriceOutcome` | home, draw, away, over, under | `prices.outcome` |
| `odds_core.vocabulary.Side` | back, lay, mid, traded | `prices.side` |
| `odds_core.vocabulary.HealthCheck` | missing_mlb_game_pk | `health_check_acknowledgements.check_name` |

## Storing a snapshot under its key

`odds_lambda.storage.snapshot_store.store_snapshot(session, snapshot, prices)` stores a snapshot with its prices under the key (event, source, market, snapshot time), `uq_odds_snapshots_event_source_market_time`. It raises `ValueError` when the snapshot has no `source` or `market`. The snapshot already holding the key is always kept, and the result (`odds_lambda.storage.snapshot_store.SnapshotWrite`) names the stored snapshot:

| Outcome | When | Effect |
|---|---|---|
| `inserted` | no snapshot holds the key | the snapshot and its prices are added |
| `identical` | the stored snapshot has the same time basis, market metadata and set of prices | nothing is written |
| `conflicting` | anything differs | nothing is written; `snapshot_key_conflict` is logged and an `odds_core.models.DataQualityLog` row (issue type `snapshot_key_conflict`) names the kept snapshot |

Prices compare by value, so a float price equal to a stored decimal is the same price. The insert runs in a savepoint: when a concurrent writer stores the key first, the unique violation rolls back only this insert and the snapshot is compared with the stored one. It flushes and never commits.

## Related

- [Odds snapshot and odds tables](odds-tables.md)
- [Events table](events.md)
- [How odds are stored](odds-storage-model.md)
