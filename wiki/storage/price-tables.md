---
type: reference
title: Prices and snapshot payloads tables
description: The prices and snapshot_payloads tables, the closed sets their text columns take, how writers fill them from each source's envelope, and the policy store_snapshot applies when a snapshot's key is already stored.
sources:
  - odds_core.models.Price
  - odds_core.models.SnapshotPayload
  - odds_core.vocabulary
  - odds_lambda.storage.snapshot_store
  - odds_lambda.storage.shaped_snapshots
  - odds_lambda.price_rows
  - odds_core.bookmakers.BOOKMAKER_ALIASES_BY_SOURCE
  - migrations/versions/9c822a7a3faf_expand_odds_data_model.py
status: stable
---

# Prices and snapshot payloads tables

`prices` holds one row per price in a snapshot, and `snapshot_payloads` holds what a source returned. They sit beside the `odds` table and `odds_snapshots.raw_data` ([Odds snapshot and odds tables](odds-tables.md)), which every reader uses. Every snapshot writer fills them alongside those ([Writing price rows](#writing-price-rows)), and `odds data backfill-shape` fills them for the snapshots stored before the writers did ([Shape backfill and verify](shape-backfill.md)). The snapshot columns they rely on (`source`, `market`, `time_basis` and the market metadata) are on [odds_snapshots](odds-tables.md#odds_snapshots).

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
| `odds_core.vocabulary.MarketStatus` | inactive, open, suspended, closed (Betfair's statuses, lower-cased) | `odds_snapshots.market_status` |
| `odds_core.vocabulary.PriceOutcome` | home, draw, away, over, under | `prices.outcome` |
| `odds_core.vocabulary.Side` | back, lay, mid, traded | `prices.side` |
| `odds_core.vocabulary.HealthCheck` | missing_mlb_game_pk | `health_check_acknowledgements.check_name` |

## Writing price rows

A writer builds one envelope per event (`raw_data`, [raw_data format](odds-tables.md#raw_data-format)), and `odds_lambda.storage.shaped_snapshots.store_shaped_snapshot` stores its source, market, time basis, market metadata, payload and price rows. It is called by `odds_lambda.storage.writers.OddsWriter.store_odds_snapshot` when the writer passes a `odds_lambda.storage.shaped_snapshots.SnapshotShape` ([OddsWriter](odds-writer.md#store_odds_snapshot)). Odds API ingestion stores its snapshots as `observed`, and the Odds API backfill as `upper_bound`: the historical endpoint returns the last snapshot at or before the requested time.

1. `odds_lambda.price_rows.expand_envelope` reads the envelope as one `odds_lambda.price_rows.MarketSnapshot` per market, in `odds_lambda.price_rows.MARKET_ORDER` (1x2, h2h, spreads, totals), with its price rows and, for Betfair, its `odds_lambda.price_rows.MarketMeta`. What it could not turn into a row comes back as an `odds_lambda.price_rows.RowIssue` of an `odds_lambda.price_rows.RowIssueType`, and each is logged to `data_quality_logs` (`odds_lambda.storage.data_quality.record_issue`).
2. The writer's snapshot takes the first market, the shape's source, time basis and `payload_id`, and the market metadata, and is stored with its prices through `odds_lambda.storage.snapshot_store.store_snapshot` ([storing a snapshot under its key](#storing-a-snapshot-under-its-key)).
3. An envelope quoting several markets (the Odds API's 1x2, spreads and totals) becomes one snapshot per market at the same time, with the same tier, `api_request_id` and payload. Each one's `raw_data` is the envelope cut to its own market (`odds_lambda.price_rows.market_envelope`), the first included, so no market is held twice. The odds rows go on the snapshot of their own market.
4. An envelope that quotes no market is added as it is, with `source`, `time_basis` and `payload_id` set and no market or prices.

A single-market envelope (OddsPortal, Betfair, FDUK, Polymarket) is one snapshot with its whole `raw_data`.

### Reading an envelope

| Source | Rows |
|---|---|
| OddsPortal, the Odds API, FDUK | a `back` row per bookmaker and outcome. `changed_at` is the book's `last_update` for the Odds API (the market's when given) and OddsPortal, and NULL for FDUK, whose stamp is ours. FDUK's `market_max` and `market_avg` (`odds_lambda.price_rows.FDUK_AGGREGATE_BOOKMAKERS`) are not written. OddsPortal live's per-outcome `betfair_matched` volume goes in `matched` on the rows of the entry carrying it (the Betfair Exchange entry). |
| The Odds API `h2h_lay` | `lay` rows of the 1x2 snapshot, Draw included, or of the h2h snapshot when it quotes no Draw (`odds_lambda.price_rows.stored_market`, which `odds_lambda.price_rows.market_envelope` also cuts by) |
| Betfair | from `betfair_meta.runners`: `back` (best back, with the back size), `lay` (best lay, with the lay size) and `traded` (last price traded); the runner name is matched after its pitcher annotation is stripped, and `The Draw` is the draw. `inplay`, `market_status` (lower-cased into `odds_core.vocabulary.MarketStatus`; an unknown status is logged and left NULL), `total_matched` and `market_start_time` go on the snapshot. |
| Polymarket | from `polymarket_meta`, as decimal odds 1/p: the Yes midpoint as `mid`, the ask as `back` and the bid as `lay`. History snapshots carry only `mid`. |

- **Outcomes** are matched once, here: a name equal to the event's home or away team after `odds_core.team.normalize_team_name`, `Draw` on 1x2, `Over` and `Under` on totals. Spreads and totals carry the outcome's `point` in `line`. An unmatched name is logged as `unmatched_outcome`; the payload keeps it for a re-parse.
- **Bookmakers** are canonicalised through the source's alias map, `odds_core.bookmakers.BOOKMAKER_ALIASES_BY_SOURCE` (`odds_core.bookmakers.canonical_bookmaker`): every source takes `odds_core.bookmakers.BOOKMAKER_ALIASES`, and the Odds API also `odds_core.bookmakers.ODDS_API_BOOKMAKER_KEY_MAP` (its regional Betfair keys, which the fetch already rewrites) and `odds_core.bookmakers.ODDS_API_KEY_ALIASES` (US BetMGM is `betmgm_us`). When two entries of one envelope name one book in a market (`7bet` and `7betuk`), the first is kept and the second logged as `bookmaker_alias_collision`. A second quote for the same book, outcome, line and side is logged as `duplicate_price`.
- **Invalid prices**, a decimal price not above 1.0 (or not a number) or a probability not strictly between 0 and 1, are not stored and are logged as `invalid_price`. An unknown market key is logged as `unknown_market`.

### time_basis and payloads

| Writer | `time_basis` | Payload |
|---|---|---|
| OddsPortal live (`odds_lambda.jobs.fetch_oddsportal.ingest_league`) | `observed` | the run's envelopes, one payload per league run |
| Betfair (`odds_lambda.jobs.fetch_betfair_exchange.ingest_sport`) | `observed` | the run's envelopes, one payload per sport run |

`odds_lambda.storage.shaped_snapshots.store_payload` adds the payload in the writer's transaction before its snapshots. A payload is the envelopes as the writer stored them, so `odds_lambda.price_rows.expand_envelope` re-reads it. A writer whose snapshots were all skipped as already stored removes the payload with `odds_lambda.storage.shaped_snapshots.discard_payload`.

## Storing a snapshot under its key

`odds_lambda.storage.snapshot_store.store_snapshot(session, snapshot, prices)` stores a snapshot with its prices under the key (event, source, market, snapshot time), `uq_odds_snapshots_event_source_market_time`. It raises `ValueError` when the snapshot has no `source` or `market`. The snapshot already holding the key is always kept, and the result (`odds_lambda.storage.snapshot_store.SnapshotWrite`) names the stored snapshot:

| Outcome | When | Effect |
|---|---|---|
| `inserted` | no snapshot holds the key | the snapshot and its prices are added |
| `identical` | the stored snapshot has the same time basis, market metadata and set of prices | nothing is written |
| `conflicting` | anything differs | nothing is written; `odds_lambda.storage.data_quality.record_issue` logs `snapshot_key_conflict` and adds a `data_quality_logs` row naming the kept snapshot |

Prices compare by value (`odds_lambda.storage.snapshot_store.PriceRow`, `odds_lambda.storage.snapshot_store.SnapshotMarketState`), so a float price equal to a stored decimal is the same price. The insert runs in a savepoint: when a concurrent writer stores the key first, the unique violation rolls back only this insert and the snapshot is compared with the stored one. It flushes and never commits.

## Related

- [Odds snapshot and odds tables](odds-tables.md)
- [Events table](events.md)
- [How odds are stored](odds-storage-model.md)
