---
type: reference
title: Prices and snapshot payloads tables
description: The prices and snapshot_payloads tables, the closed sets their text columns take, how writers fill them from each source's envelope and what each archives, how readers query them, and the policy store_snapshot applies when a snapshot's key is already stored.
sources:
  - odds_core.models.Price
  - odds_core.models.SnapshotPayload
  - odds_core.vocabulary
  - odds_lambda.storage.snapshot_store
  - odds_lambda.storage.shaped_snapshots
  - odds_lambda.price_rows
  - odds_core.bookmakers.BOOKMAKER_ALIASES_BY_SOURCE
  - migrations/versions/9c822a7a3faf_expand_odds_data_model.py
  - odds_core.quotes
  - odds_lambda.oddsportal_adapter.convert_market
status: stable
---

# Prices and snapshot payloads tables

`prices` holds one row per price in a snapshot, and `snapshot_payloads` holds the envelopes the writers built from what a source returned. Every snapshot writer fills both ([Writing price rows](#writing-price-rows)), and every price reader reads `prices` ([Reading quotes](#reading-quotes)). The snapshot columns they rely on (`source`, `market`, `time_basis` and the market metadata) are on [odds_snapshots](odds-tables.md#odds_snapshots); why prices are stored this way is on [How odds are stored](odds-storage-model.md).

## prices

`odds_core.models.Price`, one row per snapshot × bookmaker × outcome × line × side:

| Column | Type | Meaning |
|---|---|---|
| `id` | bigint identity | key |
| `snapshot_id` | integer FK `odds_snapshots.id`, `ON DELETE CASCADE` | the snapshot the price came from; deleting the snapshot deletes its prices |
| `bookmaker` | text | bookmaker key |
| `outcome` | text, closed set | `odds_core.vocabulary.PriceOutcome`: home, draw, away, over, under |
| `line` | numeric, nullable | the totals or spreads line; a spread's is signed per outcome ([sign](#spreads-line-sign)) |
| `side` | text, closed set | `odds_core.vocabulary.Side`: back, lay, mid, traded |
| `price` | numeric | decimal odds, above 1.0 (`ck_prices_price_gt_one`) |
| `size` | numeric, nullable | amount available at that price and side |
| `matched` | numeric, nullable | volume traded on the outcome |
| `changed_at` | timestamptz, nullable | the source's last-change time for this price |

`uq_prices_snapshot_book_outcome_line_side` makes (`snapshot_id`, `bookmaker`, `outcome`, `line`, `side`) unique with `NULLS NOT DISTINCT`, so two rows with no line for the same book, outcome and side collide.

## snapshot_payloads

`odds_core.models.SnapshotPayload`: `id` (bigint identity), `payload` (JSONB) and `received_at`. A snapshot points at its payload through `odds_snapshots.payload_id`, and several snapshots can share one. `odds_core.models.OddsSnapshot` maps no relationship to it, so code that has a snapshot reaches a payload only by querying the table. A payload is the envelopes as the writer stored them ([envelope format](odds-tables.md#envelope-format)), so `odds_lambda.price_rows.expand_envelope` reads it again:

| Writer | `time_basis` | Payload |
|---|---|---|
| OddsPortal live (`odds_lambda.jobs.fetch_oddsportal.ingest_league`) | `observed` | a list of the run's envelopes, one payload per league run |
| Betfair (`odds_lambda.jobs.fetch_betfair_exchange.ingest_sport`) | `observed` | a list of the run's envelopes (one per event, all its markets), one payload per sport run (EPL, MLB and NBA) |
| Polymarket live (`odds_lambda.jobs.fetch_polymarket_odds`, EPL and NBA) | `observed` | a list of the run's envelopes |
| Odds API ingestion (`odds_lambda.ingestion.OddsIngestionService.ingest_sport`) | `observed` | the response's list of event envelopes, one payload per run, beside its `fetch_logs` row |
| Odds API backfill (`odds_analytics.backfill_executor.BackfillExecutor`) | `observed`: the time is the historical endpoint's own snapshot time, the latest at or before the requested time | the response's list of event envelopes, one payload per call |
| Polymarket history (`odds_lambda.polymarket_odds_backfill.backfill_sport`, EPL and NBA) | `observed` | one envelope per grid point |
| OddsPortal history (`scripts/ingest_oddsportal.py`, any sport; NBA games with an unknown side are skipped) | `source_stamped` | a list of the game's opening and closing envelopes; none for a game whose event match is ambiguous |
| football-data.co.uk (`scripts/ingest_football_data_uk.py`) | `upper_bound` (opening) or `close` | a list of the game's opening and closing envelopes; none for a game whose event match is ambiguous |

`odds_lambda.storage.shaped_snapshots.store_payload` adds the payload in the writer's transaction before its snapshots. A writer whose snapshots were all skipped as already stored removes the payload with `odds_lambda.storage.shaped_snapshots.discard_payload`. Payloads rebuilt from stored prices carry `rebuilt_from` ([Rebuilt payloads](odds-tables.md#rebuilt-payloads)).

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

A writer builds one envelope per event ([envelope format](odds-tables.md#envelope-format)) and passes it, with its event, snapshot time and `odds_lambda.storage.shaped_snapshots.SnapshotShape` (source, time basis and payload id), to `odds_lambda.storage.shaped_snapshots.store_shaped_snapshot`, directly (the historical scripts and the Polymarket history backfill) or through `odds_lambda.storage.writers.OddsWriter.store_envelope` ([OddsWriter](odds-writer.md#store_envelope)).

1. `odds_lambda.price_rows.expand_envelope` reads the envelope as one `odds_lambda.price_rows.MarketSnapshot` per market, in `odds_lambda.price_rows.MARKET_ORDER` (1x2, h2h, spreads, totals), with its price rows and, for Betfair, its `odds_lambda.price_rows.MarketMeta`. What it could not turn into a row comes back as an `odds_lambda.price_rows.RowIssue` of an `odds_lambda.price_rows.RowIssueType`, and each is logged to `data_quality_logs` (`odds_lambda.storage.data_quality.record_issue`).
2. Each market becomes a snapshot of the event at the snapshot time with the shape's source, time basis and `payload_id`, the market and its market metadata, stored with its prices through `odds_lambda.storage.snapshot_store.store_snapshot` ([storing a snapshot under its key](#storing-a-snapshot-under-its-key)). The first market's write is returned as the envelope's snapshot. All of an envelope's snapshots share its payload, so an Odds API fetch quoting 1x2, spreads and totals is three snapshots at one time, and a Betfair EPL envelope, one `betfair_exchange` entry per market (match odds and over/under 2.5) each with its own `betfair_meta`, is two.
3. An envelope that quotes no market is stored as one snapshot with `source`, `time_basis` and `payload_id` set and no market or prices.

### Reading an envelope

| Source | Rows |
|---|---|
| OddsPortal, the Odds API, FDUK | a `back` row per bookmaker and outcome. `changed_at` is the book's `last_update` for the Odds API (the market's when given) and OddsPortal, and NULL for FDUK, whose stamp is ours. FDUK's `market_max` and `market_avg` (`odds_lambda.price_rows.FDUK_AGGREGATE_BOOKMAKERS`) are not written. OddsPortal live's per-outcome `betfair_matched` volume goes in `matched` on the rows of the entry carrying it (the Betfair Exchange entry), or of the market element carrying it (a ladder line's). |
| The Odds API `h2h_lay` | `lay` rows of the 1x2 snapshot, Draw included, or of the h2h snapshot when it quotes no Draw (`odds_lambda.price_rows.stored_market`) |
| Betfair | from `betfair_meta.runners`: `back` (best back, with the back size), `lay` (best lay, with the lay size) and `traded` (last price traded); the runner name is matched after its pitcher annotation is stripped, `The Draw` is the draw, and on `totals` an over/under runner ("Over 2.5 Goals") gives the outcome and the line (`odds_lambda.betfair.adapter.totals_runner`). `inplay`, `market_status` (lower-cased into `odds_core.vocabulary.MarketStatus`; an unknown status is logged and left NULL), `total_matched` and `market_start_time` go on the snapshot. |
| Polymarket | from `polymarket_meta`, as decimal odds 1/p: the Yes midpoint as `mid`, the ask as `back` and the bid as `lay`. History snapshots carry only `mid`. |

- **Outcomes** are matched once, here: a name equal to the event's home or away team after `odds_core.team.normalize_team_name`, `Draw` on 1x2, `Over` and `Under` on totals. Spreads and totals carry the outcome's `point` in `line`. An unmatched name is logged as `unmatched_outcome`; the payload keeps it for a re-parse.
- **Bookmakers** are canonicalised through the source's alias map, `odds_core.bookmakers.BOOKMAKER_ALIASES_BY_SOURCE` (`odds_core.bookmakers.canonical_bookmaker`): every source takes `odds_core.bookmakers.BOOKMAKER_ALIASES`, and the Odds API also `odds_core.bookmakers.ODDS_API_BOOKMAKER_KEY_MAP` (its regional Betfair keys, which the fetch already rewrites) and `odds_core.bookmakers.ODDS_API_KEY_ALIASES` (US BetMGM is `betmgm_us`). When two entries of one envelope name one book in a market (`7bet` and `7betuk`), the first is kept and the second logged as `bookmaker_alias_collision`. A second quote for the same book, outcome, line and side is logged as `duplicate_price`.
- **Invalid prices**, a decimal price not above 1.0 (or not a number) or a probability not strictly between 0 and 1, are not stored and are logged as `invalid_price`. An unknown market key is logged as `unknown_market`.

### Spreads line sign

A spread's `line` is stored signed on each outcome: the home outcome carries the home team's handicap and the away outcome its negative, so a home favourite at -5.5 is a `home` row at -5.5 and an `away` row at +5.5. A totals line is the same on `over` and `under`. A reader that pairs the two sides of a spread therefore negates one line (home line = -away line), and a reader of one team's price at a line filters on that outcome's own line. A book can quote several lines of one market in one snapshot (each is its own price rows, unique by line): OddsPortal's NBA totals and spreads store the main line and its neighbours ([ladder markets](../collection/oddsportal-scraper.md#ladder-markets)), and `odds_lambda.oddsportal_adapter.convert_market` writes the signs.

## Reading quotes

`odds_core.quotes.quotes(session, event_ids, market, *, sources, time_bases, bookmakers, sides, since, until, known_at, snapshot_ids, before_kickoff)` returns the price rows of the snapshots of one market for the given integer event ids as `odds_core.quotes.Quote`, a frozen dataclass of the row's fields (bookmaker, outcome, line, side, price, size, matched, changed_at) with its snapshot's id, time, `created_at`, source, market, time basis and market metadata. Numeric columns come back as floats. Each keyword narrows the result when given, in SQL: snapshot source and time basis, book and side, snapshot time from `since` to `until` inclusive, stored by `known_at`, among `snapshot_ids`, or within `before_kickoff` before its own event's current kick-off (a join on `events`), so one query serves many events' windows. Quotes come in snapshot time, snapshot id and price id order.

`odds_core.betfair_close.close_quotes`, the [Betfair close check](../operations/health-monitor.md#betfair-close-check), the [MCP market tools](../agent/mcp-market-tools.md) (`odds_mcp.prices.book_quotes`, which keeps back prices and Polymarket's mid), the [retail-EV rows](../modelling/clv-target-functions.md#retail-ev-rows) (`odds_analytics.retail_ev.load_retail_ev_dataset`) and the [feature pipeline](../modelling/feature-pipeline.md) (`odds_analytics.feature_groups.collect_event_data`, back prices of the `raw` target's market, with a close per source) read through it. The candidate rules read their retail, exchange and Polymarket prices through their own query, `odds_lambda.candidate_rules.inputs.rule_quotes`, which reads Polymarket only for a rule whose sharp inputs include it and returns the same `odds_core.quotes.Quote`, so the rule hash does not cover this function ([rule pinning](../evaluation/candidate-rules.md#rule-pinning)).

## Storing a snapshot under its key

`odds_lambda.storage.snapshot_store.store_snapshot(session, snapshot, prices)` stores a snapshot with its prices under the key (event, source, market, snapshot time), `uq_odds_snapshots_event_source_market_time`. It raises `ValueError` when the snapshot has no `source` or `market`. The snapshot already holding the key is always kept, and the result (`odds_lambda.storage.snapshot_store.SnapshotWrite`) names the stored snapshot:

| Outcome | When | Effect |
|---|---|---|
| `inserted` | no snapshot holds the key | the snapshot and its prices are added |
| `identical` | the stored snapshot has the same time basis, market metadata and set of prices | nothing is written |
| `conflicting` | anything differs | nothing is written; `odds_lambda.storage.data_quality.record_issue` logs `snapshot_key_conflict` (`odds_lambda.price_rows.RowIssueType.SNAPSHOT_KEY_CONFLICT`) and adds a `data_quality_logs` row naming the kept snapshot |

Prices compare by value (`odds_lambda.storage.snapshot_store.PriceRow`, `odds_lambda.storage.snapshot_store.SnapshotMarketState`), so a float price equal to a stored decimal is the same price. The insert runs in a savepoint: when a concurrent writer stores the key first, the unique violation rolls back only this insert and the snapshot is compared with the stored one. It flushes and never commits.

## Related

- [Odds snapshots table](odds-tables.md)
- [Events table](events.md)
- [How odds are stored](odds-storage-model.md)
