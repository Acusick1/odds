---
type: reference
title: Shape backfill and verify
description: What odds data backfill-shape writes for stored history (twin merges, UNMATCHED events, external ids, snapshot source, market, time basis, payloads, price rows and Odds API splits, legacy Polymarket prices), how it tells sources apart, its duplicate and decision rules, and what odds data verify-shape checks.
sources:
  - odds_cli.commands.data
  - odds_lambda.shape_backfill.run
  - odds_lambda.shape_backfill.snapshots
  - odds_lambda.shape_backfill.events
  - odds_lambda.shape_backfill.legacy_polymarket
  - odds_lambda.shape_backfill.verify
mentions:
  - odds_lambda.storage.event_resolver.lock_fixture
  - odds_lambda.storage.event_resolver.record_external_id
  - odds_lambda.storage.event_resolver.attach_events
  - odds_lambda.storage.event_merge.EventMerger.merge
  - odds_lambda.storage.event_unmatched.mark_unmatched
  - odds_lambda.price_rows.expand_envelope
  - odds_lambda.price_rows.market_envelope
status: stable
---

# Shape backfill and verify

The live writers fill the snapshot columns `source`, `market` and `time_basis`, the [`prices` and `snapshot_payloads` tables](price-tables.md) and [event external ids](events.md#external-ids) for what they store. `odds data backfill-shape` (`odds_cli.commands.data.backfill_shape`) does the same for everything stored before them, and `odds data verify-shape` (`odds_cli.commands.data.verify_shape_command`) checks the result. The steps for running them on production are in [Backfill and verify the new shape](howto-backfill-the-new-shape.md).

## Steps

`odds_lambda.shape_backfill.run.run_backfill` runs the steps in this order. It first takes the database clock as the cutoff (`odds_lambda.shape_backfill.snapshots.capture_cutoff`); the snapshot step touches only snapshots stored before it.

1. **Twin merges** (`odds_lambda.shape_backfill.events.merge_twins`). Every group of events with the same sport and teams within ±2 h (`odds_lambda.storage.event_merge.EventMerger.find_twin_groups`) is merged by `odds_lambda.storage.event_merge.EventMerger.merge`, with its clash handling, under the fixture lock (`odds_lambda.storage.event_resolver.lock_fixture`), one transaction per group. The canonical event is kept (`odds_core.event_source.canonical_event_id`). It takes the kick-off of the twin whose source ranks highest in its sport's [kick-off authority order](kickoff-authority.md) (`odds_lambda.shape_backfill.events.plan_twin_merge`): an event's source is the MLB Stats API when it holds a `gamePk`, else the source its id prefix names (`odds_lambda.shape_backfill.events.kickoff_source`). A group that fails, such as one holding two `gamePk`s, is rolled back and reported with its error.
2. **UNMATCHED** (`odds_lambda.shape_backfill.events.migrate_unmatched`). Every event with a `no_game_pk_reason` goes through `odds_lambda.storage.event_unmatched.mark_unmatched`: it becomes UNMATCHED (a FINAL event keeps its status) and gets the reason as its `missing_mlb_game_pk` acknowledgement. An event already acknowledged with the same reason is skipped. The column itself is left in place.
3. **External ids** (`odds_lambda.shape_backfill.events.stored_external_ids`, `odds_lambda.shape_backfill.events.record_stored_ids`). The ids stored data holds:

   | Source | Read from |
   |---|---|
   | `odds_api` | `raw_data.id` of the event's latest Odds API snapshot (the API reissues ids); else the key of an event keyed by an Odds API id |
   | `oddsportal` | `raw_data.oddsportal_event_id` (OddsPortal live) |
   | `betfair` | `raw_data.betfair_event_id` |
   | `polymarket` | `pm_event_id` in Polymarket envelopes, and `polymarket_events.pm_event_id` of linked legacy games |
   | `mlb_stats_api` | `events.mlb_game_pk` |

   Events go in fixture-lock order under their locks, committed every `odds_lambda.shape_backfill.events.EVENTS_PER_COMMIT` events, and each id goes through `odds_lambda.storage.event_resolver.record_external_id`: the first id seen per source wins, a second one or an id another event holds is refused and logged once to `data_quality_logs` (`external_id_refused`). An event that already holds an Odds API id keeps it (`kept_current`), since the stored one is the newer. The report counts each source's ids as `new`, `held`, `refused` or `kept_current` (`odds_lambda.shape_backfill.events.IdOutcome`).
4. **Historical OddsPortal ids**, only with `--oddsportal-dir`. `odds_lambda.shape_backfill.events.oddsportal_history_games` reads the scrapes `scripts/ingest_oddsportal.py` loads (`epl_*_<season>.json`, `nba_*_<season>.json`) and takes the id each match link ends with (`odds_lambda.oddsportal_common.event_id_from_match_link`). `odds_lambda.shape_backfill.events.attach_oddsportal_history` attaches each game to its event with the loader's ±24 h historical window (`odds_lambda.storage.event_resolver.attach_events`), never creating one.
5. **Snapshots** (`odds_lambda.shape_backfill.snapshots.backfill_snapshots`), in committed batches of `--batch-size` in id order, each batch locking its rows (`FOR UPDATE`). See [Snapshots](#snapshots).
6. **Legacy Polymarket** (`odds_lambda.shape_backfill.legacy_polymarket.migrate_legacy_polymarket`). See [Legacy Polymarket](#legacy-polymarket).

Every step skips what it has done, so a second run writes nothing, and a stopped run picks up where it stopped. Without `--apply` the command runs every step as a count on a connection whose transactions are read-only (`default_transaction_read_only`), the snapshot step through `odds_lambda.shape_backfill.snapshots.preflight_snapshots`.

## Snapshots

A batch takes the snapshots stored before the cutoff that have no `source`, or a `source` but no `payload_id`. Each legacy snapshot (no `source`) is read with the live writers' reader, `odds_lambda.price_rows.expand_envelope`, and then:

- its `raw_data`, as stored, becomes one `snapshot_payloads` row (`received_at` is the snapshot's `created_at`), and `payload_id` points at it;
- it takes the source and time basis of its `raw_data` tags (`odds_lambda.shape_backfill.snapshots.classify`), its first market, that market's metadata (Betfair `in_play`, `market_status`, `total_matched`, `market_start_time`) and that market's price rows;
- an envelope quoting several markets (the Odds API) is split as `odds_lambda.storage.shaped_snapshots.store_shaped_snapshot` splits it: the legacy row keeps its id and the first market, each other market becomes a new snapshot with the same event, time, `created_at`, tier and payload, every part's `raw_data` is cut to its market (`odds_lambda.price_rows.market_envelope`), and each `odds` row moves to the part that holds its book and market (`odds_lambda.shape_backfill.snapshots.odds_markets`). `--map-out FILE` appends each split's old id, market and new id as a JSON line (`odds_lambda.shape_backfill.snapshots.SplitMapping`); the parts of one split also share their `payload_id`.

A snapshot that has a source but no payload only gets its payload.

| `raw_data` tags | `source` | `time_basis` |
|---|---|---|
| none, stored within `odds_lambda.shape_backfill.snapshots.ODDS_API_HISTORICAL_GAP` of `snapshot_time` | `odds_api` | `observed` |
| none, stored later than that | `odds_api` | `upper_bound` |
| `oddsportal` (historical) | `oddsportal` | `source_stamped` |
| `oddsportal_live` | `oddsportal` | `observed` |
| `betfair_api` | `betfair` | `observed` |
| `polymarket` (live, and history with `api_request_id` `polymarket_history`) | `polymarket` | `observed` |
| `football_data_uk`, `timing.basis` `kickoff` | `football_data_uk` | `close` |
| `football_data_uk`, `timing.basis` `fduk_collection_deadline` | `football_data_uk` | `upper_bound` |
| `football_data_uk`, no `timing`, a later untagged FDUK snapshot on the event | `football_data_uk` | `upper_bound` |
| `football_data_uk`, no `timing`, the event's last untagged FDUK snapshot | `football_data_uk` | `close` |

The Odds API threshold sits in the gap the production data shows: live fetches are stored within 12 s of their snapshot time and historical ones 2.8 h to 340 days after it (#673). Untagged FDUK snapshots come from the loader before timing tags, which wrote an opening and a closing snapshot per game (`scripts/ingest_football_data_uk.py`). An unknown tag raises `odds_lambda.shape_backfill.snapshots.UnknownSnapshotTag`.

**Duplicates over the new key.** When a market's key (event, source, market, snapshot time) is held by a stored snapshot, or by an earlier legacy snapshot in id order, that market is not stored (`odds_lambda.shape_backfill.snapshots.assign_keys`), as `odds_lambda.storage.snapshot_store.store_snapshot` keeps the stored snapshot. Its `odds` rows are deleted, a legacy snapshot left with no market is deleted, and each dropped market is logged to `data_quality_logs` as `snapshot_key_conflict` with the kept snapshot and the payload that holds the dropped `raw_data`. The preflight lists every such duplicate.

**Candidate decisions.** Before writing a batch, the step raises `odds_lambda.shape_backfill.snapshots.SplitSnapshotReferenced` when a `candidate_decisions.input_snapshot_ids` entry names a snapshot it would split or delete (`odds_lambda.shape_backfill.snapshots.referenced_snapshot_ids`); that batch writes nothing and the command exits 2. The preflight lists those snapshots.

**Live writers.** The step only touches rows stored before the cutoff, and each batch commits on its own, so a live job that re-times one of its rows waits for one batch. A batch that loses a snapshot key to a concurrent writer (a unique violation) or a deadlock to one is rolled back and planned again, up to `odds_lambda.shape_backfill.snapshots.BATCH_ATTEMPTS` times.

**What the reader leaves out** goes to `data_quality_logs` with the `odds_lambda.price_rows.RowIssueType` the live writers use. The report counts unmatched outcome names per event: those events' team names need `odds events repair-names` ([Event merge](event-merge.md#command)) before the backfill, since a backfilled snapshot is not read again.

## Legacy Polymarket

The legacy `polymarket_*` tables ([Polymarket](../collection/polymarket.md)) hold NBA prices as probabilities. For each game linked to an event, each price snapshot of its moneyline market becomes an `h2h` snapshot with source `polymarket` and time basis `observed`, whose `raw_data` and payload are the envelope the live job writes (`odds_lambda.shape_backfill.legacy_polymarket.legacy_envelope`): `yes_probabilities` per side, with the market's two outcomes matched to the event's teams through `odds_lambda.polymarket_matching.normalize_team`. Its price rows are `mid` rows read from that envelope by `odds_lambda.price_rows.expand_envelope`. `created_at` is the snapshot time, since the legacy poller stored each row when it fetched it. One transaction per game.

Left out and counted: games with no event, market types other than moneyline (spread, total, player prop, other), moneyline outcomes that do not name the event's teams, probabilities not strictly between 0 and 1, and a time another moneyline market of the game already took. A time the event already has a Polymarket `h2h` snapshot at is skipped, so a second run writes nothing. Every `polymarket_orderbook_snapshots` row is archived as a payload under `odds_lambda.shape_backfill.legacy_polymarket.ORDER_BOOK_PAYLOAD_KEY`, which no snapshot references. The legacy tables are left in place.

## Verify

`odds_lambda.shape_backfill.verify.verify_shape` reads every snapshot stored before its own cutoff, read-only, and reports each mismatch under its `odds_lambda.shape_backfill.verify.Check`:

| Check | Mismatch |
|---|---|
| `no_source`, `no_time_basis`, `no_payload` | the column is empty |
| `wrong_source` | `source` or `time_basis` differs from what the `raw_data` tags say (the table above) |
| `unsplit` | `raw_data` quotes more than one market |
| `wrong_market` | `market` is not the one market `raw_data` quotes; an empty envelope has no market |
| `market_meta` | `in_play`, `market_status`, `total_matched` or `market_start_time` differs from the market metadata the reader reads (Betfair) |
| `prices_differ` | the price rows differ from those `odds_lambda.price_rows.expand_envelope` reads from `raw_data`, per book, outcome, line and side: price, size, matched and change time |
| `odds_market` | an `odds` row's `market_key` is not of the snapshot's market (`h2h_lay` sits on `1x2` or `h2h`) |
| `legacy_polymarket_missing` | a legacy Polymarket price the backfill would still write |

It prints the counts per source and market, the mismatch counts and up to `--examples` snapshots per check, and exits 1 when anything differs.

## Related

- [Prices and snapshot payloads tables](price-tables.md)
- [Odds snapshot and odds tables](odds-tables.md)
- [Event resolver](event-resolver.md)
