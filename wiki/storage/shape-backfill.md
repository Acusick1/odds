---
type: reference
title: Shape backfill and verify
description: What odds data backfill-shape writes for stored history (twin merges, UNMATCHED events, external ids, snapshot source, market, time basis, payloads, price rows and Odds API splits, legacy Polymarket prices), how it tells sources apart, the read-only plan and the gates that stop it before any write, its duplicate rule, and what odds data verify-shape checks.
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
  - odds_lambda.storage.event_resolver.check_external_id
  - odds_lambda.storage.event_resolver.attach_events
  - odds_lambda.storage.event_merge.EventMerger.merge
  - odds_lambda.storage.event_merge.EventMerger.merge_drops
  - odds_lambda.storage.event_kickoff.settle_merged_kickoff
  - odds_lambda.storage.event_unmatched.mark_unmatched
  - odds_lambda.price_rows.expand_envelope
  - odds_lambda.price_rows.market_envelope
status: stable
---

# Shape backfill and verify

The live writers fill the snapshot columns `source`, `market` and `time_basis`, the [`prices` and `snapshot_payloads` tables](price-tables.md) and [event external ids](events.md#external-ids) for what they store. `odds data backfill-shape` (`odds_cli.commands.data.backfill_shape`) does the same for everything stored before them, and `odds data verify-shape` (`odds_cli.commands.data.verify_shape_command`) checks the result. The steps for running them on production are in [Backfill and verify the new shape](howto-backfill-the-new-shape.md).

## Plan and gates

`odds_lambda.shape_backfill.run.run_backfill` first takes the database clock as the cutoff (`odds_lambda.shape_backfill.snapshots.capture_cutoff`); the snapshot step touches only snapshots stored before it. It then plans, read-only, the twin merges (`odds_lambda.shape_backfill.events.plan_twin_merges`) and the snapshot step (`odds_lambda.shape_backfill.snapshots.scan_snapshots`, over every pending snapshot). Without `--apply` those plans, with the counts of the other steps, are the preflight, run on a connection whose transactions are read-only (`default_transaction_read_only`), so the preflight counts exactly what the apply would write.

With `--apply`, the run stops before writing anything, and the command exits 2, when:

- a `candidate_decisions.input_snapshot_ids` entry names a snapshot the backfill would split or delete (`odds_lambda.shape_backfill.snapshots.check_references`, raising `odds_lambda.shape_backfill.snapshots.SplitSnapshotReferenced`);
- a twin merge would delete candidate decisions or paper trades (`odds_lambda.shape_backfill.events.MergeDropsRefused`), unless `--allow-merge-drops` is given.

A merge moves snapshots between events, so after merging the snapshot step is planned again; each batch also checks decisions recorded during the run before it writes.

## Steps

1. **Twin merges** (`odds_lambda.shape_backfill.events.merge_twins`). Every group of events with the same sport and teams within ±2 h (`odds_lambda.storage.event_merge.EventMerger.find_twin_groups`) is merged by `odds_lambda.storage.event_merge.EventMerger.merge`, with its clash handling, under the fixture lock (`odds_lambda.storage.event_resolver.lock_fixture`), one transaction per group. The canonical event is kept (`odds_core.event_source.canonical_event_id`). The plan (`odds_lambda.shape_backfill.events.plan_twin_merge`) lists the rows that move to the kept event (`odds_lambda.storage.event_merge.EventMerger.reference_counts`) and those the merge would delete (`odds_lambda.storage.event_merge.EventMerger.merge_drops`). After merging, the kept event takes the kick-off of the highest-ranked source in its sport's [kick-off authority order](kickoff-authority.md) that observed a twin, through `odds_lambda.storage.event_kickoff.settle_merged_kickoff`: a twin's observing sources are those holding an external id on it, the MLB Stats API when it holds a `gamePk`, and the sources of its snapshots (`odds_lambda.shape_backfill.events.observed_sources`), and the source's kick-off is the stored one of the first-stored twin it observed (`odds_lambda.shape_backfill.events.kickoff_starts`). A group that fails, such as one holding two `gamePk`s, is rolled back and reported with its error.
2. **UNMATCHED** (`odds_lambda.shape_backfill.events.migrate_unmatched`). Every event with a `no_game_pk_reason` goes through `odds_lambda.storage.event_unmatched.mark_unmatched`: it becomes UNMATCHED (a FINAL event keeps its status) and gets the reason as its `missing_mlb_game_pk` acknowledgement. An event already acknowledged with the same reason is skipped. The column itself is left in place.
3. **External ids** (`odds_lambda.shape_backfill.events.stored_external_ids`, `odds_lambda.shape_backfill.events.record_stored_ids`). The ids stored data holds:

   | Source | Read from |
   |---|---|
   | `odds_api` | `raw_data.id` of the event's latest Odds API snapshot (the API reissues ids); else the key of an event keyed by an Odds API id |
   | `oddsportal` | `raw_data.oddsportal_event_id` (OddsPortal live) |
   | `betfair` | `raw_data.betfair_event_id` |
   | `polymarket` | `pm_event_id` in Polymarket envelopes, and `polymarket_events.pm_event_id` of linked legacy games |
   | `mlb_stats_api` | `events.mlb_game_pk` |

   Events go in fixture-lock order under their locks, committed every `odds_lambda.shape_backfill.events.EVENTS_PER_COMMIT` events so a live writer waits behind a few fixtures at most. Each id is decided by the resolver's rules (`odds_lambda.storage.event_resolver.check_external_id`) and recorded through `odds_lambda.storage.event_resolver.record_external_id`: the first id seen per source wins, a second one or an id another event holds is refused and logged once to `data_quality_logs` (`external_id_refused`). An event that already holds an Odds API id keeps it (`kept_current`), since the stored one is the newer. The report counts each source's ids as `new`, `held`, `refused` or `kept_current` (`odds_lambda.shape_backfill.events.IdOutcome`).
4. **Historical OddsPortal ids**, only with `--oddsportal-dir`. `odds_lambda.shape_backfill.events.oddsportal_history_games` reads the scrapes `scripts/ingest_oddsportal.py` loads (`epl_*_<season>.json`, `nba_*_<season>.json`) and takes the id each match link ends with (`odds_lambda.oddsportal_common.event_id_from_match_link`). `odds_lambda.shape_backfill.events.attach_oddsportal_history` attaches each game to its event with the loader's ±24 h historical window (`odds_lambda.storage.event_resolver.attach_events`), never creating one.
5. **Snapshots** (`odds_lambda.shape_backfill.snapshots.backfill_snapshots`). See [Snapshots](#snapshots).
6. **Legacy Polymarket** (`odds_lambda.shape_backfill.legacy_polymarket.migrate_legacy_polymarket`). See [Legacy Polymarket](#legacy-polymarket).

Every step skips what it has done, so a second run writes nothing, and a stopped run picks up where it stopped.

## Snapshots

The step takes the snapshots stored before the cutoff that have no `source`, or a `source` but no `payload_id`. Each legacy snapshot (no `source`) is read with the live writers' reader, `odds_lambda.price_rows.expand_envelope`, and then:

- its `raw_data`, as stored, becomes one `snapshot_payloads` row (`received_at` is the snapshot's `created_at`), and `payload_id` points at it;
- it takes the source and time basis of its `raw_data` tags (`odds_lambda.shape_backfill.snapshots.classify`), its first market, that market's metadata (Betfair `in_play`, `market_status`, `total_matched`, `market_start_time`) and that market's price rows;
- an envelope quoting several markets (the Odds API) is split as `odds_lambda.storage.shaped_snapshots.store_shaped_snapshot` splits it: the legacy row keeps its id and the first market, each other market becomes a new snapshot with the same event, time, `created_at`, tier and payload, every part's `raw_data` is cut to its market (`odds_lambda.price_rows.market_envelope`), and each `odds` row moves to the part that holds its book and market (`odds_lambda.shape_backfill.snapshots.odds_actions`, `odds_lambda.shape_backfill.snapshots.odds_markets`).

The parts of one split share its `payload_id`, and the legacy row is the part with the lowest id, so the map from old to new ids can be rebuilt from `payload_id` at any time; `--map-out FILE` also appends each split's old id, market and new id as a JSON line (`odds_lambda.shape_backfill.snapshots.SplitMapping`). A snapshot that has a source but no payload only gets its payload. A legacy snapshot whose `raw_data` is not a JSON object is left alone and listed (`odds_lambda.shape_backfill.snapshots.malformed_snapshot_ids`).

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

The Odds API threshold sits in the gap the production data shows: live fetches are stored within 12 s of their snapshot time and historical ones 2.8 h to 340 days after it (#673). Untagged FDUK snapshots come from the loader before timing tags (`scripts/ingest_football_data_uk.py` before #516), which wrote an opening at kick-off − 24 h and a closing at kick-off − 1 h per game. On production all 3,800 events with untagged FDUK snapshots hold exactly one such pair at two different times, so the later one is the close (#673). An unknown tag raises `odds_lambda.shape_backfill.snapshots.UnknownSnapshotTag`.

**Duplicates over the new key.** When a market's key (event, source, market, snapshot time) is held by a stored snapshot, that market is not stored, as `odds_lambda.storage.snapshot_store.store_snapshot` keeps the stored snapshot. Among legacy snapshots sharing a key, the first stored keeps it (`odds_lambda.shape_backfill.snapshots.assign_keys`), with one exception (`odds_lambda.shape_backfill.snapshots.pair_preferences`): the historical OddsPortal loader stores a game's opening and then its closing under one `api_request_id` (`scripts/ingest_oddsportal.py`), so when exactly two historical OddsPortal snapshots with one `api_request_id`, each quoting only that key's market, share it, the later one is the closing. It keeps the key, because its prices are the state at that time. Groups of three or more, mixed markets and other sources keep the first. The scan decides this over every pending snapshot, so batch boundaries do not change it. A dropped market's `odds` rows are deleted, a legacy snapshot left with no market is deleted, and each dropped market is logged to `data_quality_logs` as `snapshot_key_conflict` (`odds_lambda.price_rows.RowIssueType`) with the kept snapshot and the payload that holds the dropped `raw_data`. The preflight lists every such duplicate.

**Live writers.** The step works in committed batches of `--batch-size` in id order. A batch locks its rows with `FOR UPDATE SKIP LOCKED`, so rows a live writer holds are skipped rather than waited for, and every statement waits at most `odds_lambda.shape_backfill.snapshots.LOCK_TIMEOUT` for a lock. A batch that times out, deadlocks or loses a snapshot key to a writer is rolled back and planned again, up to `odds_lambda.shape_backfill.snapshots.BATCH_ATTEMPTS` times. The step makes passes over the pending snapshots until one shapes nothing; what a live writer kept locked throughout is reported as left pending, for a rerun.

**What the reader leaves out** goes to `data_quality_logs` with the `odds_lambda.price_rows.RowIssueType` the live writers use. The report counts unmatched outcome names per event: those events' team names need `odds events repair-names` ([Event merge](event-merge.md#command)) before the backfill, since a backfilled snapshot is not read again.

## Legacy Polymarket

The legacy `polymarket_*` tables ([Polymarket](../collection/polymarket.md)) hold NBA prices as probabilities. For each game linked to an event, each price snapshot of its moneyline market becomes an `h2h` snapshot with source `polymarket` and time basis `observed`, whose `raw_data` and payload are the envelope the live job writes (`odds_lambda.shape_backfill.legacy_polymarket.legacy_envelope`): `yes_probabilities` per side, with the market's two outcomes matched to the event's teams through `odds_lambda.polymarket_matching.normalize_team`. Its price rows are `mid` rows read from that envelope by `odds_lambda.price_rows.expand_envelope`. `created_at` is the snapshot time, since the legacy poller stored each row when it fetched it. One transaction per game.

Left out and counted: games with no event, market types other than moneyline, moneyline outcomes that do not name the event's teams, probabilities not strictly between 0 and 1, and a time another moneyline market, or another legacy game linked to the same event, already took. Spread and total markets are left out because the Polymarket envelope has no line; on production they hold 136 price rows, all at one instant (#673). A time the event already has a Polymarket `h2h` snapshot at is skipped, so a second run writes nothing. Every `polymarket_orderbook_snapshots` row is archived as a payload under `odds_lambda.shape_backfill.legacy_polymarket.ORDER_BOOK_PAYLOAD_KEY`, which no snapshot references. The legacy tables are left in place.

## Verify

`odds_lambda.shape_backfill.verify.verify_shape` reads every snapshot stored before its own cutoff, read-only, then every archived envelope, and reports each mismatch under its `odds_lambda.shape_backfill.verify.Check`:

| Check | Mismatch |
|---|---|
| `malformed` | `raw_data` is not a JSON object |
| `no_source`, `no_time_basis`, `no_payload` | the column is empty |
| `wrong_source` | `source` or `time_basis` differs from what the `raw_data` tags say (the table above) |
| `unsplit` | `raw_data` quotes more than one market |
| `wrong_market` | `market` is not the one market `raw_data` quotes; an empty envelope has no market |
| `market_meta` | `in_play`, `market_status`, `total_matched` or `market_start_time` differs from the market metadata the reader reads (Betfair) |
| `prices_differ` | the price rows differ from those `odds_lambda.price_rows.expand_envelope` reads from `raw_data`, per book, outcome, line and side: price, size, matched and change time |
| `odds_market` | an `odds` row's `market_key` is not of the snapshot's market (`h2h_lay` sits on `1x2` or `h2h`) |
| `payload_prices` | the snapshots sharing an archived envelope do not hold the prices the envelope gives: a market on several of them, a market the envelope does not quote, or prices that differ |
| `payload_lost` | a market of an archived envelope is on no snapshot sharing it, and no `snapshot_key_conflict` log names a kept snapshot that holds its key |
| `new_orphan_odds` | an `odds` row without a snapshot written after the live writers archived their first run |
| `legacy_polymarket_missing` | a legacy Polymarket price the backfill would still write |

The per-snapshot checks catch a snapshot that disagrees with its own `raw_data`; the envelope checks (`odds_lambda.shape_backfill.verify._check_payloads`) catch what a split or a deletion lost, since a cut `raw_data` agrees with its own prices whatever it left out. They read the payloads that hold one envelope (the backfill's, and the Polymarket history's); a live run's payload holds a list of envelopes for many events and is not read. `odds` rows without a snapshot predate the link from each row to its snapshot; their count is reported, and only those written after the earliest run payload (the payload of a list of envelopes, which only the linking writers store) are mismatches (`odds_lambda.shape_backfill.verify._orphan_odds`).

It prints the counts per source and market, the mismatch counts and up to `--examples` snapshots per check, and exits 1 when anything differs.

## Related

- [Prices and snapshot payloads tables](price-tables.md)
- [Odds snapshot and odds tables](odds-tables.md)
- [Event resolver](event-resolver.md)
