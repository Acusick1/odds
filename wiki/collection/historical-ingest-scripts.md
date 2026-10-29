---
type: reference
title: Historical ingest scripts
description: "scripts/scrape_oddsportal.py, ingest_oddsportal.py, ingest_football_data_uk.py: inputs, matching, ids, snapshot shape, idempotency."
sources:
  - scripts/scrape_oddsportal.py
  - scripts/ingest_oddsportal.py
  - scripts/ingest_football_data_uk.py
  - scripts/team_names.py
  - odds_core.team.normalize_team_name
  - odds_lambda.oddsportal_common.event_id_from_match_link
  - odds_lambda.storage.event_resolver.resolve_event
  - odds_lambda.storage.event_resolver.MatchWindow
  - odds_lambda.storage.shaped_snapshots.store_shaped_snapshot
  - odds_lambda.oddsportal_common.build_envelope
  - odds_lambda.oddsportal_common.fix_odds_timestamp
  - odds_lambda.tier_utils.calculate_tier
mentions:
  - odds_core.models.Event
status: stable
---

# Historical ingest scripts

Three scripts under `scripts/` load OddsPortal scrapes and football-data.co.uk (FDUK) CSVs for EPL as `odds_core.models.Event` rows, each with `odds_core.models.OddsSnapshot` rows ([columns and constraints](../storage/odds-tables.md#odds_snapshots)) with `odds_core.models.Price` rows and a payload holding the envelopes they were built from. This is the history that models train and are evaluated on ([data-sources](data-sources.md)). None is scheduled. The steps to run them are in [howto-load-historical-epl-odds](howto-load-historical-epl-odds.md).

Each ingester finds the event of a game with `odds_lambda.storage.event_resolver.resolve_event` and stores each snapshot with `odds_lambda.storage.shaped_snapshots.store_shaped_snapshot` ([Writing price rows](../storage/price-tables.md#writing-price-rows)). Prices are read from the `prices` table ([Reading quotes](../storage/price-tables.md#reading-quotes)).

## Inputs

- `scripts/scrape_oddsportal.py` runs OddsHarvester `historic` via `uvx` (a fork branch pinned in `FORK_SPEC`; the project does not depend on OddsHarvester) and writes `data/external/oddsportal/<prefix>_<season>.json`. Resume is per match: it discovers links, then scrapes only missing or incomplete ones in batches of `BATCH_SIZE`, saving after each. It supports `--proxy-url` and `--output-dir` for geo-restricted books.
- `scripts/ingest_oddsportal.py` loads those JSON files (`--data-dirs` unions bookmakers across directories), matches or creates `odds_core.models.Event` rows, and inserts opening and closing `odds_core.models.OddsSnapshot` rows via `odds_lambda.oddsportal_common.build_envelope`. `--relink` re-links game logs and injuries (basketball).
- `scripts/ingest_football_data_uk.py` downloads `E0.csv` for the seasons listed in the script (2015-2016 to 2026-2027) and writes two snapshots per match, the opening `upper_bound` and the close `close`; a re-ingest rewrites an event's FDUK snapshots only when their time basis, snapshot time or price rows changed. FDUK is compiled by hand roughly twice weekly; its Pinnacle odds are unreliable after July 2025 (81 events; #217) and Betfair Exchange closing odds exist from 2024-25 (#253) ([ADR-0003](../../docs/adr/0003-hybrid-sharp-reference.md)). Event counts are data properties; check the database, not the repo.
- Team names go through `scripts/team_names.py`, which delegates to `odds_core.team.normalize_team_name`.

## Matching, ids and idempotency

- Both ingesters list the game as an `odds_lambda.storage.event_resolver.ListedGame` with `odds_lambda.storage.event_resolver.MatchWindow.HISTORICAL`, so the resolver matches a lone same-sport, same-teams event within ±24 h; several candidates match only the nearest within ±2 h, and when none is that close the row is ambiguous: the resolver raises `odds_lambda.storage.event_resolver.AmbiguousHistoricalMatch` and the script skips the row, counting it in `games_skipped` ([Event resolver](../storage/event-resolver.md#steps)). A game seen by one source attaches to the other's event ([event-matching](../storage/event-matching.md)). Team names are normalized by the resolver. An OddsPortal listing never moves a played event, because `odds_lambda.storage.event_kickoff.move_kickoff` refuses a past start from a source outside `odds_lambda.storage.event_kickoff.PLAYED_START_SOURCES`; FDUK moves only an event whose snapshots all come from FDUK, so an OddsPortal event holding no id yet is left alone ([Kick-off authority](../storage/kickoff-authority.md#before-and-after-the-stored-kick-off)).
- A created event takes the integer id the database assigns it; the OddsPortal id it may record is an external id, and FDUK issues none ([events](../storage/events.md#external-ids)).
- OddsPortal records the id its match link ends with (`odds_lambda.oddsportal_common.event_id_from_match_link`) as the event's `oddsportal` external id; a game found by that id matches whatever its kick-off. FDUK issues no id and writes none.
- Each game or match runs in a savepoint and is committed on its own, which releases the fixture lock the resolver took. A game that raises is reported in the summary and the run goes on.
- All scripts support `--dry-run`. OddsPortal skips an event that already holds a snapshot of the market with source `oddsportal` and time basis `source_stamped`. A snapshot another run stored under the same key (event, source, market, time) is kept, and a differing one is logged as `snapshot_key_conflict` ([storing a snapshot under its key](../storage/price-tables.md#storing-a-snapshot-under-its-key)).
- FDUK matched to an existing event gets no new event of its own. FDUK is last in the EPL kick-off order ([Kick-off authority](../storage/kickoff-authority.md)), so it moves only an event no OddsPortal or Odds API id is held on, and a match whose kick-off still differs from the event's by more than 15 min is logged (`kickoff_mismatches`). A rerun compares the event's FDUK snapshots (those with source `football_data_uk`) with what the CSV now yields and, only if they differ, deletes them with their prices and unshared payloads and writes them again (`snapshots_updated` counts the replaced snapshots), so a changed CSV, changed timing rules or a kick-off moved by another source leave one opening and one close. It replaces the whole set, not one (event, source, market, time basis) key, so an opening the CSV does not yield cannot outlive the rewrite. When a candidate decision recorded one of the snapshots as an input, the event is left as it is and logged as `fduk_replace_refused` (`replacements_refused`), so the decision's replay keeps working. Any matched event with no scores, whatever its status, gets FDUK's full-time score, FINAL and `completed_at` two hours after kick-off (`results_filled`); an event that already has scores keeps them. An event FDUK creates gets its score the same way, uncounted.

## Snapshot shape

- FDUK's `Time` column is UK local time, converted to UTC for `commence_time` (`parse_kickoff`). Seasons before 2019-20 have no `Time` column; their kick-off is assumed to be 15:00 local and the envelope's `timing.kickoff_time_assumed` is true.
- FDUK publishes no observed timestamps, so both snapshot times are approximations, recorded as the snapshot's `time_basis` (`odds_core.vocabulary.TimeBasis.UPPER_BOUND` for the opening and `odds_core.vocabulary.TimeBasis.CLOSE` for the closing) and in the envelope's `timing.basis`. The opening snapshot is stamped at the latest documented collection time (`FdukTimeBasis.COLLECTION_DEADLINE` in `scripts/ingest_football_data_uk.py`, value `fduk_collection_deadline`): Friday 17:00 UK time for Friday to Monday kick-offs, Tuesday 13:00 for Tuesday to Thursday ones, from [football-data.co.uk/matches.php](https://www.football-data.co.uk/matches.php) (`WEEKEND_SLOT`, `MIDWEEK_SLOT`, `opening_snapshot_time`). When that deadline is at or after kick-off, the opening price has no documented time before the match, so no opening snapshot is written (`openings_withheld`). The real collection was at or before the stamped time, so the opening horizon is a lower bound: 1 to 3 days for most matches and under 12 h for Friday and Tuesday evening kick-offs. An opening snapshot is never tiered `closing`, because the closing selection reads that tier (see below). The closing snapshot is stamped at kick-off (`kickoff`): FDUK documents the closing odds only as the last price before kick-off. Snapshot times use the matched event's `commence_time`.
- The bookmaker set is `BOOKMAKER_PREFIXES` (including the UK retail books BetVictor, Betfred, Paddy Power, Sky Bet, BetMGM and Coral) plus `market_max` / `market_avg`, which stay in the archived envelope and get no price rows (`odds_lambda.price_rows.FDUK_AGGREGATE_BOOKMAKERS`); the envelope's `match_stats` carries post-match stats, which no code reads. Both snapshots of a match share one payload holding their two envelopes. Every snapshot has source `football_data_uk` and market `1x2`; its prices are `back` rows.
- Outcome prices are the source decimal under `price_decimal`, with no `price` key: FDUK writes the CSV value and OddsPortal the scraped decimal, skipping a bookmaker with any price at or below 1.0 (`odds_lambda.oddsportal_common.build_envelope`) ([odds-math](../foundations/odds-math.md)).
- OddsPortal snapshots use the book's own change timestamps with the year corrected (`odds_lambda.oddsportal_common.fix_odds_timestamp`), so their `time_basis` is `odds_core.vocabulary.TimeBasis.SOURCE_STAMPED`; the source is `oddsportal`, the envelope's `source` is `"oddsportal"`, and a game's two snapshots (opening and closing) share one payload.
- No tier is stored. A snapshot's tier is derived against the event's `commence_time` by `odds_lambda.tier_utils.snapshot_tier` or `odds_lambda.tier_utils.snapshot_tiers` from its time, source, time basis and market, over the hours rule `odds_lambda.tier_utils.calculate_tier` ([fetch-tier](../storage/fetch-tier.md)): an FDUK close is `closing`, and an FDUK opening that would be `closing` is `pregame` because the source has a `close` snapshot of the market; an OddsPortal market's latest history snapshot is `closing` even when stamped after kick-off (a lone stamp before kick-off keeps its own tier). The [coverage check](../storage/coverage-checks.md) reads neither: it tiers snapshots by their time basis.
- The football three-way market key differs by source: FDUK and OddsPortal ingest both write `1x2` (`MARKET_KEY`, `MARKET_CONFIGS`) ([sport-and-market-keys](sport-and-market-keys.md#market-key-dialects)).

## Related

- [Load historical EPL odds](howto-load-historical-epl-odds.md)
- [Odds API historical backfill](odds-api-backfill.md)
- [OddsPortal scraper](oddsportal-scraper.md)
