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
  - odds_lambda.oddsportal_common.find_existing_event
  - odds_lambda.oddsportal_common.build_raw_data
  - odds_lambda.oddsportal_common.fix_odds_timestamp
  - odds_lambda.oddsportal_common.hours_to_tier
status: stable
---

# Historical ingest scripts

Three scripts under `scripts/` load OddsPortal scrapes and football-data.co.uk (FDUK) CSVs for EPL as `odds_core.models.Event` plus `odds_core.models.OddsSnapshot` ([columns and constraints](../storage/odds-tables.md#odds_snapshots)) rows in the live `raw_data` format. This is the history that trains models ([data-sources](data-sources.md), [clv-model-findings](../evaluation/clv-model-findings.md)). None is scheduled. The steps to run them are in [howto-load-historical-epl-odds](howto-load-historical-epl-odds.md).

The scripts write `odds_core.models.Event` and `odds_core.models.OddsSnapshot` rows only and no `odds_core.models.Odds` rows, so backtests and `odds_lambda.storage.readers.OddsReader.get_odds_at_time` skip these events (#435; [odds-tables](../storage/odds-tables.md)).

## Inputs

- `scripts/scrape_oddsportal.py` runs OddsHarvester `historic` via `uvx` (a fork branch pinned in `FORK_SPEC`; the project does not depend on OddsHarvester) and writes `data/external/oddsportal/<prefix>_<season>.json`. Resume is per match: it discovers links, then scrapes only missing or incomplete ones in batches of `BATCH_SIZE`, saving after each. It supports `--proxy-url` and `--output-dir` for geo-restricted books.
- `scripts/ingest_oddsportal.py` loads those JSON files (`--data-dirs` unions bookmakers across directories), matches or creates `odds_core.models.Event` rows, and inserts opening and closing `odds_core.models.OddsSnapshot` rows via `odds_lambda.oddsportal_common.build_raw_data`. `--relink` re-links game logs and injuries (basketball).
- `scripts/ingest_football_data_uk.py` downloads `E0.csv` for the seasons listed in the script (2015-2016 to 2026-2027) and writes two snapshots per match. FDUK is compiled by hand roughly twice weekly; its Pinnacle odds are unreliable after July 2025 (81 events; #217) and Betfair Exchange closing odds exist from 2024-25 (#253) ([ADR-0003](../../docs/adr/0003-hybrid-sharp-reference.md)). Event counts are data properties; check the database, not the repo.
- Team names go through `scripts/team_names.py`, which delegates to `odds_core.team.normalize_team_name`.

## Matching, ids and idempotency

- Both ingesters first call `odds_lambda.oddsportal_common.find_existing_event` (±24 h, same sport, exact names, no normalization; several candidates match only through those within ±2 h), so a game seen by one source attaches to the other's event ([event-matching](../storage/event-matching.md)).
- New ids are `op_<season>_<HOME>_<AWAY>_<date>` and `fduk_<season>_<HOME>_<AWAY>_<date>`.
- All scripts support `--dry-run`. Idempotency: a snapshot is skipped if one with the same `api_request_id` exists (`oddsportal` for h2h, `oddsportal_<market>` otherwise, `football_data_uk`).
- FDUK events matched to existing `op_` events get no new event of their own. Each row runs in a savepoint. A rerun compares the event's `football_data_uk` snapshots with those the CSV now yields and, only if they differ, deletes and rewrites them (`snapshots_updated` counts the replaced rows), so a changed CSV or changed timing rules reach existing events. A rerun also corrects `commence_time` and `completed_at` on `fduk_` events; matched `op_` events keep their own times. Any matched event with no scores, whatever its status, gets FDUK's full-time score, FINAL and `completed_at` two hours after kick-off (`results_filled`); an event that already has scores keeps them.

## Snapshot shape

- FDUK's `Time` column is UK local time, converted to UTC for `commence_time` (`parse_kickoff`). Seasons before 2019-20 have no `Time` column; their kick-off is assumed to be 15:00 local and `raw_data.timing.kickoff_time_assumed` is true.
- FDUK publishes no observed timestamps, so both snapshot times are approximations recorded in `raw_data.timing.basis`. The opening snapshot is stamped at the latest documented collection time (`odds_core.snapshot_sources.FdukTimeBasis.COLLECTION_DEADLINE`, value `fduk_collection_deadline`): Friday 17:00 UK time for Friday to Monday kick-offs, Tuesday 13:00 for Tuesday to Thursday ones, from [football-data.co.uk/matches.php](https://www.football-data.co.uk/matches.php) (`WEEKEND_SLOT`, `MIDWEEK_SLOT`, `opening_snapshot_time`). When that deadline is at or after kick-off, the opening price has no documented time before the match, so no opening snapshot is written (`openings_withheld`). The real collection was at or before the stamped time, so the opening horizon is a lower bound: 1 to 3 days for most matches and under 12 h for Friday and Tuesday evening kick-offs. An opening snapshot is never tiered `closing` (`opening_fetch_tier` raises it to `pregame`), because the closing selection reads that tier. The closing snapshot is stamped at kick-off (`kickoff`): FDUK documents the closing odds only as the last price before kick-off. Snapshot times use the matched event's `commence_time`, and a matched event whose kick-off differs from FDUK's by more than 15 min is logged (`kickoff_mismatches`).
- The bookmaker set is `BOOKMAKER_PREFIXES` (including the UK retail books BetVictor, Betfred, Paddy Power, Sky Bet, BetMGM and Coral) plus `market_max` / `market_avg`; `raw_data.match_stats` carries post-match stats. The `match_stats` feature group reads only these FDUK snapshots ([epl-context-features](../modelling/epl-context-features.md)).
- Outcome prices are the source decimal under `price_decimal`, with no `price` key: FDUK writes the CSV value and OddsPortal the scraped decimal, skipping a bookmaker with any price at or below 1.0 (`odds_lambda.oddsportal_common.build_raw_data`) ([odds-math](../foundations/odds-math.md)).
- OddsPortal snapshots use real timestamps with the year corrected (`odds_lambda.oddsportal_common.fix_odds_timestamp`) and `raw_data.source = "oddsportal"`.
- `fetch_tier` comes from `odds_lambda.oddsportal_common.hours_to_tier`, whose `<` boundaries differ from the live writer's `<=`; an FDUK snapshot exactly 24 h out is tiered `early` where the live writer would give `sharp` (#446; [fetch-tier](../storage/fetch-tier.md#boundary-encodings)).
- The football three-way market key differs by source: FDUK and OddsPortal ingest both write `1x2` (`MARKET_KEY`, `MARKET_CONFIGS`) ([sport-and-market-keys](sport-and-market-keys.md#market-key-dialects)).

## Related

- [Load historical EPL odds](howto-load-historical-epl-odds.md)
- [Odds API historical backfill](odds-api-backfill.md)
- [OddsPortal scraper](oddsportal-scraper.md)
