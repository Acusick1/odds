---
type: howto
title: Load historical odds and scores
description: "How to load history two ways, OddsPortal/football-data.co.uk scripts (snapshots only, EPL) and the Odds API historical backfill (NBA-only, paid), and what each produces and breaks."
sources:
  - scripts/scrape_oddsportal.py
  - scripts/ingest_oddsportal.py
  - scripts/ingest_football_data_uk.py
  - scripts/team_names.py
  - odds_cli.commands.discover
  - odds_cli.commands.backfill
  - odds_analytics.gap_backfill_planner.GapBackfillPlanner
  - odds_analytics.game_selector.GameSelector
  - odds_analytics.backfill_executor.BackfillExecutor
  - odds_lambda.fetch_tier.FetchTier
  - odds_analytics.backtesting.services.BacktestEngine
status: stable
---

# Load historical odds and scores

There are two unrelated ways to get history, and they feed different sports. **Path 1** (scripts under `scripts/`) loads OddsPortal scrapes and football-data.co.uk (FDUK) CSVs for EPL as `odds_core.models.Event` plus `odds_core.models.OddsSnapshot` rows in the live `raw_data` format; this is the history that trains models ([sources-and-keys](sources-and-keys.md), [clv-model-findings](../evaluation/clv-model-findings.md)). **Path 2** (`odds discover` / `odds backfill`) loads Odds API historical snapshots and scores, and is NBA-only and paid. Neither is scheduled. Both leave events without normalized `odds` rows or with gaps, so check the hazards before relying on the output.

FDUK is compiled by hand roughly twice weekly. Its Pinnacle odds are unreliable after July 2025 (81 events; #217), and Betfair Exchange closing odds exist from 2024-25 (#253) ([decision-hybrid-sharp-reference](../modelling/decision-hybrid-sharp-reference.md)). Event counts are data properties; check the database, not the repo.

## Path 1: OddsPortal and FDUK scripts (EPL)

1. `scripts/scrape_oddsportal.py` runs OddsHarvester `historic` via `uvx` (a fork branch pinned in `FORK_SPEC`, separate from the live jobs' dependency) and writes `data/external/oddsportal/<prefix>_<season>.json`. Resume is per match: it discovers links, then scrapes only missing or incomplete ones in batches of `BATCH_SIZE`, saving after each. It supports `--proxy-url` and `--output-dir` for geo-restricted books.
2. `scripts/ingest_oddsportal.py` loads those JSON files (`--data-dirs` unions bookmakers across directories), matches or creates `odds_core.models.Event` rows, and inserts opening and closing `odds_core.models.OddsSnapshot` rows via `oddsportal_common.build_raw_data`. `--relink` re-links game logs and injuries (basketball).
3. `scripts/ingest_football_data_uk.py` downloads `E0.csv` for seasons in `scripts/ingest_football_data_uk.py` (2015-2016 to 2025-2026) and writes two snapshots per match. Team names go through `scripts/team_names.py`, which delegates to `odds_core.team.normalize_team_name`.

Both ingesters first call `odds_lambda.oddsportal_common.find_existing_event` (±24 h, no sport filter, exact names), so a game seen by one source attaches to the other's event ([events](../storage/events.md#matchers)). New ids: `op_<season>_<HOME>_<AWAY>_<date>` and `fduk_<season>_<HOME>_<AWAY>_<date>`. All scripts support `--dry-run`. Idempotency: skip if a snapshot with the same `api_request_id` exists (`oddsportal` for h2h, `oddsportal_<market>` otherwise, `football_data_uk`).

### Snapshot shape

- FDUK opening snapshot is stamped 24 h before kickoff and closing 1 h before: synthetic times, not observed ones. Bookmaker set is `BOOKMAKER_PREFIXES` plus `market_max` / `market_avg`; `raw_data.match_stats` carries post-match stats. The `match_stats` feature group reads only these FDUK snapshots ([epl-context-features](../modelling/epl-context-features.md)).
- OddsPortal snapshots use real timestamps with the year corrected (`odds_lambda.oddsportal_common.fix_odds_timestamp`) and `raw_data.source = "oddsportal"`.
- `fetch_tier` comes from `odds_lambda.oddsportal_common.hours_to_tier` (`<` boundaries), not the live writer's `<=` ([fetch-tier](../storage/fetch-tier.md#four-boundary-encodings)).

## Path 2: Odds API historical backfill (NBA only)

Fills in historical Odds API snapshots (the paid `/historical/odds` endpoint, 10 units per region per market per call; the free tier has no historical endpoints; [odds-api-client-and-quota](odds-api-client-and-quota.md)) and final scores. Backtests need FINAL events plus snapshots near decision time (`odds_analytics.backtesting.services.BacktestEngine._get_events_with_results` selects FINAL events; [backtesting](../evaluation/backtesting.md)). It is also the recovery path for collection gaps (`odds_cli.commands.backfill.detect_and_plan_gaps`). Code: `packages/odds-cli/odds_cli/commands/{discover,backfill}.py` with logic in odds-analytics. Check `odds status quota` first.

### Route A: discover, plan, execute

1. `odds discover games --start D --end D` calls `get_historical_events` per day and upserts the events as SCHEDULED. Default window is yesterday to today.
2. `odds backfill plan --start D --end D [--games N] [--output plan.json]` reads `basketball_nba` events with past kickoff from the DB and builds the plan with `odds_analytics.game_selector.GameSelector`. Output defaults to `backfill_plan.json`.
3. `odds backfill execute --plan plan.json [--dry-run] [--no-skip-existing]` runs `odds_analytics.backfill_executor.BackfillExecutor`. It asks for confirmation unless `--dry-run`, sleeps 1 s after each successful fetch, and by default skips snapshots that exist within 5 minutes (`snapshot_exists`).
4. `odds backfill scores --start D --end D [--dry-run]` fills missing scores from `nba_team_game_logs`; run `odds nba-stats fetch` first ([nba-injuries-and-stats](nba-injuries-and-stats.md)). Home/away comes from the matchup string (`vs.` home, `@` away). It sets FINAL ([event-lifecycle](event-lifecycle.md)).
5. `odds backfill status` prints counts of events with snapshots and the earliest/latest snapshot time.

`odds_analytics.game_selector.GameSelector.calculate_snapshot_times` fixes five snapshots per game: 3 days, 24 h, 12 h, 3 h and 30 min before kickoff. With `--games N`, `games_per_team = max(1, N // 30)`; selection is scored per date with random jitter, so repeat plans can differ.

### Route B: tier gaps

`odds backfill gaps --start D --end D [--max-quota Q] [--output gap.json]` uses `odds_analytics.gap_backfill_planner.GapBackfillPlanner` and `odds_lambda.storage.tier_validator.TierCoverageValidator` to find games missing fetch tiers, excluding IN_PLAY ([fetch-tier](../storage/fetch-tier.md)). Games are ordered by highest missing tier (`odds_lambda.fetch_tier.FetchTier.get_priority_order`: CLOSING, PREGAME, SHARP, EARLY, OPENING). Under `--max-quota` games are included whole or not at all, and it errors if even the first game exceeds the cap. Nothing is saved without `--output`; then run `odds backfill execute --plan gap.json`.

## Hazards

- **No normalized rows from the scripts.** `scripts/ingest_oddsportal.py` and `scripts/ingest_football_data_uk.py` write no `odds_core.models.Odds` rows, so backtests and `get_odds_at_time` skip these events ([odds-storage](../storage/odds-storage.md), [backtesting](../evaluation/backtesting.md)). Odds API backfill is NBA-only, so EPL backtests have little or no data.
- **Market key differs by source for football.** FDUK writes `h2h`; OddsPortal ingest writes `1x2` for the `1x2` market (`MARKET_CONFIGS`). Readers must accept both.
- **Tier boundaries differ from live writes.** FDUK's synthetic 24 h opening gets `early` from `odds_lambda.oddsportal_common.hours_to_tier` (strict `<`), where `tier_utils.calculate_tier` would give `sharp`.
- **FDUK events matched to existing `op_` events get no new event**, and its snapshots are skipped if any `football_data_uk` snapshot exists; only `match_stats` is back-filled (`snapshots_updated`).
- **NBA hardcoding.** `basketball_nba` is literal in `odds_cli.commands.backfill` (plan, scores), `odds_analytics.backfill_executor.BackfillExecutor._process_snapshot` (`get_historical_odds`) and the tier validator's game-discovery check. `odds_cli.commands.discover` defaults to NBA but takes `--sport`, but plan finds no events for EPL and there is no `--sport` on plan, execute or scores.
- **Quota is wasted on non-NBA events.** `gaps` analysis is not sport-filtered; the executor then requests NBA historical odds and marks the event "not found in API response" after the call.
- **Quota estimate is a constant**: 30 per snapshot (`odds_analytics.gap_backfill_planner.GapBackfillPlanner.QUOTA_PER_SNAPSHOT`, a literal 30 in `odds_analytics.game_selector.GameSelector`, and `+= 30` in the executor), not derived from configured regions or markets.
- **`discover games` overwrites** status and scores of events already in the DB (`odds_lambda.storage.writers.OddsWriter.bulk_upsert_events`; [event-lifecycle](event-lifecycle.md#hazards)).
- **`backfill status`** counts all `odds_snapshots`, not only backfilled ones.
- Final scores are not available from the Odds API for games older than 3 days (`odds_cli.commands.discover` docstring), hence step 4.

## Related

- [OddsPortal scraper and job](oddsportal.md)
- [Odds snapshots, odds rows and OddsWriter](../storage/odds-storage.md)
- [Odds validation and coverage checks](../storage/data-validation.md)
- [Packages and the odds CLI](../foundations/packages-and-cli.md)
