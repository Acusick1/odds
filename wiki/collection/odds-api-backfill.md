---
type: reference
title: Odds API historical backfill
description: "Acquisition manifests and the causal call planner, odds discover, odds backfill, GapBackfillPlanner and BackfillExecutor."
sources:
  - odds_cli.commands.discover
  - odds_cli.commands.backfill
  - odds_analytics.backfill_plan
  - backfills/odds-api-epl-2020-26.yaml
  - odds_analytics.gap_backfill_planner.GapBackfillPlanner
  - odds_analytics.gap_backfill_planner.TIER_SNAPSHOT_LEADS
  - odds_analytics.backfill_executor.BackfillExecutor
  - odds_lambda.storage.tier_validator.TierCoverageValidator
  - odds_lambda.fetch_tier.FetchTier
status: stable
---

# Odds API historical backfill

Fills in historical Odds API snapshots (the paid `/historical/odds` endpoint, 10 units per region per market per call; the free tier has no historical endpoints; [odds-api-client](odds-api-client.md#cost-per-call)) and final scores. A replay backtest needs FINAL events plus snapshots before decision time (`odds_analytics.candidate_backtest.final_events` selects the events; [replay backtest](../evaluation/candidate-backtest.md)). It is also the recovery path for collection gaps (`odds_cli.commands.backfill.detect_and_plan_gaps`). Code: `packages/odds-cli/odds_cli/commands/{discover,backfill}.py` with logic in odds-analytics. The steps are on [howto-backfill-odds-api-history](howto-backfill-odds-api-history.md). It is separate from the EPL scripts ([historical-ingest-scripts](historical-ingest-scripts.md)); neither is scheduled.

## Acquisition manifests

A manifest in `backfills/` fixes what a historical purchase requests, validated by `odds_analytics.backfill_plan.Manifest`. A frozen manifest is never edited; a top-up is a new manifest. `backfills/odds-api-epl-2020-26.yaml` holds:

- **Content:** the sport, the seasons as disjoint `[start, end)` kick-off ranges, the markets in the Odds API's keys (`h2h`, and `totals` with its line), each once, and either named bookmakers or regions, not both.
- **Coverage bands:** each band (`odds_analytics.backfill_plan.Band`) is a range of leads before kick-off in whole minutes, from `min_lead_minutes` to `max_lead_minutes` inclusive, with a max input age. It promises that a decision at any whole minute in the range finds a snapshot at most the max age old, so a candidate deciding at any of those leads with at least that max input age has inputs, since it reads the latest snapshot in `[t - max age, t]` at decision time `t`, both ends inclusive (`odds_lambda.candidate_rules.market_state.load_market_state`). Bands do not overlap.
- **Snapshot grid:** steps by date, each a snapshot interval from its start until the next step's start (10 minutes from 2020-06-06, 5 from 2022-09-01).
- **Probe:** the snapshot times to request before the full run and what each must return: fixtures at least `min_fixture_lead_hours` ahead, the `required_bookmakers` on each market, and an `x-requests-last` of `units_per_call`. Validation rejects a lead shorter than the earliest band's farthest lead plus its max age, a units figure that differs from the manifest's, and required books or markets the manifest does not request. The probe also checks that the snapshot interval matches the grid.

Granularity and billing are the Odds API's documented values until the probe confirms them ([odds-api-client](odds-api-client.md#cost-per-call)).

## Planning calls

`odds_analytics.backfill_plan.plan` takes a manifest and fixtures and returns the fewest calls, each a `odds_analytics.backfill_plan.Call` with its time and the fixture bands it serves. One call returns the whole league, so the planner works on decision times, not fixtures: it merges every fixture's bands into stretches of time, each needing the strictest max age of the bands that cover it. Walking forward through decision minutes, at the first one the last call is too old for, it adds a call at the latest grid snapshot at or before that minute (`odds_analytics.backfill_plan.Manifest.floor_to_grid`). Every decision minute therefore has a call at most its max age old, and no fewer grid calls achieve that. A call serves a fixture's band (`odds_analytics.backfill_plan.targets`) when it lies between the band's farthest decision less its max age and its nearest decision. A max age with no grid snapshot inside it, or a decision before the first snapshot, raises `ValueError`.

`odds_analytics.backfill_plan.load_fixtures` reads the season's FINAL events of the manifest's sport.

A call costs `odds_analytics.backfill_plan.Manifest.units_per_call`: 10 units per market per region, where each group of up to 10 named bookmakers counts as one region.

## Commands

- `odds discover games --start D --end D` calls `odds_lambda.data_fetcher.TheOddsAPIClient.get_historical_events` per day and resolves each listed game to its event and creates a SCHEDULED one only when none matches (`odds_cli.commands.discover`; [Odds API events](../storage/event-matching.md#odds-api-events)). The default window is yesterday to today. Events already in the database, stored by any source, keep their id, status, scores, teams, MLB `gamePk` and acknowledgements, and their kick-off once it has passed, since the [kick-off authority](../storage/kickoff-authority.md) refuses a start at or before now after the stored kick-off ([events](../storage/events.md#writing-events)). It prints the events inserted, updated and dropped, and commits every 100 games.
- `odds backfill plan --manifest M [--season S ...] [--cost]` plans the manifest's calls for each season (`odds_cli.commands.backfill.plan_backfill`) and makes no API call. It prints each call's time and the number of fixture bands it serves. With `--cost` it prints a table of calls and units per season: one row for all bands, then a row per band with the calls that band adds to all the others, so band rows do not sum to the total.
- `odds backfill execute --plan plan.json --sport S [--dry-run] [--no-skip-existing]` runs `odds_analytics.backfill_executor.BackfillExecutor` on a gap plan. It asks for confirmation unless `--dry-run`, sleeps 1 s after each successful fetch, and by default skips snapshots that exist within 5 minutes (`odds_lambda.storage.readers.OddsReader.snapshot_exists`). It finds the game in the response by `odds_api_id` (a game without an `odds_api_id` is skipped), builds one event finder for the run, resolves it to its stored event, and stores each market's snapshot with source `odds_api` and time basis `upper_bound`, the price rows and the game's envelope as a payload ([Writing price rows](../storage/price-tables.md#writing-price-rows)).
- `odds backfill scores --start D --end D [--dry-run]` fills missing scores from `nba_team_game_logs`, and sets FINAL ([event-status-lifecycle](event-status-lifecycle.md)). Home/away comes from the matchup string (`vs.` home, `@` away). The game logs come from `odds nba-stats fetch` ([nba-injuries-and-stats](nba-injuries-and-stats.md)).
- `odds backfill status` prints counts of events with snapshots and the earliest/latest snapshot time. It counts all `odds_snapshots`, not only backfilled ones.
- `odds backfill gaps --start D --end D [--max-quota Q] [--output gap.json]` uses `odds_analytics.gap_backfill_planner.GapBackfillPlanner` and `odds_lambda.storage.tier_validator.TierCoverageValidator` to find games missing fetch tiers, each snapshot's tier taken from its time basis and distance to kick-off, excluding IN_PLAY ([the tier a snapshot covers](../storage/fetch-tier.md#the-tier-a-snapshot-covers)). It plans one snapshot per missing tier at that tier's lead in `odds_analytics.gap_backfill_planner.TIER_SNAPSHOT_LEADS` (72, 24, 12, 3 h and 30 min before kick-off for OPENING, EARLY, SHARP, PREGAME and CLOSING). The backfill stores its fetches as `upper_bound` and the planner tiers each planned time that way, so a filled game is not planned again. Games are ordered by highest missing tier (`odds_lambda.fetch_tier.FetchTier.get_priority_order`: CLOSING, PREGAME, SHARP, EARLY, OPENING). Under `--max-quota` games are included whole or not at all, and it errors if even the first game exceeds the cap. Each planned game carries its integer `event_id` and its `odds_api_id`, the `odds_api` external id recorded for it, or null when none is (`odds_lambda.odds_api_events.odds_api_id_of`). Nothing is saved without `--output`.

## Limits

- The manifest planner refuses a kick-off between whole minutes (`odds_analytics.backfill_plan.Fixture`), since it steps decisions a whole minute at a time. No stored EPL kick-off has seconds; 115 NBA kick-offs do (production, 2026-10-31), so an NBA manifest fails until they are handled.
- The backfill's calls spend the key of the sport they name, so the NBA backfill spends NBA's key (index 1), the key that is also `fetch-odds-nba`'s ([keys](odds-api-client.md#retry-and-keys)).
- `basketball_nba` is still literal in `odds backfill scores` (`odds_cli.commands.backfill`, which reads game logs and spends no units) and in the tier validator's game-discovery check (`odds validate daily --check-discovery`, which spends NBA's key). `odds discover games` and `odds backfill execute` require `--sport`, which the executor passes to `odds_lambda.data_fetcher.TheOddsAPIClient.get_historical_odds` (which requests the sport's configured bookmakers). `odds backfill plan` takes the manifest's sport, and with `--cost` prints that sport's key balance and what is left after the plan (the key's newest recorded quota, else `ODDS_API_QUOTA`).
- No command executes a manifest's planned calls yet: `execute` takes gap plans only.
- Historical responses for soccer carry the three-way market as `1x2`, relabelled, and bookmaker keys renamed to pipeline keys, by `odds_lambda.data_fetcher.TheOddsAPIClient.get_historical_odds` ([sport-and-market-keys](sport-and-market-keys.md#market-key-dialects)).
- The `gaps` analysis is not sport-filtered; the executor then requests NBA historical odds and marks a non-NBA event "not found in API response" after the call, so quota is spent for nothing ([howto-backfill-odds-api-history](howto-backfill-odds-api-history.md)).
- The gap plan's quota estimate is a constant 30 per snapshot (`odds_analytics.gap_backfill_planner.GapBackfillPlanner.QUOTA_PER_SNAPSHOT`, and `+= 30` in the executor), not derived from configured regions or markets (#441; [odds-api-quota](odds-api-quota.md)).
- Final scores are not available from the Odds API for games older than 3 days (`odds_cli.commands.discover` docstring), hence `backfill scores`.

## Related

- [Backfill Odds API history](howto-backfill-odds-api-history.md)
- [Odds API quota accounting](odds-api-quota.md)
- [Results and status jobs](results-jobs.md)
