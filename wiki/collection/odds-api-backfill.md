---
type: reference
title: Odds API historical backfill
description: "odds discover, odds backfill and the GameSelector, GapBackfillPlanner, BackfillExecutor modules."
sources:
  - odds_cli.commands.discover
  - odds_cli.commands.backfill
  - odds_analytics.gap_backfill_planner.GapBackfillPlanner
  - odds_analytics.game_selector.GameSelector
  - odds_analytics.backfill_executor.BackfillExecutor
  - odds_lambda.storage.tier_validator.TierCoverageValidator
  - odds_lambda.fetch_tier.FetchTier
status: stable
---

# Odds API historical backfill

Fills in historical Odds API snapshots (the paid `/historical/odds` endpoint, 10 units per region per market per call; the free tier has no historical endpoints; [odds-api-client](odds-api-client.md#cost-per-call)) and final scores. A replay backtest needs FINAL events plus snapshots before decision time (`odds_analytics.candidate_backtest.final_events` selects the events; [replay backtest](../evaluation/candidate-backtest.md)). It is also the recovery path for collection gaps (`odds_cli.commands.backfill.detect_and_plan_gaps`). Code: `packages/odds-cli/odds_cli/commands/{discover,backfill}.py` with logic in odds-analytics. The steps are on [howto-backfill-odds-api-history](howto-backfill-odds-api-history.md). It is separate from the EPL scripts ([historical-ingest-scripts](historical-ingest-scripts.md)); neither is scheduled.

## Commands

- `odds discover games --start D --end D` calls `odds_lambda.data_fetcher.TheOddsAPIClient.get_historical_events` per day and resolves each listed game to its event and creates a SCHEDULED one only when none matches (`odds_cli.commands.discover`; [Odds API events](../storage/event-matching.md#odds-api-events)). The default window is yesterday to today. Events already in the database, stored by any source, keep their surrogate key, status, scores, teams, MLB `gamePk` and acknowledgements, and their kick-off once it has passed, since the [kick-off authority](../storage/kickoff-authority.md) refuses a start at or before now after the stored kick-off ([events](../storage/events.md#writing-events)). It prints the events inserted, updated and dropped, and commits every 100 games.
- `odds backfill plan --start D --end D [--games N] [--output plan.json]` reads `basketball_nba` events with past kickoff from the DB and builds the plan with `odds_analytics.game_selector.GameSelector`. Each game in the plan (`odds backfill gaps` included) carries the event's id and `odds_api_id`, the `odds_api` external id recorded for it, or null when none is (`odds_lambda.odds_api_events.odds_api_id_of`). Output defaults to `backfill_plan.json`.
- `odds backfill execute --plan plan.json [--dry-run] [--no-skip-existing]` runs `odds_analytics.backfill_executor.BackfillExecutor`. It asks for confirmation unless `--dry-run`, sleeps 1 s after each successful fetch, and by default skips snapshots that exist within 5 minutes (`odds_lambda.storage.readers.OddsReader.snapshot_exists`). It finds the game in the response by `odds_api_id` (a game without an `odds_api_id` is skipped), builds one event finder for the run, resolves it to its stored event, and stores each market's snapshot with source `odds_api` and time basis `upper_bound`, the price rows and the game's envelope as a payload ([Writing price rows](../storage/price-tables.md#writing-price-rows)).
- `odds backfill scores --start D --end D [--dry-run]` fills missing scores from `nba_team_game_logs`, and sets FINAL ([event-status-lifecycle](event-status-lifecycle.md)). Home/away comes from the matchup string (`vs.` home, `@` away). The game logs come from `odds nba-stats fetch` ([nba-injuries-and-stats](nba-injuries-and-stats.md)).
- `odds backfill status` prints counts of events with snapshots and the earliest/latest snapshot time. It counts all `odds_snapshots`, not only backfilled ones.
- `odds backfill gaps --start D --end D [--max-quota Q] [--output gap.json]` uses `odds_analytics.gap_backfill_planner.GapBackfillPlanner` and `odds_lambda.storage.tier_validator.TierCoverageValidator` to find games missing fetch tiers, each snapshot's tier taken from its time basis and distance to kick-off, excluding IN_PLAY ([the tier a snapshot covers](../storage/fetch-tier.md#the-tier-a-snapshot-covers)). The backfill stores its fetches as `upper_bound` and the planner tiers each planned time that way, so the five planned at 72, 24, 12, 3 h and 30 min before kick-off cover OPENING, EARLY, SHARP, PREGAME and CLOSING, and a filled game is not planned again. Games are ordered by highest missing tier (`odds_lambda.fetch_tier.FetchTier.get_priority_order`: CLOSING, PREGAME, SHARP, EARLY, OPENING). Under `--max-quota` games are included whole or not at all, and it errors if even the first game exceeds the cap. Nothing is saved without `--output`.

## Snapshot selection

`odds_analytics.game_selector.GameSelector.calculate_snapshot_times` fixes five snapshots per game: 3 days, 24 h, 12 h, 3 h and 30 min before kickoff. With `--games N`, `games_per_team = max(1, N // 30)`; selection is scored per date with random jitter, so repeat plans can differ.

## Limits

- `basketball_nba` is literal in `odds_cli.commands.backfill` (plan, scores), `odds_analytics.backfill_executor.BackfillExecutor._process_snapshot` (`odds_lambda.data_fetcher.TheOddsAPIClient.get_historical_odds`, which requests the sport's configured bookmakers) and the tier validator's game-discovery check. Only `odds discover` takes `--sport`; plan finds no events for EPL and there is no `--sport` on plan, execute or scores (#448).
- Historical responses for soccer carry the three-way market as `1x2`, relabelled, and bookmaker keys renamed to pipeline keys, by `odds_lambda.data_fetcher.TheOddsAPIClient.get_historical_odds` ([sport-and-market-keys](sport-and-market-keys.md#market-key-dialects)).
- The `gaps` analysis is not sport-filtered; the executor then requests NBA historical odds and marks a non-NBA event "not found in API response" after the call, so quota is spent for nothing ([howto-backfill-odds-api-history](howto-backfill-odds-api-history.md)).
- The quota estimate is a constant 30 per snapshot (`odds_analytics.gap_backfill_planner.GapBackfillPlanner.QUOTA_PER_SNAPSHOT`, a literal 30 in `odds_analytics.game_selector.GameSelector`, and `+= 30` in the executor), not derived from configured regions or markets (#441; [odds-api-quota](odds-api-quota.md)).
- Final scores are not available from the Odds API for games older than 3 days (`odds_cli.commands.discover` docstring), hence `backfill scores`.

## Related

- [Backfill Odds API history](howto-backfill-odds-api-history.md)
- [Odds API quota accounting](odds-api-quota.md)
- [Results and status jobs](results-jobs.md)
