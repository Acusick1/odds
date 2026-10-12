---
type: howto
title: Backfill historical odds and scores
summary: Discover, plan, execute, scores, status (or gaps, then execute) for loading historical Odds API snapshots, with the gap planner, game selector and executor and their NBA-only defaults.
sources:
  - packages/odds-cli/odds_cli/commands/discover.py
  - packages/odds-cli/odds_cli/commands/backfill.py
  - packages/odds-analytics/odds_analytics/gap_backfill_planner.py::GapBackfillPlanner
  - packages/odds-analytics/odds_analytics/game_selector.py::GameSelector
  - packages/odds-analytics/odds_analytics/backfill_executor.py::BackfillExecutor
  - packages/odds-lambda/odds_lambda/fetch_tier.py::FetchTier
  - docs/CLI.md#discovery-commands
  - docs/CLI.md#backfill-commands
  - docs/CLI.md#historical-backfill-workflow
  - docs/DEPLOYMENT.md#operations
  - docs/BACKTESTING_GUIDE.md#data-requirements
verified_at: 7887039
---

# Backfill historical odds and scores

Fills in historical Odds API snapshots (the paid `/historical/odds` endpoint, 10 units per region per market per call; [Odds API quota](odds-api-quota.md)) and final scores. Backtests need FINAL events plus snapshots near decision time (Backtesting data requirements (`docs/BACKTESTING_GUIDE.md#data-requirements`; [Look-ahead bias](look-ahead-bias.md))). It is also the recovery path for collection gaps (`docs/DEPLOYMENT.md#operations`). Code: `packages/odds-cli/odds_cli/commands/{discover,backfill}.py` with logic in odds-analytics.

**The whole path is NBA-only** (see Hazards). Check `odds status quota` first.

## Route A: discover, plan, execute

1. `odds discover games --start D --end D` calls `get_historical_events` per day and upserts the events as SCHEDULED. Default window is yesterday to today.
2. `odds backfill plan --start D --end D [--games N] [--output plan.json]` reads `basketball_nba` events with past kickoff from the DB and builds the plan with `GameSelector`. Output defaults to `backfill_plan.json`.
3. `odds backfill execute --plan plan.json [--dry-run] [--no-skip-existing]` runs `BackfillExecutor`. It asks for confirmation unless `--dry-run`, sleeps 1 s after each successful fetch, and by default skips snapshots that exist within 5 minutes (`snapshot_exists`).
4. `odds backfill scores --start D --end D [--dry-run]` fills missing scores from `nba_team_game_logs`; run `odds nba-stats fetch` first ([NBA stats](nba-stats-data.md)). Home/away comes from the matchup string (`vs.` home, `@` away). It sets FINAL ([Event lifecycle](event-lifecycle.md)).
5. `odds backfill status` prints counts of events with snapshots and the earliest/latest snapshot time.

`GameSelector.calculate_snapshot_times` fixes five snapshots per game: 3 days, 24 h, 12 h, 3 h and 30 min before kickoff. With `--games N`, `games_per_team = max(1, N // 30)`; selection is scored per date with a random jitter, so repeat plans can differ.

## Route B: tier gaps

`odds backfill gaps --start D --end D [--max-quota Q] [--output gap.json]` uses `GapBackfillPlanner` and `TierCoverageValidator` to find games missing fetch tiers ([Fetch tiers](fetch-tier.md)), excluding IN_PLAY. Games are ordered by highest missing tier (`FetchTier.get_priority_order`: CLOSING, PREGAME, SHARP, EARLY, OPENING). Under `--max-quota` games are included whole or not at all, and it errors if even the first game exceeds the cap. Nothing is saved without `--output`; then run `odds backfill execute --plan gap.json`.

## Hazards

- **NBA hardcoding.** `basketball_nba` is literal in `backfill.py` (plan, scores), `BackfillExecutor._process_snapshot` (`get_historical_odds`) and the tier validator's game-discovery check. `discover` defaults to NBA but takes `--sport`. `docs/CLI.md#historical-backfill-workflow` shows `--sport soccer_epl` for discovery, but plan finds no events for EPL and there is no `--sport` on plan, execute or scores.
- **Quota is wasted on non-NBA events.** `gaps` analysis is not sport-filtered; the executor then requests NBA historical odds and marks the event "not found in API response" after the call.
- **Quota estimate is a constant**: 30 per snapshot (`GapBackfillPlanner.QUOTA_PER_SNAPSHOT`, a literal 30 in `GameSelector`, and `+= 30` in the executor), not derived from configured regions or markets.
- **`discover games` overwrites** status and scores of events already in the DB (`OddsWriter.bulk_upsert_events`; [Event lifecycle](event-lifecycle.md)). Docs describe it only as discovery.
- `backfill status` counts all `odds_snapshots`, not only backfilled ones.
- Final scores are not available from the Odds API for games older than 3 days (`discover.py` docstring), hence step 4.

## Related

- [Odds writer](odds-writer.md)
- [Completeness validation](completeness-validation.md)
- [CLI overview](cli-overview.md)
