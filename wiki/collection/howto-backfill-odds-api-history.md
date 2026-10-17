---
type: howto
title: Backfill Odds API history (NBA)
description: Steps for discover, plan, execute, scores and status, and the tier-gap route.
sources:
  - odds_cli.commands.discover
  - odds_cli.commands.backfill
  - odds_analytics.gap_backfill_planner.GapBackfillPlanner
  - odds_analytics.backfill_executor.BackfillExecutor
status: stable
---

# Backfill Odds API history (NBA)

Loads historical Odds API snapshots and scores for NBA. It spends paid quota and works for NBA only; command behaviour and limits are on [odds-api-backfill](odds-api-backfill.md). Check `odds status quota` first ([odds-api-quota](odds-api-quota.md)).

## Route A: discover, plan, execute

1. Discover the events.
   ```bash
   uv run odds discover games --start 2025-01-01 --end 2025-01-07
   ```
   - It upserts the events as SCHEDULED. Events already in the database lose their status and scores, so use ranges you have not finalised (#440).
2. Build a plan.
   ```bash
   uv run odds backfill plan --start 2025-01-01 --end 2025-01-07 --games 20 --output plan.json
   ```
   - The plan fixes five snapshots per game, so cost is known before you spend it.
3. Dry-run, then execute.
   ```bash
   uv run odds backfill execute --plan plan.json --dry-run
   uv run odds backfill execute --plan plan.json
   ```
   > [!WARNING]
   > `odds backfill execute` spends paid historical quota, 10 units per region per market per call. `gaps` is not sport-filtered and the executor requests NBA odds for every event, so non-NBA events consume quota and return nothing (#448).

   - Confirm the prompt only after the dry-run estimate fits the quota; the estimate is a constant 30 units per snapshot (#441).
   - Existing snapshots within 5 minutes are skipped unless you pass `--no-skip-existing`.
4. Fill the scores.
   ```bash
   uv run odds nba-stats fetch
   uv run odds backfill scores --start 2025-01-01 --end 2025-01-07 --dry-run
   ```
   - The scores come from `nba_team_game_logs`, so fetch the game logs first; rerun without `--dry-run` to write them.
5. Check the result with `uv run odds backfill status`.

## Route B: tier gaps

1. Find games missing fetch tiers.
   ```bash
   uv run odds backfill gaps --start 2025-01-01 --end 2025-01-07 --max-quota 500 --output gap.json
   ```
   - Nothing is saved without `--output`.
   - Under `--max-quota` games are included whole or not at all, and it errors if even the first game exceeds the cap.
2. Execute the gap plan.
   ```bash
   uv run odds backfill execute --plan gap.json --dry-run
   ```
   - Rerun without `--dry-run` once the estimate fits the quota; the warning above applies.

## Related

- [Odds API historical backfill](odds-api-backfill.md)
- [Load historical EPL odds](howto-load-historical-epl-odds.md)
- [Odds API quota accounting](odds-api-quota.md)
