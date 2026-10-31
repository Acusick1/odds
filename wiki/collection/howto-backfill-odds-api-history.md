---
type: howto
title: Backfill Odds API history
description: Steps to cost an acquisition manifest, and the NBA tier-gap route through discover, gaps, execute, scores and status.
sources:
  - odds_cli.commands.discover
  - odds_cli.commands.backfill
  - odds_analytics.backfill_plan.Manifest
  - odds_analytics.gap_backfill_planner.GapBackfillPlanner
  - odds_analytics.backfill_executor.BackfillExecutor
status: stable
---

# Backfill Odds API history

Costs a historical purchase before it is bought, and fills NBA tier gaps with historical Odds API snapshots and scores. Executing spends paid quota and works for NBA only; command behaviour and limits are on [odds-api-backfill](odds-api-backfill.md). Check `odds status quota` first ([odds-api-quota](odds-api-quota.md)).

## Route A: cost a manifest

1. Write or pick a manifest in `backfills/` ([acquisition manifests](odds-api-backfill.md#acquisition-manifests)).
2. Make sure the manifest's seasons are stored as FINAL events; the planner reads fixtures from the database.
3. Print the cost.
   ```bash
   uv run odds backfill plan --manifest backfills/odds-api-epl-2020-26.yaml --cost
   ```
   - It makes no API call. Add `--season 2024-25` to cost one season, and drop `--cost` to list the planned calls.
   - A window's row is what that window adds to the others, so dropping it saves that many calls.

## Route B: NBA tier gaps

1. Discover the events.
   ```bash
   uv run odds discover games --start 2025-01-01 --end 2025-01-07
   ```
   - It resolves each listed game to its event: a game already in the database (stored by any source) keeps its status, scores and teams, and a new one is created as SCHEDULED. It commits every 100 games and prints dropped counts.
2. Find games missing fetch tiers.
   ```bash
   uv run odds backfill gaps --start 2025-01-01 --end 2025-01-07 --max-quota 500 --output gap.json
   ```
   - A game is planned only for its missing tiers; the planner tiers each fetch as the `upper_bound` snapshot the backfill stores ([the tier a snapshot covers](../storage/fetch-tier.md#the-tier-a-snapshot-covers)).
   - Nothing is saved without `--output`.
   - Under `--max-quota` games are included whole or not at all, and it errors if even the first game exceeds the cap.
3. Dry-run, then execute the gap plan.
   ```bash
   uv run odds backfill execute --plan gap.json --dry-run
   uv run odds backfill execute --plan gap.json
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

## Related

- [Odds API historical backfill](odds-api-backfill.md)
- [Load historical EPL odds](howto-load-historical-epl-odds.md)
- [Odds API quota accounting](odds-api-quota.md)
