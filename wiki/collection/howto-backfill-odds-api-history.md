---
type: howto
title: Backfill Odds API history
description: Steps to cost an acquisition manifest, execute its calls, and fill scores and check status.
sources:
  - odds_cli.commands.discover
  - odds_cli.commands.backfill
  - odds_analytics.backfill_plan.Manifest
  - odds_analytics.backfill_executor.BackfillExecutor
status: stable
---

# Backfill Odds API history

Costs a historical purchase before it is bought, buys it, and fills scores. Executing spends paid quota; command behaviour and limits are on [odds-api-backfill](odds-api-backfill.md). Check `odds status quota` first ([odds-api-quota](odds-api-quota.md)).

## Cost and execute a manifest

1. Write or pick a manifest in `backfills/` ([acquisition manifests](odds-api-backfill.md#acquisition-manifests)).
2. Make sure the manifest's seasons are stored as FINAL events; the planner reads fixtures from the database. `odds discover games --start D --end D --sport S` stores the events the Odds API lists for a window, resolving each to an existing event or creating a SCHEDULED one ([commands](odds-api-backfill.md#commands)).
3. Print the cost.
   ```bash
   uv run odds backfill plan --manifest backfills/odds-api-epl-2020-26.yaml --cost
   ```
   - It makes no API call. Add `--season 2024-25` to cost one season, and drop `--cost` to list the planned calls.
   - A band's row is what that band adds to the others, so dropping it saves that many calls.
   - The caption is the sport's key balance left after the whole plan.
4. Dry-run the execution.
   ```bash
   uv run odds backfill execute --manifest backfills/odds-api-epl-2020-26.yaml --season 2024-25 --dry-run
   ```
   - It makes no API call. It prints the calls, the units they would cost and the key's balance, and counts the calls already stored as skipped.
5. Execute.
   ```bash
   uv run odds backfill execute --manifest backfills/odds-api-epl-2020-26.yaml --season 2024-25
   ```
   > [!WARNING]
   > `odds backfill execute` spends paid historical quota on the manifest's sport key, 10 units per region per market per call. Confirm the prompt only when the estimated units fit the key's balance.

   - A call that fails stops the run. Rerun the same command: calls whose fixtures already have their snapshot are skipped, unless you pass `--no-skip-existing`.
   - Each call logs its real cost and the key's remaining quota; `odds status quota` shows the balance afterwards.

## Fill scores

1. Fetch the game logs and fill the scores.
   ```bash
   uv run odds nba-stats fetch
   uv run odds backfill scores --start 2025-01-01 --end 2025-01-07 --dry-run
   ```
   - The scores come from `nba_team_game_logs`, so fetch the game logs first; rerun without `--dry-run` to write them.
2. Check the result with `uv run odds backfill status`.

## Related

- [Odds API historical backfill](odds-api-backfill.md)
- [Load historical EPL odds](howto-load-historical-epl-odds.md)
- [Odds API quota accounting](odds-api-quota.md)
