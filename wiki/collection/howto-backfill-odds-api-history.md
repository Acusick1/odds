---
type: howto
title: Backfill Odds API history
description: Steps to cost an acquisition manifest, execute its calls, and fill scores and check status.
sources:
  - odds_cli.commands.discover
  - odds_cli.commands.events.sync_mlb_seasons
  - odds_cli.commands.backfill
  - odds_analytics.backfill_plan.Manifest
  - odds_analytics.backfill_executor.BackfillExecutor
status: stable
---

# Backfill Odds API history

Costs a historical purchase before it is bought, buys it, and fills scores. Executing spends paid quota; command behaviour and limits are on [odds-api-backfill](odds-api-backfill.md). Check `odds status quota` first ([odds-api-quota](odds-api-quota.md)).

## Cost and execute a manifest

1. Write or pick a manifest in `backfills/` ([acquisition manifests](odds-api-backfill.md#acquisition-manifests)).
   - List market keys only (`markets: [h2h, totals]`); a call stores every line the books return.
   - Put each probe snapshot inside a season and on the snapshot grid; loading the manifest rejects one that is not.
2. Make sure the manifest's seasons are stored as FINAL events; the planner reads fixtures from the database, and the executor stores prices only on stored events. `odds discover games --start D --end D --sport S` stores the events the Odds API lists for a window, spending sport S's key, resolving each to an existing event or creating a SCHEDULED one at its pre-game start; a game a snapshot lists already under way is skipped, since the API lists it at its actual start ([commands](odds-api-backfill.md#commands)). For MLB, `odds events sync-mlb-seasons --season YEAR` stores every played game as a FINAL event from the free MLB Stats API instead ([MLB season load](mlb-season-load.md)).
3. Print the cost.
   ```bash
   uv run odds backfill plan --manifest backfills/odds-api-epl-2020-26.yaml --cost
   ```
   - It makes no API call. Add `--season 2024-25` to cost one season, and drop `--cost` to list the planned calls.
   - A band's row is what that band adds to the others, so dropping it saves that many calls.
   - The caption is the sport's key balance left after the whole plan, or says the balance is unknown when no recent fetch recorded a quota.
4. Dry-run the execution.
   ```bash
   uv run odds backfill execute --manifest backfills/odds-api-epl-2020-26.yaml --season 2024-25 --dry-run
   ```
   - It makes no API call. It prints the calls, the units they would cost and the acquisition key's balance (unknown when no recent fetch recorded a quota), and counts the calls a backfill already stored as skipped; live Odds API snapshots do not count.
5. Execute.
   ```bash
   uv run odds backfill execute --manifest backfills/odds-api-epl-2020-26.yaml --season 2024-25
   ```
   > [!WARNING]
   > `odds backfill execute` spends paid historical quota on the acquisition key (`ODDS_API_ACQUISITION_KEY_INDEX`, which must be set), whatever the manifest's sport, 10 units per region per market per call. Confirm the prompt only when the estimated units fit the key's balance.

   - A call that fails stops the run. Rerun the same command: calls with a stored snapshot for any fixture they serve are skipped, unless you pass `--no-skip-existing`.
   - Each call logs its real cost, and the run ends with the key's remaining quota. Backfill calls record no quota in `fetch_logs`, so `odds status quota` does not show their spend.
   - The summary's dropped games are distinct games the calls listed but no stored event took ([historical listings](../storage/event-matching.md#historical-odds-api-listings)); an `id_refused` drop names a second event holding the game's Odds API id.

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
