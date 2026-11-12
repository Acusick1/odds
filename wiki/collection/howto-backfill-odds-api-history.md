---
type: howto
title: Backfill Odds API history
description: Steps to cost an acquisition manifest, execute its calls, audit what they stored, and fill scores and check status.
sources:
  - odds_cli.commands.discover
  - odds_cli.commands.events.sync_mlb_seasons
  - odds_cli.commands.backfill
  - odds_analytics.backfill_plan.Manifest
  - odds_analytics.backfill_executor.BackfillExecutor
  - odds_analytics.backfill_executor.BackfillExecutor.probe
  - scripts.audit_backfill.main
  - scripts.audit_backfill.band_coverage
  - scripts.audit_backfill.reconcile_calls
  - scripts.audit_backfill.is_sealed
  - scripts.audit_backfill.near_close_lines
  - odds_analytics.holdout.intersecting
status: stable
---

# Backfill Odds API history

Costs a historical purchase before it is bought, buys it, audits what it stored, and fills scores. Executing spends paid quota; command behaviour and limits are on [odds-api-backfill](odds-api-backfill.md). Check `odds status quota` first ([odds-api-quota](odds-api-quota.md)).

## Cost and execute a manifest

1. Write or pick a manifest in `backfills/` ([acquisition manifests](odds-api-backfill.md#acquisition-manifests)).
   - List market keys only (`markets: [h2h]`); a call stores every line the books return.
   - Put each probe snapshot inside a season and on the snapshot grid, seconds included (such as `2025-09-13T09:55:38Z`); loading the manifest rejects one that is not.
2. Make sure the manifest's seasons are stored as FINAL events; the planner reads fixtures from the database, and the executor stores prices only on stored events. `odds discover games --start D --end D --sport S` stores the events the Odds API lists for a window, spending sport S's key, resolving each to an existing event or creating a SCHEDULED one at its pre-game start; a game a snapshot lists already under way is skipped, since the API lists it at its actual start ([commands](odds-api-backfill.md#commands)). For MLB, `odds events sync-mlb-seasons --season YEAR` stores every played game as a FINAL event from the free MLB Stats API instead ([MLB season load](mlb-season-load.md)).
3. Print the cost.
   ```bash
   uv run odds backfill plan --manifest backfills/odds-api-epl-2022-26.yaml --cost
   ```
   - It makes no API call. Add `--season 2024-25` to cost one season, and drop `--cost` to list the planned calls.
   - A band's row is what that band adds to the others, so dropping it saves that many calls.
   - The caption is the sport's key balance left after the whole plan, or says the balance is unknown when no recent fetch recorded a quota.
4. Dry-run the execution.
   ```bash
   uv run odds backfill execute --manifest backfills/odds-api-epl-2022-26.yaml --season 2024-25 --dry-run
   ```
   - It makes no API call. It prints the calls, the units they would cost and the acquisition key's balance (unknown when no recent fetch recorded a quota), and counts the calls a backfill already stored as skipped; live Odds API snapshots do not count.
5. Probe the manifest.
   ```bash
   uv run odds backfill probe --manifest backfills/odds-api-epl-2022-26.yaml --dry-run
   uv run odds backfill probe --manifest backfills/odds-api-epl-2022-26.yaml
   ```
   - The dry run lists each call's request time, the snapshot it requests and its units, without a key or a database; the real run spends them on the acquisition key and stores what comes back.
   - It exits non-zero if the returned snapshot is not the requested one, or the billed units, the grid gaps, the farthest fixture's lead, the required books or how far ahead they are listed disagree with the manifest ([probe](odds-api-backfill.md#probing-a-manifest)). Fix the manifest, or stop, before executing.
6. Execute.
   ```bash
   uv run odds backfill execute --manifest backfills/odds-api-epl-2022-26.yaml --season 2024-25
   ```
   > [!WARNING]
   > `odds backfill execute` spends paid historical quota on the acquisition key (`ODDS_API_ACQUISITION_KEY_INDEX`, which must be set), whatever the manifest's sport, 10 units per region per market per call. Confirm the prompt only when the estimated units fit the key's balance.

   - Add `--max-units N` to cap the units the run may bill. The run also stops when a call's billed units differ from the manifest's, when a call returns a snapshot later than its request time or stamped for the next grid time, when several consecutive calls are sparse (no events, none of their fixtures stored, or a required book missing on a fixture near kick-off; `--max-consecutive-gaps`), or when several consecutive calls are provider gaps (`--max-consecutive-provider-gaps`) ([guards](odds-api-backfill.md#executing-calls)).
   - A call that returns an earlier snapshot hit a hole in the provider's history. The run stores it, records it as a provider gap and carries on, and a rerun skips it without paying again ([provider gaps](odds-api-backfill.md#provider-gaps)). A call that returns the planned step's snapshot stamped off the grid's phase is stored and counted, and the run carries on ([off phase](odds-api-backfill.md#where-the-returned-snapshot-lies)).
   - A guard that stops the run exits non-zero with its reason, and so does a call that fails. Rerun the same command: calls with a stored snapshot for any fixture they serve, and calls a recorded provider gap covers, are skipped, unless you pass `--no-skip-existing`.
   - A later-snapshot stop means the manifest's grid is wrong. When the snapshot is after the request time, an unchanged rerun pays for the same call and stops again; when it is stamped for the next grid time at or before the request time, an unchanged rerun skips the call and carries on with the wrong grid. Correct the manifest's `snapshot_grid` to the returned snapshot's phase before rerunning.
   - The run makes its calls back to back. A call the API rate-limits (HTTP 429) is retried with backoff; add `--pause S` to wait S seconds after each paid call ([pace](odds-api-backfill.md#executing-calls)).
   - Manifests of different sports can run at once, one process each ([limits](odds-api-backfill.md#limits)).
   - Each call logs its real cost, and the run ends with the key's remaining quota. Backfill calls record no quota in `fetch_logs`, so `odds status quota` does not show their spend.
   - The summary's dropped games are distinct games the calls listed but no stored event took ([historical listings](../storage/event-matching.md#historical-odds-api-listings)); an `id_refused` drop names a second event holding the game's Odds API id.

## Audit a finished manifest

1. Print the manifest's audit, by season, and commit it beside the manifest as `backfills/<manifest stem>.audit.md`.
   ```bash
   uv run python scripts/audit_backfill.py --manifest backfills/odds-api-bundesliga-2024-26.yaml
   ```
   - It only reads the database and spends no units; add `--season 2024-25` to audit one season. It reads every historical Odds API snapshot and recorded gap of the manifest's sport, market and seasons, so runs of other sports may write meanwhile, and an earlier manifest over the same sport and seasons is reported with this one.
   - It reports FINAL fixtures without a stored snapshot, listings in stored responses that attached to no event, each book's fixture coverage, the share of each band's whole-minute decisions served within the band's max age at the snapshots' actual times, the latest Betfair Exchange price at or before kick-off with its age (a season with a kick-off in a sealed holdout window, `odds_analytics.holdout.intersecting`, gets the counts and ages of any quote only, without the overround and back-price checks), the provider gaps and off-phase snapshots, and the planned calls against the stored snapshots, judging each call as the executor's skip does: stored when a fixture it serves has a snapshot in the call's window, else covered by a recorded gap, else missing ([skip](odds-api-backfill.md#executing-calls)).
   - Its units are distinct stored snapshot times times the manifest's units per call. A call bought twice stores one snapshot time and the database keeps no bill, so take the billed units from `odds backfill execute`'s output: each `fetched` line is one paid call with its units, and each run's summary has its units used.

## Fill scores

1. Fetch the game logs and fill the scores.
   ```bash
   uv run odds nba-stats fetch
   uv run odds backfill scores --start 2025-01-01 --end 2025-01-07 --dry-run
   ```
   - The scores come from `nba_team_game_logs`, so fetch the game logs first; rerun without `--dry-run` to write them.
2. Check the result with `uv run odds backfill status`, which also lists the recorded provider gaps.

## Related

- [Odds API historical backfill](odds-api-backfill.md)
- [Load historical EPL odds](howto-load-historical-epl-odds.md)
- [Odds API quota accounting](odds-api-quota.md)
