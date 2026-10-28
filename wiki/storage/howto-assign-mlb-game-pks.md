---
type: howto
title: Assign MLB events their gamePk
description: Steps to give stored MLB events without a gamePk their MLB Stats API gamePk, merge the events of one game, and settle the events the backfill cannot resolve.
sources:
  - odds_cli.commands.events.assign_game_pks
  - odds_cli.commands.events.set_game_pk
  - odds_cli.commands.events.mark_unassignable
status: stable
---

# Assign MLB events their gamePk

MLB events stored before the live jobs keyed events to the MLB Stats API `gamePk` hold no `gamePk`, and some games are split across several events. This backfill gives each event its `gamePk` and merges the events of one game. What each command does is on [Event merge](event-merge.md#assigning-gamepks).

> [!WARNING]
> `--apply` merges and deletes events in the database `DATABASE_URL` names. On the production host that is the production database; elsewhere point it at a copy ([environments](../foundations/databases-and-environments.md#environments)).

## Steps

1. **Dry-run the backfill.** It resolves every event, runs every merge and rolls it all back, so the counts are exact.

   ```bash
   odds events assign-game-pks --since 2026-03-01 > game-pks-dry-run.txt
   ```

2. **Read the summary, then the lines.** Each line starts with its kind, so `grep '^MERGE '` lists the merges.
   - `MERGE_READ_ONLY` merges move paper trades, match briefs or candidate decisions to the kept event. Check that each kept event is the game the trade or brief was about.
   - `REFUSED reason=paper_trade_clash` games would delete a candidate decision or paper trade, so they are left as they are; settle them in step 5.
   - `moved_from=` shows the kept event's old and new start. A postponed slot can move by weeks, and all its snapshots' tiers follow the new start. `start=tbd` marks a game replayed with a TBD start: a kept event on the played date keeps its listed time, and a postponed slot's event moves to MLB's placeholder start, a few minutes after game 1, on the played date.
   - `score=` and `settled=` show where the kept event's result comes from. `settled=mlb` settles it from MLB's Final score; any `conflict` is left as stored for `repair-scores` (step 6).
   - `UNRESOLVED` events are left as they are; settle them in step 5.
   - `GHOST` events have no MLB game and nothing references them.
3. **Stop the scheduler, apply, start it again.** A live MLB run writing to an event while it is merged makes that game `FAILED` or that run's write fail, so stop the jobs for the few minutes the apply takes, as `deploy.sh` does for migrations ([scheduler service](../scheduling/howto-run-scheduler-service.md)). Each game commits in its own transaction; one that fails is printed as `FAILED` and the rest go ahead. Add `--delete-ghosts` to delete the ghosts too.

   ```bash
   sudo systemctl stop odds-scheduler
   odds events assign-game-pks --since 2026-03-01 --apply
   sudo systemctl start odds-scheduler
   ```

4. **Run it again.** The second dry run assigns nothing new; it lists only the events left unresolved, refused or failed.
5. **Settle each unresolved or refused event by hand.** Find its game on the MLB schedule, then dry-run before adding `--apply`:
   - it is a game: `odds events set-game-pk EVENT_ID GAME_PK`. Add `--merge` when another event already holds that `gamePk`, to merge into it. A merge that would delete a candidate decision or trade is refused here too; resolve the clashing decision first.
   - it has no game: `odds events mark-unassignable EVENT_ID --reason "..."`, so the `missing_game_pk` warning leaves it out.
6. **Re-check the scores.** Merging moves results between events, so run `odds events repair-scores` ([results jobs](../collection/results-jobs.md#verifying-mlb-scores)).

## Related

- [Event merge](event-merge.md)
- [Event matching across sources](event-matching.md#live-mlb-events)
