---
type: howto
title: Backfill and verify the new shape
description: Steps to move production's stored history into the new snapshot, price and event-identity shape with odds data backfill-shape, check it with odds data verify-shape, and recover when either fails.
sources:
  - odds_cli.commands.data.backfill_shape
  - odds_cli.commands.data.verify_shape_command
mentions:
  - odds_cli.commands.events.repair_names
status: stable
---

# Backfill and verify the new shape

`odds data backfill-shape` fills the new-shape columns, payloads, price rows and external ids for everything stored before the live writers wrote them, and `odds data verify-shape` checks every snapshot against its `raw_data`. What each step writes is on [Shape backfill and verify](shape-backfill.md). Run these on the production host from the repo checkout, after the code is deployed.

> [!WARNING]
> `--apply` writes to and deletes from the database `DATABASE_URL` names: on the production host, the production database. Take the dump in step 1 first; nothing in the repo backs production up ([environments](../foundations/databases-and-environments.md#environments)).

## Steps

1. **Dump the database.** The dump is the way back from a bad run; keep it until the old shape has been dropped and verified.

   ```bash
   docker compose exec -T postgres pg_dump -U postgres -Fc odds > ~/odds-before-shape-backfill.dump
   ls -lh ~/odds-before-shape-backfill.dump
   ```

2. **Preflight.** It runs every step as a count on read-only connections, so the scheduler can keep running. `--oddsportal-dir` is optional: give it only where the historical OddsPortal scrapes are on disk.

   ```bash
   uv run odds data backfill-shape --oddsportal-dir data/external/oddsportal > shape-preflight.txt
   ```

3. **Read the preflight.**
   - The twin groups, the events to mark UNMATCHED and the external-id counts are what the event steps will write. A twin line with `deleted=` naming `candidate_decisions` or `paper_trades` stops the apply; decide on those decisions first, or pass `--allow-merge-drops` to let the merge delete them.
   - `duplicates over the new key` lists each snapshot whose key another holds; that market is dropped and archived ([duplicates](shape-backfill.md#snapshots)).
   - `referenced by decisions` must be 0: a decision's input that would be split or deleted stops the run.
   - Events with unmatched outcome names: run `odds events repair-names` for their sport, dry run first ([Event merge](event-merge.md#command)), then the preflight again. A backfilled snapshot is not read again.
4. **Apply.** It plans again and stops before writing if a gate fails ([plan and gates](shape-backfill.md#plan-and-gates)). Batches commit one at a time, skip rows a live job holds and wait at most a moment for a lock, so the scheduler can keep running. `--map-out` keeps the old and new ids of every split snapshot. Its counts should equal the preflight's.

   ```bash
   uv run odds data backfill-shape --apply --oddsportal-dir data/external/oddsportal \
     --map-out shape-split-map.jsonl > shape-apply.txt
   ```

5. **Verify.** It exits 1 and lists examples when any snapshot differs from its `raw_data`.

   ```bash
   uv run odds data verify-shape > shape-verify.txt; echo "exit $?"
   ```

6. **Run the backfill again.** Every count should be 0 or `held`/`already`: nothing is left to write, including the snapshots a live job kept locked during step 4 (`left pending`). Then verify again.

   ```bash
   uv run odds data backfill-shape --apply > shape-apply-2.txt
   uv run odds data verify-shape
   ```

## When something fails

- **The apply stops before writing** (`Stopped: candidate decisions recorded snapshots …`, or twin merges that would delete decisions or trades): nothing was written. Find the decisions named and decide on them before rerunning.
- **The apply stops part-way** (an error, a lost connection): what it committed stays, and running it again continues from there.
- **Verify reports `no_source` or `no_payload`:** the backfill has not reached those snapshots; run it again.
- **Verify reports another check:** the listed snapshots disagree with their `raw_data`. Leave the data in place, read the examples against [the checks](shape-backfill.md#verify), and fix the code before running anything else.
- **The data is wrong and must be undone:** stop the scheduler, restore the dump into an empty `odds` database and start it again ([run the scheduler service](../scheduling/howto-run-scheduler-service.md)). Everything collected since the dump is lost.

  ```bash
  sudo systemctl stop odds-scheduler
  docker compose exec -T postgres dropdb -U postgres --force odds
  docker compose exec -T postgres createdb -U postgres odds
  docker compose exec -T postgres pg_restore -U postgres -d odds < ~/odds-before-shape-backfill.dump
  sudo systemctl start odds-scheduler
  ```

## Related

- [Prices and snapshot payloads tables](price-tables.md)
- [Assign MLB events their gamePk](howto-assign-mlb-game-pks.md)
