---
type: howto
title: Delete same-time snapshot pairs
description: Steps to delete the MLB snapshot pairs that hold two games' prices and the duplicate snapshots of one fetch, with their odds rows, and to check that every odds row then names its snapshot.
sources:
  - odds_cli.commands.events.delete_same_time_pairs_command
  - odds_lambda.storage.same_time_pairs.find_same_time_groups
  - odds_lambda.storage.same_time_pairs.delete_same_time_pairs
status: stable
---

# Delete same-time snapshot pairs

Some events hold two snapshots from one source at one time. An OddsPortal live pair on an MLB event with different prices is two games collapsed onto one event, which cannot be attributed to either game; a pair with identical prices is one fetch stored twice. Their shared odds rows have no `snapshot_id`. `odds events delete-same-time-pairs` removes both kinds. What counts as a pair and what happens to each kind is on [Odds snapshot and odds tables](odds-tables.md#same-time-pairs).

> [!WARNING]
> `--apply` deletes snapshots and odds rows in the database `DATABASE_URL` names. On the production host that is the production database, and nothing in the repo backs it up ([environments](../foundations/databases-and-environments.md#environments)).

## Steps

1. **Assign MLB gamePks first.** Merging events moves their snapshots, so a merge can create pairs; run [Assign MLB events their gamePk](howto-assign-mlb-game-pks.md) before this.
2. **Dry-run.** It deletes and links everything in one transaction and rolls it back, so the counts are exact.

   ```bash
   odds events delete-same-time-pairs > same-time-pairs-dry-run.txt
   ```

3. **Read the summary, then the lines.** One line per event and time, starting with its kind, so `grep '^CONFLICTING '` lists the MLB pairs that go.
   - `CONFLICTING`: both snapshots of each conflicting OddsPortal live MLB pair go, with `odds_deleted` rows.
   - `DUPLICATE`: `deleted` lists the copies that go; `odds_linked` rows are kept and linked to the snapshot that stays.
   - `CONFLICTING_LEFT`: any other conflicting pairs, left in place; identical copies at the same time still go.
   - `REFUSED`: a candidate decision used a snapshot that would go (`used_by_decision`); the group is left in place.
   - `odds rows without a snapshot` is the count after the run, split into orphans, which no snapshot quotes and the command never resolves ([Snapshot link](odds-tables.md#snapshot-link)), and rows several snapshots quote, such as the shared rows of `CONFLICTING_LEFT` and `REFUSED` groups.
4. **Apply.** The command touches only stored snapshot times, so the scheduler can keep running.

   ```bash
   odds events delete-same-time-pairs --apply
   ```

5. **Run the dry run again.** It lists only `CONFLICTING_LEFT` and `REFUSED` groups, and deletes nothing.

## Related

- [Odds snapshot and odds tables](odds-tables.md#snapshot-link)
- [Event merge](event-merge.md)
