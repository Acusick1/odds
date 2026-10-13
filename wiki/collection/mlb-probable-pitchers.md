---
type: component
title: MLB probable pitchers
description: "The MLB Stats API fetcher, the fetch-mlb-probables job, the append-only mlb_probable_pitchers table and its reader, and how the MCP tool reads it."
sources:
  - packages/odds-lambda/odds_lambda/mlb_stats_fetcher.py::MlbStatsFetcher
  - packages/odds-lambda/odds_lambda/jobs/fetch_mlb_probables.py::main
  - packages/odds-lambda/odds_lambda/storage/mlb_pitcher_writer.py::MlbPitcherWriter
  - packages/odds-lambda/odds_lambda/storage/mlb_pitcher_reader.py::MlbPitcherReader
  - packages/odds-core/odds_core/mlb_data_models.py::MlbProbablePitchers
  - packages/odds-lambda/odds_lambda/scheduling/jobs.py::_JOB_CRON_MAP
  - packages/odds-mcp/odds_mcp/tools/mlb.py::get_probable_pitchers
  - deployment/aws/terraform/eventbridge.tf
status: stable
---

# MLB probable pitchers

Point-in-time snapshots of MLBAM's announced probable starters, stored in `mlb_probable_pitchers` (`packages/odds-core/odds_core/mlb_data_models.py::MlbProbablePitchers`). MLBAM only exposes current state, so the announcement timeline exists only if every fetch is written. MLB odds themselves are `h2h` ([sources-and-keys](sources-and-keys.md)).

## Flow

1. `packages/odds-lambda/odds_lambda/jobs/fetch_mlb_probables.py::main` computes `dates_for_window(now, 72h)`: UTC dates from today to D+3, padded one day each side because MLBAM resolves `date=` in US Eastern, not UTC.
2. `packages/odds-lambda/odds_lambda/mlb_stats_fetcher.py::MlbStatsFetcher.fetch_dates` calls `https://statsapi.mlb.com/api/v1/schedule` with `hydrate=probablePitcher` (httpx, 3 attempts, 0.25 s pacing). One `packages/odds-core/odds_core/mlb_data_models.py::MlbProbablePitchersRecord` per game; a missing `probablePitcher` yields null name/id fields, deliberately kept as the "not announced" signal.
3. `packages/odds-lambda/odds_lambda/storage/mlb_pitcher_writer.py::MlbPitcherWriter.insert_snapshots` is append-only, `ON CONFLICT DO NOTHING` on `(game_pk, fetched_at)`. All records in one run share one `fetched_at`, and it returns the input count, not rows inserted.

The job raises unless `ctx.sport` is `baseball_mlb`.

## Where it runs

Schedule `0 6 * * *`, MLB only (`_JOB_CRON_MAP` in `packages/odds-lambda/odds_lambda/scheduling/jobs.py`), in the default local bootstrap. It never runs on AWS: Terraform `sport_configs` has MLB commented out and no rule exists ([jobs-and-deployment-matrix](../scheduling/jobs-and-deployment-matrix.md)). Deployed state is unverifiable from the repo: `odds scheduler list-jobs --backend aws`. The MLB overnight window (05:00-14:00 UTC) belongs to the self-scheduling jobs, not this cron ([self-scheduling](../scheduling/self-scheduling.md)).

## Reading

`MlbPitcherReader.get_latest_in_window(start, end)` returns domain records (not SQLModel rows), filtered by `commence_time`, latest `fetched_at` per `game_pk`, via `odds_core.mlb_data_models::select_latest_in_window`. The same function is applied to live records, so both read paths return identical shapes.

The MCP tool `packages/odds-mcp/odds_mcp/tools/mlb.py::get_probable_pitchers` ([mcp-sport-context-tools](../agent/mcp-sport-context-tools.md)) live-fetches by default (`refresh=True`), never writes, and falls back to the reader when MLBAM fails. Its `fetch_status` is `live`, `stale_db_only` or `db_only`; a null pitcher means "not announced", and openers and starters are indistinguishable.

## Using the data

- **Current state**: use `packages/odds-lambda/odds_lambda/storage/mlb_pitcher_reader.py::MlbPitcherReader.get_latest_in_window`, which collapses the append-only history to the latest snapshot per `game_pk`; do not query the table directly for "the" probable starter, or you will see every historical snapshot.
- **Announcement timeline**: group rows by `game_pk` and order by `fetched_at`. Resolution is one snapshot per daily run, so intra-day changes are invisible, and a pitcher change made between runs shows only as a differing name or id on consecutive snapshots.
- **Null fields** mean the starter is not yet announced (kept deliberately), not a fetch failure. A game with no row at all was outside the 72 h window on every run.
- **Joining to odds**: there is no foreign key to `events`; match on teams and `commence_time` yourself, remembering that MLBAM team names are not the pipeline's canonical aliases.

## Hazards

- **Misleading cron comment.** The comment on `fetch-mlb-probables` in `_JOB_CRON_MAP` says the MCP tool "writes through every call"; the tool and the job docstring say it never persists. Code: read-only. The daily cron is the sole writer.
- **Unbounded growth.** The table grows by every game on every daily run (no dedup of unchanged state, no retention).
- **Indexes.** `commence_time` and `fetched_at` are the only indexed columns besides the unique key; there is no standalone `game_pk` index (the unique key's prefix serves it).
- **No event link.** MLB rows carry MLBAM team names, not `events` ids ([events](../storage/events.md#matchers)).

## Related

- [EPL context data](epl-context-data.md), the sibling daily-cron job
- [MCP sport context tools](../agent/mcp-sport-context-tools.md)
- [Sources and keys](sources-and-keys.md)
