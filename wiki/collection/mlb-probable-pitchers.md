---
type: reference
title: MLB probable pitchers
description: "The MLB Stats API fetcher, the fetch-mlb-probables job, the append-only mlb_probable_pitchers table and its reader, and how the MCP tool reads it."
sources:
  - odds_lambda.mlb_stats_fetcher.MlbStatsFetcher
  - odds_lambda.jobs.fetch_mlb_probables.main
  - odds_lambda.storage.mlb_pitcher_writer.MlbPitcherWriter
  - odds_lambda.storage.mlb_pitcher_reader.MlbPitcherReader
  - odds_core.mlb_data_models.MlbProbablePitchers
  - odds_lambda.scheduling.jobs._JOB_CRON_MAP
  - odds_lambda.scheduling.jobs._JOB_SPORTS
  - odds_mcp.tools.mlb.get_probable_pitchers
  - deployment/aws/terraform/eventbridge.tf
status: stable
---

# MLB probable pitchers

Point-in-time snapshots of MLBAM's announced probable starters, stored in `mlb_probable_pitchers` (`odds_core.mlb_data_models.MlbProbablePitchers`). MLBAM only exposes current state, so the announcement timeline exists only if every fetch is written. MLB odds themselves are `h2h` ([sport-and-market-keys](sport-and-market-keys.md#market-key-dialects)).

## Flow

1. `odds_lambda.jobs.fetch_mlb_probables.main` computes `dates_for_window(now, 72h)`: UTC dates from today to D+3, padded one day each side because MLBAM resolves `date=` in US Eastern, not UTC.
2. `odds_lambda.mlb_stats_fetcher.MlbStatsFetcher.fetch_dates` calls `https://statsapi.mlb.com/api/v1/schedule` with `hydrate=probablePitcher` (httpx, 3 attempts, 0.25 s pacing). One `odds_core.mlb_data_models.MlbProbablePitchersRecord` per game; a missing `probablePitcher` yields null name/id fields, deliberately kept as the "not announced" signal.
3. `odds_lambda.storage.mlb_pitcher_writer.MlbPitcherWriter.insert_snapshots` is append-only, `ON CONFLICT DO NOTHING` on `(game_pk, fetched_at)`. All records in one run share one `fetched_at`, and it returns the input count, not rows inserted.

The job raises unless `ctx.sport` is `baseball_mlb`.

## Where it runs

Schedule `0 6 * * *` (`odds_lambda.scheduling.jobs._JOB_CRON_MAP`), MLB only through `odds_lambda.scheduling.jobs._JOB_SPORTS`, in the default local bootstrap, so it runs on the mini-PC ([job-deployment-matrix](../scheduling/job-deployment-matrix.md#where-each-job-runs)). Terraform declares no rule for it. It keeps running through the MLB off-season (#487). The MLB overnight window (05:00-14:00 UTC) belongs to the self-scheduling jobs, not this cron ([decision-engine](../scheduling/decision-engine.md#overnight-windows)).

## Reading

`MlbPitcherReader.get_latest_in_window(start, end)` returns domain records (not SQLModel rows), filtered by `commence_time`, latest `fetched_at` per `game_pk`, via `odds_core.mlb_data_models.select_latest_in_window`. The same function is applied to live records, so both read paths return identical shapes.

The MCP tool `odds_mcp.tools.mlb.get_probable_pitchers` ([mcp-sport-context-tools](../agent/mcp-sport-context-tools.md)) live-fetches by default (`refresh=True`), never writes, and falls back to the reader when MLBAM fails. Its `fetch_status` is `live`, `stale_db_only` or `db_only`; a null pitcher means "not announced", and openers and starters are indistinguishable.

## Using the data

- **Current state**: use `odds_lambda.storage.mlb_pitcher_reader.MlbPitcherReader.get_latest_in_window`, which collapses the append-only history to the latest snapshot per `game_pk`; do not query the table directly for "the" probable starter, or you will see every historical snapshot.
- **Announcement timeline**: group rows by `game_pk` and order by `fetched_at`. Resolution is one snapshot per daily run, so intra-day changes are invisible, and a pitcher change made between runs shows only as a differing name or id on consecutive snapshots.
- **Null fields** mean the starter is not yet announced (kept deliberately), not a fetch failure. A game with no row at all was outside the 72 h window on every run.
- **Joining to odds**: there is no foreign key to `events`; match on teams and `commence_time` yourself, remembering that MLBAM team names are not the pipeline's canonical aliases.

## Storage characteristics

- The daily cron is the only writer; the MCP tool and the job docstring both state that the tool never persists.
- The table grows by every game on every daily run: there is no dedup of unchanged state and no retention.
- `commence_time` and `fetched_at` are the only indexed columns besides the unique key; there is no standalone `game_pk` index (the unique key's prefix serves it).
- MLB rows carry MLBAM team names, not `events` ids ([event-matching](../storage/event-matching.md)).

## Related

- [EPL context data](epl-context-data.md), the sibling daily-cron job
- [MCP sport context tools](../agent/mcp-sport-context-tools.md)
- [Data sources](data-sources.md)
