---
type: component
title: MLB probable pitchers
summary: The MLB Stats API fetcher, the fetch-mlb-probables job, the mlb_probable_pitchers snapshot table and its domain-record reader.
sources:
  - packages/odds-lambda/odds_lambda/mlb_stats_fetcher.py::MlbStatsFetcher
  - packages/odds-lambda/odds_lambda/jobs/fetch_mlb_probables.py::main
  - packages/odds-lambda/odds_lambda/storage/mlb_pitcher_writer.py::MlbPitcherWriter
  - packages/odds-lambda/odds_lambda/storage/mlb_pitcher_reader.py::MlbPitcherReader
  - packages/odds-core/odds_core/mlb_data_models.py::MlbProbablePitchers
  - packages/odds-mcp/odds_mcp/tools/mlb.py::get_probable_pitchers
  - docs/ARCHITECTURE.md#fixture-and-probables-jobs
  - docs/DATABASE.md#sport-context-tables
  - docs/AGENT_DATA_SOURCES.md#scheduled-collection
  - docs/audits/comprehension-audit-2026-06-09.md#5-what-was-not-found-worth-saying
verified_at: 7887039
---

# MLB probable pitchers

Point-in-time snapshots of MLBAM's announced probable starters, stored in `mlb_probable_pitchers` (`MlbProbablePitchers`, `packages/odds-core/odds_core/mlb_data_models.py`). MLBAM only exposes current state, so the announcement timeline exists only if every fetch is written.

## Flow

1. `fetch_mlb_probables.py::main` computes `dates_for_window(now, 72h)`: UTC dates from today to D+3, padded one day each side because MLBAM resolves `date=` in US Eastern, not UTC.
2. `MlbStatsFetcher.fetch_dates` calls `https://statsapi.mlb.com/api/v1/schedule` with `hydrate=probablePitcher` (httpx, 3 attempts, 0.25s pacing). One `MlbProbablePitchersRecord` per game; a missing `probablePitcher` yields null name/id fields, deliberately kept as the "not announced" signal.
3. `MlbPitcherWriter.insert_snapshots` is append-only, `ON CONFLICT DO NOTHING` on `(game_pk, fetched_at)`. All records in one run share one `fetched_at`, and it returns the input count, not rows inserted.

The job raises unless `ctx.sport` is `baseball_mlb`. Schedule: `0 6 * * *`, MLB only (`_JOB_CRON_MAP` in `packages/odds-lambda/odds_lambda/scheduling/jobs.py`), bootstrapped locally. Not in any Terraform rule per [DEPLOYMENT](../docs/DEPLOYMENT.md#job-deployment-matrix); deployed state is unverifiable from the repo: `odds scheduler list-jobs --backend aws`.

## Reading

`MlbPitcherReader.get_latest_in_window(start, end)` returns domain records (not SQLModel rows), filtered by `commence_time`, latest `fetched_at` per `game_pk`, via `odds_core.mlb_data_models::select_latest_in_window`. The same function is applied to live records, so both read paths return identical shapes.

The MCP tool `get_probable_pitchers` live-fetches by default (`refresh=True`), never writes, and falls back to the reader when MLBAM fails (`fetch_status` = `live`, `stale_db_only` or `db_only`). It cannot distinguish openers from starters.

## Hazards

- The comment on `fetch-mlb-probables` in `_JOB_CRON_MAP` says the MCP tool "writes through every call"; the tool and the job docstring say it never persists. Code: read-only. The daily cron is the sole writer.
- Table grows by every game on every daily run (no dedup of unchanged state, no retention).
- `commence_time` and `fetched_at` are the only indexed columns besides the unique key; no `game_pk` standalone index (the unique key's prefix serves it).

## Related

- [EPL context data](epl-context-data.md) - sibling daily job
- [Event matching](event-matching.md) - MLB rows carry MLBAM team names, not `events` ids
- [Data sources](data-sources.md)
