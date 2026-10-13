---
type: component
title: EPL context data (ESPN, FPL, FBref)
description: "ESPN fixtures (the fetch-espn-fixtures job) and lineups, FPL availability and FBref stats, covering fetchers, scripts, tables and readers, and which of them any feature or tool actually consumes."
sources:
  - packages/odds-lambda/odds_lambda/espn_fixture_fetcher.py::EspnFixtureFetcher
  - packages/odds-lambda/odds_lambda/jobs/fetch_espn_fixtures.py::main
  - packages/odds-lambda/odds_lambda/storage/espn_fixture_writer.py::EspnFixtureWriter
  - packages/odds-lambda/odds_lambda/storage/espn_fixture_reader.py::EspnFixtureReader
  - packages/odds-lambda/odds_lambda/storage/espn_lineup_reader.py::EspnLineupReader
  - packages/odds-lambda/odds_lambda/storage/fpl_availability_reader.py::FplAvailabilityReader
  - packages/odds-core/odds_core/epl_data_models.py::EspnFixture
  - packages/odds-lambda/odds_lambda/scheduling/jobs.py::_JOB_CRON_MAP
  - packages/odds-mcp/odds_mcp/tools/epl.py::_is_final
  - scripts/ingest_espn_lineups.py
  - scripts/ingest_fbref_player_stats.py
  - deployment/aws/terraform/eventbridge.tf
status: stable
---

# EPL context data (ESPN, FPL, FBref)

Supplementary EPL data that feeds the `epl_schedule` and `epl_lineup` [feature groups](../modelling/epl-context-features.md) and the MCP team-context tool ([mcp-sport-context-tools](../agent/mcp-sport-context-tools.md)). Models are in `packages/odds-core/odds_core/epl_data_models.py`; fetchers and storage in `packages/odds-lambda/odds_lambda/`.

| Source | Table (model) | Written by | Read by |
|---|---|---|---|
| ESPN fixtures | `espn_fixtures` (`packages/odds-core/odds_core/epl_data_models.py::EspnFixture`) | the `fetch-espn-fixtures` job; `scripts/ingest_espn_fixtures.py` | `packages/odds-lambda/odds_lambda/storage/espn_fixture_reader.py::EspnFixtureReader`, `odds_analytics.feature_groups::load_fixtures_df` |
| ESPN lineups | `espn_lineups` (`packages/odds-core/odds_core/epl_data_models.py::EspnLineup`) | `scripts/ingest_espn_lineups.py` (manual) | `packages/odds-lambda/odds_lambda/storage/espn_lineup_reader.py::EspnLineupReader`, `odds_analytics.feature_groups::load_lineup_cache` |
| FPL availability | `fpl_availability` (`packages/odds-core/odds_core/epl_data_models.py::FplAvailability`) | `scripts/ingest_fpl_availability.py` (manual) | `packages/odds-lambda/odds_lambda/storage/fpl_availability_reader.py::FplAvailabilityReader` |
| FBref player stats | none | `scripts/ingest_fbref_player_stats.py` (CSV only) | n/a |

## ESPN fixtures

`packages/odds-lambda/odds_lambda/espn_fixture_fetcher.py::EspnFixtureFetcher` wraps the free ESPN Site API over httpx (3 attempts, 1 s backoff). `LEAGUE_SLUGS` covers the Premier League, FA Cup, League Cup and three European competitions (the `epl_schedule` group uses these all-competition fixtures). Two endpoints are combined:

- `fetch_season` walks each EPL team's `teams/{id}/schedule` (completed matches only).
- `fetch_upcoming` reads the date-range scoreboard, emitting two mirrored rows per match.

`packages/odds-lambda/odds_lambda/jobs/fetch_espn_fixtures.py::main` runs both (30-day lookahead), dedupes on `(date, team, opponent)` with the scoreboard row winning (fresher `state`), and upserts through `EspnFixtureWriter.upsert_fixtures`, idempotent on `(date, team, competition)`. It raises for any sport but `soccer_epl`. Each match is two rows, one per team.

**Schedule**: `0 6 * * *`, EPL only, from `_JOB_CRON_MAP` in `packages/odds-lambda/odds_lambda/scheduling/jobs.py`, in the default local `SchedulerConfig.bootstrap_jobs` (`packages/odds-core/odds_core/config.py`). It runs on the local scheduler only; no EventBridge rule exists in `deployment/aws/terraform/eventbridge.tf`. Deployed state cannot be verified from the repo: `odds scheduler list-jobs --backend aws` ([jobs-and-deployment-matrix](../scheduling/jobs-and-deployment-matrix.md)). The MCP `packages/odds-mcp/odds_mcp/tools/epl.py::get_team_context` tool reads `espn_fixtures`, so it goes stale if the local scheduler is not running this cron.

## Lineups, FPL, FBref

All three are manual scripts writing CSVs under `data/`, with `--skip-db` on the lineup and FPL scripts to skip the upsert. Lineup coverage is 2020-21 onward, FPL 2021-22 onward (`SEASONS` in each script). FBref pages must be saved by hand from a browser (Cloudflare) and parsed; there is no model or table.

`epl_lineup` features use `espn_lineups`, with a look-ahead caveat: the XI is public about KO-60 while training samples 3-24 h out. FPL availability hurt in the ablation (R² -0.017 sliding) and has no feature group ([exp-epl-feature-ablation-and-search-space](../evaluation/exp-epl-feature-ablation-and-search-space.md)).

No structured lineup API was usable for live pre-kickoff data; ESPN lineups were rejected for the agent (non-monotonic near KO-60/KO-45, 403 from cloud IPs) and the agent reads BBC Sport instead (inferred from manual probes on 2026-04-13/14; no results file; [agent-data-sources](../agent/agent-data-sources.md)). That is why no scheduled lineup job exists, though ESPN is still used for fixtures and a manual lineup script exists.

## Hazards

- **Wrong cadence claim.** The `packages/odds-lambda/odds_lambda/jobs/fetch_espn_fixtures.py` module docstring says the cadence is "managed in Terraform"; it is local-cron only (`_JOB_CRON_MAP`, Terraform comment at `deployment/aws/terraform/eventbridge.tf`).
- **FPL is stored but unused.** `packages/odds-lambda/odds_lambda/storage/fpl_availability_reader.py::FplAvailabilityReader` has no caller outside the module; no feature code in `packages/` consumes FPL data.
- **Nullable `state`.** `espn_fixtures.state` is nullable; consumers fall back to `status` strings (`packages/odds-mcp/odds_mcp/tools/epl.py::_is_final`).
- **Name pass-through.** Team names pass through `odds_core.team.normalize_team_name`, which passes unknown names through unchanged ([events](../storage/events.md#matchers)).

## Related

- [Sources and keys](sources-and-keys.md), for where ESPN sits among feeds
- [EPL context feature groups](../modelling/epl-context-features.md) and [feature pipeline](../modelling/feature-pipeline.md), the consumers
- [MLB probable pitchers](mlb-probable-pitchers.md), the sibling daily-cron job
