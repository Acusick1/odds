---
type: component
title: EPL context data (ESPN, FPL, FBref)
summary: ESPN fixtures (with the fetch-espn-fixtures job) and lineups, FPL availability and FBref stats - fetchers, ingestion scripts, tables, readers and writers.
sources:
  - packages/odds-lambda/odds_lambda/espn_fixture_fetcher.py::EspnFixtureFetcher
  - packages/odds-lambda/odds_lambda/jobs/fetch_espn_fixtures.py::main
  - packages/odds-lambda/odds_lambda/storage/espn_fixture_writer.py::EspnFixtureWriter
  - packages/odds-lambda/odds_lambda/storage/espn_fixture_reader.py::EspnFixtureReader
  - packages/odds-lambda/odds_lambda/storage/espn_lineup_reader.py::EspnLineupReader
  - packages/odds-lambda/odds_lambda/storage/fpl_availability_reader.py::FplAvailabilityReader
  - packages/odds-core/odds_core/epl_data_models.py::EspnFixture
  - scripts/ingest_espn_lineups.py
  - scripts/ingest_fbref_player_stats.py
  - docs/ARCHITECTURE.md#fixture-and-probables-jobs
  - docs/DATABASE.md#sport-context-tables
  - docs/AGENT_DATA_SOURCES.md#scheduled-collection
  - docs/AGENT_DATA_SOURCES.md#lineup-data-2026-04-13
verified_at: 7887039
---

# EPL context data (ESPN, FPL, FBref)

Supplementary EPL data that feeds the `epl_schedule` and `epl_lineup` [feature groups](feature-groups.md) and the MCP team-context tool. Models are in `packages/odds-core/odds_core/epl_data_models.py`; fetchers and storage in `packages/odds-lambda/odds_lambda/`.

| Source | Table (model) | Written by | Read by |
|---|---|---|---|
| ESPN fixtures | `espn_fixtures` (`EspnFixture`) | the `fetch-espn-fixtures` job; `scripts/ingest_espn_fixtures.py` | `EspnFixtureReader`, `odds_analytics.feature_groups::load_fixtures_df` |
| ESPN lineups | `espn_lineups` (`EspnLineup`) | `scripts/ingest_espn_lineups.py` (manual) | `EspnLineupReader`, `odds_analytics.feature_groups::load_lineup_cache` |
| FPL availability | `fpl_availability` (`FplAvailability`) | `scripts/ingest_fpl_availability.py` (manual) | `FplAvailabilityReader` |
| FBref player stats | none | `scripts/ingest_fbref_player_stats.py` (CSV only) | n/a |

## ESPN fixtures

`EspnFixtureFetcher` wraps the free ESPN Site API over httpx (3 attempts, 1s backoff). `LEAGUE_SLUGS` covers the Premier League, FA Cup, League Cup and three European competitions. Two endpoints are combined:

- `fetch_season` walks each EPL team's `teams/{id}/schedule` (completed matches only).
- `fetch_upcoming` reads the date-range scoreboard, emitting two mirrored rows per match.

`fetch_espn_fixtures.py::main` runs both (30-day lookahead), dedupes on `(date, team, opponent)` with the scoreboard row winning (fresher `state`), and upserts through `EspnFixtureWriter.upsert_fixtures`, idempotent on `(date, team, competition)`. It raises for any sport but `soccer_epl`. Each match is two rows, one per team.

Schedule: `0 6 * * *`, EPL only, from `_JOB_CRON_MAP` in `packages/odds-lambda/odds_lambda/scheduling/jobs.py`, bootstrapped by `SchedulerConfig.bootstrap_jobs` in `packages/odds-core/odds_core/config.py`. Local scheduler only; no EventBridge rule exists in `deployment/aws/terraform/eventbridge.tf`. Deployed state cannot be verified from the repo: `odds scheduler list-jobs --backend aws`.

## Lineups, FPL, FBref

All three are manual scripts writing CSVs under `data/`, with `--skip-db` on the lineup and FPL scripts to skip the upsert. Lineup coverage is 2020-21 onward, FPL 2021-22 onward (`SEASONS` in each script). FBref pages must be saved by hand from a browser (Cloudflare) and parsed; there is no model or table.

No structured lineup API was usable for live pre-kickoff data; the agent reads BBC Sport instead ([AGENT_DATA_SOURCES](../docs/AGENT_DATA_SOURCES.md#lineup-data-2026-04-13)). ESPN lineups are non-monotonic near KO-60/KO-45 per that doc, which is why no scheduled lineup job exists.

## Hazards

- `fetch_espn_fixtures.py` module docstring says the cadence is "managed in Terraform"; it is local-cron only (`_JOB_CRON_MAP`, Terraform comment at `deployment/aws/terraform/eventbridge.tf`).
- `FplAvailabilityReader` has no caller outside the module; FPL data is stored but not consumed by any feature code in `packages/`.
- `espn_fixtures.state` is nullable; consumers fall back to `status` strings (`packages/odds-mcp/odds_mcp/tools/epl.py::_is_final`).
- Team names pass through `odds_core.team.normalize_team_name`, which passes unknown names through unchanged ([Event matching](event-matching.md)).

## Related

- [Data sources](data-sources.md) - where ESPN sits among feeds
- [Feature groups](feature-groups.md) and [EPL context features](features-epl-context.md) - consumers
- [MLB probable pitchers](mlb-probable-pitchers.md) - the sibling daily-cron job
