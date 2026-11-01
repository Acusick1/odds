---
type: reference
title: EPL context data (ESPN, FPL, FBref)
description: "ESPN fixtures (the fetch-espn-fixtures job) and lineups, FPL availability and FBref stats, covering fetchers, scripts, tables and readers, and which of them any feature or tool actually consumes."
sources:
  - odds_lambda.espn_fixture_fetcher.EspnFixtureFetcher
  - odds_lambda.jobs.fetch_espn_fixtures.main
  - odds_lambda.jobs.fetch_espn_fixtures._mark_postponed_events
  - odds_lambda.storage.espn_fixture_writer.EspnFixtureWriter
  - odds_lambda.storage.espn_fixture_reader.EspnFixtureReader
  - odds_lambda.storage.espn_lineup_reader.EspnLineupReader
  - odds_lambda.storage.fpl_availability_reader.FplAvailabilityReader
  - odds_core.epl_data_models.EspnFixture
  - odds_lambda.scheduling.jobs._JOB_CRON_MAP
  - odds_mcp.tools.epl._is_final
  - scripts/ingest_espn_lineups.py
  - scripts/ingest_fbref_player_stats.py
  - deployment/aws/terraform/eventbridge.tf
mentions:
  - odds_lambda.scheduling.jobs._JOB_SPORTS
  - odds_core.alerts.job_alert_context
status: stable
---

# EPL context data (ESPN, FPL, FBref)

Supplementary EPL data that feeds the retail-EV schedule features ([EPL context features](../modelling/epl-context-features.md)) and the MCP team-context tool ([mcp-sport-context-tools](../agent/mcp-sport-context-tools.md)). Models are in `odds_core.epl_data_models`; fetchers and storage in `packages/odds-lambda/odds_lambda/`.

| Source | Table (model) | Written by | Read by |
|---|---|---|---|
| ESPN fixtures | `espn_fixtures` (`odds_core.epl_data_models.EspnFixture`) | the `fetch-espn-fixtures` job; `scripts/ingest_espn_fixtures.py` | `odds_lambda.storage.espn_fixture_reader.EspnFixtureReader`, `odds_analytics.feature_groups.load_fixtures_df` (all fixtures, or those up to an optional date) |
| ESPN lineups | `espn_lineups` (`odds_core.epl_data_models.EspnLineup`) | `scripts/ingest_espn_lineups.py` (manual) | `odds_lambda.storage.espn_lineup_reader.EspnLineupReader` |
| FPL availability | `fpl_availability` (`odds_core.epl_data_models.FplAvailability`) | `scripts/ingest_fpl_availability.py` (manual) | `odds_lambda.storage.fpl_availability_reader.FplAvailabilityReader` |
| FBref player stats | none | `scripts/ingest_fbref_player_stats.py` (CSV only) | n/a |

## ESPN fixtures

`odds_lambda.espn_fixture_fetcher.EspnFixtureFetcher` wraps the free ESPN Site API over httpx (3 attempts, 1 s backoff). `odds_lambda.espn_fixture_fetcher.LEAGUE_SLUGS` covers the Premier League, FA Cup, League Cup and three European competitions (`odds_analytics.as_of_team_features.load_team_history` loads these all-competition fixtures for the schedule features). Two endpoints are combined:

- `odds_lambda.espn_fixture_fetcher.EspnFixtureFetcher.fetch_season` walks each EPL team's `teams/{id}/schedule` (completed matches only).
- `odds_lambda.espn_fixture_fetcher.EspnFixtureFetcher.fetch_upcoming` reads the scoreboard, emitting two mirrored rows per match. ESPN rejects `dates=` ranges with HTTP 400, so it requests one calendar month per competition (`dates=YYYYMM&limit=300`; a single day or a year also work) and keeps events from the start of today to the end of the window.

`odds_lambda.jobs.fetch_espn_fixtures.main` runs both (30-day lookahead), dedupes on `(date, team, opponent)` with the scoreboard row winning (fresher `state`), and upserts through `odds_lambda.storage.espn_fixture_writer.EspnFixtureWriter.upsert_fixtures`, idempotent on `(date, team, competition)`. For `basketball_nba` it instead syncs the NBA schedule into events (warning on an empty listing only while NBA events are scheduled) ([NBA schedule](polymarket-odds-snapshots.md#nba-schedule-from-espn)); it raises for any other sport. Inside [`odds_core.alerts.job_alert_context`](../operations/alerts.md#delivery) it fails the run, raising a critical alert (once per outage) and skipping the heartbeat, after upserting the rows already fetched (past rows when the upcoming fetch fails), in these cases: `odds_lambda.espn_fixture_fetcher.EspnUpcomingFetchError` when the Premier League upcoming request failed (a cup or European failure goes through `odds_core.alerts.send_job_warning`, at most every 30 minutes, and the run continues), `odds_lambda.espn_fixture_fetcher.EspnSeasonFetchError` when the Premier League schedule request failed for every team (the upcoming fetch still runs and its rows are upserted first), and `odds_lambda.collection_calendar.EmptyCollectionError` when ESPN returned no upcoming Premier League fixture while `odds_lambda.collection_calendar.event_calendar` reports SCHEDULED EPL `events` rows in the window (excluding no fixture). Each match is two rows, one per team. After the upsert, `odds_lambda.jobs.fetch_espn_fixtures._mark_postponed_events` sets the event of every Premier League match whose ESPN status is `odds_core.epl_data_models.ESPN_STATUS_POSTPONED` POSTPONED, skipping a match that ESPN also lists with another status within `odds_lambda.storage.event_merge.EVENT_MATCH_WINDOW` of the postponed row, matching the home-side row's teams and date with `odds_lambda.storage.event_merge.find_match_id` (the integer id of the event within ±2 h; of several, the one a merge would keep) and touching only SCHEDULED or LIVE events ([event-status-lifecycle](event-status-lifecycle.md#who-sets-what)). A cron job keeps its cron when a run fails ([Local scheduler](../scheduling/local-scheduler.md#scheduling-primitives)).

**Schedule**: it requires `ctx.sport` to be `soccer_epl` or `basketball_nba` (a missing sport raises; the EPL description here covers the former); its fixed cron `0 6 * * *` (UTC) in `odds_lambda.scheduling.jobs._JOB_CRON_MAP`, the same as `fetch-mlb-probables`, EPL only through its `odds_lambda.scheduling.jobs._JOB_SPORTS` allowlist, in the default production `odds_core.config.SchedulerConfig.bootstrap_jobs` with the odds collectors, the results jobs, the NBA injury-report job and the candidate runner, so it runs on the production host ([job-deployment-matrix](../scheduling/job-deployment-matrix.md#where-each-job-runs)). `odds_lambda.jobs.fetch_espn_fixtures` states in its module docstring that the cadence is "managed in Terraform"; the cadence is the local cron above, and `deployment/aws/terraform/eventbridge.tf` declares no rule for it (its per-sport rules cover only the jobs listed there). The MCP `odds_mcp.tools.epl.get_team_context` tool reads `espn_fixtures`, so it goes stale whenever this cron does not succeed.

## Lineups, FPL, FBref

All three are manual scripts writing CSVs under `data/`, with `--skip-db` on the lineup and FPL scripts to skip the upsert. Lineup coverage is 2020-21 onward, FPL 2021-22 onward (`scripts/ingest_espn_lineups.py`, `scripts/ingest_fpl_availability.py`). FBref pages must be saved by hand from a browser (Cloudflare) and parsed; there is no model or table.

No feature code reads `espn_lineups` or FPL availability.

No scheduled lineup job exists: lineups are sourced from live web research, and the manual lineup script feeds training data only ([ADR-0006](../../docs/adr/0006-lineups-from-web-research.md)). ESPN is still used for fixtures.

## Consumption and nullability

- FPL is stored but unused: `odds_lambda.storage.fpl_availability_reader.FplAvailabilityReader` has no caller outside the module and no feature code in `packages/` consumes FPL data (related: #412, a proposed MCP tool over it).
- `espn_fixtures.state` is nullable; consumers fall back to `odds_core.epl_data_models.EspnFixture.status` strings (`odds_mcp.tools.epl._is_final`).
- Team names pass through `odds_core.team.normalize_team_name`, which passes unknown names through unchanged ([event-matching](../storage/event-matching.md)).

## Related

- [Data sources](data-sources.md), for where ESPN sits among feeds
- [EPL context feature groups](../modelling/epl-context-features.md) and [feature pipeline](../modelling/feature-pipeline.md), the consumers
- [MLB probable pitchers](mlb-probable-pitchers.md), the sibling daily-cron job
