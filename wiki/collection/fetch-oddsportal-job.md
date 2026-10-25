---
type: reference
title: The fetch-oddsportal job
description: LEAGUE_SPECS, flow, cadence, alerts, heartbeats of odds_lambda.jobs.fetch_oddsportal.
sources:
  - odds_lambda.jobs.fetch_oddsportal.main
  - odds_lambda.jobs.fetch_oddsportal.LEAGUE_SPECS
  - odds_lambda.jobs.fetch_oddsportal.LeagueSpec
  - odds_lambda.jobs.fetch_oddsportal.CADENCE
  - odds_lambda.jobs.fetch_oddsportal.SOURCE
  - odds_lambda.jobs.fetch_oddsportal.ingest_league
  - odds_lambda.jobs.fetch_oddsportal.scrape_league
  - odds_lambda.scheduling.decision.OVERNIGHT_WINDOWS
  - odds_core.config.AlertConfig.heartbeat_expectations
mentions:
  - odds_core.alerts.job_alert_context
status: stable
---

# The fetch-oddsportal job

`odds_lambda.jobs.fetch_oddsportal.main(ctx)` reads pre-match odds over HTTP ([oddsportal-scraper](oddsportal-scraper.md)), converts them to the canonical shape and stores snapshots and their odds rows through `odds_lambda.storage.writers.OddsWriter.store_odds_snapshot`. It is in the default local `bootstrap_jobs` and runs on the production host ([job-deployment-matrix](../scheduling/job-deployment-matrix.md#where-each-job-runs)). It has no Terraform rule or Lambda; a rule left in the AWS account from before their removal would scrape alongside the production host, and no code checks for a job enabled in both places. Deployed state: `odds scheduler list-jobs --backend aws`. On the production host, one hung run ends the local chains (#485).

## Leagues

`odds_lambda.jobs.fetch_oddsportal.LEAGUE_SPECS` holds `odds_lambda.jobs.fetch_oddsportal.LeagueSpec` rows: EPL (`england-premier-league`, listing `/football/england/premier-league/`, markets `1x2` and `over_under_2_5`, 3 outcomes, overnight 22-06) and MLB (`mlb`, listing `/baseball/usa/mlb/`, market `home_away`, 2 outcomes, overnight 05-14). `ctx.sport` is required and only its spec runs; a missing sport raises `ValueError` and an unknown sport logs an error and returns. The `odds_lambda.jobs.fetch_oddsportal.LeagueSpec` overnight fields duplicate `odds_lambda.scheduling.decision.OVERNIGHT_WINDOWS` in the decision engine ([decision-engine](../scheduling/decision-engine.md#overnight-windows)).

## Flow

1. Pre-schedule using `odds_lambda.collection_calendar.decide_collection_run` with `odds_lambda.jobs.fetch_oddsportal.CADENCE`, timed from the events' `commence_time` ([decision-engine](../scheduling/decision-engine.md#cadence)). With no fixture in the stored events or the sport's published schedule within 28 days the run is dormant, the same decision for every collector: it books a 6 h check-in (or the moment a known fixture enters the 28 days), writes its heartbeat and returns without scraping ([dormancy](../scheduling/decision-engine.md#dormancy)). A scheduler restart keeps the pre-scheduled run ([bootstrap](../scheduling/local-scheduler.md#bootstrap-jobs-and-cron-table)).
2. Inside `job_alert_context(compound_name)`, `odds_lambda.jobs.fetch_oddsportal.ingest_league` per spec: `odds_lambda.jobs.fetch_oddsportal.scrape_league` (listing, bookmaker context, then one feed per upcoming event and market, converted by `odds_lambda.oddsportal_adapter.convert_market` with the feed's decimal prices kept as `price_decimal` and the OddsPortal event id as `raw_data.oddsportal_event_id`; a failed event or market is recorded in the stats errors and skipped, but a league where every market fetch failed raises `odds_lambda.oddsportal_client.OddsPortalError`, as does a listing with no events and no empty-state marker), the event of each match through `odds_lambda.live_events.LiveEventFinder.find_or_create` (which never merges; for MLB, one MLB schedule request per run resolves every listed game to its `gamePk` before the database session opens, and a game left unresolved gets no event or snapshot and is counted in `games_unresolved`, [live MLB events](../storage/event-matching.md#live-mlb-events)) and `odds_lambda.storage.writers.OddsWriter.store_odds_snapshot` per match (tier assigned by the writer, see [fetch-tier](../storage/fetch-tier.md)), tagging `api_request_id` as `oddsportal_live_<time>`. A league failure, including those errors, is logged and the others continue.
3. A run that dropped MLB games sends the `mlb_games_dropped:<job>` warning (`odds_lambda.live_events.alert_dropped_games`, [alerts](../operations/alerts.md#alert-history)), which never fails the run. If any league failed, the job raises after the others finish. Otherwise `odds_lambda.collection_calendar.require_rows_for_fixtures` raises `odds_lambda.collection_calendar.EmptyCollectionError` for a sport with zero stored snapshots and a SCHEDULED fixture within 48 hours (`odds_lambda.jobs.fetch_oddsportal.SOURCE`; MLB fixtures come from the MLB Stats API schedule, [stale-data check](../operations/health-monitor.md#stale-data-check)), with the run's scrape errors in the message. Both fail inside [`odds_core.alerts.job_alert_context`](../operations/alerts.md#delivery): a critical alert (once per outage) and no heartbeat ([empty collection](../operations/alerting-and-heartbeats.md#empty-collection)). A sport with no fixtures in that window passes, and so does one whose listing parsed with events but every listed event had already started (`odds_lambda.jobs.fetch_oddsportal.IngestionStats.events_listed` above zero, `odds_lambda.jobs.fetch_oddsportal.IngestionStats.upcoming_listed` zero): no scrapeable event existed, so the job logs `oddsportal_empty_run_check_skipped` (a warning carrying the latest listed start time) and leaves that sport out of the check (it passes no `accounted_by_sport`, which only the [Betfair job](betfair-exchange.md) uses). A listing with no events and no empty-state marker, a sport with an upcoming event that stored nothing, and an empty-state listing while fixtures are scheduled still raise. On the local scheduler a run past `odds_lambda.scheduling.runner.JOB_TIMEOUT` is cancelled and alerted, and a failed run that booked nothing is re-booked ([Local scheduler](../scheduling/local-scheduler.md#scheduling-primitives)).
4. Post-schedule only when at least one match was scraped. A zero-match run skips it, so the next run keeps the earlier computed time.

## Heartbeats

Heartbeats are keyed `heartbeat:fetch-oddsportal-<sport>`; both sports have entries (11 h EPL, 12 h MLB, covering the overnight window); the results jobs `fetch-oddsportal-results-epl` and `fetch-mlb-results` (26 h each), `fetch-mlb-probables` (30 h), `fetch-espn-fixtures` (26 h) and `fetch-betfair-exchange` (10 h EPL, 11 h MLB), `fetch-polymarket-odds` (11 h EPL), `fetch-odds-epl` (49 h) and `fetch-scores-epl` (13 h) share the map in `odds_core.config.AlertConfig.heartbeat_expectations` ([alerting-and-heartbeats](../operations/alerting-and-heartbeats.md#heartbeats)).

## Related

- [OddsPortal scraper](oddsportal-scraper.md)
- [Results and status jobs](results-jobs.md)
- [Self-scheduling](../scheduling/self-scheduling.md)
