---
type: reference
title: The fetch-oddsportal job
description: LEAGUE_SPECS, flow, cadence, runs before candidate decision times, alerts, heartbeats of odds_lambda.jobs.fetch_oddsportal.
sources:
  - odds_lambda.jobs.fetch_oddsportal.main
  - odds_lambda.jobs.fetch_oddsportal.LEAGUE_SPECS
  - odds_lambda.jobs.fetch_oddsportal.LeagueSpec
  - odds_lambda.jobs.fetch_oddsportal.CADENCE
  - odds_lambda.jobs.fetch_oddsportal.SCRAPE_LEAD
  - odds_lambda.jobs.fetch_oddsportal.candidate_scrape_leads
  - odds_lambda.candidate_rules.rules.decision_leads
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

`odds_lambda.jobs.fetch_oddsportal.main(ctx)` reads pre-match odds over HTTP ([oddsportal-scraper](oddsportal-scraper.md)), converts them to the canonical shape and stores one snapshot per market with its price rows through `odds_lambda.storage.writers.OddsWriter.store_envelope`, and a snapshot already stored under its key is skipped ([Writing price rows](../storage/price-tables.md#writing-price-rows)). It is in the default local `bootstrap_jobs` and runs on the production host ([job-deployment-matrix](../scheduling/job-deployment-matrix.md#where-each-job-runs)). It has no Terraform rule or Lambda; a rule left in the AWS account from before their removal would scrape alongside the production host, and no code checks for a job enabled in both places. Deployed state: `odds scheduler list-jobs --backend aws`. On the production host, one hung run ends the local chains (#485).

## Leagues

`odds_lambda.jobs.fetch_oddsportal.LEAGUE_SPECS` holds `odds_lambda.jobs.fetch_oddsportal.LeagueSpec` rows: EPL (`england-premier-league`, listing `/football/england/premier-league/`, markets `1x2` and `over_under_2_5`, 3 outcomes, overnight 22-06), MLB (`mlb`, listing `/baseball/usa/mlb/`, market `home_away`, 2 outcomes, overnight 05-14) and NBA (`nba`, listing `/basketball/usa/nba/`, markets `home_away`, `totals_ladder` and `spreads_ladder`, 2 outcomes, overnight 07-15; the ladders store each game's main line and its neighbours as [ladder markets](oddsportal-scraper.md#ladder-markets)). OddsPortal lists NBA games by full team name, which are the canonical names. `ctx.sport` is required and only its spec runs; a missing sport raises `ValueError` and an unknown sport logs an error and returns. The job reads its overnight window from `odds_lambda.scheduling.decision.OVERNIGHT_WINDOWS` by the spec's sport ([decision-engine](../scheduling/decision-engine.md#overnight-windows)). NBA fixtures come from the ESPN NBA schedule (#725), so the NBA job stays dormant until that is scheduled.

## Flow

1. Pre-schedule using `odds_lambda.collection_calendar.decide_collection_run` with `odds_lambda.jobs.fetch_oddsportal.CADENCE`, timed from the events' `commence_time` ([decision-engine](../scheduling/decision-engine.md#cadence)), and the [candidate leads](#runs-before-candidate-decisions). With no fixture in the stored events or the sport's published schedule within 28 days the run is dormant, the same decision for every collector: it books a 6 h check-in (or the moment a known fixture enters the 28 days), writes its heartbeat and returns without scraping ([dormancy](../scheduling/decision-engine.md#dormancy)). A scheduler restart keeps the pre-scheduled run ([bootstrap](../scheduling/local-scheduler.md#bootstrap-jobs-and-cron-table)).
2. Inside `job_alert_context(compound_name)`, `odds_lambda.jobs.fetch_oddsportal.ingest_league` per spec: `odds_lambda.jobs.fetch_oddsportal.scrape_league` (listing, bookmaker context, then one feed per upcoming event and market, converted by `odds_lambda.oddsportal_adapter.convert_market` with the feed's decimal prices kept as `price_decimal` and the OddsPortal event id as the envelope's `oddsportal_event_id`; a failed event or market is recorded in the stats errors and skipped, but a league where every market fetch failed raises `odds_lambda.oddsportal_client.OddsPortalError`, as does a listing with no events and no empty-state marker), one payload archiving the run's envelopes, the event of each match through `odds_lambda.live_events.LiveEventFinder.resolve` (shared with the Odds API writers) and the [event resolver](../storage/event-resolver.md), recording the OddsPortal event id (which never merges; for MLB, one MLB schedule request per run resolves every listed game to its `gamePk` before the database session opens, and a game left unresolved gets no event or snapshot and is counted in `games_unresolved`, [live MLB events](../storage/event-matching.md#live-mlb-events)) and `odds_lambda.storage.writers.OddsWriter.store_envelope` per match under the event's integer id, which stores each market's snapshot once per key (source `oddsportal`, time basis `observed`; tiers are derived, see [fetch-tier](../storage/fetch-tier.md)). Matches are resolved and stored in fixture-lock order, each in its own savepoint, so a failed match is recorded in the errors and the others are kept. A league failure, including those errors, is logged and the others continue.
3. A run that dropped MLB games sends the `mlb_games_dropped:<job>` warning (`odds_lambda.live_events.alert_dropped_games`, [alerts](../operations/alerts.md#alert-history)), which never fails the run. If any league failed, the job raises after the others finish. Otherwise `odds_lambda.collection_calendar.require_rows_for_fixtures` raises `odds_lambda.collection_calendar.EmptyCollectionError` for a sport with zero stored snapshots and a SCHEDULED fixture within 48 hours (`odds_lambda.jobs.fetch_oddsportal.SOURCE`, whose snapshots are source `oddsportal`, time basis `observed`; MLB fixtures come from the MLB Stats API schedule, [stale-data check](../operations/health-monitor.md#stale-data-check)), with the run's scrape errors in the message. Both fail inside [`odds_core.alerts.job_alert_context`](../operations/alerts.md#delivery): a critical alert (once per outage) and no heartbeat ([empty collection](../operations/alerting-and-heartbeats.md#empty-collection)). A sport with no fixtures in that window passes, and so does one whose listing parsed with events but every listed event had already started (`odds_lambda.jobs.fetch_oddsportal.IngestionStats.events_listed` above zero, `odds_lambda.jobs.fetch_oddsportal.IngestionStats.upcoming_listed` zero): no scrapeable event existed, so the job logs `oddsportal_empty_run_check_skipped` (a warning carrying the latest listed start time) and leaves that sport out of the check (it passes no `accounted_by_sport`, which only the [Betfair job](betfair-exchange.md) uses). A listing with no events and no empty-state marker, a sport with an upcoming event that stored nothing, and an empty-state listing while fixtures are scheduled still raise. On the local scheduler a run past `odds_lambda.scheduling.runner.JOB_TIMEOUT` is cancelled and alerted, and a failed run that booked nothing is re-booked ([Local scheduler](../scheduling/local-scheduler.md#scheduling-primitives)).
4. Post-schedule only when at least one match was scraped. A zero-match run skips it, so the next run keeps the earlier computed time.

## Runs before candidate decisions

A [mechanical candidate](../evaluation/candidate-rules.md#market-state) decides from the latest snapshot at or before its decision time that carries UK retail prices and a Betfair Exchange row together, and only from one within its maximum input age; a live OddsPortal snapshot carries both. `odds_lambda.jobs.fetch_oddsportal.candidate_scrape_leads` reads the candidate registry at the start of each run and, through `odds_lambda.candidate_rules.rules.decision_leads`, takes every decision time in the `minutes_before_kickoff` list of every active mechanical candidate of the job's sport whose rule and params resolve, whatever its `start_date` ([decision times](../evaluation/candidate-rules.md#decision-times)). Each becomes a [lead](../scheduling/decision-engine.md#lead-runs) `odds_lambda.jobs.fetch_oddsportal.SCRAPE_LEAD` (5 min) before the decision time that still runs late until the decision time: a run that much before every stored kickoff, inside the overnight window too. The job passes its start time, so a cadence run that started before a lead time and finished after it books the lead run straight away. A league scrape took about a minute in October 2026 (two at most) for EPL and MLB, so the snapshots are stored before the decision and are about 5 minutes old at it. An NBA scrape of 22 upcoming games, with five feed requests per game, took 6 minutes: longer than the lead, which an NBA candidate would have to raise. A registry that cannot be read logs `candidate_registry_invalid` and adds no runs; the [run-candidates job](../evaluation/run-candidates-job.md#one-run) alerts on it. Matches that kick off together share a run, because one run scrapes the whole league.

## Heartbeats

Heartbeats are keyed `heartbeat:fetch-oddsportal-<sport>`; every sport has an entry (11 h EPL, 12 h MLB, 11 h NBA, covering the overnight window); the results jobs `fetch-oddsportal-results-epl` and `fetch-mlb-results` (26 h each), `fetch-mlb-probables` (30 h), `fetch-espn-fixtures` and `fetch-espn-fixtures-nba` (26 h) and `fetch-betfair-exchange` (10 h EPL, 11 h MLB, 11 h NBA), `fetch-polymarket-odds` (11 h EPL, 11 h NBA), `fetch-odds-epl` (49 h) and `fetch-scores-epl` (13 h) share the map in `odds_core.config.AlertConfig.heartbeat_expectations` ([alerting-and-heartbeats](../operations/alerting-and-heartbeats.md#heartbeats)).

## Related

- [OddsPortal scraper](oddsportal-scraper.md)
- [Results and status jobs](results-jobs.md)
- [Self-scheduling](../scheduling/self-scheduling.md)
