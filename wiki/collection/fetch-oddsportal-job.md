---
type: reference
title: The fetch-oddsportal job
description: LEAGUE_SPECS, flow, cadence, alerts, heartbeats and inline scoring of odds_lambda.jobs.fetch_oddsportal.
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
  - deployment/aws/terraform/scraper.tf
  - deployment/aws/Dockerfile.scraper
status: stable
---

# The fetch-oddsportal job

`odds_lambda.jobs.fetch_oddsportal.main(ctx)` reads pre-match odds over HTTP ([oddsportal-scraper](oddsportal-scraper.md)), converts them to the canonical shape and stores snapshots through `odds_lambda.storage.writers.OddsWriter.store_odds_snapshot`. It is in the default local `bootstrap_jobs` and runs on the mini-PC ([job-deployment-matrix](../scheduling/job-deployment-matrix.md#where-each-job-runs)). Its scraper Lambda rules (gated by `enable_oddsportal_scraper`) are dormant; a re-enabled rule would scrape alongside the mini-PC, and no code checks for a job enabled in both places. Deployed state: `odds scheduler list-jobs --backend aws`. On the mini-PC, one hung run ends the local chains (#485).

## Leagues

`odds_lambda.jobs.fetch_oddsportal.LEAGUE_SPECS` holds `odds_lambda.jobs.fetch_oddsportal.LeagueSpec` rows: EPL (`football`, `england-premier-league`, listing `/football/england/premier-league/`, markets `1x2` and `over_under_2_5`, 3 outcomes, overnight 22-06) and MLB (`baseball`, `mlb`, listing `/baseball/usa/mlb/`, market `home_away`, 2 outcomes, overnight 05-14). With a `ctx.sport` only that spec runs; an unknown sport logs an error and returns. The `odds_lambda.jobs.fetch_oddsportal.LeagueSpec` overnight fields duplicate `odds_lambda.scheduling.decision.OVERNIGHT_WINDOWS` in the decision engine ([decision-engine](../scheduling/decision-engine.md#overnight-windows)).

## Flow

1. Pre-schedule using `odds_lambda.scheduling.decision.decide_forward_resilient` with `odds_lambda.jobs.fetch_oddsportal.CADENCE` ([decision-engine](../scheduling/decision-engine.md#cadence)). This job is always-run: only `next_execution` is used, not `should_execute`. A scheduler restart keeps the pre-scheduled run ([bootstrap](../scheduling/local-scheduler.md#bootstrap-jobs-and-cron-table)).
2. Inside `job_alert_context(compound_name)`, `odds_lambda.jobs.fetch_oddsportal.ingest_league` per spec: `odds_lambda.jobs.fetch_oddsportal.scrape_league` (listing, bookmaker context, then one feed per upcoming event and market, converted by `odds_lambda.oddsportal_adapter.convert_market` with the feed's decimal prices kept as `price_decimal`; a failed event or market is recorded in the stats errors and skipped, but a league where every market fetch failed raises `odds_lambda.oddsportal_client.OddsPortalError`, as does a listing with no events and no empty-state marker), `odds_lambda.storage.writers.OddsWriter.find_or_create_event` (which returns the canonical event, the Odds API one when it exists, and never merges) and `odds_lambda.storage.writers.OddsWriter.store_odds_snapshot` per match (tier assigned by the writer, see [fetch-tier](../storage/fetch-tier.md)), tagging `api_request_id` as `oddsportal_live_<time>`. A league failure, including those errors, is logged and the others continue.
3. If any league failed, the job raises after the others finish. Otherwise `odds_lambda.collection_calendar.require_rows_for_fixtures` raises `odds_lambda.collection_calendar.EmptyCollectionError` for a sport with zero stored snapshots and a SCHEDULED fixture within 48 hours (`odds_lambda.jobs.fetch_oddsportal.SOURCE`; MLB fixtures come from the MLB Stats API schedule, [stale-data check](../operations/health-monitor.md#stale-data-check)), with the run's scrape errors in the message. Both fail inside `odds_core.alerts.job_alert_context`: a critical alert and no heartbeat ([empty collection](../operations/alerting-and-heartbeats.md#empty-collection)). A sport with no fixtures in that window passes.
4. Outside the alert context: if `model_supports_sport(ctx.sport)`, `score_events(sport=...)` runs inline ([prediction-scoring](../modelling/prediction-scoring.md)). Any exception there is logged as `fetch_oddsportal_scoring_failed` and swallowed so it cannot fire a scrape alert.
5. Post-schedule only when at least one match was scraped. A zero-match run skips it, so the next run keeps the earlier computed time.

## Scraper Lambda limits

These apply only if the dormant scraper Lambda runs again.

- The scraper image installs only `odds-core` and `odds-lambda`; `odds_lambda.model_loader` imports `odds_analytics` at module top and `xgboost` is not installed there (`deployment/aws/Dockerfile.scraper`). On the scraper Lambda the scoring import fails and is logged as `fetch_oddsportal_scoring_failed` (#454).
- `deployment/aws/terraform/scraper.tf` sets no `MODEL_*` or alert variables, so scraper-side alerts are off ([aws-infrastructure](../operations/aws-infrastructure.md)). Check the deployed function with `aws lambda get-function-configuration --function-name odds-scheduler-scraper`.

## Heartbeats

Heartbeats are keyed `heartbeat:fetch-oddsportal-<sport>`; both sports have entries (11 h EPL, 12 h MLB, covering the overnight window); `fetch-mlb-probables` (30 h) and `fetch-betfair-exchange` (10 h EPL, 11 h MLB) share the map in `odds_core.config.AlertConfig.heartbeat_expectations` ([alerting-and-heartbeats](../operations/alerting-and-heartbeats.md#heartbeats)).

## Related

- [OddsPortal scraper](oddsportal-scraper.md)
- [Results and status jobs](results-jobs.md)
- [Self-scheduling](../scheduling/self-scheduling.md)
