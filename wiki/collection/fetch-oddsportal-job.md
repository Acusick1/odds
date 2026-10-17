---
type: reference
title: The fetch-oddsportal job
description: LEAGUE_SPECS, flow, cadence, alerts, heartbeats and inline scoring of odds_lambda.jobs.fetch_oddsportal.
sources:
  - odds_lambda.jobs.fetch_oddsportal.main
  - odds_lambda.jobs.fetch_oddsportal.LEAGUE_SPECS
  - odds_lambda.jobs.fetch_oddsportal.LeagueSpec
  - odds_lambda.jobs.fetch_oddsportal.CADENCE
  - odds_lambda.jobs.fetch_oddsportal.ingest_league
  - odds_lambda.scheduling.decision.OVERNIGHT_WINDOWS
  - odds_core.config.AlertConfig.heartbeat_expectations
  - deployment/aws/terraform/scraper.tf
  - deployment/aws/Dockerfile.scraper
status: stable
---

# The fetch-oddsportal job

`odds_lambda.jobs.fetch_oddsportal.main(ctx)` scrapes pre-match odds ([oddsportal-scraper](oddsportal-scraper.md)), converts them to the canonical shape and stores snapshots through `odds_lambda.storage.writers.OddsWriter`. It runs on the scraper Lambda (per-sport rules, gated by `enable_oddsportal_scraper`) and in the default local `bootstrap_jobs` ([job-deployment-matrix](../scheduling/job-deployment-matrix.md#where-each-job-runs)). If the scraper is enabled on AWS and bootstrapped locally, both run and scrape twice; no code checks for a job enabled in both places. Deployed state: `odds scheduler list-jobs --backend aws`.

## Leagues

`odds_lambda.jobs.fetch_oddsportal.LEAGUE_SPECS` holds `odds_lambda.jobs.fetch_oddsportal.LeagueSpec` rows: EPL (`football`, `england-premier-league`, markets `1x2` and `over_under_2_5`, 3 outcomes, overnight 22-06) and MLB (`baseball`, `mlb`, market `home_away`, 2 outcomes, overnight 05-14). With a `ctx.sport` only that spec runs; an unknown sport logs an error and returns. The `odds_lambda.jobs.fetch_oddsportal.LeagueSpec` overnight fields duplicate `odds_lambda.scheduling.decision.OVERNIGHT_WINDOWS` in the decision engine ([decision-engine](../scheduling/decision-engine.md#overnight-windows)).

## Flow

1. Pre-schedule using `odds_lambda.scheduling.decision.decide_forward_resilient` with `odds_lambda.jobs.fetch_oddsportal.CADENCE` ([decision-engine](../scheduling/decision-engine.md#cadence)). This job is always-run: only `next_execution` is used, not `should_execute`.
2. Inside `job_alert_context(compound_name)`, `odds_lambda.jobs.fetch_oddsportal.ingest_league` per spec: scrape under the Playwright semaphore, `odds_lambda.oddsportal_adapter.convert_upcoming_matches` per market, `odds_lambda.storage.writers.OddsWriter.find_or_create_event` and `odds_lambda.storage.writers.OddsWriter.store_odds_snapshot` per match (tier assigned by the writer, see [fetch-tier](../storage/fetch-tier.md)), tagging `api_request_id` as `oddsportal_live_<time>`. A league failure is logged and the others continue.
3. If nothing was scraped, `send_job_warning("scrape_empty:<job>")`.
4. Outside the alert context: if `model_supports_sport(ctx.sport)`, `score_events(sport=...)` runs inline ([prediction-scoring](../modelling/prediction-scoring.md)). Any exception there is logged as `fetch_oddsportal_scoring_failed` and swallowed so it cannot fire a scrape alert.
5. Post-schedule only when at least one match was scraped. A zero-match run skips it, so the next run keeps the earlier computed time.

## Scraper Lambda limits

- The scraper image installs only `odds-core` and `odds-lambda`; `odds_lambda.model_loader` imports `odds_analytics` at module top and `xgboost` is not installed there (`deployment/aws/Dockerfile.scraper`). On the scraper Lambda the scoring import fails and is logged as `fetch_oddsportal_scoring_failed` (#454).
- `deployment/aws/terraform/scraper.tf` sets no `MODEL_*` or alert variables, so scraper-side alerts are off ([aws-infrastructure](../operations/aws-infrastructure.md)). Check the deployed function with `aws lambda get-function-configuration --function-name odds-scheduler-scraper`.

## Heartbeats

Heartbeats are keyed `heartbeat:fetch-oddsportal-<sport>`; both sports have 2 h expectations in `odds_core.config.AlertConfig.heartbeat_expectations` ([alerting-and-heartbeats](../operations/alerting-and-heartbeats.md#heartbeats)).

## Related

- [OddsPortal scraper](oddsportal-scraper.md)
- [Results and status jobs](results-jobs.md)
- [Self-scheduling](../scheduling/self-scheduling.md)
