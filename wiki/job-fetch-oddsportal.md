---
type: component
title: fetch-oddsportal job
summary: "The per-league OddsPortal upcoming-odds scrape (scraper Lambda or local): kickoff-proximity cadence, tiering and inline CLV scoring."
sources:
  - docs/ARCHITECTURE.md#oddsportal-scrape-job
  - docs/AGENT_DATA_SOURCES.md#scheduled-collection
  - packages/odds-lambda/odds_lambda/jobs/fetch_oddsportal.py::main
  - packages/odds-lambda/odds_lambda/jobs/fetch_oddsportal.py::LEAGUE_SPECS
  - packages/odds-lambda/odds_lambda/jobs/fetch_oddsportal.py::CADENCE
verified_at: 7887039
---

# fetch-oddsportal job

`packages/odds-lambda/odds_lambda/jobs/fetch_oddsportal.py::main(ctx)` scrapes pre-match odds with OddsHarvester ([OddsPortal scraper](oddsportal-scraper.md)), converts them to the canonical `raw_data` shape and stores snapshots through `OddsWriter`. It runs on the scraper Lambda (per-sport rules, gated by `enable_oddsportal_scraper`) and in the default local `bootstrap_jobs` ([deployment matrix](job-deployment-matrix.md)).

## Leagues

`LEAGUE_SPECS` holds `LeagueSpec` rows: EPL (`football`, `england-premier-league`, markets `1x2` and `over_under_2_5`, 3 outcomes, overnight 22-06) and MLB (`baseball`, `mlb`, market `home_away`, 2 outcomes, overnight 05-14). With a `ctx.sport` only that spec runs; an unknown sport logs an error and returns. `LEAGUE_SPEC_BY_NAME` is shared with scrape CLI and MCP tools, and the [results job](job-fetch-oddsportal-results.md) reuses the specs.

## Flow

1. Pre-schedule using `decide_forward_resilient` with `CADENCE`: 30 min under 3 h, 1 h at 3-12 h, 2 h beyond or with no games, 1 h if the database is down. This job is always-run: only `next_execution` is used, not `should_execute` ([decision engine](scheduling-decision-engine.md), [self-scheduling](self-scheduling.md)).
2. Inside `job_alert_context(compound_name)`, `ingest_league` per spec: scrape under the Playwright semaphore, `convert_upcoming_matches` per market, `find_or_create_event` and `store_odds_snapshot` per match (tier assigned by the writer, see [fetch tier](fetch-tier.md)), tagging `api_request_id` as `oddsportal_live_<time>`. A league failure is logged and the others continue.
3. If nothing was scraped, `send_job_warning("scrape_empty:<job>")`.
4. Outside the alert context: if `model_supports_sport(ctx.sport)`, `score_events(sport=...)` runs inline ([inline scoring](job-score-predictions.md)). Any exception there is logged as `fetch_oddsportal_scoring_failed` and swallowed so it cannot fire a scrape alert.
5. Post-schedule only when at least one match was scraped.

## Hazards

- The scraper image installs only `odds-core` and `odds-lambda`; `model_loader` imports `odds_analytics` at module top and `xgboost` is not installed there (`deployment/aws/Dockerfile.scraper`). On the scraper Lambda the scoring import fails and is swallowed, and `deployment/aws/terraform/scraper.tf` sets no `MODEL_*` variables. Inline scoring therefore runs only on a local scheduler. Check the deployed function with `aws lambda get-function-configuration --function-name odds-scheduler-scraper`.
- Heartbeats are keyed `heartbeat:fetch-oddsportal-<sport>`; both sports have expectations (2 h) in `AlertConfig.heartbeat_expectations` ([health monitor](health-monitor.md)).
- When the scrape returns zero matches, the job does not reschedule after the pre-schedule, so the next run keeps the earlier computed time.
- The same job can run on AWS and locally at once; check `odds scheduler list-jobs --backend aws`.

## Related

- [Job registry](job-registry.md), [data sources](data-sources.md).
