---
type: component
title: OddsPortal scraper and fetch-oddsportal job
description: "OddsHarvester/Playwright scraping, LEAGUE_SPECS, the raw_data adapter and retries, the fetch-oddsportal job's cadence and alerts, and why its inline CLV scoring only works on a local scheduler."
sources:
  - odds_lambda.oddsportal_common.run_scraper_with_retry
  - odds_lambda.oddsportal_common.build_raw_data
  - odds_lambda.oddsportal_common.find_existing_event
  - odds_lambda.oddsportal_common.decimal_to_american
  - odds_lambda.oddsportal_common.hours_to_tier
  - odds_lambda.oddsportal_adapter.convert_upcoming_matches
  - odds_lambda.browser_lock.playwright_semaphore
  - odds_lambda.jobs.fetch_oddsportal.main
  - odds_lambda.jobs.fetch_oddsportal.LEAGUE_SPECS
  - odds_lambda.jobs.fetch_oddsportal.CADENCE
  - odds_cli.commands.scrape.scrape_upcoming
  - deployment/aws/terraform/scraper.tf
status: stable
---

# OddsPortal scraper and fetch-oddsportal job

Headless-browser scraping of OddsPortal (UK bookmakers [the Odds API](odds-api-client-and-quota.md) lacks) through the OddsHarvester fork, pinned in `pyproject.toml` (git tag `odds-fork-2026.05.29`). It is used by the fetch-oddsportal job below, by the results job ([event-lifecycle](event-lifecycle.md)) and by `odds scrape upcoming`. Historical bulk loads use scripts instead ([howto-load-historical-odds](howto-load-historical-odds.md)).

## Scraper modules

- `odds_lambda.jobs.fetch_oddsportal`: `odds_lambda.jobs.fetch_oddsportal.LEAGUE_SPECS` (EPL: `1x2` + `over_under_2_5`; MLB: `home_away`), `odds_lambda.jobs.fetch_oddsportal.run_harvester_upcoming`, `odds_lambda.jobs.fetch_oddsportal.ingest_league`. `odds_lambda.jobs.fetch_oddsportal.LEAGUE_SPEC_BY_NAME` serves the CLI and MCP scrape tools.
- `odds_lambda.oddsportal_common`: `odds_lambda.oddsportal_common.BOOKMAKER_KEY_MAP`/`odds_lambda.oddsportal_common.normalize_bookmaker_key`, `odds_lambda.oddsportal_common.decimal_to_american`, `odds_lambda.oddsportal_common.hours_to_tier`, `odds_lambda.oddsportal_common.build_raw_data` (historical records), `odds_lambda.oddsportal_common.run_scraper_with_retry`, `odds_lambda.oddsportal_common.find_existing_event`.
- `odds_lambda.oddsportal_adapter`: pure conversion of live OddsHarvester upcoming-matches output (`odds_lambda.oddsportal_adapter.convert_upcoming_matches`, dispatch via `_MARKET_CONVERTERS`: `1x2`, `over_under_2_5`, `home_away`). Fractional odds become American; Betfair Exchange rows carry `betfair_matched` liquidity.
- `odds_lambda.browser_lock`: `playwright_semaphore` (value 1) serializes browser use in one process. The results job reuses `odds_lambda.jobs.fetch_oddsportal.LEAGUE_SPECS` but not this semaphore.

### Behaviour

- Output is The Odds API `raw_data` shape with `source: oddsportal_live`, stored by `odds_lambda.storage.writers.OddsWriter.store_odds_snapshot` ([odds-storage](../storage/odds-storage.md)); events resolve by `find_or_create_event` (±2 h, `op_live_` ids) ([events](../storage/events.md#matchers)).
- `odds_lambda.oddsportal_adapter.convert_upcoming_matches` drops matches whose start is more than `odds_lambda.oddsportal_adapter.STALE_MATCH_REJECT_THRESHOLD` (12 h) before scrape time, a guard against OddsPortal hydrating a past meeting.
- `odds_lambda.oddsportal_common.run_scraper_with_retry` retries an all-empty scrape and then re-scrapes failed URLs (except 404s) in fresh browser sessions, with a fresh `fake_useragent` Chrome UA each time. It raises `RuntimeError` if OddsHarvester returns `None`.
- `odds scrape upcoming --league --market --dry-run --from-file` replaces the spec's markets with `--market` values; `--dry-run` still scrapes unless `--from-file` is given. The MCP `odds_mcp.server.refresh_scrape` tool queues `odds_lambda.jobs.fetch_oddsportal.ingest_league` in the local scheduler process and needs it running ([mcp-server](../agent/mcp-server.md)).

## The fetch-oddsportal job

`odds_lambda.jobs.fetch_oddsportal.main(ctx)` scrapes pre-match odds, converts them to the canonical shape and stores snapshots through `odds_lambda.storage.writers.OddsWriter`. It runs on the scraper Lambda (per-sport rules, gated by `enable_oddsportal_scraper`) and in the default local `bootstrap_jobs` ([jobs-and-deployment-matrix](../scheduling/jobs-and-deployment-matrix.md)). The same job on AWS and locally double-scrapes.

### Leagues

`odds_lambda.jobs.fetch_oddsportal.LEAGUE_SPECS` holds `odds_lambda.jobs.fetch_oddsportal.LeagueSpec` rows: EPL (`football`, `england-premier-league`, markets `1x2` and `over_under_2_5`, 3 outcomes, overnight 22-06) and MLB (`baseball`, `mlb`, market `home_away`, 2 outcomes, overnight 05-14). With a `ctx.sport` only that spec runs; an unknown sport logs an error and returns. The `odds_lambda.jobs.fetch_oddsportal.LeagueSpec` overnight fields duplicate `odds_lambda.scheduling.decision.OVERNIGHT_WINDOWS` in the decision engine ([self-scheduling](../scheduling/self-scheduling.md)).

### Flow

1. Pre-schedule using `odds_lambda.scheduling.decision.decide_forward_resilient` with `odds_lambda.jobs.fetch_oddsportal.CADENCE`: 30 min under 3 h, 1 h at 3-12 h, 2 h beyond or with no games, 1 h if the database is down. This job is always-run: only `next_execution` is used, not `should_execute`.
2. Inside `job_alert_context(compound_name)`, `odds_lambda.jobs.fetch_oddsportal.ingest_league` per spec: scrape under the Playwright semaphore, `odds_lambda.oddsportal_adapter.convert_upcoming_matches` per market, `find_or_create_event` and `store_odds_snapshot` per match (tier assigned by the writer, see [fetch-tier](../storage/fetch-tier.md)), tagging `api_request_id` as `oddsportal_live_<time>`. A league failure is logged and the others continue.
3. If nothing was scraped, `send_job_warning("scrape_empty:<job>")`.
4. Outside the alert context: if `model_supports_sport(ctx.sport)`, `score_events(sport=...)` runs inline ([model-publishing-and-scoring](../modelling/model-publishing-and-scoring.md)). Any exception there is logged as `fetch_oddsportal_scoring_failed` and swallowed so it cannot fire a scrape alert.
5. Post-schedule only when at least one match was scraped.

Heartbeats are keyed `heartbeat:fetch-oddsportal-<sport>`; both sports have 2 h expectations in `odds_core.config.AlertConfig.heartbeat_expectations` ([alerts-and-health](../operations/alerts-and-health.md#heartbeats)).

## Hazards

- **Inline scoring only works locally.** The scraper image installs only `odds-core` and `odds-lambda`; `model_loader` imports `odds_analytics` at module top and `xgboost` is not installed there (`deployment/aws/Dockerfile.scraper`). On the scraper Lambda the scoring import fails and is swallowed (no alert), and `deployment/aws/terraform/scraper.tf` sets no `MODEL_*` variables. Check the deployed function with `aws lambda get-function-configuration --function-name odds-scheduler-scraper`.
- **Scraper-side alerts are off by default.** `scraper.tf` sets no `ALERT_ENABLED` or webhook, so the scraper Lambda's alerts do not fire unless added ([aws-infrastructure](../operations/aws-infrastructure.md)).
- **`MAX_SCRAPER_RETRIES = 1` means one attempt, zero retries** (the loop is `range(1, MAX+1)`); the docstring says "up to 3 passes" but `MAX_FAILED_URL_RETRY_PASSES = 1` too. The empty-page retry never fires.
- **Market spelling.** The converter key is `over_under_2_5`; anything else (such as `over_under_2.5`) raises `ValueError("Unsupported market")` in `odds_lambda.oddsportal_adapter.convert_upcoming_matches`.
- **Two decimal-to-American converters.** `odds_lambda.oddsportal_common.decimal_to_american` rounds and returns `-10000` for decimal <= 1.0; `odds_core.odds_math.decimal_to_american` (used by Betfair) truncates with `int()` and divides by zero at exactly 1.0. Both feed stored prices ([odds-math-and-devigging](../foundations/odds-math-and-devigging.md)).
- **Tier boundaries disagree.** `odds_lambda.oddsportal_common.hours_to_tier` uses `<`; `tier_utils.calculate_tier` (live writes) uses `<=`, so a snapshot exactly 3, 12, 24 or 72 h out is tiered differently by historical vs live paths ([fetch-tier](../storage/fetch-tier.md#four-boundary-encodings)).
- **Not just "common".** `odds_lambda.oddsportal_common` also holds the retry state machine and a DB query; `odds_lambda.oddsportal_common.find_existing_event` (±24 h, returns `None` on ambiguity) is a different matcher from the writer's. `odds_lambda.betfair.adapter.build_raw_data` is an unrelated function of the same name.
- **Zero-match runs don't reschedule.** When the scrape returns zero matches the job skips post-schedule, so the next run keeps the earlier computed time.
- **Double scrape.** If the scraper is enabled on AWS and bootstrapped locally, both run; check `odds scheduler list-jobs --backend aws`.

## Related

- [Sources and keys](sources-and-keys.md)
- [Betfair Exchange client and job](betfair-exchange.md)
- [Event lifecycle and results jobs](event-lifecycle.md)
- [Self-scheduling](../scheduling/self-scheduling.md)
