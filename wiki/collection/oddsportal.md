---
type: component
title: OddsPortal scraper and fetch-oddsportal job
description: "OddsHarvester/Playwright scraping, LEAGUE_SPECS, the raw_data adapter and retries, the fetch-oddsportal job's cadence and alerts, and why its inline CLV scoring only works on a local scheduler."
sources:
  - packages/odds-lambda/odds_lambda/oddsportal_common.py::run_scraper_with_retry
  - packages/odds-lambda/odds_lambda/oddsportal_common.py::build_raw_data
  - packages/odds-lambda/odds_lambda/oddsportal_common.py::find_existing_event
  - packages/odds-lambda/odds_lambda/oddsportal_common.py::decimal_to_american
  - packages/odds-lambda/odds_lambda/oddsportal_common.py::hours_to_tier
  - packages/odds-lambda/odds_lambda/oddsportal_adapter.py::convert_upcoming_matches
  - packages/odds-lambda/odds_lambda/browser_lock.py::playwright_semaphore
  - packages/odds-lambda/odds_lambda/jobs/fetch_oddsportal.py::main
  - packages/odds-lambda/odds_lambda/jobs/fetch_oddsportal.py::LEAGUE_SPECS
  - packages/odds-lambda/odds_lambda/jobs/fetch_oddsportal.py::CADENCE
  - packages/odds-cli/odds_cli/commands/scrape.py::scrape_upcoming
  - deployment/aws/terraform/scraper.tf
  - docs/ARCHITECTURE.md#oddsportal-via-oddsharvester-scraper
  - docs/ARCHITECTURE.md#oddsportal-scrape-job
  - docs/AGENT_DATA_SOURCES.md#scheduled-collection
  - docs/CLI.md#scrape-oddsportal
  - AGENTS.md#conventions
verified: "7887039"
status: stable
---

# OddsPortal scraper and fetch-oddsportal job

Headless-browser scraping of OddsPortal (UK bookmakers [the Odds API](odds-api-client-and-quota.md) lacks) through the OddsHarvester fork, pinned in `pyproject.toml` (git tag `odds-fork-2026.05.29`). It is used by the fetch-oddsportal job below, by the results job ([event-lifecycle](event-lifecycle.md)) and by `odds scrape upcoming`. Historical bulk loads use scripts instead ([howto-load-historical-odds](howto-load-historical-odds.md)).

## Scraper modules

- `packages/odds-lambda/odds_lambda/jobs/fetch_oddsportal.py`: `LEAGUE_SPECS` (EPL: `1x2` + `over_under_2_5`; MLB: `home_away`), `run_harvester_upcoming`, `ingest_league`. `LEAGUE_SPEC_BY_NAME` serves the CLI and MCP scrape tools.
- `oddsportal_common.py`: `BOOKMAKER_KEY_MAP`/`normalize_bookmaker_key`, `decimal_to_american`, `hours_to_tier`, `build_raw_data` (historical records), `run_scraper_with_retry`, `find_existing_event`.
- `oddsportal_adapter.py`: pure conversion of live `upcoming` output (`convert_upcoming_matches`, dispatch via `_MARKET_CONVERTERS`: `1x2`, `over_under_2_5`, `home_away`). Fractional odds become American; Betfair Exchange rows carry `betfair_matched` liquidity.
- `browser_lock.py`: `playwright_semaphore` (value 1) serializes browser use in one process. The results job reuses `LEAGUE_SPECS` but not this semaphore.

### Behaviour

- Output is The Odds API `raw_data` shape with `source: oddsportal_live`, stored by `OddsWriter.store_odds_snapshot` ([odds-storage](../storage/odds-storage.md)); events resolve by `find_or_create_event` (±2 h, `op_live_` ids) ([events](../storage/events.md#matchers)).
- `convert_upcoming_matches` drops matches whose start is more than `STALE_MATCH_REJECT_THRESHOLD` (12 h) before scrape time, a guard against OddsPortal hydrating a past meeting.
- `run_scraper_with_retry` retries an all-empty scrape and then re-scrapes failed URLs (except 404s) in fresh browser sessions, with a fresh `fake_useragent` Chrome UA each time. It raises `RuntimeError` if OddsHarvester returns `None`.
- `odds scrape upcoming --league --market --dry-run --from-file` replaces the spec's markets with `--market` values; `--dry-run` still scrapes unless `--from-file` is given. The MCP `refresh_scrape` tool queues `ingest_league` in the local scheduler process and needs it running ([mcp-server](../agent/mcp-server.md)).

## The fetch-oddsportal job

`packages/odds-lambda/odds_lambda/jobs/fetch_oddsportal.py::main(ctx)` scrapes pre-match odds, converts them to the canonical shape and stores snapshots through `OddsWriter`. It runs on the scraper Lambda (per-sport rules, gated by `enable_oddsportal_scraper`) and in the default local `bootstrap_jobs` ([jobs-and-deployment-matrix](../scheduling/jobs-and-deployment-matrix.md)). The same job on AWS and locally double-scrapes.

### Leagues

`LEAGUE_SPECS` holds `LeagueSpec` rows: EPL (`football`, `england-premier-league`, markets `1x2` and `over_under_2_5`, 3 outcomes, overnight 22-06) and MLB (`baseball`, `mlb`, market `home_away`, 2 outcomes, overnight 05-14). With a `ctx.sport` only that spec runs; an unknown sport logs an error and returns. The `LeagueSpec` overnight fields duplicate `OVERNIGHT_WINDOWS` in the decision engine ([self-scheduling](../scheduling/self-scheduling.md)).

### Flow

1. Pre-schedule using `decide_forward_resilient` with `CADENCE`: 30 min under 3 h, 1 h at 3-12 h, 2 h beyond or with no games, 1 h if the database is down. This job is always-run: only `next_execution` is used, not `should_execute`.
2. Inside `job_alert_context(compound_name)`, `ingest_league` per spec: scrape under the Playwright semaphore, `convert_upcoming_matches` per market, `find_or_create_event` and `store_odds_snapshot` per match (tier assigned by the writer, see [fetch-tier](../storage/fetch-tier.md)), tagging `api_request_id` as `oddsportal_live_<time>`. A league failure is logged and the others continue.
3. If nothing was scraped, `send_job_warning("scrape_empty:<job>")`.
4. Outside the alert context: if `model_supports_sport(ctx.sport)`, `score_events(sport=...)` runs inline ([model-publishing-and-scoring](../modelling/model-publishing-and-scoring.md)). Any exception there is logged as `fetch_oddsportal_scoring_failed` and swallowed so it cannot fire a scrape alert.
5. Post-schedule only when at least one match was scraped.

Heartbeats are keyed `heartbeat:fetch-oddsportal-<sport>`; both sports have 2 h expectations in `AlertConfig.heartbeat_expectations` ([alerts-and-health](../operations/alerts-and-health.md#heartbeats)).

## Hazards

- **Inline scoring only works locally.** The scraper image installs only `odds-core` and `odds-lambda`; `model_loader` imports `odds_analytics` at module top and `xgboost` is not installed there (`deployment/aws/Dockerfile.scraper`). On the scraper Lambda the scoring import fails and is swallowed (no alert), and `deployment/aws/terraform/scraper.tf` sets no `MODEL_*` variables. Check the deployed function with `aws lambda get-function-configuration --function-name odds-scheduler-scraper`.
- **Scraper-side alerts are off by default.** `scraper.tf` sets no `ALERT_ENABLED` or webhook, so the scraper Lambda's alerts do not fire unless added ([aws-infrastructure](../operations/aws-infrastructure.md)).
- **`MAX_SCRAPER_RETRIES = 1` means one attempt, zero retries** (the loop is `range(1, MAX+1)`); the docstring says "up to 3 passes" but `MAX_FAILED_URL_RETRY_PASSES = 1` too. The empty-page retry never fires.
- **CLI docs use the wrong market spelling**: `docs/CLI.md` shows `--market over_under_2.5`; the converter key is `over_under_2_5`, and anything else raises `ValueError("Unsupported market")`.
- **Two `decimal_to_american` implementations.** `oddsportal_common.decimal_to_american` rounds and returns `-10000` for decimal <= 1.0; `odds_core.odds_math.decimal_to_american` (used by Betfair) truncates with `int()` and divides by zero at exactly 1.0. Both feed stored prices ([odds-math-and-devigging](../foundations/odds-math-and-devigging.md)).
- **Tier boundaries disagree.** `hours_to_tier` uses `<`; `tier_utils.calculate_tier` (live writes) uses `<=`, so a snapshot exactly 3, 12, 24 or 72 h out is tiered differently by historical vs live paths ([fetch-tier](../storage/fetch-tier.md#four-boundary-encodings)).
- **Not just "common".** `oddsportal_common.py` also holds the retry state machine and a DB query; `find_existing_event` (±24 h, returns `None` on ambiguity) is a different matcher from the writer's. A `build_raw_data` also exists in `packages/odds-lambda/odds_lambda/betfair/adapter.py`; they are unrelated.
- **Zero-match runs don't reschedule.** When the scrape returns zero matches the job skips post-schedule, so the next run keeps the earlier computed time.
- **Double scrape.** If the scraper is enabled on AWS and bootstrapped locally, both run; check `odds scheduler list-jobs --backend aws`.

## Related

- [Sources and keys](sources-and-keys.md)
- [Betfair Exchange client and job](betfair-exchange.md)
- [Event lifecycle and results jobs](event-lifecycle.md)
- [Self-scheduling](../scheduling/self-scheduling.md)
