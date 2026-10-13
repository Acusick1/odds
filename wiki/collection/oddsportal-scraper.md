---
type: reference
title: OddsPortal scraper
description: oddsportal_common, oddsportal_adapter, browser_lock, retries, and odds scrape upcoming.
sources:
  - odds_lambda.oddsportal_common.run_scraper_with_retry
  - odds_lambda.oddsportal_common.build_raw_data
  - odds_lambda.oddsportal_common.find_existing_event
  - odds_lambda.oddsportal_common.BOOKMAKER_KEY_MAP
  - odds_lambda.oddsportal_common.normalize_bookmaker_key
  - odds_lambda.oddsportal_common.decimal_to_american
  - odds_lambda.oddsportal_common.hours_to_tier
  - odds_lambda.oddsportal_common.MAX_SCRAPER_RETRIES
  - odds_lambda.oddsportal_common.MAX_FAILED_URL_RETRY_PASSES
  - odds_lambda.oddsportal_adapter.convert_upcoming_matches
  - odds_lambda.oddsportal_adapter.STALE_MATCH_REJECT_THRESHOLD
  - odds_lambda.browser_lock.playwright_semaphore
  - odds_cli.commands.scrape.scrape_upcoming
  - pyproject.toml
status: stable
---

# OddsPortal scraper

Headless-browser scraping of OddsPortal (UK bookmakers [the Odds API](odds-api-client.md) lacks) through the OddsHarvester fork, pinned in `pyproject.toml` (git tag `odds-fork-2026.05.29`). It is used by [the fetch-oddsportal job](fetch-oddsportal-job.md), by the results job ([results-jobs](results-jobs.md)) and by `odds scrape upcoming`. Historical bulk loads use scripts instead ([historical-ingest-scripts](historical-ingest-scripts.md)).

## Modules

- `odds_lambda.jobs.fetch_oddsportal`: `odds_lambda.jobs.fetch_oddsportal.LEAGUE_SPECS`, `odds_lambda.jobs.fetch_oddsportal.run_harvester_upcoming`, `odds_lambda.jobs.fetch_oddsportal.ingest_league`. `odds_lambda.jobs.fetch_oddsportal.LEAGUE_SPEC_BY_NAME` serves the CLI and MCP scrape tools.
- `odds_lambda.oddsportal_common`: `odds_lambda.oddsportal_common.BOOKMAKER_KEY_MAP`/`odds_lambda.oddsportal_common.normalize_bookmaker_key`, `odds_lambda.oddsportal_common.decimal_to_american`, `odds_lambda.oddsportal_common.hours_to_tier`, `odds_lambda.oddsportal_common.build_raw_data` (historical records), `odds_lambda.oddsportal_common.run_scraper_with_retry`, `odds_lambda.oddsportal_common.find_existing_event`. Besides conversion helpers it holds the retry state machine and a DB query. `odds_lambda.betfair.adapter.build_raw_data` is an unrelated function of the same name.
- `odds_lambda.oddsportal_adapter`: pure conversion of live OddsHarvester upcoming-matches output (`odds_lambda.oddsportal_adapter.convert_upcoming_matches`, dispatch via `_MARKET_CONVERTERS`: `1x2`, `over_under_2_5`, `home_away`). Fractional odds become American; Betfair Exchange rows carry `betfair_matched` liquidity.
- `odds_lambda.browser_lock`: `playwright_semaphore` (value 1) serializes browser use in one process. The results job does not use it.

## Behaviour

- Output is The Odds API `raw_data` shape with `source: oddsportal_live`, stored by `odds_lambda.storage.writers.OddsWriter.store_odds_snapshot` ([odds-writer](../storage/odds-writer.md)); events resolve by `find_or_create_event` (±2 h, `op_live_` ids) ([event-matching](../storage/event-matching.md)). `odds_lambda.oddsportal_common.find_existing_event` (±24 h, returns `None` on ambiguity) is a different matcher from the writer's.
- `odds_lambda.oddsportal_adapter.convert_upcoming_matches` drops matches whose start is more than `odds_lambda.oddsportal_adapter.STALE_MATCH_REJECT_THRESHOLD` (12 h) before scrape time, a guard against OddsPortal hydrating a past meeting. The converter key for totals is `over_under_2_5`; anything else raises `ValueError("Unsupported market")` ([sport-and-market-keys](sport-and-market-keys.md#market-key-dialects)).
- `odds_lambda.oddsportal_common.run_scraper_with_retry` retries an all-empty scrape and then re-scrapes failed URLs (except 404s) in fresh browser sessions, with a fresh `fake_useragent` Chrome UA each time. It raises `RuntimeError` if OddsHarvester returns `None`. `odds_lambda.oddsportal_common.MAX_SCRAPER_RETRIES` and `odds_lambda.oddsportal_common.MAX_FAILED_URL_RETRY_PASSES` are 1, so an empty scrape is not retried and failed URLs get one pass; the docstring's "up to 3 passes" does not match (#447).
- `odds_lambda.oddsportal_common.decimal_to_american` and `odds_lambda.oddsportal_common.hours_to_tier` have different edge behaviour from the copies used by Betfair and live writes ([odds-math](../foundations/odds-math.md#conversion-copies), [fetch-tier](../storage/fetch-tier.md#boundary-encodings)).

## odds scrape upcoming

`odds scrape upcoming --league --market --dry-run --from-file` (`odds_cli.commands.scrape.scrape_upcoming`) replaces the spec's markets with `--market` values; `--dry-run` still scrapes unless `--from-file` is given. The MCP `odds_mcp.server.refresh_scrape` tool queues `odds_lambda.jobs.fetch_oddsportal.ingest_league` in the local scheduler process and needs it running ([mcp-scheduling-tools](../agent/mcp-server.md#scrape-and-scheduling-tools)).

## Related

- [fetch-oddsportal job](fetch-oddsportal-job.md)
- [Sport, market and bookmaker keys](sport-and-market-keys.md)
- [Betfair Exchange client and job](betfair-exchange.md)
