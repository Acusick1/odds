---
type: component
title: OddsPortal scraper integration
summary: OddsHarvester/Playwright scraping, bookmaker mapping, the raw_data adapter, retry logic, browser lock and LEAGUE_SPECS, shared by the OddsPortal jobs and `odds scrape`.
sources:
  - packages/odds-lambda/odds_lambda/oddsportal_common.py::run_scraper_with_retry
  - packages/odds-lambda/odds_lambda/oddsportal_common.py::build_raw_data
  - packages/odds-lambda/odds_lambda/oddsportal_common.py::find_existing_event
  - packages/odds-lambda/odds_lambda/oddsportal_adapter.py::convert_upcoming_matches
  - packages/odds-lambda/odds_lambda/browser_lock.py::playwright_semaphore
  - packages/odds-lambda/odds_lambda/jobs/fetch_oddsportal.py::LEAGUE_SPECS
  - packages/odds-cli/odds_cli/commands/scrape.py::scrape_upcoming
  - docs/ARCHITECTURE.md#oddsportal-via-oddsharvester-scraper
  - docs/CLI.md#scrape-oddsportal
  - ./AGENTS.md#conventions
verified_at: 7887039
---

# OddsPortal scraper integration

Headless-browser scraping of OddsPortal (UK bookmakers the [Odds API](odds-api-client.md) lacks) through the OddsHarvester fork, pinned in `pyproject.toml` (git tag `odds-fork-2026.05.29`). Used by [fetch-oddsportal](job-fetch-oddsportal.md), [fetch-oddsportal-results](job-fetch-oddsportal-results.md) and `odds scrape upcoming`. Historical bulk loads use scripts instead: [Historical odds ingestion](historical-odds-ingestion.md).

## Modules

- `packages/odds-lambda/odds_lambda/jobs/fetch_oddsportal.py`: `LEAGUE_SPECS` (EPL: `1x2` + `over_under_2_5`; MLB: `home_away`), `run_harvester_upcoming`, `ingest_league`. `LEAGUE_SPEC_BY_NAME` serves the CLI.
- `oddsportal_common.py`: `BOOKMAKER_KEY_MAP`/`normalize_bookmaker_key`, `decimal_to_american`, `hours_to_tier`, `build_raw_data` (historical records), `run_scraper_with_retry`, `find_existing_event`.
- `oddsportal_adapter.py`: pure conversion of live `upcoming` output (`convert_upcoming_matches`, dispatch via `_MARKET_CONVERTERS`: `1x2`, `over_under_2_5`, `home_away`). Fractional odds become American; Betfair Exchange rows carry `betfair_matched` liquidity.
- `browser_lock.py`: `playwright_semaphore` (value 1) serializes browser use in one process.

## Behaviour

- Output is The Odds API `raw_data` shape with `source: oddsportal_live`, stored by `OddsWriter.store_odds_snapshot` ([Odds writer](odds-writer.md)); events resolve by `find_or_create_event` (+/-2 h, `op_live_` ids) ([Event matching](event-matching.md)).
- `convert_upcoming_matches` drops matches whose start is more than `STALE_MATCH_REJECT_THRESHOLD` (12 h) before scrape time, a guard against OddsPortal hydrating a past meeting.
- `run_scraper_with_retry` retries an all-empty scrape and then re-scrapes failed URLs (except 404s) in fresh browser sessions, with a fresh `fake_useragent` Chrome UA each time. It raises `RuntimeError` if OddsHarvester returns `None`.
- `odds scrape upcoming --league --market --dry-run --from-file` replaces the spec's markets with `--market` values; `--dry-run` still scrapes unless `--from-file` is given.
- After ingest, `fetch-oddsportal` runs `score_events` inline ([Job: score predictions](job-score-predictions.md)); the job self-schedules ([Self-scheduling](self-scheduling.md)).

## Hazards

- **`MAX_SCRAPER_RETRIES = 1` means one attempt, zero retries** (the loop is `range(1, MAX+1)`); the docstring says "up to 3 passes" but `MAX_FAILED_URL_RETRY_PASSES = 1` too. The empty-page retry never fires.
- **CLI docs use the wrong market spelling**: `docs/CLI.md` shows `--market over_under_2.5`; the converter key is `over_under_2_5`, and anything else raises `ValueError("Unsupported market")`.
- **Two `decimal_to_american` implementations.** `oddsportal_common.decimal_to_american` rounds and returns `-10000` for decimal <= 1.0; `odds_core.odds_math.decimal_to_american` (used by Betfair) truncates with `int()` and divides by zero at exactly 1.0. Both feed stored prices ([Odds math](odds-math.md)).
- **Tier boundaries disagree.** `hours_to_tier` uses `<`; `tier_utils.calculate_tier` (live writes) uses `<=`, so a snapshot exactly 3, 12, 24 or 72 h out is tiered differently by historical vs live paths ([Fetch tier](fetch-tier.md)).
- `oddsportal_common.py` also holds the retry state machine and a DB query, not just "common" helpers; `find_existing_event` (+/-24 h, returns `None` on ambiguity) is a different matcher from the writer's.
- A `build_raw_data` also exists in `packages/odds-lambda/odds_lambda/betfair/adapter.py`; they are unrelated.
- Whether the scraper Lambda is scheduled is deployed state: `odds scheduler list-jobs --backend aws`.

## Related

- [Data sources](data-sources.md)
- [Betfair Exchange client](betfair-exchange-client.md)
- [Job deployment matrix](job-deployment-matrix.md)
