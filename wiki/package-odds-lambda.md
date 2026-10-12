---
type: component
title: odds-lambda package
summary: "Module map for odds-lambda (fetchers, ingestion, storage, scheduling, jobs, services) and the cross-package imports its pyproject does not declare."
sources:
  - packages/odds-lambda/odds_lambda/__init__.py
  - packages/odds-lambda/odds_lambda/scheduling/jobs.py::_JOB_MODULE_MAP
  - packages/odds-lambda/odds_lambda/model_loader.py::load_model
  - packages/odds-lambda/odds_lambda/jobs/score_predictions.py::score_events
  - packages/odds-lambda/odds_lambda/paper_trading.py::place_trade
  - packages/odds-lambda/odds_lambda/copy_from_production.py::copy_from_prod
  - packages/odds-lambda/pyproject.toml
verified_at: 7887039
---

# odds-lambda package

`packages/odds-lambda/odds_lambda/` holds everything that collects, stores and schedules data. It depends on [odds-core](package-odds-core.md) and is the base for [odds-analytics](package-odds-analytics.md), `odds-mcp` and `odds-cli`. The name is historical: the same code also runs under the local scheduler. Layout: [AGENTS.md](../AGENTS.md#package-structure).

## Module map

| Area | Modules | Page |
|---|---|---|
| Odds API | `data_fetcher.py`, `api_key_manager.py`, `event_sync.py` | [Client and key rotation](odds-api-client.md) |
| Ingestion | `ingestion.py`, `odds_lambda.storage.validators` | [Ingestion flow](odds-ingestion-flow.md) |
| Scrapers | `oddsportal_common.py`, `oddsportal_adapter.py`, `browser_lock.py` | [OddsPortal scraper](oddsportal-scraper.md) |
| Exchange | `betfair/` | [Betfair client](betfair-exchange-client.md) |
| Sport context fetchers | `espn_fixture_fetcher.py`, `mlb_stats_fetcher.py`, `injury_fetcher.py`, `game_log_fetcher.py`, `pbpstats_fetcher.py`, `polymarket_fetcher.py` | [EPL context](epl-context-data.md), [MLB pitchers](mlb-probable-pitchers.md), [NBA injuries](nba-injury-reports.md), [NBA stats](nba-stats-data.md), [Polymarket](polymarket-pipeline.md) |
| Storage | `storage/` (one reader/writer pair per source) | [Storage layer](storage-layer.md) |
| Scheduling | `scheduling/` (`jobs.py`, `decision.py`, `helpers.py`, `backends/`), `lambda_handler.py`, `fetch_tier.py`, `tier_utils.py` | [Job registry](job-registry.md), [Scheduling decision engine](scheduling-decision-engine.md), [Fetch tier](fetch-tier.md) |
| Jobs | `jobs/*.py`, one entry point (`main`) per file | `_JOB_MODULE_MAP` lists all 14 |
| Services | `paper_trading.py`, `model_loader.py`, `health_monitor.py`, `copy_from_production.py` | [Paper trading](paper-trading.md), [S3 model store](s3-model-store.md), [Health monitor](health-monitor.md), [Copy from production](copy-from-production.md) |

## Behaviour to know

- `packages/odds-lambda/odds_lambda/__init__.py` re-exports `TheOddsAPIClient`, `OddsIngestionService`, `FetchTier`, `calculate_tier` and `calculate_hours_until_commence`.
- `odds_lambda.jobs` is a package and `odds_lambda.scheduling.jobs` is the registry module, so "jobs" imports from "jobs" ([Job registry](job-registry.md)).
- Job modules are imported lazily by name from `packages/odds-lambda/odds_lambda/scheduling/jobs.py::_JOB_MODULE_MAP`, so a scraper-only Lambda never imports the analytics stack.
- Two modules here are service code used by the CLI and MCP server rather than Lambda code: `paper_trading.py` and `copy_from_production.py` (the latter prints with a Rich console).

## Hazards

- `packages/odds-lambda/pyproject.toml` does not list `odds-analytics`, but `model_loader.py` imports `odds_analytics.training.config` at module level and `packages/odds-lambda/odds_lambda/jobs/score_predictions.py` imports `odds_analytics.feature_groups`. Analytics in turn declares a dependency on odds-lambda, so the dependency is circular and undeclared on this side. Scoring requires the analytics extras installed.

## Related

- [Project overview](project-overview.md) — repo layout
- [Job deployment matrix](job-deployment-matrix.md) — where each job runs
