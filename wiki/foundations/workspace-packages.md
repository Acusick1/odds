---
type: reference
title: Workspace packages and repository layout
description: The five uv workspace packages, their modules and declared dependencies, the repo layout and stack.
sources:
  - odds_core
  - odds_core.database
  - odds_core.odds_math
  - odds_core.snapshot_utils.extract_odds_from_snapshot
  - odds_core.utils.raw_data_has_market
  - odds_core.paper_trade_models.PaperTrade
  - packages/odds-core/pyproject.toml
  - odds_lambda
  - odds_lambda.scheduling.jobs._JOB_MODULE_MAP
  - odds_lambda.model_loader.load_model
  - odds_lambda.jobs.score_predictions.score_events
  - odds_lambda.paper_trading.place_trade
  - odds_lambda.copy_from_production.copy_from_prod
  - packages/odds-lambda/pyproject.toml
  - odds_analytics
  - odds_analytics.sequence_loader.calculate_devigged_bookmaker_target
  - odds_analytics.quality_metrics
  - odds_analytics.ml_strategy_example
  - odds_analytics.utils.per_book_market_holds
  - packages/odds-analytics/pyproject.toml
  - packages/odds-cli/pyproject.toml
  - packages/odds-mcp/pyproject.toml
  - pyproject.toml
  - docker-compose.yml
status: stable
---

# Workspace packages and repository layout

The repo is a uv workspace of five packages under `packages/` ([AGENTS.md](../../AGENTS.md)): `odds-core` at the bottom, then `odds-lambda`, `odds-analytics`, and the two front ends `odds-mcp` and `odds-cli`. Dependencies point downward: odds-lambda depends on odds-core, odds-analytics on both, and odds-mcp and odds-cli on all three. The stated rule is that core stays light; two exceptions are under [Dependency exceptions](#dependency-exceptions).

## Repository layout

| Path | Contents |
|---|---|
| `packages/` | The five uv workspace members |
| `docs/adr/`, `docs/agents/` | Decision records and agent tooling conventions, see [Documentation and issue tracking](documentation-and-issue-tracking.md) |
| `experiments/` | `configs/`, `scripts/`, `results/` for modelling runs |
| `scripts/` | Operational scripts (ingestion, tier checks, `ci/`, `wiki_lint.py`) |
| `migrations/`, `alembic.ini` | Alembic, see [migrations](howto-database-migration.md) |
| `tests/` | `unit/`, `integration/`, `migration/`, see [Test suite](test-suite.md) |
| `deployment/aws/` | Terraform, Dockerfiles, bootstrap, see [AWS infrastructure](../operations/aws-infrastructure.md) |
| `.github/workflows/` | [CI/CD workflows](../operations/ci-cd-workflows.md) |
| `wiki/` | The wiki |

`.claude/`, `worktrees/` and `logs/` are gitignored. The root `pyproject.toml` description reads "NBA betting odds data collection and analysis system" (#465).

## Stack

Python 3.12+ (`pyproject.toml`), a uv workspace. Per package dependencies are in each `packages/<name>/pyproject.toml`:

| Concern | Library | Declared in |
|---|---|---|
| ORM and driver | SQLModel, asyncpg, on PostgreSQL 15 (`docker-compose.yml`) | `packages/odds-core/pyproject.toml` |
| Settings and logging | Pydantic Settings, structlog | `packages/odds-core/pyproject.toml` |
| Local scheduler | APScheduler 4 (pinned alpha), tenacity retries | `packages/odds-lambda/pyproject.toml` |
| Migrations | Alembic | `packages/odds-lambda/pyproject.toml` |
| NBA stats and browser scraping | nba_api, Playwright | `packages/odds-lambda/pyproject.toml` |
| CLI | Typer, Rich | `packages/odds-cli/pyproject.toml` |
| MCP server | FastMCP | `packages/odds-mcp/pyproject.toml` |
| Modelling | XGBoost | `packages/odds-analytics/pyproject.toml` |
| Tests | pytest, pytest-asyncio, py-pglite | `pyproject.toml` |

## odds-core

`packages/odds-core/odds_core/` declares only sqlmodel, asyncpg, pydantic-settings, python-dotenv, structlog and aiohttp (`packages/odds-core/pyproject.toml`).

| Module | Holds |
|---|---|
| `models.py` | `odds_core.models.Event`, `odds_core.models.OddsSnapshot`, `odds_core.models.Odds`, `odds_core.models.FetchLog`, `odds_core.models.DataQualityLog`, `odds_core.models.AlertHistory`, `odds_core.models.utc_now` ([events](../storage/events.md), [odds tables](../storage/odds-tables.md)) |
| `odds_core.prediction_models`, `odds_core.match_brief_models`, `odds_core.paper_trade_models`, `odds_core.agent_wakeup_models` | Prediction and agent tables ([match briefs](../agent/match-briefs.md), [paper trading](../agent/paper-trading.md)) |
| `odds_core.epl_data_models`, `odds_core.mlb_data_models`, `odds_core.injury_models`, `odds_core.game_log_models`, `odds_core.player_stats_models`, `odds_core.polymarket_models` | Sport-context and Polymarket tables |
| `config.py` | Pydantic Settings, see [Configuration](configuration.md) |
| `odds_core.database` | Engine and `odds_core.database.async_session_maker`, see [Databases and environments](databases-and-environments.md) |
| `odds_core.alerts` | `odds_core.alerts.AlertManager`, `odds_core.alerts.DiscordAlert`, see [Alerts](../operations/alerts.md) |
| `odds_core.odds_math` | Price conversion, implied probability, devig, see [Odds arithmetic helpers](odds-math.md) |
| `odds_core.snapshot_utils`, `utils.py` | `odds_core.snapshot_utils.extract_odds_from_snapshot` (parse `raw_data` into `odds_core.models.Odds`), `odds_core.utils.raw_data_has_market` |
| `odds_core.sports`, `odds_core.team`, `odds_core.time`, `odds_core.logging_setup` | [Sport keys](../collection/sport-and-market-keys.md), [team-name normalization](../storage/event-matching.md), [timezone helpers](timezones.md), structlog setup |
| `odds_core.api_models` | The Odds API response models and event factories |

`odds_core.snapshot_utils.extract_odds_from_snapshot` matches market keys exactly, so `h2h` and `1x2` snapshots are different markets to every reader that uses it.

Importing any `odds_core` submodule runs `odds_core.database`, which builds the engine from `odds_core.config.get_settings()`. Every `odds` command therefore needs `ODDS_API_KEY` and `DATABASE_URL` to resolve, even commands that never touch the API or database ([Engine creation](databases-and-environments.md#engine-creation), [Configuration](configuration.md)).

## odds-lambda

`packages/odds-lambda/odds_lambda/` holds everything that collects, stores and schedules data. The name is historical: the same code runs under the local scheduler.

| Area | Modules | Page |
|---|---|---|
| Odds API | `odds_lambda.data_fetcher`, `odds_lambda.api_key_manager`, `odds_lambda.event_sync` (event discovery and `op_live_` twin merge) | [Odds API client](../collection/odds-api-client.md) |
| Ingestion | `odds_lambda.ingestion`, `odds_lambda.storage.validators` | [Odds API ingestion service](../collection/odds-ingestion-service.md) |
| OddsPortal scrapers | `odds_lambda.oddsportal_common`, `odds_lambda.oddsportal_adapter`, `odds_lambda.oddsportal_client` | [OddsPortal scraper](../collection/oddsportal-scraper.md) |
| Exchange | `betfair/` | [Betfair Exchange](../collection/betfair-exchange.md) |
| Sport context fetchers | `odds_lambda.espn_fixture_fetcher`, `odds_lambda.mlb_stats_fetcher`, `odds_lambda.injury_fetcher`, `odds_lambda.game_log_fetcher`, `odds_lambda.pbpstats_fetcher`, `odds_lambda.polymarket_fetcher` | [EPL](../collection/epl-context-data.md), [MLB](../collection/mlb-probable-pitchers.md), [NBA](../collection/nba-injuries-and-stats.md), [Polymarket](../collection/polymarket.md) |
| Storage | `storage/` (one reader/writer pair per source) | [Storage layer](../storage/storage-layer.md) |
| Scheduling | `scheduling/` (`odds_lambda.scheduling.jobs`, `odds_lambda.scheduling.decision`, `odds_lambda.scheduling.helpers`, `odds_lambda.scheduling.backends`), `odds_lambda.lambda_handler`, `odds_lambda.fetch_tier`, `odds_lambda.tier_utils` | [Job registry](../scheduling/job-registry.md), [Self-scheduling](../scheduling/self-scheduling.md), [Fetch tier](../storage/fetch-tier.md) |
| Jobs | `jobs/*.py`, one `main(ctx)` per file | `odds_lambda.scheduling.jobs._JOB_MODULE_MAP` lists all 14 |
| Services | `odds_lambda.paper_trading`, `odds_lambda.model_loader`, `odds_lambda.health_monitor`, `odds_lambda.copy_from_production` | [Paper trading](../agent/paper-trading.md), [model store](../modelling/model-store.md), [health monitor](../operations/health-monitor.md), [copy production data](howto-copy-production-data.md) |

`odds_lambda` re-exports `odds_lambda.data_fetcher.TheOddsAPIClient`, `odds_lambda.ingestion.OddsIngestionService`, `odds_lambda.fetch_tier.FetchTier`, `odds_lambda.tier_utils.calculate_tier` and `odds_lambda.tier_utils.calculate_hours_until_commence`. Job modules are imported lazily by name from `odds_lambda.scheduling.jobs._JOB_MODULE_MAP`, so a scraper-only Lambda never imports the analytics stack. `odds_lambda.paper_trading` and `odds_lambda.copy_from_production` are service code for the CLI and MCP server rather than Lambda code; the copier is a Rich-console tool.

## odds-analytics

`packages/odds-analytics/odds_analytics/` holds backtesting, strategies, feature extraction, model training and the backfill planner. It depends on core and odds-lambda (storage readers, `odds_lambda.fetch_tier.FetchTier`). Heavy dependencies (torch, mlflow, scikit-learn, optuna) sit behind the `training` extra; `odds_analytics` lazy-loads the modules that need them via `odds_analytics.__getattr__`.

| Area | Modules | Page |
|---|---|---|
| Backtesting | `backtesting/` (`config.py`, `models.py`, `odds_analytics.backtesting.services`), `odds_analytics.strategies` | [Backtest engine](../evaluation/backtest-engine.md), [Backtest strategies](../evaluation/backtest-strategies.md) |
| Features | `odds_analytics.feature_extraction`, `odds_analytics.feature_groups` (snapshot samplers and the training-array orchestrator), `odds_analytics.standings_features`, `odds_analytics.match_stats_features`, `odds_analytics.epl_schedule_features`, `odds_analytics.schedule_features`, `odds_analytics.injury_features`, `odds_analytics.polymarket_features` | [Feature pipeline](../modelling/feature-pipeline.md) |
| Training | `training/` (`odds_analytics.training.config`, `odds_analytics.training.data_preparation` (config-driven event selection and training-data preparation), `odds_analytics.training.cross_validation`, `odds_analytics.training.tuner` (Optuna wrapper with a seedable sampler), `odds_analytics.training.feature_selection`, `odds_analytics.training.tracking`) | [Training pipeline](../modelling/training-pipeline.md) |
| Line-movement models | `odds_analytics.xgboost_line_movement`, `odds_analytics.lstm_line_movement`, `odds_analytics.sequence_loader` | [Line-movement strategies](../modelling/line-movement-models.md) |
| Backfill | `odds_analytics.gap_backfill_planner`, `odds_analytics.game_selector`, `odds_analytics.backfill_executor` | [Odds API historical backfill](../collection/odds-api-backfill.md) |
| Coverage reporting | `odds_analytics.quality_metrics` | [Coverage checks](../storage/coverage-checks.md) |
| Helpers | `utils.py` (EV, holds, arbitrage), `odds_analytics.quality_metrics` | |

### Names that mislead

- `odds_analytics.sequence_loader` is not LSTM-only: it holds the target maths for every model (`odds_analytics.sequence_loader.calculate_devigged_bookmaker_target`, `odds_analytics.sequence_loader.DeviggedProbs`), the label for every model ([Target functions](../modelling/clv-target-functions.md), [Devigging](devigging.md)).
- `odds_analytics.quality_metrics` never touches `odds_core.models.DataQualityLog`; it counts game and odds coverage. Data-quality rows are a separate system ([Odds validation](../storage/odds-validator.md), [Coverage checks](../storage/coverage-checks.md)).
- `odds_analytics.ml_strategy_example` (329 lines) is tested infrastructure for an XGBoost win-probability classifier, not the CLV regression the system uses ([Line-movement strategies](../modelling/line-movement-models.md)).
- `odds_analytics.utils.per_book_market_holds` is the hold maths the MCP screener calls ([retail edges](../agent/mcp-market-tools.md#retail-edges)); it is the only reason `odds-mcp` imports analytics ([odds-mcp server](../agent/mcp-server.md)).

## Dependency exceptions

- `packages/odds-lambda/pyproject.toml` does not declare `odds-analytics`, but `odds_lambda.model_loader` imports `odds_analytics.training.config` (where the feature group names and target fields are validated) and `odds_lambda.jobs.score_predictions` imports `odds_analytics.feature_groups` (the training-array module) at module level (#454). Analytics declares a dependency on odds-lambda, so the cycle is undeclared on this side. The scraper-side consequence is on [Prediction scoring](../modelling/prediction-scoring.md).
- `odds_core.paper_trade_models.PaperTrade` imports `odds_analytics.backtesting.models` (type-checking plus a lazy import inside `odds_core.paper_trade_models.PaperTrade.to_bet_record`), so odds-core depends on odds-analytics in practice (#459).
- `python-dotenv` is declared in core but nothing imports it; `.env` is read only by pydantic-settings (#459).

## Related

- [About the system and its pivots](project-overview.md)
- [The odds CLI](odds-cli.md)
- [Configuration](configuration.md)
