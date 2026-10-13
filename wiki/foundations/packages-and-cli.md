---
type: component
title: Packages and the odds CLI
description: Which of the five workspace packages holds what, the dependency rule and the undeclared odds-lambda to odds-analytics cycle, misleading module names, and the odds CLI's command groups, owning pages and shared flags.
sources:
  - packages/odds-core/odds_core/__init__.py
  - packages/odds-core/odds_core/database.py
  - packages/odds-core/odds_core/odds_math.py
  - packages/odds-core/odds_core/snapshot_utils.py::extract_odds_from_snapshot
  - packages/odds-core/odds_core/utils.py::raw_data_has_market
  - packages/odds-core/odds_core/paper_trade_models.py::PaperTrade
  - packages/odds-core/pyproject.toml
  - packages/odds-lambda/odds_lambda/__init__.py
  - packages/odds-lambda/odds_lambda/scheduling/jobs.py::_JOB_MODULE_MAP
  - packages/odds-lambda/odds_lambda/model_loader.py::load_model
  - packages/odds-lambda/odds_lambda/jobs/score_predictions.py::score_events
  - packages/odds-lambda/odds_lambda/paper_trading.py::place_trade
  - packages/odds-lambda/odds_lambda/copy_from_production.py::copy_from_prod
  - packages/odds-lambda/pyproject.toml
  - packages/odds-analytics/odds_analytics/__init__.py
  - packages/odds-analytics/odds_analytics/sequence_loader.py::calculate_devigged_bookmaker_target
  - packages/odds-analytics/odds_analytics/quality_metrics.py
  - packages/odds-analytics/odds_analytics/ml_strategy_example.py
  - packages/odds-analytics/odds_analytics/utils.py::per_book_market_holds
  - packages/odds-analytics/pyproject.toml
  - packages/odds-cli/odds_cli/main.py::app
  - packages/odds-cli/odds_cli/main.py::callback
  - packages/odds-cli/odds_cli/db_override.py::override_database_url
  - packages/odds-cli/odds_cli/db_override.py::swap_dbname
  - packages/odds-cli/odds_cli/commands/fetch.py
  - packages/odds-cli/odds_cli/commands/backtest.py::STRATEGIES
  - packages/odds-cli/odds_cli/commands/scheduler.py::run_once
  - packages/odds-cli/odds_cli/commands/train.py
  - packages/odds-cli/odds_cli/commands/validate.py
status: stable
---

# Packages and the odds CLI

The repo is a uv workspace of five packages under `packages/` ([AGENTS.md](../../AGENTS.md)): `odds-core` at the bottom, then `odds-lambda`, `odds-analytics`, and the two front ends `odds-mcp` and `odds-cli`. Dependencies point downward: odds-lambda depends on odds-core, odds-analytics on both, and odds-mcp and odds-cli on all three. The stated rule is that core stays light; two exceptions are under Hazards.

## odds-core

`packages/odds-core/odds_core/` declares only sqlmodel, asyncpg, pydantic-settings, python-dotenv, structlog and aiohttp (`packages/odds-core/pyproject.toml`).

| Module | Holds |
|---|---|
| `models.py` | `packages/odds-core/odds_core/models.py::Event`, `packages/odds-core/odds_core/models.py::OddsSnapshot`, `packages/odds-core/odds_core/models.py::Odds`, `packages/odds-core/odds_core/models.py::FetchLog`, `packages/odds-core/odds_core/models.py::DataQualityLog`, `packages/odds-core/odds_core/models.py::AlertHistory`, `packages/odds-core/odds_core/models.py::utc_now` ([events](../storage/events.md), [odds storage](../storage/odds-storage.md)) |
| `packages/odds-core/odds_core/prediction_models.py`, `packages/odds-core/odds_core/match_brief_models.py`, `packages/odds-core/odds_core/paper_trade_models.py`, `packages/odds-core/odds_core/agent_wakeup_models.py` | Prediction and agent tables ([briefs and paper trades](../agent/match-briefs-and-paper-trades.md)) |
| `packages/odds-core/odds_core/epl_data_models.py`, `packages/odds-core/odds_core/mlb_data_models.py`, `packages/odds-core/odds_core/injury_models.py`, `packages/odds-core/odds_core/game_log_models.py`, `packages/odds-core/odds_core/player_stats_models.py`, `packages/odds-core/odds_core/polymarket_models.py` | Sport-context and Polymarket tables |
| `config.py` | Pydantic Settings, see [Configuration](configuration.md) |
| `packages/odds-core/odds_core/database.py` | Engine and `async_session_maker`, see [Databases](databases-and-environments.md) |
| `packages/odds-core/odds_core/alerts.py` | `packages/odds-core/odds_core/alerts.py::AlertManager`, `packages/odds-core/odds_core/alerts.py::DiscordAlert`, see [Alerts and health](../operations/alerts-and-health.md) |
| `packages/odds-core/odds_core/odds_math.py` | Price conversion, implied probability, devig, see [Odds arithmetic](odds-math-and-devigging.md) |
| `packages/odds-core/odds_core/snapshot_utils.py`, `utils.py` | `packages/odds-core/odds_core/snapshot_utils.py::extract_odds_from_snapshot` (parse `raw_data` into `packages/odds-core/odds_core/models.py::Odds`), `packages/odds-core/odds_core/utils.py::raw_data_has_market` |
| `packages/odds-core/odds_core/sports.py`, `packages/odds-core/odds_core/team.py`, `packages/odds-core/odds_core/time.py`, `packages/odds-core/odds_core/logging_setup.py` | [Sport keys](../collection/sources-and-keys.md), team-name normalization, [timezone helpers](conventions.md#timezones), structlog setup |
| `packages/odds-core/odds_core/api_models.py` | The Odds API response models and event factories |

`packages/odds-core/odds_core/snapshot_utils.py::extract_odds_from_snapshot` matches market keys exactly, so `h2h` and `1x2` snapshots are different markets to every reader that uses it.

Importing any `odds_core` submodule runs `odds_core.database`, which builds the engine from `packages/odds-core/odds_core/config.py::get_settings()`. Every `odds` command therefore needs `ODDS_API_KEY` and `DATABASE_URL` to resolve, even commands that never touch the API or database ([Databases](databases-and-environments.md#engine-creation), [Configuration](configuration.md)).

## odds-lambda

`packages/odds-lambda/odds_lambda/` holds everything that collects, stores and schedules data. The name is historical: the same code runs under the local scheduler.

| Area | Modules | Page |
|---|---|---|
| Odds API | `packages/odds-lambda/odds_lambda/data_fetcher.py`, `packages/odds-lambda/odds_lambda/api_key_manager.py`, `packages/odds-lambda/odds_lambda/event_sync.py` | [Client and quota](../collection/odds-api-client-and-quota.md) |
| Ingestion | `packages/odds-lambda/odds_lambda/ingestion.py`, `odds_lambda.storage.validators` | [Odds API ingestion](../collection/odds-api-ingestion.md) |
| Scrapers | `packages/odds-lambda/odds_lambda/oddsportal_common.py`, `packages/odds-lambda/odds_lambda/oddsportal_adapter.py`, `packages/odds-lambda/odds_lambda/browser_lock.py` | [OddsPortal](../collection/oddsportal.md) |
| Exchange | `betfair/` | [Betfair Exchange](../collection/betfair-exchange.md) |
| Sport context fetchers | `packages/odds-lambda/odds_lambda/espn_fixture_fetcher.py`, `packages/odds-lambda/odds_lambda/mlb_stats_fetcher.py`, `packages/odds-lambda/odds_lambda/injury_fetcher.py`, `packages/odds-lambda/odds_lambda/game_log_fetcher.py`, `packages/odds-lambda/odds_lambda/pbpstats_fetcher.py`, `packages/odds-lambda/odds_lambda/polymarket_fetcher.py` | [EPL](../collection/epl-context-data.md), [MLB](../collection/mlb-probable-pitchers.md), [NBA](../collection/nba-injuries-and-stats.md), [Polymarket](../collection/polymarket.md) |
| Storage | `storage/` (one reader/writer pair per source) | [OddsReader and the storage layer](../storage/odds-reader-and-storage-layer.md) |
| Scheduling | `scheduling/` (`packages/odds-lambda/odds_lambda/scheduling/jobs.py`, `packages/odds-lambda/odds_lambda/scheduling/decision.py`, `packages/odds-lambda/odds_lambda/scheduling/helpers.py`, `backends/`), `packages/odds-lambda/odds_lambda/lambda_handler.py`, `packages/odds-lambda/odds_lambda/fetch_tier.py`, `packages/odds-lambda/odds_lambda/tier_utils.py` | [Jobs](../scheduling/jobs-and-deployment-matrix.md), [Self-scheduling](../scheduling/self-scheduling.md), [Fetch tier](../storage/fetch-tier.md) |
| Jobs | `jobs/*.py`, one `main(ctx)` per file | `_JOB_MODULE_MAP` lists all 14 |
| Services | `packages/odds-lambda/odds_lambda/paper_trading.py`, `packages/odds-lambda/odds_lambda/model_loader.py`, `packages/odds-lambda/odds_lambda/health_monitor.py`, `packages/odds-lambda/odds_lambda/copy_from_production.py` | [Paper trades](../agent/match-briefs-and-paper-trades.md), [model store](../modelling/model-publishing-and-scoring.md), [health monitor](../operations/alerts-and-health.md), [copy from production](databases-and-environments.md#copy-production-data-locally) |

`packages/odds-lambda/odds_lambda/__init__.py` re-exports `packages/odds-lambda/odds_lambda/data_fetcher.py::TheOddsAPIClient`, `packages/odds-lambda/odds_lambda/ingestion.py::OddsIngestionService`, `packages/odds-lambda/odds_lambda/fetch_tier.py::FetchTier`, `packages/odds-lambda/odds_lambda/tier_utils.py::calculate_tier` and `packages/odds-lambda/odds_lambda/tier_utils.py::calculate_hours_until_commence`. Job modules are imported lazily by name from `_JOB_MODULE_MAP`, so a scraper-only Lambda never imports the analytics stack. `packages/odds-lambda/odds_lambda/paper_trading.py` and `packages/odds-lambda/odds_lambda/copy_from_production.py` are service code for the CLI and MCP server rather than Lambda code.

## odds-analytics

`packages/odds-analytics/odds_analytics/` holds backtesting, strategies, feature extraction, model training and the backfill planner. It depends on core and odds-lambda (storage readers, `packages/odds-lambda/odds_lambda/fetch_tier.py::FetchTier`). Heavy dependencies (torch, mlflow, scikit-learn, optuna) sit behind the `training` extra; `packages/odds-analytics/odds_analytics/__init__.py` lazy-loads the modules that need them via `packages/odds-analytics/odds_analytics/__init__.py::__getattr__`.

| Area | Modules | Page |
|---|---|---|
| Backtesting | `backtesting/` (`config.py`, `models.py`, `packages/odds-analytics/odds_analytics/backtesting/services.py`), `packages/odds-analytics/odds_analytics/strategies.py` | [Backtesting](../evaluation/backtesting.md), [Strategies](../evaluation/backtest-strategies.md) |
| Features | `packages/odds-analytics/odds_analytics/feature_extraction.py`, `packages/odds-analytics/odds_analytics/feature_groups.py`, `packages/odds-analytics/odds_analytics/standings_features.py`, `packages/odds-analytics/odds_analytics/match_stats_features.py`, `packages/odds-analytics/odds_analytics/epl_schedule_features.py`, `packages/odds-analytics/odds_analytics/epl_lineup_features.py`, `packages/odds-analytics/odds_analytics/schedule_features.py`, `packages/odds-analytics/odds_analytics/injury_features.py`, `packages/odds-analytics/odds_analytics/polymarket_features.py` | [Feature pipeline](../modelling/feature-pipeline.md) |
| Training | `training/` (`config.py`, `packages/odds-analytics/odds_analytics/training/data_preparation.py`, `packages/odds-analytics/odds_analytics/training/cross_validation.py`, `packages/odds-analytics/odds_analytics/training/tuner.py`, `packages/odds-analytics/odds_analytics/training/feature_selection.py`, `packages/odds-analytics/odds_analytics/training/tracking.py`) | [Training pipeline](../modelling/training-pipeline.md) |
| Line-movement models | `packages/odds-analytics/odds_analytics/xgboost_line_movement.py`, `packages/odds-analytics/odds_analytics/lstm_line_movement.py`, `packages/odds-analytics/odds_analytics/sequence_loader.py` | [Line-movement models](../modelling/line-movement-models.md) |
| Backfill | `packages/odds-analytics/odds_analytics/gap_backfill_planner.py`, `packages/odds-analytics/odds_analytics/game_selector.py`, `packages/odds-analytics/odds_analytics/backfill_executor.py` | [Historical loads](../collection/howto-load-historical-odds.md) |
| Helpers | `utils.py` (EV, holds, arbitrage), `packages/odds-analytics/odds_analytics/quality_metrics.py` | |

Names that mislead:

- `packages/odds-analytics/odds_analytics/sequence_loader.py` is not LSTM-only: it owns all target maths (`packages/odds-analytics/odds_analytics/sequence_loader.py::calculate_devigged_bookmaker_target`, `packages/odds-analytics/odds_analytics/sequence_loader.py::DeviggedProbs`), the label for every model ([CLV target](../modelling/clv-target.md), [Devigging](odds-math-and-devigging.md#devigging)).
- `packages/odds-analytics/odds_analytics/quality_metrics.py` never touches `packages/odds-core/odds_core/models.py::DataQualityLog`; it counts game and odds coverage. Data-quality rows are a separate system ([Validation and coverage](../storage/data-validation.md)).
- `packages/odds-analytics/odds_analytics/ml_strategy_example.py` (430 lines) is tested infrastructure for an XGBoost win-probability classifier, not the CLV regression the system uses ([Line-movement models](../modelling/line-movement-models.md)).
- `packages/odds-analytics/odds_analytics/utils.py::per_book_market_holds` is the hold maths the MCP screener calls ([find_retail_edges](../agent/mcp-market-tools.md)); it is the only reason `odds-mcp` imports analytics ([odds-mcp server](../agent/mcp-server.md)).

## The odds CLI

`packages/odds-cli/odds_cli/main.py::app` is a Typer app installed as the `odds` script. Each group is a module in `packages/odds-cli/odds_cli/commands/` registered with `add_typer`. each command's `--help` is authoritative for flags and defaults. Commands are documented on the page that owns their domain; none is repeated here. The `packages/odds-cli/odds_cli/main.py::callback` configures human-readable logging ([Conventions](conventions.md#logging)).

| Group | Page |
|---|---|
| `odds fetch` | [Odds API ingestion](../collection/odds-api-ingestion.md), [Event lifecycle](../collection/event-lifecycle.md) |
| `scrape` | [OddsPortal](../collection/oddsportal.md) |
| `packages/odds-cli/odds_cli/commands/discover.py`, `packages/odds-cli/odds_cli/commands/backfill.py` | [Historical loads](../collection/howto-load-historical-odds.md) |
| `backtest` | [Backtesting](../evaluation/backtesting.md), [Strategies](../evaluation/backtest-strategies.md) |
| `train` | [Training pipeline](../modelling/training-pipeline.md), [Tuner](../modelling/hyperparameter-tuner.md) |
| `model` | [Publishing and scoring](../modelling/model-publishing-and-scoring.md) |
| `odds status` | [Reporting](../operations/reporting-and-daily-digest.md), [Odds API quota](../collection/odds-api-client-and-quota.md) |
| `validate`, `quality` | [Validation and coverage](../storage/data-validation.md) |
| `copy` | [Databases](databases-and-environments.md#copy-production-data-locally) |
| `scheduler` | [Local scheduler](../scheduling/local-scheduler.md) |
| `polymarket` | [Polymarket](../collection/polymarket.md) |
| `injuries`, `nba-stats`, `pbpstats` | [NBA feeds](../collection/nba-injuries-and-stats.md) |
| `paper` | [Paper trades](../agent/match-briefs-and-paper-trades.md) |
| `agent` | [Betting agent](../agent/betting-agent.md) |

Shared conventions:

- **Sport.** `--sport` takes a [sport key](../collection/sources-and-keys.md); defaults differ per group (Hazards).
- **Database override.** `scheduler start` and `agent run` accept `--db NAME`, which calls `packages/odds-cli/odds_cli/db_override.py::override_database_url`: it swaps the dbname in `DATABASE_URL`, writes `os.environ`, and clears the settings cache (`packages/odds-cli/odds_cli/db_override.py::swap_dbname` rejects names containing `/` or `?`). The engine already built in-process keeps the original URL; only subprocesses see the swap ([Configuration](configuration.md#hazards)). On `odds agent run` it has no effect at all when `AGENT_DATABASE_URL` is set ([Betting agent](../agent/betting-agent.md#database-role)).
- **Async.** Commands wrap async code in `asyncio.run()`.

## Hazards

- **Undeclared cycle.** `packages/odds-lambda/pyproject.toml` does not list `odds-analytics`, but `packages/odds-lambda/odds_lambda/model_loader.py` imports `odds_analytics.training.config` at module level and `packages/odds-lambda/odds_lambda/jobs/score_predictions.py` imports `odds_analytics.feature_groups`. Analytics declares a dependency on odds-lambda, so the cycle is undeclared on this side. The scraper image lacks odds-analytics and xgboost, so inline scoring inside `fetch-oddsportal` fails there and the failure is swallowed ([Publishing and scoring](../modelling/model-publishing-and-scoring.md)).
- **Reverse edge in core.** `packages/odds-core/odds_core/paper_trade_models.py::PaperTrade` imports `odds_analytics.backtesting.models` (type-checking plus a lazy import inside `to_bet_record`).
- `python-dotenv` is declared in core but nothing imports it; `.env` is read only by pydantic-settings ([Configuration](configuration.md#hazards)).
- `--sport` defaults to `basketball_nba` in `fetch current`, `fetch scores`, `odds discover` and `odds quality`; `copy from-prod` defaults to `soccer_epl`; `backtest` and `train` take an optional filter; `agent run` requires it. Pass it explicitly ([Project overview](project-overview.md#nba-assumptions-still-in-live-code)).
- Two exit styles coexist: `typer.Exit(code=...)` in most groups and `sys.exit(1)` in `validate` (10 sites) and `quality` (3).
- The root help text and callback docstring say "NBA". `backtest run --strategy` help lists `lstm`, which is not a key of `packages/odds-cli/odds_cli/commands/backtest.py::STRATEGIES`. `scheduler run-once` help names three of the 14 jobs, and `scheduler info` has no help text because its docstring follows a statement.
- `train validate` warns "Very small test split" for `test_split < 0.1`, although `test_split: 0.0` is the protocol ([Walk-forward CV](../modelling/decision-walk-forward-cv.md)).
- `copy from-prod` takes dates as positional arguments where `odds backfill` and `odds quality` use `--start`/`--end`.
- `odds agent run` imports the underscore-private `packages/odds-lambda/odds_lambda/jobs/agent_run.py::_run_claude_agent` from odds-lambda.

## Related

- [Project overview and pivots](project-overview.md)
- [Configuration](configuration.md)
