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
  - docs/CLI.md#command-groups
verified: "7887039"
status: stable
---

# Packages and the odds CLI

The repo is a uv workspace of five packages under `packages/` ([AGENTS.md](../../AGENTS.md#package-structure)): `odds-core` at the bottom, then `odds-lambda`, `odds-analytics`, and the two front ends `odds-mcp` and `odds-cli`. Dependencies point downward: odds-lambda depends on odds-core, odds-analytics on both, and odds-mcp and odds-cli on all three. The stated rule is that core stays light; two exceptions are under Hazards.

## odds-core

`packages/odds-core/odds_core/` declares only sqlmodel, asyncpg, pydantic-settings, python-dotenv, structlog and aiohttp (`packages/odds-core/pyproject.toml`).

| Module | Holds |
|---|---|
| `models.py` | `Event`, `OddsSnapshot`, `Odds`, `FetchLog`, `DataQualityLog`, `AlertHistory`, `utc_now` ([events](../storage/events.md), [odds storage](../storage/odds-storage.md)) |
| `prediction_models.py`, `match_brief_models.py`, `paper_trade_models.py`, `agent_wakeup_models.py` | Prediction and agent tables ([briefs and paper trades](../agent/match-briefs-and-paper-trades.md)) |
| `epl_data_models.py`, `mlb_data_models.py`, `injury_models.py`, `game_log_models.py`, `player_stats_models.py`, `polymarket_models.py` | Sport-context and Polymarket tables |
| `config.py` | Pydantic Settings, see [Configuration](configuration.md) |
| `database.py` | Engine and `async_session_maker`, see [Databases](databases-and-environments.md) |
| `alerts.py` | `AlertManager`, `DiscordAlert`, see [Alerts and health](../operations/alerts-and-health.md) |
| `odds_math.py` | Price conversion, implied probability, devig, see [Odds arithmetic](odds-math-and-devigging.md) |
| `snapshot_utils.py`, `utils.py` | `extract_odds_from_snapshot` (parse `raw_data` into `Odds`), `raw_data_has_market` |
| `sports.py`, `team.py`, `time.py`, `logging_setup.py` | [Sport keys](../collection/sources-and-keys.md), team-name normalization, [timezone helpers](conventions.md#timezones), structlog setup |
| `api_models.py` | The Odds API response models and event factories |

`extract_odds_from_snapshot` matches market keys exactly, so `h2h` and `1x2` snapshots are different markets to every reader that uses it.

Importing any `odds_core` submodule runs `odds_core.database`, which builds the engine from `get_settings()`. Every `odds` command therefore needs `ODDS_API_KEY` and `DATABASE_URL` to resolve, even commands that never touch the API or database ([Databases](databases-and-environments.md#engine-creation), [Configuration](configuration.md)).

## odds-lambda

`packages/odds-lambda/odds_lambda/` holds everything that collects, stores and schedules data. The name is historical: the same code runs under the local scheduler.

| Area | Modules | Page |
|---|---|---|
| Odds API | `data_fetcher.py`, `api_key_manager.py`, `event_sync.py` | [Client and quota](../collection/odds-api-client-and-quota.md) |
| Ingestion | `ingestion.py`, `odds_lambda.storage.validators` | [Odds API ingestion](../collection/odds-api-ingestion.md) |
| Scrapers | `oddsportal_common.py`, `oddsportal_adapter.py`, `browser_lock.py` | [OddsPortal](../collection/oddsportal.md) |
| Exchange | `betfair/` | [Betfair Exchange](../collection/betfair-exchange.md) |
| Sport context fetchers | `espn_fixture_fetcher.py`, `mlb_stats_fetcher.py`, `injury_fetcher.py`, `game_log_fetcher.py`, `pbpstats_fetcher.py`, `polymarket_fetcher.py` | [EPL](../collection/epl-context-data.md), [MLB](../collection/mlb-probable-pitchers.md), [NBA](../collection/nba-injuries-and-stats.md), [Polymarket](../collection/polymarket.md) |
| Storage | `storage/` (one reader/writer pair per source) | [OddsReader and the storage layer](../storage/odds-reader-and-storage-layer.md) |
| Scheduling | `scheduling/` (`jobs.py`, `decision.py`, `helpers.py`, `backends/`), `lambda_handler.py`, `fetch_tier.py`, `tier_utils.py` | [Jobs](../scheduling/jobs-and-deployment-matrix.md), [Self-scheduling](../scheduling/self-scheduling.md), [Fetch tier](../storage/fetch-tier.md) |
| Jobs | `jobs/*.py`, one `main` per file | `_JOB_MODULE_MAP` lists all 14 |
| Services | `paper_trading.py`, `model_loader.py`, `health_monitor.py`, `copy_from_production.py` | [Paper trades](../agent/match-briefs-and-paper-trades.md), [model store](../modelling/model-publishing-and-scoring.md), [health monitor](../operations/alerts-and-health.md), [copy from production](databases-and-environments.md#copy-production-data-locally) |

`packages/odds-lambda/odds_lambda/__init__.py` re-exports `TheOddsAPIClient`, `OddsIngestionService`, `FetchTier`, `calculate_tier` and `calculate_hours_until_commence`. Job modules are imported lazily by name from `_JOB_MODULE_MAP`, so a scraper-only Lambda never imports the analytics stack. `paper_trading.py` and `copy_from_production.py` are service code for the CLI and MCP server rather than Lambda code.

## odds-analytics

`packages/odds-analytics/odds_analytics/` holds backtesting, strategies, feature extraction, model training and the backfill planner. It depends on core and odds-lambda (storage readers, `FetchTier`). Heavy dependencies (torch, mlflow, scikit-learn, optuna) sit behind the `training` extra; `packages/odds-analytics/odds_analytics/__init__.py` lazy-loads the modules that need them via `__getattr__`.

| Area | Modules | Page |
|---|---|---|
| Backtesting | `backtesting/` (`config.py`, `models.py`, `services.py`), `strategies.py` | [Backtesting](../evaluation/backtesting.md), [Strategies](../evaluation/backtest-strategies.md) |
| Features | `feature_extraction.py`, `feature_groups.py`, `standings_features.py`, `match_stats_features.py`, `epl_schedule_features.py`, `epl_lineup_features.py`, `schedule_features.py`, `injury_features.py`, `polymarket_features.py` | [Feature pipeline](../modelling/feature-pipeline.md) |
| Training | `training/` (`config.py`, `data_preparation.py`, `cross_validation.py`, `tuner.py`, `feature_selection.py`, `tracking.py`) | [Training pipeline](../modelling/training-pipeline.md) |
| Line-movement models | `xgboost_line_movement.py`, `lstm_line_movement.py`, `sequence_loader.py` | [Line-movement models](../modelling/line-movement-models.md) |
| Backfill | `gap_backfill_planner.py`, `game_selector.py`, `backfill_executor.py` | [Historical loads](../collection/howto-load-historical-odds.md) |
| Helpers | `utils.py` (EV, holds, arbitrage), `quality_metrics.py` | |

Names that mislead:

- `sequence_loader.py` is not LSTM-only: it owns all target maths (`calculate_devigged_bookmaker_target`, `DeviggedProbs`), the label for every model ([CLV target](../modelling/clv-target.md), [Devigging](odds-math-and-devigging.md#devigging)).
- `quality_metrics.py` never touches `DataQualityLog`; it counts game and odds coverage. Data-quality rows are a separate system ([Validation and coverage](../storage/data-validation.md)).
- `ml_strategy_example.py` (430 lines) is tested infrastructure for an XGBoost win-probability classifier, not the CLV regression the system uses ([Line-movement models](../modelling/line-movement-models.md)).
- `utils.py::per_book_market_holds` is the hold maths the MCP screener calls ([find_retail_edges](../agent/mcp-market-tools.md)); it is the only reason `odds-mcp` imports analytics ([odds-mcp server](../agent/mcp-server.md)).

## The odds CLI

`packages/odds-cli/odds_cli/main.py::app` is a Typer app installed as the `odds` script. Each group is a module in `packages/odds-cli/odds_cli/commands/` registered with `add_typer`. `docs/CLI.md` is the reference; each command's `--help` is authoritative for flags and defaults. Commands are documented on the page that owns their domain; none is repeated here. The `callback` configures human-readable logging ([Conventions](conventions.md#logging)).

| Group | Page |
|---|---|
| `fetch` | [Odds API ingestion](../collection/odds-api-ingestion.md), [Event lifecycle](../collection/event-lifecycle.md) |
| `scrape` | [OddsPortal](../collection/oddsportal.md) |
| `discover`, `backfill` | [Historical loads](../collection/howto-load-historical-odds.md) |
| `backtest` | [Backtesting](../evaluation/backtesting.md), [Strategies](../evaluation/backtest-strategies.md) |
| `train` | [Training pipeline](../modelling/training-pipeline.md), [Tuner](../modelling/hyperparameter-tuner.md) |
| `model` | [Publishing and scoring](../modelling/model-publishing-and-scoring.md) |
| `status` | [Reporting](../operations/reporting-and-daily-digest.md), [Odds API quota](../collection/odds-api-client-and-quota.md) |
| `validate`, `quality` | [Validation and coverage](../storage/data-validation.md) |
| `copy` | [Databases](databases-and-environments.md#copy-production-data-locally) |
| `scheduler` | [Local scheduler](../scheduling/local-scheduler.md) |
| `polymarket` | [Polymarket](../collection/polymarket.md) |
| `injuries`, `nba-stats`, `pbpstats` | [NBA feeds](../collection/nba-injuries-and-stats.md) |
| `paper` | [Paper trades](../agent/match-briefs-and-paper-trades.md) |
| `agent` | [Betting agent](../agent/betting-agent.md) |

Shared conventions:

- **Sport.** `--sport` takes a [sport key](../collection/sources-and-keys.md); defaults differ per group (Hazards).
- **Database override.** `scheduler start` and `agent run` accept `--db NAME`, which calls `override_database_url`: it swaps the dbname in `DATABASE_URL`, writes `os.environ`, and clears the settings cache (`swap_dbname` rejects names containing `/` or `?`). The engine already built in-process keeps the original URL; only subprocesses see the swap ([Configuration](configuration.md#hazards)). On `odds agent run` it has no effect at all when `AGENT_DATABASE_URL` is set ([Betting agent](../agent/betting-agent.md#database-role)).
- **Async.** Commands wrap async code in `asyncio.run()`.

## Hazards

- **Undeclared cycle.** `packages/odds-lambda/pyproject.toml` does not list `odds-analytics`, but `model_loader.py` imports `odds_analytics.training.config` at module level and `packages/odds-lambda/odds_lambda/jobs/score_predictions.py` imports `odds_analytics.feature_groups`. Analytics declares a dependency on odds-lambda, so the cycle is undeclared on this side. The scraper image lacks odds-analytics and xgboost, so inline scoring inside `fetch-oddsportal` fails there and the failure is swallowed ([Publishing and scoring](../modelling/model-publishing-and-scoring.md)).
- **Reverse edge in core.** `packages/odds-core/odds_core/paper_trade_models.py::PaperTrade` imports `odds_analytics.backtesting.models` (type-checking plus a lazy import inside `to_bet_record`).
- `python-dotenv` is declared in core but nothing imports it; `.env` is read only by pydantic-settings ([Configuration](configuration.md#hazards)).
- `--sport` defaults to `basketball_nba` in `fetch current`, `fetch scores`, `discover` and `quality`; `copy from-prod` defaults to `soccer_epl`; `backtest` and `train` take an optional filter; `agent run` requires it. Pass it explicitly ([Project overview](project-overview.md#nba-assumptions-still-in-live-code)).
- Two exit styles coexist: `typer.Exit(code=...)` in most groups and `sys.exit(1)` in `validate` (10 sites) and `quality` (3).
- The root help text and callback docstring say "NBA". `backtest run --strategy` help lists `lstm`, which is not a key of `STRATEGIES`. `scheduler run-once` help names three of the 14 jobs, and `scheduler info` has no help text because its docstring follows a statement.
- `train validate` warns "Very small test split" for `test_split < 0.1`, although `test_split: 0.0` is the protocol ([Walk-forward CV](../modelling/decision-walk-forward-cv.md)).
- `copy from-prod` takes dates as positional arguments where `backfill` and `quality` use `--start`/`--end`.
- `odds agent run` imports the underscore-private `_run_claude_agent` from odds-lambda.

## Related

- [Project overview and pivots](project-overview.md)
- [Configuration](configuration.md)
