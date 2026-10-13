---
type: component
title: Packages and the odds CLI
description: Which of the five workspace packages holds what, the dependency rule and the undeclared odds-lambda to odds-analytics cycle, misleading module names, and the odds CLI's command groups, owning pages and shared flags.
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
  - odds_cli.main.app
  - odds_cli.main.callback
  - odds_cli.db_override.override_database_url
  - odds_cli.db_override.swap_dbname
  - odds_cli.commands.fetch
  - odds_cli.commands.backtest.STRATEGIES
  - odds_cli.commands.scheduler.run_once
  - odds_cli.commands.train
  - odds_cli.commands.validate
status: stable
---

# Packages and the odds CLI

The repo is a uv workspace of five packages under `packages/` ([AGENTS.md](../../AGENTS.md)): `odds-core` at the bottom, then `odds-lambda`, `odds-analytics`, and the two front ends `odds-mcp` and `odds-cli`. Dependencies point downward: odds-lambda depends on odds-core, odds-analytics on both, and odds-mcp and odds-cli on all three. The stated rule is that core stays light; two exceptions are under Hazards.

## odds-core

`packages/odds-core/odds_core/` declares only sqlmodel, asyncpg, pydantic-settings, python-dotenv, structlog and aiohttp (`packages/odds-core/pyproject.toml`).

| Module | Holds |
|---|---|
| `models.py` | `odds_core.models.Event`, `odds_core.models.OddsSnapshot`, `odds_core.models.Odds`, `odds_core.models.FetchLog`, `odds_core.models.DataQualityLog`, `odds_core.models.AlertHistory`, `odds_core.models.utc_now` ([events](../storage/events.md), [odds storage](../storage/odds-storage.md)) |
| `odds_core.prediction_models`, `odds_core.match_brief_models`, `odds_core.paper_trade_models`, `odds_core.agent_wakeup_models` | Prediction and agent tables ([briefs and paper trades](../agent/match-briefs-and-paper-trades.md)) |
| `odds_core.epl_data_models`, `odds_core.mlb_data_models`, `odds_core.injury_models`, `odds_core.game_log_models`, `odds_core.player_stats_models`, `odds_core.polymarket_models` | Sport-context and Polymarket tables |
| `config.py` | Pydantic Settings, see [Configuration](configuration.md) |
| `odds_core.database` | Engine and `async_session_maker`, see [Databases](databases-and-environments.md) |
| `odds_core.alerts` | `odds_core.alerts.AlertManager`, `odds_core.alerts.DiscordAlert`, see [Alerts and health](../operations/alerts-and-health.md) |
| `odds_core.odds_math` | Price conversion, implied probability, devig, see [Odds arithmetic](odds-math-and-devigging.md) |
| `odds_core.snapshot_utils`, `utils.py` | `odds_core.snapshot_utils.extract_odds_from_snapshot` (parse `raw_data` into `odds_core.models.Odds`), `odds_core.utils.raw_data_has_market` |
| `odds_core.sports`, `odds_core.team`, `odds_core.time`, `odds_core.logging_setup` | [Sport keys](../collection/sources-and-keys.md), team-name normalization, [timezone helpers](conventions.md#timezones), structlog setup |
| `odds_core.api_models` | The Odds API response models and event factories |

`odds_core.snapshot_utils.extract_odds_from_snapshot` matches market keys exactly, so `h2h` and `1x2` snapshots are different markets to every reader that uses it.

Importing any `odds_core` submodule runs `odds_core.database`, which builds the engine from `odds_core.config.get_settings()`. Every `odds` command therefore needs `ODDS_API_KEY` and `DATABASE_URL` to resolve, even commands that never touch the API or database ([Databases](databases-and-environments.md#engine-creation), [Configuration](configuration.md)).

## odds-lambda

`packages/odds-lambda/odds_lambda/` holds everything that collects, stores and schedules data. The name is historical: the same code runs under the local scheduler.

| Area | Modules | Page |
|---|---|---|
| Odds API | `odds_lambda.data_fetcher`, `odds_lambda.api_key_manager`, `odds_lambda.event_sync` | [Client and quota](../collection/odds-api-client-and-quota.md) |
| Ingestion | `odds_lambda.ingestion`, `odds_lambda.storage.validators` | [Odds API ingestion](../collection/odds-api-ingestion.md) |
| Scrapers | `odds_lambda.oddsportal_common`, `odds_lambda.oddsportal_adapter`, `odds_lambda.browser_lock` | [OddsPortal](../collection/oddsportal.md) |
| Exchange | `betfair/` | [Betfair Exchange](../collection/betfair-exchange.md) |
| Sport context fetchers | `odds_lambda.espn_fixture_fetcher`, `odds_lambda.mlb_stats_fetcher`, `odds_lambda.injury_fetcher`, `odds_lambda.game_log_fetcher`, `odds_lambda.pbpstats_fetcher`, `odds_lambda.polymarket_fetcher` | [EPL](../collection/epl-context-data.md), [MLB](../collection/mlb-probable-pitchers.md), [NBA](../collection/nba-injuries-and-stats.md), [Polymarket](../collection/polymarket.md) |
| Storage | `storage/` (one reader/writer pair per source) | [OddsReader and the storage layer](../storage/odds-reader-and-storage-layer.md) |
| Scheduling | `scheduling/` (`odds_lambda.scheduling.jobs`, `odds_lambda.scheduling.decision`, `odds_lambda.scheduling.helpers`, `backends/`), `odds_lambda.lambda_handler`, `odds_lambda.fetch_tier`, `odds_lambda.tier_utils` | [Jobs](../scheduling/jobs-and-deployment-matrix.md), [Self-scheduling](../scheduling/self-scheduling.md), [Fetch tier](../storage/fetch-tier.md) |
| Jobs | `jobs/*.py`, one `main(ctx)` per file | `_JOB_MODULE_MAP` lists all 14 |
| Services | `odds_lambda.paper_trading`, `odds_lambda.model_loader`, `odds_lambda.health_monitor`, `odds_lambda.copy_from_production` | [Paper trades](../agent/match-briefs-and-paper-trades.md), [model store](../modelling/model-publishing-and-scoring.md), [health monitor](../operations/alerts-and-health.md), [copy from production](databases-and-environments.md#copy-production-data-locally) |

`odds_lambda` re-exports `odds_lambda.data_fetcher.TheOddsAPIClient`, `odds_lambda.ingestion.OddsIngestionService`, `odds_lambda.fetch_tier.FetchTier`, `odds_lambda.tier_utils.calculate_tier` and `odds_lambda.tier_utils.calculate_hours_until_commence`. Job modules are imported lazily by name from `_JOB_MODULE_MAP`, so a scraper-only Lambda never imports the analytics stack. `odds_lambda.paper_trading` and `odds_lambda.copy_from_production` are service code for the CLI and MCP server rather than Lambda code.

## odds-analytics

`packages/odds-analytics/odds_analytics/` holds backtesting, strategies, feature extraction, model training and the backfill planner. It depends on core and odds-lambda (storage readers, `odds_lambda.fetch_tier.FetchTier`). Heavy dependencies (torch, mlflow, scikit-learn, optuna) sit behind the `training` extra; `odds_analytics` lazy-loads the modules that need them via `odds_analytics.__getattr__`.

| Area | Modules | Page |
|---|---|---|
| Backtesting | `backtesting/` (`config.py`, `models.py`, `odds_analytics.backtesting.services`), `odds_analytics.strategies` | [Backtesting](../evaluation/backtesting.md), [Strategies](../evaluation/backtest-strategies.md) |
| Features | `odds_analytics.feature_extraction`, `odds_analytics.feature_groups`, `odds_analytics.standings_features`, `odds_analytics.match_stats_features`, `odds_analytics.epl_schedule_features`, `odds_analytics.epl_lineup_features`, `odds_analytics.schedule_features`, `odds_analytics.injury_features`, `odds_analytics.polymarket_features` | [Feature pipeline](../modelling/feature-pipeline.md) |
| Training | `training/` (`config.py`, `odds_analytics.training.data_preparation`, `odds_analytics.training.cross_validation`, `odds_analytics.training.tuner`, `odds_analytics.training.feature_selection`, `odds_analytics.training.tracking`) | [Training pipeline](../modelling/training-pipeline.md) |
| Line-movement models | `odds_analytics.xgboost_line_movement`, `odds_analytics.lstm_line_movement`, `odds_analytics.sequence_loader` | [Line-movement models](../modelling/line-movement-models.md) |
| Backfill | `odds_analytics.gap_backfill_planner`, `odds_analytics.game_selector`, `odds_analytics.backfill_executor` | [Historical loads](../collection/howto-load-historical-odds.md) |
| Helpers | `utils.py` (EV, holds, arbitrage), `odds_analytics.quality_metrics` | |

Names that mislead:

- `odds_analytics.sequence_loader` is not LSTM-only: it owns all target maths (`odds_analytics.sequence_loader.calculate_devigged_bookmaker_target`, `odds_analytics.sequence_loader.DeviggedProbs`), the label for every model ([CLV target](../modelling/clv-target.md), [Devigging](odds-math-and-devigging.md#devigging)).
- `odds_analytics.quality_metrics` never touches `odds_core.models.DataQualityLog`; it counts game and odds coverage. Data-quality rows are a separate system ([Validation and coverage](../storage/data-validation.md)).
- `odds_analytics.ml_strategy_example` (430 lines) is tested infrastructure for an XGBoost win-probability classifier, not the CLV regression the system uses ([Line-movement models](../modelling/line-movement-models.md)).
- `odds_analytics.utils.per_book_market_holds` is the hold maths the MCP screener calls ([find_retail_edges](../agent/mcp-market-tools.md)); it is the only reason `odds-mcp` imports analytics ([odds-mcp server](../agent/mcp-server.md)).

## The odds CLI

`odds_cli.main.app` is a Typer app installed as the `odds` script. Each group is a module in `packages/odds-cli/odds_cli/commands/` registered with `add_typer`. each command's `--help` is authoritative for flags and defaults. Commands are documented on the page that owns their domain; none is repeated here. The `odds_cli.main.callback` configures human-readable logging ([Conventions](conventions.md#logging)).

| Group | Page |
|---|---|
| `odds fetch` | [Odds API ingestion](../collection/odds-api-ingestion.md), [Event lifecycle](../collection/event-lifecycle.md) |
| `scrape` | [OddsPortal](../collection/oddsportal.md) |
| `odds_cli.commands.discover`, `odds_cli.commands.backfill` | [Historical loads](../collection/howto-load-historical-odds.md) |
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
- **Database override.** `scheduler start` and `agent run` accept `--db NAME`, which calls `odds_cli.db_override.override_database_url`: it swaps the dbname in `DATABASE_URL`, writes `os.environ`, and clears the settings cache (`odds_cli.db_override.swap_dbname` rejects names containing `/` or `?`). The engine already built in-process keeps the original URL; only subprocesses see the swap ([Configuration](configuration.md#hazards)). On `odds agent run` it has no effect at all when `AGENT_DATABASE_URL` is set ([Betting agent](../agent/betting-agent.md#database-role)).
- **Async.** Commands wrap async code in `asyncio.run()`.

## Hazards

- **Undeclared cycle.** `packages/odds-lambda/pyproject.toml` does not list `odds-analytics`, but `odds_lambda.model_loader` imports `odds_analytics.training.config` at module level and `odds_lambda.jobs.score_predictions` imports `odds_analytics.feature_groups`. Analytics declares a dependency on odds-lambda, so the cycle is undeclared on this side. The scraper image lacks odds-analytics and xgboost, so inline scoring inside `fetch-oddsportal` fails there and the failure is swallowed ([Publishing and scoring](../modelling/model-publishing-and-scoring.md)).
- **Reverse edge in core.** `odds_core.paper_trade_models.PaperTrade` imports `odds_analytics.backtesting.models` (type-checking plus a lazy import inside `to_bet_record`).
- `python-dotenv` is declared in core but nothing imports it; `.env` is read only by pydantic-settings ([Configuration](configuration.md#hazards)).
- `--sport` defaults to `basketball_nba` in `fetch current`, `fetch scores`, `odds discover` and `odds quality`; `copy from-prod` defaults to `soccer_epl`; `backtest` and `train` take an optional filter; `agent run` requires it. Pass it explicitly ([Project overview](project-overview.md#nba-assumptions-still-in-live-code)).
- Two exit styles coexist: `typer.Exit(code=...)` in most groups and `sys.exit(1)` in `validate` (10 sites) and `quality` (3).
- The root help text and callback docstring say "NBA". `backtest run --strategy` help lists `lstm`, which is not a key of `odds_cli.commands.backtest.STRATEGIES`. `scheduler run-once` help names three of the 14 jobs, and `scheduler info` has no help text because its docstring follows a statement.
- `train validate` warns "Very small test split" for `test_split < 0.1`, although `test_split: 0.0` is the protocol ([Walk-forward CV](../modelling/decision-walk-forward-cv.md)).
- `copy from-prod` takes dates as positional arguments where `odds backfill` and `odds quality` use `--start`/`--end`.
- `odds agent run` imports the underscore-private `odds_lambda.jobs.agent_run._run_claude_agent` from odds-lambda.

## Related

- [Project overview and pivots](project-overview.md)
- [Configuration](configuration.md)
