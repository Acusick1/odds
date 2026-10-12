---
type: component
title: odds-analytics package
summary: "Module map for odds-analytics (backtesting, strategies, features, training, backfill planning) and the modules whose names mislead."
sources:
  - packages/odds-analytics/odds_analytics/__init__.py
  - packages/odds-analytics/odds_analytics/sequence_loader.py::calculate_devigged_bookmaker_target
  - packages/odds-analytics/odds_analytics/quality_metrics.py
  - packages/odds-analytics/odds_analytics/ml_strategy_example.py
  - packages/odds-analytics/odds_analytics/utils.py::per_book_market_holds
  - packages/odds-analytics/pyproject.toml
verified_at: 7887039
---

# odds-analytics package

`packages/odds-analytics/odds_analytics/` holds backtesting, betting strategies, feature extraction, model training and the historical-backfill planner. It depends on [odds-core](package-odds-core.md) and [odds-lambda](package-odds-lambda.md) (for storage readers and `FetchTier`); `odds-mcp` and `odds-cli` depend on it. Heavy dependencies (torch, mlflow, scikit-learn, optuna) sit behind the `training` extra, and `packages/odds-analytics/odds_analytics/__init__.py` lazy-loads the modules that need them via `__getattr__`.

## Module map

| Area | Modules | Page |
|---|---|---|
| Backtesting | `backtesting/` (`config.py`, `models.py`, `services.py`), `strategies.py` | [Backtest engine](backtest-engine.md), [Strategies](backtest-strategies.md), [Metrics](backtest-metrics.md) |
| Features | `feature_extraction.py`, `feature_groups.py`, `standings_features.py`, `match_stats_features.py`, `epl_schedule_features.py`, `epl_lineup_features.py`, `schedule_features.py`, `injury_features.py`, `polymarket_features.py` | [Feature pipeline](feature-pipeline.md), [Feature groups](feature-groups.md) |
| Training | `training/` (`config.py`, `data_preparation.py`, `cross_validation.py`, `tuner.py`, `feature_selection.py`, `tracking.py`) | [Training pipeline](training-pipeline.md), [Training config](training-config.md) |
| Line-movement models | `xgboost_line_movement.py`, `lstm_line_movement.py`, `sequence_loader.py` | [Line-movement models](line-movement-models.md) |
| Backfill | `gap_backfill_planner.py`, `game_selector.py`, `backfill_executor.py` | [Historical backfill](howto-historical-backfill.md) |
| Helpers | `utils.py` (EV, holds, arbitrage), `quality_metrics.py` | |

## Names that mislead

- `sequence_loader.py` is not LSTM-only: it owns all target maths (`calculate_devigged_bookmaker_target`, `DeviggedProbs`), the label for every model. See [CLV](clv.md) and [Devigging](devigging.md).
- `quality_metrics.py` never touches `DataQualityLog`; it counts game and odds coverage ([Completeness validation](completeness-validation.md)). Data-quality rows are [Odds validation](data-quality-validation.md).
- `ml_strategy_example.py` (430 lines) is tested infrastructure for an XGBoost win-probability classifier, not the CLV regression the system uses ([Line-movement models](line-movement-models.md)).
- `utils.py::per_book_market_holds` is the hold maths the MCP screener calls ([find_retail_edges](mcp-find-retail-edges.md)); it is the reason `odds-mcp` needs this whole package.

## Hazards

- Several modules assume NBA defaults, see [Project pivots](project-pivots.md).

## Related

- [Project overview](project-overview.md) — repo layout
- [odds-lambda package](package-odds-lambda.md) — the layer below
