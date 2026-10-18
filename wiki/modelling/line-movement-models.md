---
type: reference
title: Line-movement strategies (XGBoost, LSTM)
description: The two strategy classes, their shared confidence mapping, defaults, save and load, and ml_strategy_example.
sources:
  - odds_analytics.xgboost_line_movement.XGBoostLineMovementStrategy
  - odds_analytics.lstm_line_movement.LSTMLineMovementStrategy
  - odds_analytics.ml_strategy_example.XGBoostStrategy
  - odds_analytics.sequence_loader.load_sequences_for_event
  - odds_analytics.training.config.TrainingConfig
status: stable
---

# Line-movement strategies (XGBoost, LSTM)

Two `odds_analytics.backtesting.services.BettingStrategy` subclasses wrap a trained CLV regressor so the [backtest engine](../evaluation/backtest-engine.md) can bet on it; they are also the trainers behind `odds train` ([training pipeline](training-pipeline.md)). Registry names are `xgb_line_movement` and `lstm_line_movement` ([backtest strategies](../evaluation/backtest-strategies.md)); config `strategy_type` accepts only `xgboost_line_movement` and `lstm_line_movement`.

## Shared behaviour

`odds_analytics.backtesting.services.BettingStrategy.evaluate_opportunity` implementations predict a movement delta for the home and away outcome, skips any below `min_predicted_movement` (0.02), takes the best price among its bookmakers, and sets `confidence = base_confidence + (predicted - min) * movement_confidence_scale`, clamped to 0.5-0.95 (defaults 0.52 and 5.0). The engine passes `confidence` to Kelly sizing as the win probability, and zero or negative Kelly is floored to `min_bet_size` ([backtest engine](../evaluation/backtest-engine.md)). Draws are never bet.

- **XGBoost** (`odds_analytics.xgboost_line_movement`). `odds_analytics.xgboost_line_movement.XGBoostLineMovementStrategy.train` fits an `XGBRegressor` with the train set and validation set as `eval_set` (metrics and pruning only; there is no early stopping, so all `n_estimators` trees are built); `odds_analytics.xgboost_line_movement.XGBoostLineMovementStrategy.train_from_config` requires `strategy_type == "xgboost_line_movement"` and an `odds_analytics.training.config.XGBoostConfig`, and adds an Optuna pruning callback when given a trial. `odds_analytics.xgboost_line_movement.XGBoostLineMovementStrategy.save_model` writes joblib `{model, feature_names, params}` plus a `.yaml` of metadata beside it; `odds_analytics.xgboost_line_movement.XGBoostLineMovementStrategy.load_model` falls back to pickle when joblib fails.
- **LSTM** (`odds_analytics.lstm_line_movement`). Needs torch (training extra). Defaults: 72 h lookback, 24 timesteps, hidden 64, 2 layers, dropout 0.2, MSE loss. Inputs come from `odds_analytics.sequence_loader.load_sequences_for_event` through `odds_analytics.feature_extraction.SequenceFeatureExtractor`, whose sharp features need a target outcome ([feature pipeline](feature-pipeline.md)); packed sequences handle padding (`odds_analytics.lstm_line_movement._compact_sequences`, the 2026-02-27 mask fix, #162). The module duplicates the XGBoost training scaffold. It is selectable through `strategy_type: lstm_line_movement`; its results are on [Early NBA signal search and LSTM](../evaluation/exp-early-nba-signal-and-lstm.md#lstm-sequence-models), and neither LSTM config sets the required `target_type`.
- **`odds_analytics.ml_strategy_example`.** `odds_analytics.ml_strategy_example.XGBoostStrategy` is an XGBoost win-probability classifier kept as a reference implementation, with no config-driven training entry point; it is not in any registry and is referenced only by tests, so it is not the model strategy.

## Backtest behaviour

- **Feature vector.** The XGBoost strategy builds `odds_analytics.feature_extraction.TabularFeatures.to_array` (6 values). Models trained from the arrays that `odds_analytics.feature_groups.prepare_training_data` builds add `hours_until_event` (from the XGBoost adapter) and any other configured groups, so a model trained through it receives a different width (#476; read from code, not run). Model strategies need `--model-path`.
- **Market and bookmakers.** Both strategies default to `market="h2h"`, while EPL odds use `1x2`; the CLI constructs them with defaults and cannot override ([market keys](../collection/sport-and-market-keys.md#market-key-dialects), #465). XGBoost defaults to eight bookmakers (Pinnacle plus seven US books); the extractor's sharp and retail rosters default to `pinnacle` and `fanduel/draftkings/betmgm`, ignoring the hybrid sharp reference (#465; [sharp benchmark](sharp-benchmark.md#code-copies)).
- **Snapshots.** LSTM backtests load all snapshots for the event; the cutoff arithmetic is under [look-ahead bias](look-ahead-bias.md#where-the-rule-breaks).
- **Decision time.** The model learns from 3-24 h snapshots while backtests decide 1 h out, so a model-driven backtest feeds it inputs unlike its training rows (same page).

## Related

- [CLV target](clv-target.md), [Add a backtest strategy](../evaluation/howto-add-backtest-strategy.md)
