---
type: component
title: Line-movement model strategies (XGBoost, LSTM)
summary: XGBoostLineMovementStrategy and LSTMLineMovementStrategy - training, prediction and evaluate_opportunity, Kelly-confidence parameters, LSTM status, and ml_strategy_example.
sources:
  - packages/odds-analytics/odds_analytics/xgboost_line_movement.py::XGBoostLineMovementStrategy
  - packages/odds-analytics/odds_analytics/lstm_line_movement.py::LSTMLineMovementStrategy
  - packages/odds-analytics/odds_analytics/ml_strategy_example.py::XGBoostStrategy
  - packages/odds-analytics/odds_analytics/sequence_loader.py::load_sequences_for_event
  - packages/odds-analytics/odds_analytics/training/config.py::TrainingConfig
  - docs/BACKTESTING_GUIDE.md#model-driven-strategies
  - docs/MODELING.md#lstm-sequence-models
verified_at: 7887039
---

# Line-movement model strategies (XGBoost, LSTM)

Two `BettingStrategy` subclasses wrap a trained CLV regressor so the [backtest engine](backtest-engine.md) can bet on it; they are also the trainers behind `odds train` ([training-pipeline](training-pipeline.md)). Registry names are `xgb_line_movement` and `lstm_line_movement` ([backtest-strategies](backtest-strategies.md)); config `strategy_type` values are `xgboost_line_movement` and `lstm_line_movement`.

## Shared behaviour

`evaluate_opportunity` predicts a movement delta for the home and away outcome, skips any below `min_predicted_movement` (0.02), takes the best price among its bookmakers, and sets `confidence = base_confidence + (predicted - min) * movement_confidence_scale`, clamped to 0.5-0.95 (defaults 0.52 and 5.0). The engine passes `confidence` to Kelly sizing as the win probability ([backtest-engine](backtest-engine.md)). Draws are never bet.

- **XGBoost** (`packages/odds-analytics/odds_analytics/xgboost_line_movement.py`). `train` fits an `XGBRegressor` with the train set and validation set as `eval_set`; `train_from_config` requires `strategy_type == "xgboost_line_movement"` and an `XGBoostConfig`, and adds an Optuna pruning callback when given a trial. `save_model` writes joblib `{model, feature_names, params}` plus a `.yaml` of metadata beside it; `load_model` falls back to pickle.
- **LSTM** (`packages/odds-analytics/odds_analytics/lstm_line_movement.py`). Needs torch (training extra). Defaults: 72 h lookback, 24 timesteps, hidden 64, 2 layers, dropout 0.2, MSE loss. Inputs come from `load_sequences_for_event` through `SequenceFeatureExtractor` ([feature-groups](feature-groups.md)); packed sequences handle padding. **Status: ruled out.** All variants were worse than predicting the mean ([exp-lstm-sequence-models](exp-lstm-sequence-models.md)); the module docstring does not say so and duplicates the XGBoost training scaffold (AUD B5, re-verified).
- **`ml_strategy_example.py`.** `XGBoostStrategy` is an XGBoost win-probability classifier kept as a reference implementation; it is not in any registry and is referenced only by tests.

## Hazards

- **Backtest vectors may not match trained models.** The XGBoost strategy builds `TabularFeatures.to_array()` (6 values). Models trained by `prepare_training_data` add `hours_until_event` and any other configured groups, so a trained model would receive a different width (not run; check before relying on `--model-path`).
- **EPL needs a market override.** Both strategies default to `market="h2h"`, while EPL odds use `1x2`; the CLI constructs them with defaults ([sport-and-market-keys](sport-and-market-keys.md)).
- **Bookmaker roster is US-flavoured.** XGBoost defaults to eight bookmakers (Pinnacle plus seven US books); the extractor's sharp and retail rosters default to `pinnacle` and `fanduel/draftkings/betmgm`, ignoring the hybrid sharp reference ([sharp-benchmark](sharp-benchmark.md)).
- **LSTM backtests load all snapshots** for the event; see [look-ahead-bias](look-ahead-bias.md).
- **Model decision time mismatch.** The model learns from 3-24 h snapshots; backtests decide 1 h out ([decision-time](decision-time.md)).

## Related

- [CLV](clv.md), [Howto: add a betting strategy](howto-add-betting-strategy.md)
