---
type: component
title: Line-movement models (XGBoost, LSTM)
description: The XGBoost and LSTM strategy classes that train the CLV regressor and bet on it in backtests, their confidence-to-Kelly mapping, LSTM's ruled-out status, and the misleading ml_strategy_example.
sources:
  - odds_analytics.xgboost_line_movement.XGBoostLineMovementStrategy
  - odds_analytics.lstm_line_movement.LSTMLineMovementStrategy
  - odds_analytics.ml_strategy_example.XGBoostStrategy
  - odds_analytics.sequence_loader.load_sequences_for_event
  - odds_analytics.training.config.TrainingConfig
status: stable
---

# Line-movement models (XGBoost, LSTM)

Two `odds_analytics.backtesting.services.BettingStrategy` subclasses wrap a trained CLV regressor so the [backtest engine](../evaluation/backtesting.md) can bet on it; they are also the trainers behind `odds train` ([training pipeline](training-pipeline.md)). Registry names are `xgb_line_movement` and `lstm_line_movement` ([backtest strategies](../evaluation/backtest-strategies.md)); config `strategy_type` values are `xgboost_line_movement` and `lstm_line_movement`.

## Shared behaviour

`evaluate_opportunity` predicts a movement delta for the home and away outcome, skips any below `min_predicted_movement` (0.02), takes the best price among its bookmakers, and sets `confidence = base_confidence + (predicted - min) * movement_confidence_scale`, clamped to 0.5-0.95 (defaults 0.52 and 5.0). The engine passes `confidence` to Kelly sizing as the win probability, and zero or negative Kelly is floored to `min_bet_size` ([backtesting](../evaluation/backtesting.md)). Draws are never bet.

- **XGBoost** (`odds_analytics.xgboost_line_movement`). `train` fits an `XGBRegressor` with the train set and validation set as `eval_set`; `train_from_config` requires `strategy_type == "xgboost_line_movement"` and an `odds_analytics.training.config.XGBoostConfig`, and adds an Optuna pruning callback when given a trial. `save_model` writes joblib `{model, feature_names, params}` plus a `.yaml` of metadata beside it; `odds_analytics.xgboost_line_movement.XGBoostLineMovementStrategy.load_model` falls back to pickle when joblib fails.
- **LSTM** (`odds_analytics.lstm_line_movement`). Needs torch (training extra). Defaults: 72 h lookback, 24 timesteps, hidden 64, 2 layers, dropout 0.2, MSE loss. Inputs come from `odds_analytics.sequence_loader.load_sequences_for_event` through `odds_analytics.feature_extraction.SequenceFeatureExtractor` ([feature pipeline](feature-pipeline.md)); packed sequences handle padding (`odds_analytics.lstm_line_movement._compact_sequences`, the 2026-02-27 mask fix, #162). The module docstring does not state the status below and duplicates the XGBoost training scaffold (AUD B5, re-verified).
- **`odds_analytics.ml_strategy_example`.** `odds_analytics.ml_strategy_example.XGBoostStrategy` is an XGBoost win-probability classifier kept as a reference implementation; it is not in any registry and is referenced only by tests, so do not copy it as the model strategy.

## LSTM status: ruled out

No LSTM variant beat predicting the mean. All runs used the Pinnacle target on NBA:

| Date | Variant | Data | CV R² |
|---|---|---|---|
| 2026-02-21 | Sequence only (15 features x 24 timesteps) | 800 events | -0.010 ± 0.039 |
| 2026-02-21 | Sequence + 17 static features | 800 events | -0.122 ± 0.115 |
| 2026-02-28 | Sequence only, packed sequences, 15 x 8 timesteps, 50 trials, 5-fold walk-forward | about 1K Odds API events | -0.075 ± 0.113 |

Config: `experiments/configs/lstm_pinnacle_tuning.yaml` (also `experiments/configs/lstm_line_movement_tuning.yaml`). The 2026-02-21 rows were recorded in PR #160 and the 2026-02-28 row has no results file, only the config, so it is unverified. A cross-check on 230 events gave R² about 0 at both decision times once the `IN_PLAY` leak was fixed; the earlier +0.02 was contaminated ([early NBA signal and LSTM](../evaluation/exp-early-nba-signal-and-lstm.md)). LSTM was never retried with the bet365 target or on EPL: neither LSTM config sets `target_bookmaker` (the code default is Pinnacle) and both are NBA runs. It it is still selectable through `strategy_type: lstm_line_movement`; nothing prevents using it. XGBoost remains the model ([CLV model findings](../evaluation/clv-model-findings.md)).

## Hazards

- **Backtest vectors may not match trained models.** The XGBoost strategy builds `odds_analytics.feature_extraction.TabularFeatures.to_array` (6 values). Models trained by `odds_analytics.feature_groups.prepare_training_data` add `hours_until_event` and any other configured groups, so a trained model would receive a different width (not run). Model strategies need `--model-path`; check the feature width before relying on it.
- **EPL needs a market override.** Both strategies default to `market="h2h"`, while EPL odds use `1x2`; the CLI constructs them with defaults and cannot override ([sources and keys](../collection/sources-and-keys.md)).
- **Bookmaker roster is US-flavoured.** XGBoost defaults to eight bookmakers (Pinnacle plus seven US books); the extractor's sharp and retail rosters default to `pinnacle` and `fanduel/draftkings/betmgm`, ignoring the hybrid sharp reference ([sharp benchmark](sharp-benchmark.md)).
- **LSTM backtests load all snapshots** for the event ([look-ahead bias](decision-time-and-look-ahead.md#hazards), which also covers the cutoff arithmetic).
- **Model decision time mismatch.** The model learns from 3-24 h snapshots; backtests decide 1 h out, so a model-driven backtest feeds it inputs unlike its training rows ([decision time](decision-time-and-look-ahead.md)).

## Related

- [CLV target](clv-target.md), [Add a betting strategy](../evaluation/backtest-strategies.md)
