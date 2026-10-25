---
type: reference
title: Line-movement model (XGBoost)
description: The XGBoost regressor wrapper behind odds train, covering fitting, base margin, config-driven training, and saving and loading.
sources:
  - odds_analytics.xgboost_line_movement.XGBoostLineMovementStrategy
  - odds_analytics.training.config.TrainingConfig
status: stable
---

# Line-movement model (XGBoost)

`odds_analytics.xgboost_line_movement.XGBoostLineMovementStrategy` wraps an XGBoost regressor of the closing line move. It is the trainer behind `odds train` ([training pipeline](training-pipeline.md)) and the model of the [retail-EV evaluation](../evaluation/retail-ev-evaluation.md); it places no bets. Config `strategy_type` accepts only `xgboost_line_movement`.

## Behaviour

`odds_analytics.xgboost_line_movement.XGBoostLineMovementStrategy.train` fits an `XGBRegressor` with the train set and validation set as `eval_set` (metrics and pruning only; there is no early stopping, so all `n_estimators` trees are built), with optional `sample_weight_train` and `sample_weight_val` passed to the fit, the evaluation set and the metrics ([sample weights](training-pipeline.md#sample-weights)). Its optional `margin_feature` names a feature column that XGBoost takes as each row's base margin, in the fit and the evaluation set; `odds_analytics.xgboost_line_movement.XGBoostLineMovementStrategy.predict` adds that column back, and the train and validation metrics go through it. `odds_analytics.xgboost_line_movement.XGBoostLineMovementStrategy.train_from_config` requires `strategy_type == "xgboost_line_movement"` and an `odds_analytics.training.config.XGBoostConfig`, sets `margin_feature` to `baseline_ev` for the `retail_ev` residual arm ([retail-EV arm](training-config.md#retail-ev-arm)), and adds an Optuna pruning callback when given a trial. `odds_analytics.xgboost_line_movement.XGBoostLineMovementStrategy.save_model` writes joblib `{model, feature_names, margin_feature}` plus a `.yaml` of metadata beside it; `odds_analytics.xgboost_line_movement.XGBoostLineMovementStrategy.load_model` falls back to pickle when joblib fails.

## Feature vector

`odds_analytics.xgboost_line_movement.XGBoostLineMovementStrategy` builds `odds_analytics.feature_extraction.TabularFeatures.to_array` (6 values) when it extracts features itself. Models trained from the arrays that `odds_analytics.feature_groups.prepare_training_data` builds add `hours_until_event` (from the XGBoost adapter) and any other configured groups, so a model trained through it receives a different width (#476; read from code, not run).

## Related

- [CLV target](clv-target.md)
