---
type: reference
title: Cross-validation
description: How cross_validation.py builds walk-forward folds, the other CV methods, per-fold metrics and train_with_cv.
sources:
  - odds_analytics.training.cross_validation.make_walk_forward_splits
  - odds_analytics.training.cross_validation.run_cv
  - odds_analytics.training.cross_validation.train_with_cv
  - odds_analytics.training.cross_validation.CVResult
  - odds_analytics.training.cross_validation.CVFoldResult
  - odds_analytics.training.config.DataConfig
  - odds_analytics.training.data_preparation.prepare_training_data_from_config
  - experiments/configs/xgboost_epl_production_tabular_match_stats.yaml
status: stable
---

# Cross-validation

The production protocol is `cv_method: walk_forward`, `window_type: sliding`, `min_train_events = max_train_events = 380`, `val_step_events: 50`, `test_split: 0.0`, set in the training YAML (`experiments/configs/xgboost_epl_production_tabular_match_stats.yaml`, whose metrics are on [CLV model findings](../evaluation/clv-model-findings.md#deployed-model)); the code defaults are `cv_method: timeseries` and `test_split: 0.2` ([training config](training-config.md)). The decision and its evidence are [ADR-0004](../../docs/adr/0004-walk-forward-cv-protocol.md) and [CV protocol grid](../evaluation/exp-cv-protocol-grid.md).

## How folds are built

`odds_analytics.training.cross_validation.make_walk_forward_splits(event_ids, min_train_events, val_step_events, window_type, max_train_events)` takes row-level event ids in chronological order and works on unique events:

- The first validation block starts after `min_train_events`; each block holds exactly `val_step_events` events; the last partial block is dropped. Fold count is `floor((n_events - min_train_events) / val_step_events)`; fewer than `min + step` events raises `ValueError`. That gives about 26 folds on 1.73K events and 28 on 1,813 events.
- `expanding` trains on every earlier event; `sliding` trains on the previous `max_train_events` only.
- Splits are row indices, so multi-row events (time-range sampling) stay whole on one side.
- Chronological order is assumed, not checked. Order comes from `odds_lambda.storage.readers.OddsReader.get_events_by_date_range`; a caller passing shuffled data gets no error ([feature pipeline](feature-pipeline.md)).

## CV methods and metrics

`odds_analytics.training.cross_validation.run_cv` switches on `data.cv_method`: `walk_forward`, `timeseries` (sklearn `TimeSeriesSplit(n_folds)`, row-level, requires sorted data) or `kfold`.

- `n_folds` is ignored for `walk_forward` and mandatory for the other methods, but only `odds_analytics.training.cross_validation.run_cv` enforces it.
- `walk_forward` without `event_ids` logs a warning and runs `timeseries` with `n_folds` (default 5), changing the protocol and the metric meaning (#437).
- Per fold it calls `odds_analytics.training.cross_validation.TrainableStrategy.train_from_config` and records train and validation MSE, MAE and R² (`odds_analytics.training.cross_validation.CVFoldResult`). `odds_analytics.training.cross_validation.CVResult` reports mean and population standard deviation. Each fold's R² is against that fold's own mean.
- XGBoost has no early stopping: every fold builds all `n_estimators` trees. The strategy passes the fold as `eval_set` for metrics and Optuna pruning only, so the validation rows never influence the fitted model (#474).
- `odds_analytics.training.cross_validation.train_with_cv` then fits a final model on all rows and merges the CV metrics into the history. Tuning optimises the mean CV metric ([tuner](hyperparameter-tuner.md)).

## Test split

The splitter takes `max(1, int(n * test_split))` events, so `test_split: 0.0` still holds out the last event from `odds train run` CV (#437). That event is included in tuning CV ([training pipeline](training-pipeline.md#steps-of-odds-train-run)).

## Related

- [Look-ahead bias](look-ahead-bias.md), [Feature pipeline](feature-pipeline.md), [Hyperparameter tuner](hyperparameter-tuner.md)
- [CLV model findings](../evaluation/clv-model-findings.md)
