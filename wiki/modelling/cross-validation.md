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
  - experiments/configs/xgboost_epl_cv_protocol_expanding.yaml
status: stable
---

# Cross-validation

The EPL evaluation protocol is `cv_method: walk_forward`, `window_type: expanding`, `min_train_events: 380`, `val_step_events: 50`, `test_split: 0.0`, from 2019-08-01, set in `experiments/configs/xgboost_epl_cv_protocol_expanding.yaml` (which also carries an `n_folds: 5` that `walk_forward` ignores). Hyperparameters are tuned only on events before 2024-08-01, the tuning range. Events from that date on, the evaluation range, are scored by folds that walk forward with the tuned values frozen, starting at `min_train_events` equal to the tuning-range event count. The decision and its evidence are [ADR-0010](../../docs/adr/0010-expanding-walk-forward-and-evaluation-range.md) and [CV protocol on a common validation range](../evaluation/exp-cv-protocol-common-range.md), which also gives the fold layout.

The deployed model's config, `experiments/configs/xgboost_epl_production_tabular_match_stats.yaml`, predates that decision and uses sliding-380:50 (`window_type: sliding`, `min_train_events = max_train_events = 380`); its metrics are on [CLV model findings](../evaluation/clv-model-findings.md#deployed-model). The code defaults are `cv_method: timeseries` and `test_split: 0.2` ([training config](training-config.md)).

## How folds are built

`odds_analytics.training.cross_validation.make_walk_forward_splits(event_ids, min_train_events, val_step_events, window_type, max_train_events, commence_times, gap)` takes row-level event ids in chronological order and works on unique events:

- The first validation block starts after `min_train_events`; each block holds exactly `val_step_events` events; the last partial block is dropped. Fold count is `floor((n_events - min_train_events) / val_step_events)`; fewer than `min + step` events raises `ValueError`. That gives 30 folds on the 1,900 tuning-range events with `min_train_events: 380`, and 22 with 760.
- `expanding` trains on every earlier event; `sliding` trains on the previous `max_train_events` only.
- With a positive `gap` and per-row `commence_times`, training events must kick off at least `gap` before the block's first validation event. `odds_analytics.training.cross_validation.run_cv` sets it to `odds_analytics.retail_ev.DECISION_WINDOW` (7 days) for `retail_ev`, whose validation decisions start that long before kick-off, so no training close falls after one. A fold left with no training event raises. The gap filter runs after the window is cut, so a sliding window then holds fewer than `max_train_events` events.
- Splits are row indices, so multi-row events (time-range sampling) stay whole on one side.
- Chronological order is assumed, not checked. Order comes from `odds_lambda.storage.readers.OddsReader.get_events_by_date_range`, which sorts by `commence_time` and then event ID, so kick-off ties have a fixed order that does not depend on the query plan; a caller passing shuffled data gets no error ([feature pipeline](feature-pipeline.md)).

## CV methods and metrics

`odds_analytics.training.cross_validation.run_cv` raises for `retail_ev` unless `sample_weights` and `commence_times` are given and `cv_method` is `walk_forward`. It switches on `data.cv_method`: `walk_forward`, `timeseries` (sklearn `TimeSeriesSplit(n_folds)`, row-level, requires sorted data) or `kfold`.

- `n_folds` is ignored for `walk_forward` and mandatory for the other methods, but only `odds_analytics.training.cross_validation.run_cv` enforces it.
- `walk_forward` without `event_ids` raises `ValueError`.
- Per fold it calls `odds_analytics.training.cross_validation.TrainableStrategy.train_from_config` with that fold's rows of `sample_weights`, when given, and records train and validation MSE, MAE and R² (`odds_analytics.training.cross_validation.CVFoldResult`), weighted when there are weights ([sample weights](training-pipeline.md#sample-weights)). `odds_analytics.training.cross_validation.CVResult` reports mean and population standard deviation. Each fold's R² is against that fold's own mean.
- XGBoost has no early stopping: every fold builds all `n_estimators` trees. The strategy passes the fold as `eval_set` for metrics and Optuna pruning only, so the validation rows never influence the fitted model (#474).
- `odds_analytics.training.cross_validation.train_with_cv` then fits a final model on all training rows and merges the CV metrics into the history. Tuning optimises the mean CV metric ([tuner](hyperparameter-tuner.md)).

## Test split

The events are those `odds_analytics.training.data_preparation.prepare_training_data_from_config` selects. The event-level split takes `int(n * test_split)` events, at least one when `test_split > 0`; `test_split: 0.0` holds out none ([training pipeline](training-pipeline.md#steps-of-odds-train-run)). With no test set, `odds_analytics.training.cross_validation.train_with_cv` fits the final model without a validation set, so its history has no `val_*` metrics beyond the merged CV means.

## Related

- [Look-ahead bias](look-ahead-bias.md), [Feature pipeline](feature-pipeline.md), [Hyperparameter tuner](hyperparameter-tuner.md)
- [CLV model findings](../evaluation/clv-model-findings.md)
