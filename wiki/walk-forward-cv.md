---
type: component
title: Walk-forward cross-validation
summary: cross_validation.py - sliding and expanding event windows, validation steps, fold construction and metrics, plus the test_split and silent-downgrade traps.
sources:
  - packages/odds-analytics/odds_analytics/training/cross_validation.py::make_walk_forward_splits
  - packages/odds-analytics/odds_analytics/training/cross_validation.py::run_cv
  - packages/odds-analytics/odds_analytics/training/cross_validation.py::train_with_cv
  - packages/odds-analytics/odds_analytics/training/cross_validation.py::CVResult
  - packages/odds-analytics/odds_analytics/training/config.py::DataConfig
  - packages/odds-analytics/odds_analytics/training/data_preparation.py::prepare_training_data_from_config
  - docs/DATA_MODELS.md#cross-validation-protocol
  - docs/DATA_MODELS.md#cross-validation-trainingdata
  - AGENTS.md
verified_at: 7887039
---

# Walk-forward cross-validation

`packages/odds-analytics/odds_analytics/training/cross_validation.py` evaluates a model the way production retrains: train on the past, predict the next block of events, roll forward. The protocol and why it was chosen are in [decision-cv-protocol](decision-cv-protocol.md); the config fields are in [training-config](training-config.md).

## Fold construction

`make_walk_forward_splits(event_ids, min_train_events, val_step_events, window_type, max_train_events)` takes row-level event ids in chronological order and works on unique events:

- The first validation block starts after `min_train_events`; each block holds exactly `val_step_events` events; the last partial block is dropped. Fold count is `floor((n_events - min_train_events) / val_step_events)`; fewer than `min + step` events raises `ValueError`.
- `expanding` trains on every earlier event; `sliding` trains on the previous `max_train_events` only.
- Splits are row indices, so multi-row events (time-range sampling) stay whole in one side.

The production setting is `walk_forward`, sliding, `min = max = 380`, step 50, `test_split: 0.0`. That gave about 26 folds on 1.73K events and 28 on 1,813 events.

## `run_cv` and metrics

`run_cv` switches on `data.cv_method`: `walk_forward` (above), `timeseries` (sklearn `TimeSeriesSplit(n_folds)`, row-level, requires sorted data) or `kfold`. Per fold it calls `train_from_config` and records train and validation MSE, MAE and R² (`CVFoldResult`); `CVResult` reports mean and population standard deviation. Each fold's R² is against that fold's own mean. `train_with_cv` then fits a final model on all rows and merges the CV metrics into the history ([training-pipeline](training-pipeline.md)). Tuning optimises the mean CV metric ([hyperparameter-tuner](hyperparameter-tuner.md)); a tuned CV R² is optimistically biased (best of N trials), but comparisons between equally tuned experiments are fair (`AGENTS.md`).

## Hazards

- **Silent downgrade.** `walk_forward` without `event_ids` logs a warning and runs `timeseries` with `n_folds` (default 5), changing the protocol and the metric meaning (AUD A3/A4, re-verified).
- **`test_split: 0.0` still holds out one event** because the splitter uses `max(1, int(n * test_split))`; that event is excluded from `odds train run` CV and included in tuning CV ([training-pipeline](training-pipeline.md)).
- **Early stopping sees the validation fold.** The XGBoost strategy passes the fold as an `eval_set` and `early_stopping_rounds` is applied by XGBoost; fold R² can therefore be mildly optimistic when early stopping is on (read from code, not measured).
- **Different cells validate different events.** Expanding cells in the protocol grid start at 700 events and sliding-380 at 380, so MSEs compare different validation sets ([exp-cv-protocol-grid](exp-cv-protocol-grid.md)).
- **Chronological order is assumed, not checked.** Order comes from `OddsReader.get_events_by_date_range`; a caller passing shuffled data gets no error (AUD T4).
- `n_folds` is ignored for `walk_forward` and mandatory for the other methods, but only `run_cv` enforces it.

## Related

- [Look-ahead bias](look-ahead-bias.md), [Feature pipeline](feature-pipeline.md)
