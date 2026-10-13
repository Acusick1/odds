---
type: decision
title: Walk-forward CV (sliding-380:50, no test split)
description: Why the CV protocol is walk-forward, sliding 380 events with 50-event steps and no test split; the grid evidence; how cross_validation.py builds folds and metrics; and the traps (one-event holdout, silent downgrade, early stopping).
sources:
  - packages/odds-analytics/odds_analytics/training/cross_validation.py::make_walk_forward_splits
  - packages/odds-analytics/odds_analytics/training/cross_validation.py::run_cv
  - packages/odds-analytics/odds_analytics/training/cross_validation.py::train_with_cv
  - packages/odds-analytics/odds_analytics/training/cross_validation.py::CVResult
  - packages/odds-analytics/odds_analytics/training/config.py::DataConfig
  - packages/odds-analytics/odds_analytics/training/data_preparation.py::prepare_training_data_from_config
  - experiments/configs/xgboost_epl_production_tabular_match_stats.yaml
  - experiments/scripts/cv_protocol_grid.py
status: stable
---

# Walk-forward CV (sliding-380:50, no test split)

**Decision.** Evaluate with walk-forward CV, `window_type: sliding`, `min_train_events = max_train_events = 380` (about one EPL season), `val_step_events: 50` (about 5 matchdays), `test_split: 0.0`. No ADR exists; the evidence is PRs #234 and #361 and `experiments/results/cv_protocol_grid/grid_results.csv`; the retrain-cycle and no-test-split rationale below is `(inferred)` (the protocol was introduced in PR #230). The protocol must be set in the training YAML; code defaults are `cv_method: timeseries` and `test_split: 0.2` ([training pipeline](training-pipeline.md)).

## Rationale

- **Walk-forward is the primary metric** because it simulates the production retrain cycle: train on the past, predict the next chunk, roll forward `(inferred)`.
- **No test split.** A fixed chronological test split wastes training data and does not reflect deployment, where the model is retrained on all available data before predicting new events. There is no held-out test set.
- **Window and step** were chosen by grid search (below): sliding-380 gave the lowest MSE at validation steps 10 and 50; step 50 was chosen so the tuner can tell feature sets apart, since step 10 selects identical hyperparameters for every feature set.
- Once the search space stopped collapsing to constant predictors, sliding-380:50 also edged expanding windows. Sliding-380:50 only works with the principled search space; the original space collapses to constant predictors ([tuner](hyperparameter-tuner.md#the-principled-search-space), [search-space experiment](../evaluation/exp-epl-feature-ablation-and-search-space.md)).

Caveats recorded: CV R² from tuning is optimistically biased (best of N trials); relative comparisons between experiments tuned alike are fair. Under the original search space the protocol favoured expanding windows with more features; the production retraining strategy is the deciding factor `(inferred)` ([hybrid sharp reference](decision-hybrid-sharp-reference.md)).

## Grid evidence

2026-03-16, about 1.7K EPL events, tabular features, Pinnacle sharp, 100 Optuna trials per cell. `experiments/scripts/cv_protocol_grid.py` crosses window type (expanding; sliding 95, 190, 380, 760 events) with validation step (10, 50, 150) and writes `experiments/results/cv_protocol_grid/grid_results.csv` (PR #234). CV MSE:

| val_step | expanding | sliding-95 | sliding-190 | sliding-380 | sliding-760 |
|---|---|---|---|---|---|
| 10 | 0.000728 | 0.000752 | 0.000749 | **0.000706** | 0.000789 |
| 50 | 0.000757 | 0.000748 | 0.000756 | **0.000732** | 0.000794 |
| 150 | 0.000793 | 0.000758 | 0.000742 | 0.000752 | 0.000758 |

- Sliding-380 (one season) is the best window; shorter windows starve the model and longer ones include stale data `(inferred)`; PR #234 states this as the rationale but the CSV does not measure it.
- `val_step = 10` gives the lowest MSE but leads the tuner to pick identical hyperparameters for every feature set. `val_step = 50` keeps the sliding-380 advantage with enough per-fold signal.
- A rerun with 32 features (2026-03-19) gave the same lowest-MSE cell (sliding-380:10, 0.000693), but R² favoured expanding windows: expanding:50 0.061, expanding:150 0.052, sliding-380:50 -0.013. Recorded reading at the time: expanding windows favour more features, sliding-380 penalises them. The weakness was later traced to the default search space, not the protocol (PRs #253 and #361). Those 32-feature figures are as recorded in the PRs; they are not in `grid_results.csv`, which holds the tabular run.
- MSE favours sliding-380:10 while R² in the same CSV does not (sliding-380:10 R² -0.038, sliding-380:50 +0.026 on tabular only); the choice of step trades the two. Lowest MSE alone reflects target variance as much as skill.
- Expanding cells start validation after 700 training events and sliding-380 after 380, so MSEs are computed on different validation sets.

## How folds are built

`packages/odds-analytics/odds_analytics/training/cross_validation.py::make_walk_forward_splits(event_ids, min_train_events, val_step_events, window_type, max_train_events)` takes row-level event ids in chronological order and works on unique events:

- The first validation block starts after `min_train_events`; each block holds exactly `val_step_events` events; the last partial block is dropped. Fold count is `floor((n_events - min_train_events) / val_step_events)`; fewer than `min + step` events raises `ValueError`. That gives about 26 folds on 1.73K events and 28 on 1,813 events (earlier docs said about 26 for the latter).
- `expanding` trains on every earlier event; `sliding` trains on the previous `max_train_events` only.
- Splits are row indices, so multi-row events (time-range sampling) stay whole in one side.

`packages/odds-analytics/odds_analytics/training/cross_validation.py::run_cv` switches on `data.cv_method`: `walk_forward`, `timeseries` (sklearn `TimeSeriesSplit(n_folds)`, row-level, requires sorted data) or `kfold`. Per fold it calls `train_from_config` and records train and validation MSE, MAE and R² (`packages/odds-analytics/odds_analytics/training/cross_validation.py::CVFoldResult`); `packages/odds-analytics/odds_analytics/training/cross_validation.py::CVResult` reports mean and population standard deviation. Each fold's R² is against that fold's own mean. `packages/odds-analytics/odds_analytics/training/cross_validation.py::train_with_cv` then fits a final model on all rows and merges the CV metrics into the history. Tuning optimises the mean CV metric ([tuner](hyperparameter-tuner.md)).

## Hazards

- **`test_split: 0.0` still holds out one event** because the splitter uses `max(1, int(n * test_split))`; that event is excluded from `odds train run` CV and included in tuning CV. `odds train validate` also warns about `test_split < 0.1` although 0.0 is the protocol.
- **Silent downgrade.** `walk_forward` without `event_ids` logs a warning and runs `timeseries` with `n_folds` (default 5), changing the protocol and the metric meaning (AUD A3/A4, re-verified).
- **Early stopping sees the validation fold.** The XGBoost strategy passes the fold as an `eval_set` and `early_stopping_rounds` is applied by XGBoost; fold R² can therefore be mildly optimistic when early stopping is on (read from code, not measured).
- **Chronological order is assumed, not checked.** Order comes from `packages/odds-lambda/odds_lambda/storage/readers.py::OddsReader.get_events_by_date_range`; a caller passing shuffled data gets no error (AUD T4). See [feature pipeline](feature-pipeline.md).
- **`n_folds` is ignored for `walk_forward`** and mandatory for the other methods, but only `packages/odds-analytics/odds_analytics/training/cross_validation.py::run_cv` enforces it.

## Related

- [Look-ahead bias](decision-time-and-look-ahead.md), [Feature pipeline](feature-pipeline.md)
- [CLV model findings](../evaluation/clv-model-findings.md), [Hyperparameter tuner](hyperparameter-tuner.md)
