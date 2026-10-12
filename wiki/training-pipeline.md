---
type: process
title: Training pipeline
summary: From YAML config to a trained model - odds train run and tune, data preparation, CV, tuning, MLflow, verification artifacts and saved models.
sources:
  - packages/odds-cli/odds_cli/commands/train.py::run_training
  - packages/odds-cli/odds_cli/commands/train.py::run_tuning
  - packages/odds-analytics/odds_analytics/training/data_preparation.py::prepare_training_data_from_config
  - packages/odds-analytics/odds_analytics/feature_groups.py::prepare_training_data
  - packages/odds-analytics/odds_analytics/training/cross_validation.py::train_with_cv
  - packages/odds-analytics/odds_analytics/training/tracking.py::MLflowTracker
  - packages/odds-analytics/odds_analytics/training/tracking.py::create_tracker
  - docs/CLI.md#ml-training-commands
  - docs/DEVELOPMENT.md#running-experiments
verified_at: 7887039
---

# Training pipeline

`odds train run` and `odds train tune` (`packages/odds-cli/odds_cli/commands/train.py`) turn an `MLTrainingConfig` YAML ([training-config](training-config.md)) into a pickled model. Steps for `run`:

1. **Load.** `load_config` parses `.yaml`/`.yml`/`.json`; `base:` inheritance deep-merges a parent file.
2. **Select events.** `prepare_training_data_from_config` converts the date range to UTC, then `filter_events_by_date_range` keeps FINAL events, filtered by `data_source` (event-id prefix), `min_snapshots` and `sport_key`.
3. **Build features.** `packages/odds-analytics/odds_analytics/feature_groups.py::prepare_training_data` samples snapshots, computes targets and features, and drops constant columns ([feature-pipeline](feature-pipeline.md)).
4. **Split.** With event ids, the last `max(1, int(n_events * test_split))` events become the test set; `validation_split` is cut the same way.
5. **Verification artifacts.** `feature_stats.csv`, `feature_sample.csv` and `feature_target_corr.csv` are written to `experiments/results/<experiment.name>/verification/` from the concatenated splits ([howto-run-experiment](howto-run-experiment.md)).
6. **Train.** If `use_kfold` (default true), `train_with_cv` runs [walk-forward CV](walk-forward-cv.md) and then fits a final model on all training rows; otherwise a single `train_from_config`. Only `xgboost_line_movement` and `lstm_line_movement` are in `STRATEGY_CLASSES` ([line-movement-models](line-movement-models.md)).
7. **Save.** The model goes to `<output_path>/<strategy_type>_<experiment.name>.pkl` (or `model_name`), and the full config is written beside it as `.yaml`.

`tune` ([hyperparameter-tuner](hyperparameter-tuner.md)) prepares data once, pre-computes features per `feature_groups` choice if that is searched, runs Optuna (study `<experiment>_tuning`), writes the best config to `<config>_best.yaml`, and with `--train-best` trains and saves a model. Publishing the result is in [howto-train-and-publish-model](howto-train-and-publish-model.md).

**MLflow.** `--track` or `tracking.enabled` creates an `packages/odds-analytics/odds_analytics/training/tracking.py::MLflowTracker` through `create_tracker`; if setup fails the run continues untracked. Under `tune`, a parent run holds study metadata and each trial is a nested run.

## Hazards

- **`test_split: 0.0` still holds out one event** (`max(1, ...)`), so the last event never reaches CV in `run`. Under `tune` the objective re-stacks train and test rows, so tuning CV includes it.
- **`--track` logs nothing during CV.** `train_with_cv` is called without the tracker, so with the default `use_kfold` the run only has start and end (read from code).
- **`strategy_type: xgboost` validates but cannot train.** The schema accepts it, `STRATEGY_CLASSES` and `XGBoostLineMovementStrategy.train_from_config` reject it; in `tune` the resulting error is swallowed per trial ([hyperparameter-tuner](hyperparameter-tuner.md)).
- **`tune` never persists studies by default.** The default storage lookup imports `get_config`, which does not exist in `odds_core.config` (only `get_settings`); the blanket `except` falls back to in-memory storage.
- **`tune --train-best` leaves a metadata-only `.yaml`** beside the model; only `run` overwrites it with the full config that the S3 loader expects ([s3-model-store](s3-model-store.md)). `odds model publish` uploads whichever `.yaml` sits beside the `.pkl`.

## Related

- [Walk-forward CV](walk-forward-cv.md), [Decision time](decision-time.md)
