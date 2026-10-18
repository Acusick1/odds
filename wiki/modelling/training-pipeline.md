---
type: reference
title: Training pipeline
description: What odds train run, tune, validate and list-configs do step by step, the splits, verification artifacts and MLflow tracking.
sources:
  - odds_cli.commands.train.run_training
  - odds_cli.commands.train.run_tuning
  - odds_cli.commands.train.validate_config
  - odds_cli.commands.train.list_configs
  - odds_cli.commands.train.load_config
  - odds_cli.commands.train.STRATEGY_CLASSES
  - odds_analytics.training.data_preparation.prepare_training_data_from_config
  - odds_analytics.training.data_preparation.filter_events_by_date_range
  - odds_analytics.feature_groups.prepare_training_data
  - odds_analytics.training.cross_validation.train_with_cv
  - odds_analytics.training.tracking.MLflowTracker
  - odds_analytics.training.tracking.create_tracker
status: stable
---

# Training pipeline

`odds train run` and `odds train tune` (`odds_cli.commands.train`) turn an `odds_analytics.training.config.MLTrainingConfig` YAML ([training config](training-config.md)) into a pickled model. `odds train validate` loads the file and checks strategy, dates and splits; it warns "Very small test split" for `test_split < 0.1`, including 0.0 (`odds_cli.commands.train.validate_config`); 0.0 is the production value, and step 4 shows what it does in `odds_cli.commands.train.run_training` (#437). `odds train list-configs` tabulates a directory.

## Steps of `odds train run`

1. **Load.** `odds_cli.commands.train.load_config` parses `.yaml`/`.yml`/`.json`; `base:` inheritance deep-merges a parent file.
2. **Select events.** `odds_analytics.training.data_preparation.prepare_training_data_from_config` converts the date range to UTC, then `odds_analytics.training.data_preparation.filter_events_by_date_range` keeps FINAL events, filtered by `data_source`, `min_snapshots` and `sport_key`.
3. **Build features.** `odds_analytics.feature_groups.prepare_training_data` samples snapshots, computes targets and features, and drops constant columns ([feature pipeline](feature-pipeline.md)). Group names were already validated when the config loaded.
4. **Split.** With event ids, the last `max(1, int(n_events * test_split))` events become the test set; `validation_split` is cut the same way. `test_split: 0.0` therefore still holds out the last event from CV in `odds train run` (#437); under `tune` the objective re-stacks train and test rows, so tuning CV includes it ([cross-validation](cross-validation.md)).
5. **Verification artifacts.** `feature_stats.csv`, `feature_sample.csv` and `feature_target_corr.csv` are written to `experiments/results/<experiment.name>/verification/` from the concatenated splits ([run an experiment](../evaluation/experiment-scripts.md#verification-artifacts)).
6. **Train.** If `use_kfold` (default true), `odds_analytics.training.cross_validation.train_with_cv` runs [walk-forward CV](cross-validation.md) and then fits a final model on all training rows; otherwise a single `odds_analytics.training.cross_validation.TrainableStrategy.train_from_config`. Only `xgboost_line_movement` and `lstm_line_movement` are in `odds_cli.commands.train.STRATEGY_CLASSES` ([line-movement models](line-movement-models.md)).
7. **Save.** The model goes to `<output_path>/<strategy_type>_<experiment.name>.pkl` (or `model_name`), and the full config is written beside it as `.yaml`. Only `odds train run` writes the full `.yaml` the loader needs.

## Tune

`odds train tune` ([hyperparameter tuner](hyperparameter-tuner.md)) prepares data once, pre-computes features per `feature_groups` choice if that is searched, runs Optuna (study `<experiment>_tuning`), writes the best config to `<config>_best.yaml`, and with `--train-best` trains and saves a model. Studies are in-memory unless `--storage` is passed (#437). `--train-best` leaves a metadata-only `.yaml` beside the model, with no `training` section, so scoring refuses it; only `odds train run` overwrites it with the full config the S3 loader expects, and `odds model publish` uploads whichever `.yaml` sits beside the `.pkl` (#437; [model store](model-store.md), [prediction scoring](prediction-scoring.md)).

## MLflow

`--track` or `tracking.enabled` creates an `odds_analytics.training.tracking.MLflowTracker` through `odds_analytics.training.tracking.create_tracker`; if setup fails the run continues untracked. Under `tune`, a parent run holds study metadata and each trial is a nested run. `odds_analytics.training.cross_validation.train_with_cv` is called without the tracker, so under `odds train run` with the default `use_kfold` the run logs only start and end and nothing during CV (#437; read from code).

## Related

- [Training config](training-config.md), [Cross-validation](cross-validation.md), [Train and publish a model](howto-train-and-publish-model.md)
