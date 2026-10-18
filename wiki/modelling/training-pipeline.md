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
  - odds_analytics.training.utils.compute_regression_metrics
  - odds_analytics.xgboost_line_movement.XGBoostLineMovementStrategy.train
status: stable
---

# Training pipeline

`odds train run` and `odds train tune` (`odds_cli.commands.train`) turn an `odds_analytics.training.config.MLTrainingConfig` YAML (its `training.features` block is required), validated at load (strategy, feature groups, tuning metric and search-space keys) ([training config](training-config.md)), into a pickled model. `odds train validate` loads the file and checks strategy, dates and splits; it warns "Very small test split" for `test_split < 0.1`, including 0.0 (`odds_cli.commands.train.validate_config`); 0.0 is the production value and holds out nothing (step 4). `odds train list-configs` tabulates a directory.

## Steps of `odds train run`

1. **Load.** `odds_cli.commands.train.load_config` parses `.yaml`/`.yml`/`.json`; `base:` inheritance deep-merges a parent file.
2. **Select events.** `odds_analytics.training.data_preparation.prepare_training_data_from_config` converts the date range to UTC, then `odds_analytics.training.data_preparation.filter_events_by_date_range` keeps FINAL events, filtered by `data_source` (event-id prefix, `odds_core.event_ids`), `min_snapshots` and `sport_key`. Its optional `sampler` argument is passed through to `odds_analytics.feature_groups.prepare_training_data`.
3. **Build features.** `odds_analytics.feature_groups.prepare_training_data` samples snapshots (with the `sampler` argument when given, else the one `config.sampling` specifies), computes targets (per `target_type` and `outcome`) and features, and drops columns whose variance is at or below `variance_threshold` ([feature pipeline](feature-pipeline.md)).
4. **Split.** With event ids, the last `int(n_events * test_split)` events become the test set (at least one when `test_split > 0`); `validation_split` is cut the same way. `test_split: 0.0` holds out no events, so `odds_analytics.training.data_preparation.TrainingDataResult` has no test arrays (`None`), as it has none for validation without a `validation_split`. Under `tune` the objective re-stacks train and validation rows for CV ([cross-validation](cross-validation.md)); test rows are never used for tuning.
5. **Verification artifacts.** `feature_stats.csv`, `feature_sample.csv` and `feature_target_corr.csv` are written to `experiments/results/<experiment.name>/verification/` from the concatenated splits ([run an experiment](../evaluation/experiment-scripts.md#verification-artifacts)).
6. **Train.** If `use_kfold` (default true), `odds_analytics.training.cross_validation.train_with_cv` runs [walk-forward CV](cross-validation.md) and then fits a final model on all training rows; otherwise a single `odds_analytics.training.cross_validation.TrainableStrategy.train_from_config`. `odds_cli.commands.train.STRATEGY_CLASSES` has exactly the `odds_analytics.training.config.StrategyType` values, `xgboost_line_movement` and `lstm_line_movement` ([line-movement models](line-movement-models.md)).
7. **Save.** The model goes to `<output_path>/<strategy_type>_<experiment.name>.pkl` (or `model_name`), and the full config is written beside it as `.yaml`, which the loader needs; a failed config write fails the command. `tune --train-best` saves through the same helper.

## Sample weights

`odds_analytics.feature_groups.PreparedFeatureData.sample_weights` is set only by the `retail_ev` target ([target functions](clv-target-functions.md#retail-ev-rows)). `odds_analytics.training.data_preparation.TrainingDataResult` splits it with the rows into `sample_weights_train`, `sample_weights_val` and `sample_weights_test`. `odds_cli.commands.train.run_training` and the tuning objective (`odds_analytics.training.tuner.create_objective`) pass them to `odds_analytics.training.cross_validation.run_cv` and `odds_analytics.training.cross_validation.train_with_cv`, which slice them per fold, and on to `odds_analytics.training.cross_validation.TrainableStrategy.train_from_config`. `odds_analytics.xgboost_line_movement.XGBoostLineMovementStrategy.train` passes them to `fit` as `sample_weight` and `sample_weight_eval_set`, so the validation RMSE used for Optuna pruning is weighted, and scores train and validation with `odds_analytics.training.utils.compute_regression_metrics`, whose MSE, MAE and R² are weighted when weights are given. The LSTM strategy raises if given weights. Feature selection for `top_k_features` and the verification artifacts ignore weights.

## Tune

`odds train tune` ([hyperparameter tuner](hyperparameter-tuner.md)) prepares data once, pre-computes features per `feature_groups` choice if that is searched, runs Optuna (study `<experiment>_<UTC timestamp>` unless `--study-name` is given), writes the best config to `<config>_best.yaml`, and with `--train-best` trains and saves a model. Studies are stored in the SQLite file `optuna.db` in `experiments/`, relative to the working directory, unless `--storage` names another store or `--in-memory` is passed; they never use `DATABASE_URL`. `--train-best` writes the full best config beside the model, as `odds train run` does, so `odds model publish` uploads a `.yaml` with the `training.features` section the S3 loader reads ([model store](model-store.md), [prediction scoring](prediction-scoring.md)).

## MLflow

`--track` or `tracking.enabled` creates an `odds_analytics.training.tracking.MLflowTracker` through `odds_analytics.training.tracking.create_tracker`; if setup fails the run continues untracked. Under `tune`, a parent run holds study metadata and each trial is a nested run. `odds_analytics.training.cross_validation.train_with_cv` is called without the tracker, so under `odds train run` with the default `use_kfold` the run logs only start and end and nothing during CV (#502).

## Related

- [Training config](training-config.md), [Cross-validation](cross-validation.md), [Train and publish a model](howto-train-and-publish-model.md)
