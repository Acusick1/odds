---
type: component
title: Training pipeline and config
description: From an MLTrainingConfig YAML to a saved model with odds train run/tune — config schema and defaults, data selection, splits, verification artifacts, MLflow — and the invalid values the schema silently accepts.
sources:
  - odds_cli.commands.train.run_training
  - odds_cli.commands.train.run_tuning
  - odds_cli.commands.train.validate_config
  - odds_cli.commands.train.list_configs
  - odds_analytics.training.data_preparation.prepare_training_data_from_config
  - odds_analytics.feature_groups.prepare_training_data
  - odds_analytics.training.cross_validation.train_with_cv
  - odds_analytics.training.tracking.MLflowTracker
  - odds_analytics.training.tracking.create_tracker
  - odds_analytics.training.config.MLTrainingConfig
  - odds_analytics.training.config.DataConfig
  - odds_analytics.training.config.FeatureConfig
  - odds_analytics.training.config.SamplingConfig
  - odds_analytics.training.config.TuningConfig
  - experiments/configs/xgboost_epl_production_tabular_match_stats.yaml
status: stable
---

# Training pipeline and config

`odds train run` and `odds train tune` (`odds_cli.commands.train`) turn an `odds_analytics.training.config.MLTrainingConfig` YAML into a pickled model. `odds_analytics.training.config` is authoritative for fields and defaults; configs live in `experiments/configs/` (the production example is `experiments/configs/xgboost_epl_production_tabular_match_stats.yaml`).

## The config

Every model forbids unknown keys (`extra="forbid"`). A file has `experiment` (name, tags), `training` (`strategy_type`, `data`, `model`, `features`, `feature_selection`, output settings), optional `tuning` and `tracking`, and optional `base:` inheritance (deep-merged). `odds train validate` loads the file and checks strategy, dates and splits; `odds train list-configs` tabulates a directory.

**`training.data`**

- `data_source`: `oddsportal` (`op_` ids), `football_data_uk` (`fduk_`), `oddsapi` (everything else), `all` or null (no filter). The filter is on event-id prefix ([events](../storage/events.md)). `sport_key`, `start_date`/`end_date` (required), `min_snapshots`.
- CV: `cv_method` (`walk_forward`, `timeseries` default, `kfold`), `window_type` (`expanding` default or `sliding`), `min_train_events`, `max_train_events`, `val_step_events`, `n_folds` ([walk-forward CV](decision-walk-forward-cv.md)). `use_kfold` defaults true and `test_split` to 0.2.

**The CV protocol must be set in YAML.** The decided protocol is `cv_method: walk_forward`, `window_type: sliding`, 380/380/50 and `test_split: 0.0`; the code defaults are `timeseries` and 0.2, so it exists only where a YAML sets it. `odds train validate` also warns when `test_split < 0.1` although 0.0 is the protocol (`odds_cli.commands.train.validate_config`).

**`training.features`**

- `target_type` (`raw` default, `devigged_bookmaker`), `target_bookmaker` (default `pinnacle`), `sharp_bookmakers` (priority order, default `["pinnacle"]`), `retail_bookmakers`, `closing_source_priority`, `outcome`, `markets` (first entry is the primary market), `feature_groups`, `form_window`, `match_stats_window`, `variance_threshold`, `sampling` ([CLV target](clv-target.md), [sharp benchmark](sharp-benchmark.md), [decision time](decision-time-and-look-ahead.md), [feature pipeline](feature-pipeline.md)). Set `target_type` and `target_bookmaker` explicitly: the production target is devigged bet365 ([bet365 target](decision-bet365-target.md)).

**Choosing data.** `target_bookmaker` must belong to the selected source. EPL with the bet365 target and hybrid sharp uses `data_source: null`, `sharp_bookmakers: [pinnacle, betfair_exchange]` and `closing_source_priority: [football_data_uk]` ([hybrid sharp reference](decision-hybrid-sharp-reference.md)). NBA bet365 needs `oddsportal`; NBA Pinnacle needs `oddsapi`. LSTM runs also set `min_snapshots: 5` or more.

Model and feature settings such as `target_type` and `sharp_bookmakers` are YAML config, not environment Settings ([configuration](../foundations/configuration.md)).

## Steps of `odds train run`

1. **Load.** `odds_cli.commands.train.load_config` parses `.yaml`/`.yml`/`.json`; `base:` inheritance deep-merges a parent file.
2. **Select events.** `odds_analytics.training.data_preparation.prepare_training_data_from_config` converts the date range to UTC, then `odds_analytics.training.data_preparation.filter_events_by_date_range` keeps FINAL events, filtered by `data_source`, `min_snapshots` and `sport_key`.
3. **Build features.** `odds_analytics.feature_groups.prepare_training_data` samples snapshots, computes targets and features, and drops constant columns ([feature pipeline](feature-pipeline.md)).
4. **Split.** With event ids, the last `max(1, int(n_events * test_split))` events become the test set; `validation_split` is cut the same way.
5. **Verification artifacts.** `feature_stats.csv`, `feature_sample.csv` and `feature_target_corr.csv` are written to `experiments/results/<experiment.name>/verification/` from the concatenated splits ([run an experiment](../evaluation/howto-run-experiment.md)).
6. **Train.** If `use_kfold` (default true), `odds_analytics.training.cross_validation.train_with_cv` runs [walk-forward CV](decision-walk-forward-cv.md) and then fits a final model on all training rows; otherwise a single `train_from_config`. Only `xgboost_line_movement` and `lstm_line_movement` are in `odds_cli.commands.train.STRATEGY_CLASSES` ([line-movement models](line-movement-models.md)).
7. **Save.** The model goes to `<output_path>/<strategy_type>_<experiment.name>.pkl` (or `model_name`), and the full config is written beside it as `.yaml`. Only `odds train run` writes the full `.yaml` the loader needs.

`tune` ([hyperparameter tuner](hyperparameter-tuner.md)) prepares data once, pre-computes features per `feature_groups` choice if that is searched, runs Optuna (study `<experiment>_tuning`), writes the best config to `<config>_best.yaml`, and with `--train-best` trains and saves a model. Publishing the result is in [publishing and scoring](model-publishing-and-scoring.md).

**MLflow.** `--track` or `tracking.enabled` creates a `odds_analytics.training.tracking.MLflowTracker` through `odds_analytics.training.tracking.create_tracker`; if setup fails the run continues untracked. Under `tune`, a parent run holds study metadata and each trial is a nested run.

## Hazards

Values the schema accepts without error:

- **Unknown `feature_groups`.** The field is a free `tuple[str, ...]`; a typo is ignored. Its description still lists `trajectory`, which no code handles.
- **`strategy_type: xgboost` / `lstm` validate but cannot train.** The schema accepts them, `odds_cli.commands.train.STRATEGY_CLASSES` and `odds_analytics.xgboost_line_movement.XGBoostLineMovementStrategy.train_from_config` reject them; in `tune` the resulting error is swallowed per trial ([tuner](hyperparameter-tuner.md#hazards)).
- **Search-space keys that match nothing** are dropped silently.
- **`target_type` defaults to `raw`**, not the documented devigged target.
- **Target bookmaker absent from the source** drops events, counted only in a log line.
- **`n_folds` is ignored for `walk_forward`** and is required (checked at run time, not by the schema) for `kfold` and `timeseries`.
- **One class, many concerns.** `odds_analytics.training.config.FeatureConfig` carries bookmaker rosters, sampling, LSTM, Polymarket and EPL knobs, so every committed YAML serialises about 30 unrelated fields, and the primary market is `markets[0]` by position (AUD B5, re-verified).

Pipeline behaviour:

- **`test_split: 0.0` still holds out one event** (`max(1, ...)`), so the last event never reaches CV in `odds train run`. Under `tune` the objective re-stacks train and test rows, so tuning CV includes it.
- **`--track` logs nothing during CV.** `odds_analytics.training.cross_validation.train_with_cv` is called without the tracker, so with the default `use_kfold` the run only has start and end (read from code).
- **`tune` never persists studies by default.** The default storage lookup imports `get_config`, which does not exist in `odds_core.config` (only `odds_core.config.get_settings`); the blanket `except` falls back to in-memory storage.
- **`tune --train-best` is not scorable.** It leaves a metadata-only `.yaml` beside the model (no `training` section), so scoring refuses it; only `odds train run` overwrites it with the full config that the S3 loader expects ([publishing and scoring](model-publishing-and-scoring.md)). `odds model publish` uploads whichever `.yaml` sits beside the `.pkl`.

## Related

- [Walk-forward CV](decision-walk-forward-cv.md), [Decision time](decision-time-and-look-ahead.md), [Hyperparameter tuner](hyperparameter-tuner.md)
