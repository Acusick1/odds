---
type: reference
title: Hyperparameter tuner (Optuna)
description: OptunaTuner, the objective, search-space schema, feature selection and CLI options, with the example search spaces cited by file.
sources:
  - odds_analytics.training.tuner.OptunaTuner
  - odds_analytics.training.tuner.create_objective
  - odds_analytics.training.tuner.suggest_params_from_search_space
  - odds_analytics.training.config.SearchSpace
  - odds_analytics.training.config.TuningConfig
  - odds_analytics.training.feature_selection.FeatureSelector
  - odds_analytics.training.feature_selection.get_feature_selector
  - odds_cli.commands.train.run_tuning
  - experiments/configs/xgboost_tuning_without_feature_selection.yaml
  - experiments/configs/xgboost_feature_selection_tuning.yaml
status: stable
---

# Hyperparameter tuner (Optuna)

`odds_analytics.training.tuner` wraps Optuna. `odds train tune` builds an `odds_analytics.training.tuner.OptunaTuner` and an objective from a config's `tuning` section ([training pipeline](training-pipeline.md)); search-space fields are part of the config schema ([training config](training-config.md)).

- **Tuner.** `OptunaTuner(study_name, direction, sampler, pruner, storage, tracking_config, sampler_seed)`. Samplers `tpe`, `random`, `cmaes` are all seeded with `sampler_seed` (default 42; `tests/unit/test_tuner.py` covers it in `TestOptunaTunerSamplerSeed`), and `odds train tune` does not set it; pruners `median`, `hyperband`, `none`. The study is created with `load_if_exists=True`, so a name that already exists in the storage is resumed and its new trials are appended, including trials from a run with different data or protocol. The CLI default name is unique per run (experiment name plus a UTC timestamp), so only an explicit `--study-name` resumes, and the CLI prints the existing trial count first. With MLflow tracking, a parent run holds study metadata and each trial is a nested run.
- **Search spaces.** `odds_analytics.training.config.SearchSpace` is `int`, `float` (optional `step` and a log-scale flag) or `categorical`; list choices become tuples. A key may be a model field that is not also a feature-config field, `feature_groups` (pre-computed once per choice) or `top_k_features`; any other key, including every other `odds_analytics.training.config.FeatureConfig` field (`retail_ev_arm` too), fails `odds_analytics.training.config.MLTrainingConfig` validation, because the data is prepared once and such a key would only add noise. So does a `feature_groups` choice that is not a valid group list for the target (a `retail_ev` choice may name only its as-of-t groups), and a `tuning` section with `use_kfold: false` and `validation_split: 0`. A `retail_ev` residual-arm config may not tune `top_k_features` with feature selection enabled, because the selection could drop the `baseline_ev` column its base margin is read from ([retail-EV arm](training-config.md#retail-ev-arm)). `odds_analytics.training.tuner.apply_trial_params` applies the values in the objective and for `--train-best`.
- **Objective.** Each trial deep-copies the config, applies the sampled values, and (with `use_kfold`) stacks train and validation rows and runs `odds_analytics.training.cross_validation.run_cv` ([cross-validation](cross-validation.md)), passing the rows' sample weights and kick-off times, which `odds_analytics.training.tuner.create_objective` requires for `retail_ev` ([sample weights](training-pipeline.md#sample-weights)). `tuning.metric` is one of `val_mse` (default), `val_mae` or `val_r2` and maps to the mean CV value. The direction follows the metric (`val_r2` maximises, the others minimise) through `odds_analytics.training.config.TuningConfig.direction`; the config has no `direction` key. Tuning never reads test rows: a non-CV trial is scored on the validation rows and needs a validation split.
- **Failed trials.** The objective catches nothing: an exception (or a metric missing from a non-CV trial's history) fails the trial, Optuna records it as `FAIL` and `odds_analytics.training.tuner.OptunaTuner.optimize` re-raises, aborting the study. `TrialPruned` is recorded as pruned.
- **Feature selection.** With `top_k_features` in the search space and `feature_selection.enabled`, `odds_analytics.training.feature_selection.get_feature_selector` (`manual`, `filter`, `ensemble`, `hybrid`) ranks features once per study (when the study has no stored ranking) and later trials keep the top k.
- **CLI.** `--n-trials`, `--timeout`, `--study-name` (default `<experiment>_<UTC timestamp>`; an existing name resumes), `--storage`, `--output` (default `<config>_best.yaml`), `--train-best`. `--storage` defaults to the SQLite file `optuna.db` in `experiments/`, relative to the working directory (gitignored), never `DATABASE_URL`: Optuna's schema check fails against a database that already holds another `alembic_version` row (reproduced on Optuna 4.6.0: `RuntimeError: ... is no longer compatible with the table schema`, in the storage-schema test in `tests/unit/test_tuner.py`). `--in-memory` opts out and conflicts with `--storage` (`odds_cli.commands.train.run_tuning`). `--train-best` is rejected when `top_k_features` is searched, because the selected subset is not saved with the config. `--train-best` saves the model with the full config `.yaml` beside it, as `odds train run` does ([training pipeline](training-pipeline.md#steps-of-odds-train-run)).

## Example search spaces

`experiments/configs/xgboost_tuning_without_feature_selection.yaml` and `experiments/configs/xgboost_feature_selection_tuning.yaml` are the committed XGBoost tuning examples, and the values are authoritative there; the second adds `top_k_features` with feature selection enabled. Both set `retail_bookmakers` to `["fanduel", "draftkings", "betmgm_us"]`, the `odds_analytics.training.config.FeatureConfig` default, where US BetMGM is the canonical key `betmgm_us`. `odds_analytics.training.config.XGBoostConfig` has no early-stopping field, so a search space tunes `n_estimators` as the only control on boosting length. Neither file sets `target_type`, which `odds_analytics.training.config.FeatureConfig` requires, so a copy needs one added.

## Related

- [Cross-validation](cross-validation.md), [Feature pipeline](feature-pipeline.md)
