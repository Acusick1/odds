---
type: reference
title: Hyperparameter tuner (Optuna)
description: OptunaTuner, the objective, search-space schema, feature selection and CLI options, with the production search space cited by file.
sources:
  - odds_analytics.training.tuner.OptunaTuner
  - odds_analytics.training.tuner.create_objective
  - odds_analytics.training.tuner.suggest_params_from_search_space
  - odds_analytics.training.config.SearchSpace
  - odds_analytics.training.config.TuningConfig
  - odds_analytics.training.feature_selection.FeatureSelector
  - odds_analytics.training.feature_selection.get_feature_selector
  - odds_cli.commands.train.run_tuning
  - experiments/configs/xgboost_epl_production_tabular_match_stats.yaml
status: stable
---

# Hyperparameter tuner (Optuna)

`odds_analytics.training.tuner` wraps Optuna. `odds train tune` builds an `odds_analytics.training.tuner.OptunaTuner` and an objective from a config's `tuning` section ([training pipeline](training-pipeline.md)); search-space fields are part of the config schema ([training config](training-config.md)).

- **Tuner.** `OptunaTuner(study_name, direction, sampler, pruner, storage, tracking_config)`. Samplers `tpe`, `random`, `cmaes` are all seeded 42; pruners `median`, `hyperband`, `none`. The study is created with `load_if_exists=True`, so rerunning a name with the same storage appends trials to the old study, including one run with different data or protocol. With MLflow tracking, a parent run holds study metadata and each trial is a nested run.
- **Search spaces.** `odds_analytics.training.config.SearchSpace` is `int`, `float` (optional `step` and a log-scale flag) or `categorical`; list choices become tuples. `odds_analytics.training.tuner.apply_trial_params` applies sampled values, in the objective and for `--train-best`: a key that is a model field goes to the model config, a key that is an `odds_analytics.training.config.FeatureConfig` field goes to the features config (for example `feature_groups`, pre-computed once per choice), and a key on both (the LSTM `lookback_hours` and `timesteps`) goes to both. A key that matches neither (and is not `top_k_features`) fails `odds_analytics.training.config.MLTrainingConfig` validation, as does any `feature_groups` choice that is not a valid group list.
- **Objective.** Each trial deep-copies the config, applies the sampled values, and (with `use_kfold`) stacks train and validation rows and runs `odds_analytics.training.cross_validation.run_cv` ([cross-validation](cross-validation.md)). `tuning.metric` is one of `val_mse` (default), `val_mae` or `val_r2` and maps to the mean CV value. The direction follows the metric (`val_r2` maximises, the others minimise) through `odds_analytics.training.config.TuningConfig.direction`; the config has no `direction` key. Tuning never reads test rows: a non-CV trial is scored on the validation rows and needs a validation split.
- **Failed trials.** The objective catches nothing: an exception (or a metric missing from a non-CV trial's history) fails the trial, Optuna records it as `FAIL` and `odds_analytics.training.tuner.OptunaTuner.optimize` re-raises, aborting the study. `TrialPruned` is recorded as pruned.
- **Feature selection.** With `top_k_features` in the search space and `feature_selection.enabled`, `odds_analytics.training.feature_selection.get_feature_selector` (`manual`, `filter`, `ensemble`, `hybrid`) ranks features once on trial 0 and later trials keep the top k. Not supported for LSTM.
- **CLI.** `--n-trials`, `--timeout`, `--study-name` (default `<experiment>_tuning`), `--storage`, `--output` (default `<config>_best.yaml`), `--train-best`. `--storage` defaults to the local SQLite file the local SQLite file `optuna.db` under `experiments/` (gitignored), never `DATABASE_URL`: Optuna's tables would clash with the application database's alembic history. `--in-memory` opts out and conflicts with `--storage` (`odds_cli.commands.train.run_tuning`). `--train-best` is rejected when `top_k_features` is searched, because the selected subset is not saved with the config. The model from `--train-best` carries the full config `.yaml`, so it is scorable ([prediction scoring](prediction-scoring.md)).

## Production search space

The search space of the deployed model is in `experiments/configs/xgboost_epl_production_tabular_match_stats.yaml` (PR #362), and the values are authoritative there. The deployed model `v308058c` was trained from that YAML as at commit `2bf3712`, which set early stopping on the validation fold ([caveat](../evaluation/clv-model-findings.md#caveats)). The search space tunes `n_estimators`, which is the only control on boosting length. Most older committed YAMLs carry a different search space, so copy it from the `experiments/configs/xgboost_epl_production_*` configs when starting a new sliding-window run. Why it replaced the original, and the results under each, are on [EPL feature ablation and principled search space](../evaluation/exp-epl-feature-ablation-and-search-space.md#principled-search-space-re-test).

## Related

- [Cross-validation](cross-validation.md), [Feature pipeline](feature-pipeline.md)
