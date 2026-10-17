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
- **Search spaces.** `odds_analytics.training.config.SearchSpace` is `int`, `float` (optional `step` and a log-scale flag) or `categorical`; list choices become tuples. Keys that are `odds_analytics.training.config.FeatureConfig` fields are applied to the features config (for example `feature_groups`, pre-computed once per choice); everything else is applied to the model config. Keys that match nothing are dropped by the `hasattr` guards and never raise (#437).
- **Objective.** Each trial deep-copies the config, applies the sampled values, and (with `use_kfold`) stacks train and validation rows and runs `odds_analytics.training.cross_validation.run_cv` ([cross-validation](cross-validation.md)). `tuning.metric` (default `val_mse`, direction `minimize`) maps to the mean CV value; `cv_*` keys read the CV dict. A metric outside `val_mse`, `val_mae`, `val_r2` and `cv_*` uses `val_mse` with a warning, and `direction` is never checked against the metric, so `val_r2` with `minimize` minimises R² (#437).
- **Failed trials.** Any exception except `TrialPruned`, and a missing metric, make the trial return `inf` (or `-inf` when maximising); a study whose trials all fail still reports a best trial (#437). An `ImportError` is re-raised whenever Optuna exposes `TrialPruned`, so it aborts the study (#437).
- **Feature selection.** With `top_k_features` in the search space and `feature_selection.enabled`, `odds_analytics.training.feature_selection.get_feature_selector` (`manual`, `filter`, `ensemble`, `hybrid`) ranks features once on trial 0 and later trials keep the top k. Not supported for LSTM.
- **CLI.** `--n-trials`, `--timeout`, `--study-name` (default `<experiment>_tuning`), `--storage`, `--output` (default `<config>_best.yaml`), `--train-best`. `--storage` is meant to default to `DATABASE_URL`, but the lookup in `odds_cli.commands.train.run_tuning` imports a non-existent `get_config`, so studies are in-memory unless `--storage` is passed (#437). The model from `--train-best` is not scorable: its `.yaml` lacks `training` ([prediction scoring](prediction-scoring.md)).

## Production search space

The search space of the deployed model is in `experiments/configs/xgboost_epl_production_tabular_match_stats.yaml` (PR #362), and the values are authoritative there. The deployed model `v308058c` was trained from that YAML as at commit `2bf3712`, which set early stopping on the validation fold ([caveat](../evaluation/clv-model-findings.md#caveats)). The search space tunes `n_estimators`, which is the only control on boosting length. Most older committed YAMLs carry a different search space, so copy it from the `experiments/configs/xgboost_epl_production_*` configs when starting a new sliding-window run. Why it replaced the original, and the results under each, are on [EPL feature ablation and principled search space](../evaluation/exp-epl-feature-ablation-and-search-space.md#principled-search-space-re-test).

## Related

- [Cross-validation](cross-validation.md), [Feature pipeline](feature-pipeline.md)
