---
type: component
title: Hyperparameter tuner (Optuna)
summary: OptunaTuner and its objective, search spaces (including the principled space), study storage, feature selection, and the tuner's error-swallowing behaviour.
sources:
  - packages/odds-analytics/odds_analytics/training/tuner.py::OptunaTuner
  - packages/odds-analytics/odds_analytics/training/tuner.py::create_objective
  - packages/odds-analytics/odds_analytics/training/tuner.py::suggest_params_from_search_space
  - packages/odds-analytics/odds_analytics/training/config.py::SearchSpace
  - packages/odds-analytics/odds_analytics/training/config.py::TuningConfig
  - packages/odds-analytics/odds_analytics/training/feature_selection.py::FeatureSelector
  - packages/odds-analytics/odds_analytics/training/feature_selection.py::get_feature_selector
  - packages/odds-cli/odds_cli/commands/train.py::run_tuning
  - docs/CLI.md#tune-hyperparameters
  - docs/MODELING.md#principled-search-space-and-protocol-re-test
verified_at: 7887039
---

# Hyperparameter tuner (Optuna)

`packages/odds-analytics/odds_analytics/training/tuner.py` wraps Optuna. `odds train tune` builds an `OptunaTuner` and an objective from a config's `tuning` section ([training-pipeline](training-pipeline.md)); search-space fields are in [training-config](training-config.md).

- **Tuner.** `OptunaTuner(study_name, direction, sampler, pruner, storage, tracking_config)`. Samplers `tpe`, `random`, `cmaes` are all seeded 42; pruners `median`, `hyperband`, `none`. The study is created with `load_if_exists=True`, so rerunning a name with the same storage appends trials to the old study. With MLflow tracking, a parent run holds study metadata and each trial is a nested run.
- **Search spaces.** `SearchSpace` is `int`, `float` (optional `step`, `log`) or `categorical`; list choices become tuples. Keys that are `FeatureConfig` fields are applied to the features config (for example `feature_groups`, pre-computed once per choice); everything else is applied to the model config.
- **Objective.** Each trial deep-copies the config, applies the sampled values, and (with `use_kfold`) stacks train and validation rows and runs `run_cv` ([walk-forward-cv](walk-forward-cv.md)). `tuning.metric` (default `val_mse`, direction `minimize`) maps to the mean CV value; `cv_*` keys read the CV dict.
- **Feature selection.** With `top_k_features` in the search space and `feature_selection.enabled`, `get_feature_selector` (`manual`, `filter`, `ensemble`, `hybrid`) ranks features once on trial 0 and later trials keep the top k. Not supported for LSTM.
- **CLI.** `--n-trials`, `--timeout`, `--study-name` (default `<experiment>_tuning`), `--storage`, `--output` (default `<config>_best.yaml`), `--train-best`.

## The principled search space

The default space let sliding-380:50 collapse to constant predictors; the revised space used for the deployed model is `n_estimators` 100-600 (step 50), `max_depth` 2-4, `learning_rate` 0.01-0.1 (log), `min_child_weight` 1-15, `subsample` and `colsample_bytree` 0.5-1.0, `reg_lambda` 0.01-5.0 (log), `gamma` and `reg_alpha` fixed at 0, `early_stopping_rounds` 50. Results and reasons (mostly `(inferred)`) are in [exp-principled-search-space](exp-principled-search-space.md). The committed config is `experiments/configs/xgboost_epl_production_tabular_match_stats.yaml`.

## Hazards

- **Trial errors become ±infinity.** Any exception other than `TrialPruned` is logged and the trial returns `inf` (or `-inf` when maximising); a missing metric does too. A study of all-failed trials still reports a "best" trial. Combined with `strategy_type: xgboost` (rejected by `train_from_config`), every trial can fail silently (AUD T5/A3, re-verified).
- **Unsupported CV metrics fall back.** A metric outside `val_mse`, `val_mae`, `val_r2` and `cv_*` uses `val_mse` with a warning; `direction` is never checked against the metric (`val_r2` with `minimize` minimises R²).
- **Unknown search-space keys are dropped** by the `hasattr` guards; they never raise.
- **`ImportError` handling is a no-op.** The block re-raises whenever Optuna exposes `TrialPruned`, which it always does, so import errors abort the study rather than failing the trial.
- **Storage default does not work.** `--storage` is documented as defaulting to `DATABASE_URL`; the lookup calls a non-existent `get_config`, so studies are in-memory unless `--storage` is passed ([training-pipeline](training-pipeline.md)).
- **Results are tied to the study name.** Reusing a name continues an old study with different data or protocol.

## Related

- [Walk-forward CV](walk-forward-cv.md), [Feature pipeline](feature-pipeline.md)
