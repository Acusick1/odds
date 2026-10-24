---
type: reference
title: Training config
description: The MLTrainingConfig schema, every section's fields and defaults, data-source selection, and values the schema accepts that cannot train.
sources:
  - odds_analytics.training.config.MLTrainingConfig
  - odds_analytics.training.config.DataConfig
  - odds_analytics.training.config.FeatureConfig
  - odds_analytics.training.config.RetailEvArm
  - odds_analytics.retail_ev.BASELINE_FEATURE
  - odds_analytics.training.config.SamplingConfig
  - odds_analytics.training.config.TuningConfig
  - odds_cli.commands.train.STRATEGY_CLASSES
  - odds_analytics.xgboost_line_movement.XGBoostLineMovementStrategy.train_from_config
  - experiments/configs/xgboost_epl_retail_ev.yaml
status: stable
---

# Training config

`odds_analytics.training.config` is authoritative for fields and defaults; configs live in `experiments/configs/`, and `experiments/configs/xgboost_epl_retail_ev.yaml` is the EPL example. `odds_analytics.training.config.XGBoostConfig` has no early-stopping field and rejects unknown keys, so boosting length is set only by `n_estimators`. What `odds train` does with a config is on [Training pipeline](training-pipeline.md).

## Sections

Every model forbids unknown keys (`extra="forbid"`). A file has `experiment` (name, tags), `training` (`strategy_type`, `data`, `model`, `features`, `feature_selection`, output settings), optional `tuning` and `tracking`, and optional `base:` inheritance (deep-merged). Model and feature settings such as `target_type` and `sharp_bookmakers` are YAML config, not environment Settings ([configuration](../foundations/configuration.md)).

**`training.data`**

- `data_source`: `oddsportal` (`op_` ids), `football_data_uk` (`fduk_`), `oddsapi` (everything else), `all` or null (no filter). The filter is on event-id prefix ([events](../storage/events.md)). `sport_key`, `start_date`/`end_date` (required), `min_snapshots`.
- CV: `cv_method` (`walk_forward`, `timeseries` default, `kfold`), `window_type` (`expanding` default or `sliding`), `min_train_events`, `max_train_events`, `val_step_events`, `n_folds` ([cross-validation](cross-validation.md)). `use_kfold` defaults true and `test_split` to 0.2.
- The code defaults are `cv_method: timeseries` and `test_split: 0.2`, so a walk-forward protocol (for example `experiments/configs/xgboost_epl_retail_ev.yaml`: `cv_method: walk_forward`, `window_type: expanding`, `min_train_events: 380`, `val_step_events: 50`, `test_split: 0.0`) exists only where a YAML sets it ([cross-validation](cross-validation.md)).
- `strategy_type` is `xgboost_line_movement` or `lstm_line_movement` (`odds_analytics.training.config.StrategyType`), the keys of `odds_cli.commands.train.STRATEGY_CLASSES`; any other value fails validation.
- `n_folds` is ignored for `walk_forward` and is required (checked at run time by `odds_analytics.training.cross_validation.run_cv`, not by the schema) for `kfold` and `timeseries`.

**`training.features`**

- `target_type` (required: `raw` or `retail_ev`), `retail_ev_arm` (`direct` default, or `residual` under `retail_ev` only; [retail-EV arm](#retail-ev-arm)), `sharp_bookmakers` (priority order, default `["pinnacle"]`), `retail_bookmakers`, `closing_tier`, `outcome`, `markets`, `feature_groups`, `form_window`, `variance_threshold`, `sampling` ([target functions](clv-target-functions.md), [sharp benchmark](sharp-benchmark.md), [decision time](look-ahead-bias.md), [feature pipeline](feature-pipeline.md)).
- `odds_analytics.training.config.SamplingConfig`: `strategy` (`time_range` default, or `tier`), `decision_tier` (default `pregame`, used by `tier`), and `min_hours` 3.0, `max_hours` 12.0 and `max_samples_per_event` 5 (used by `time_range`). ([feature pipeline](feature-pipeline.md)).
- `training.features` and `target_type` have no defaults; `retail_ev` is the target of [ADR-0012](../../docs/adr/0012-retail-ev-against-betfair-exchange-close.md).
- `retail_ev` needs the `xgboost` adapter, market `1x2` or `h2h` and `feature_groups` drawn only from `odds_analytics.training.config.RetailEvFeatureGroupName` (`polymarket_at_t`, `dynamics`, `schedule_as_of_t`, `standings_as_of_t`; empty is the price features alone) (`odds_analytics.training.config.FeatureConfig`), and `use_kfold` with `cv_method: walk_forward`, `test_split: 0` and `validation_split: 0` (`odds_analytics.training.config.TrainingConfig`, and again in `odds_analytics.training.cross_validation.run_cv`). It rejects a value other than the default for the fields it does not read: `outcome`, `sampling`, `sharp_bookmakers`, `retail_bookmakers`, `closing_tier`, `lookback_hours` and `timesteps`. Its price features are built in, and `standings_as_of_t` reads `form_window` ([target functions](clv-target-functions.md#retail-ev-features)). `experiments/configs/xgboost_epl_retail_ev.yaml` is its EPL config, from 2024-08-01 under expanding-380:50. Its loss weights have a fixed scale, so `min_child_weight` and `reg_lambda` are in units of about 156 per event ([target functions](clv-target-functions.md#retail-ev-rows)).
- `sharp_bookmakers` defaults to Pinnacle only, while the hybrid reference is `[pinnacle, betfair_exchange]`; the MCP default differs on purpose (#421, [ADR-0003](../../docs/adr/0003-hybrid-sharp-reference.md)).
- `feature_groups` accepts the names in `odds_analytics.training.config.FeatureGroupName` for the `raw` target and only `retail_ev`'s own groups for `retail_ev`, must be non-empty for the `xgboost` adapter under `raw` (the `lstm` adapter may leave it empty), and each `feature_groups` choice in a tuning search space is validated by `odds_analytics.training.config.MLTrainingConfig` at load.
- The primary market is `markets[0]` by position (#475). `odds_analytics.training.config.FeatureConfig` carries bookmaker rosters, sampling, LSTM, Polymarket and standings settings, so every committed YAML serialises about 30 unrelated fields.

**`tuning`**

`metric` is `val_mse`, `val_mae` or `val_r2`; the optimization direction follows from it and is not configurable. Search-space keys must be model fields, `feature_groups` or `top_k_features`, and a non-CV `tuning` section needs `validation_split > 0` ([tuner](hyperparameter-tuner.md)).

## Retail-EV arm

`retail_ev_arm` (`odds_analytics.training.config.RetailEvArm`) chooses what XGBoost fits under `target_type: retail_ev`:

- `direct`: the EV target.
- `residual`: the EV target minus the baseline. `odds_analytics.xgboost_line_movement.XGBoostLineMovementStrategy.train_from_config` names the `baseline_ev` feature (`odds_analytics.retail_ev.BASELINE_FEATURE`) as each row's base margin, so the trees start from the baseline, with no fitted intercept, and a model that learns nothing predicts the baseline. Training, CV, tuning and prediction all score the target itself, because the margin is read from the feature matrix ([XGBoost strategy](line-movement-models.md)). A feature matrix without that column raises `ValueError`.

Any other target rejects `residual`. `odds_analytics.training.config.MLTrainingConfig` also rejects a residual-arm config with MLflow `tracking` enabled, because MLflow logs the bare XGBoost model, whose predictions omit the margin, and one that tunes `top_k_features` with feature selection enabled, which could drop `baseline_ev`. A saved residual model carries its `margin_feature`. XGBoost ignores `base_score` when given a base margin, so the residual arm has no intercept; a script that fits the target minus the baseline with XGBoost's default fits one, the residual's weighted mean.

## Related

- [Training pipeline](training-pipeline.md), [Hyperparameter tuner](hyperparameter-tuner.md), [CLV target](clv-target.md)
