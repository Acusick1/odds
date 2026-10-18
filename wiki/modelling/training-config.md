---
type: reference
title: Training config
description: The MLTrainingConfig schema, every section's fields and defaults, data-source selection, and values the schema accepts that cannot train.
sources:
  - odds_analytics.training.config.MLTrainingConfig
  - odds_analytics.training.config.DataConfig
  - odds_analytics.training.config.FeatureConfig
  - odds_analytics.training.config.SamplingConfig
  - odds_analytics.training.config.TuningConfig
  - odds_cli.commands.train.STRATEGY_CLASSES
  - odds_analytics.xgboost_line_movement.XGBoostLineMovementStrategy.train_from_config
  - experiments/configs/xgboost_epl_production_tabular_match_stats.yaml
status: stable
---

# Training config

`odds_analytics.training.config` is authoritative for fields and defaults; configs live in `experiments/configs/` and the production example is `experiments/configs/xgboost_epl_production_tabular_match_stats.yaml`, whose metrics are on [CLV model findings](../evaluation/clv-model-findings.md#deployed-model). `odds_analytics.training.config.XGBoostConfig` has no early-stopping field and rejects unknown keys, so boosting length is set only by `n_estimators`. What `odds train` does with a config is on [Training pipeline](training-pipeline.md).

## Sections

Every model forbids unknown keys (`extra="forbid"`). A file has `experiment` (name, tags), `training` (`strategy_type`, `data`, `model`, `features`, `feature_selection`, output settings), optional `tuning` and `tracking`, and optional `base:` inheritance (deep-merged). Model and feature settings such as `target_type` and `sharp_bookmakers` are YAML config, not environment Settings ([configuration](../foundations/configuration.md)).

**`training.data`**

- `data_source`: `oddsportal` (`op_` ids), `football_data_uk` (`fduk_`), `oddsapi` (everything else), `all` or null (no filter). The filter is on event-id prefix ([events](../storage/events.md)). `sport_key`, `start_date`/`end_date` (required), `min_snapshots`.
- CV: `cv_method` (`walk_forward`, `timeseries` default, `kfold`), `window_type` (`expanding` default or `sliding`), `min_train_events`, `max_train_events`, `val_step_events`, `n_folds` ([cross-validation](cross-validation.md)). `use_kfold` defaults true and `test_split` to 0.2.
- The code defaults are `cv_method: timeseries` and `test_split: 0.2`, so the production protocol (`cv_method: walk_forward`, `window_type: sliding`, 380/380/50, `test_split: 0.0`) exists only where a YAML sets it ([ADR-0004](../../docs/adr/0004-walk-forward-cv-protocol.md)).
- `n_folds` is ignored for `walk_forward` and is required (checked at run time by `odds_analytics.training.cross_validation.run_cv`, not by the schema) for `kfold` and `timeseries`.

**`training.features`**

- `target_type` (required: `raw` or `devigged_bookmaker`), `target_bookmaker` (required), `sharp_bookmakers` (priority order, default `["pinnacle"]`), `retail_bookmakers`, `closing_source_priority`, `outcome`, `markets`, `feature_groups`, `form_window`, `match_stats_window`, `variance_threshold`, `sampling` ([target functions](clv-target-functions.md), [sharp benchmark](sharp-benchmark.md), [decision time](look-ahead-bias.md), [feature pipeline](feature-pipeline.md)).
- `odds_analytics.training.config.SamplingConfig`: `strategy` (`time_range` default, or `tier`), `decision_tier` (default `pregame`, used by `tier`), and `min_hours` 3.0, `max_hours` 12.0 and `max_samples_per_event` 5 (used by `time_range`). Production sets `strategy: tier` with `decision_tier: sharp` ([feature pipeline](feature-pipeline.md)).
- `training.features` and its `target_type` and `target_bookmaker` have no defaults; the production target is devigged bet365 ([ADR-0002](../../docs/adr/0002-bet365-clv-target.md)).
- `sharp_bookmakers` defaults to Pinnacle only, while the hybrid reference sets `[pinnacle, betfair_exchange]`; the MCP default differs on purpose (#421, [ADR-0003](../../docs/adr/0003-hybrid-sharp-reference.md)).
- `feature_groups` accepts only the names in `odds_analytics.training.config.FeatureGroupName`, must be non-empty for the `xgboost` adapter (the `lstm` adapter may leave it empty), and each `feature_groups` choice in a tuning search space is validated by `odds_analytics.training.config.MLTrainingConfig` at load.
- The primary market is `markets[0]` by position (#475). `odds_analytics.training.config.FeatureConfig` carries bookmaker rosters, sampling, LSTM, Polymarket and EPL settings, so every committed YAML serialises about 30 unrelated fields.
- The production YAML lists bet365 in `retail_bookmakers` as well as using it as the target bookmaker.

## Data source and bookmaker

`target_bookmaker` must belong to the selected `data_source`; a target bookmaker absent from the source drops events, counted only in a log line. The data-source and bookmaker combinations used by the committed configs are set out in [Train and publish a model](howto-train-and-publish-model.md#choose-data).

## Values accepted but not trainable

- `strategy_type: xgboost` and `lstm` validate, but `odds_cli.commands.train.STRATEGY_CLASSES` and `odds_analytics.xgboost_line_movement.XGBoostLineMovementStrategy.train_from_config` reject them; under `tune` the resulting error is swallowed per trial (#437, [tuner](hyperparameter-tuner.md)).
- Search-space keys that match nothing are dropped silently (#437).

## Related

- [Training pipeline](training-pipeline.md), [Hyperparameter tuner](hyperparameter-tuner.md), [CLV target](clv-target.md)
