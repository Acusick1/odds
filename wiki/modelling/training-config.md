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

`odds_analytics.training.config` is authoritative for fields and defaults; configs live in `experiments/configs/` and the production example is `experiments/configs/xgboost_epl_production_tabular_match_stats.yaml`. What `odds train` does with a config is on [Training pipeline](training-pipeline.md).

## Sections

Every model forbids unknown keys (`extra="forbid"`). A file has `experiment` (name, tags), `training` (`strategy_type`, `data`, `model`, `features`, `feature_selection`, output settings), optional `tuning` and `tracking`, and optional `base:` inheritance (deep-merged). Model and feature settings such as `target_type` and `sharp_bookmakers` are YAML config, not environment Settings ([configuration](../foundations/configuration.md)).

**`training.data`**

- `data_source`: `oddsportal` (`op_` ids), `football_data_uk` (`fduk_`), `oddsapi` (everything else), `all` or null (no filter). The filter is on event-id prefix ([events](../storage/events.md)). `sport_key`, `start_date`/`end_date` (required), `min_snapshots`.
- CV: `cv_method` (`walk_forward`, `timeseries` default, `kfold`), `window_type` (`expanding` default or `sliding`), `min_train_events`, `max_train_events`, `val_step_events`, `n_folds` ([cross-validation](cross-validation.md)). `use_kfold` defaults true and `test_split` to 0.2.
- The code defaults are `cv_method: timeseries` and `test_split: 0.2`, so the production protocol (`cv_method: walk_forward`, `window_type: sliding`, 380/380/50, `test_split: 0.0`) exists only where a YAML sets it ([ADR-0004](../../docs/adr/0004-walk-forward-cv-protocol.md)).
- `n_folds` is ignored for `walk_forward` and is required (checked at run time by `odds_analytics.training.cross_validation.run_cv`, not by the schema) for `kfold` and `timeseries`.

**`training.features`**

- `target_type` (`raw` default, `devigged_bookmaker`), `target_bookmaker` (default `pinnacle`), `sharp_bookmakers` (priority order, default `["pinnacle"]`), `retail_bookmakers`, `closing_source_priority`, `outcome`, `markets`, `feature_groups`, `form_window`, `match_stats_window`, `variance_threshold`, `sampling` ([target functions](clv-target-functions.md), [sharp benchmark](sharp-benchmark.md), [decision time](look-ahead-bias.md), [feature pipeline](feature-pipeline.md)).
- A YAML that omits `target_type` and `target_bookmaker` trains the raw mean-implied-probability delta against Pinnacle (#434); the production target is devigged bet365 ([ADR-0002](../../docs/adr/0002-bet365-clv-target.md)).
- `sharp_bookmakers` defaults to Pinnacle only, while the hybrid reference sets `[pinnacle, betfair_exchange]`; the MCP default differs on purpose (#421, [ADR-0003](../../docs/adr/0003-hybrid-sharp-reference.md)).
- The primary market is `markets[0]` by position (#475). `odds_analytics.training.config.FeatureConfig` carries bookmaker rosters, sampling, LSTM, Polymarket and EPL settings, so every committed YAML serialises about 30 unrelated fields.
- The production YAML lists bet365 in `retail_bookmakers` as well as using it as the target bookmaker.

## Choosing data

`target_bookmaker` must belong to the selected source. EPL with the bet365 target and hybrid sharp uses `data_source: null`, `sharp_bookmakers: [pinnacle, betfair_exchange]` and `closing_source_priority: [football_data_uk]`. NBA bet365 needs `oddsportal`; NBA Pinnacle needs `oddsapi`. LSTM runs also set `min_snapshots: 5` or more. A target bookmaker absent from the source drops events, counted only in a log line.

## Values accepted but not trainable

- `feature_groups` is a free `tuple[str, ...]`, so an unknown name is ignored and its description still lists `trajectory`, which no code handles (#475, #473).
- `strategy_type: xgboost` and `lstm` validate, but `odds_cli.commands.train.STRATEGY_CLASSES` and `odds_analytics.xgboost_line_movement.XGBoostLineMovementStrategy.train_from_config` reject them; under `tune` the resulting error is swallowed per trial (#437, [tuner](hyperparameter-tuner.md)).
- Search-space keys that match nothing are dropped silently (#437).

## Related

- [Training pipeline](training-pipeline.md), [Hyperparameter tuner](hyperparameter-tuner.md), [CLV target](clv-target.md)
