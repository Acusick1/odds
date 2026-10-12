---
type: component
title: Training config (MLTrainingConfig)
summary: The YAML schema for training.data, training.model and training.features - field meanings, choosing data for an experiment, and invalid values that are accepted silently.
sources:
  - packages/odds-analytics/odds_analytics/training/config.py::MLTrainingConfig
  - packages/odds-analytics/odds_analytics/training/config.py::DataConfig
  - packages/odds-analytics/odds_analytics/training/config.py::FeatureConfig
  - packages/odds-analytics/odds_analytics/training/config.py::SamplingConfig
  - packages/odds-analytics/odds_analytics/training/config.py::TuningConfig
  - packages/odds-cli/odds_cli/commands/train.py::validate_config
  - packages/odds-cli/odds_cli/commands/train.py::list_configs
  - experiments/configs/xgboost_epl_production_tabular_match_stats.yaml
  - docs/DATA_MODELS.md#configuration
  - docs/CLI.md#validate-config
verified_at: 7887039
---

# Training config (MLTrainingConfig)

`packages/odds-analytics/odds_analytics/training/config.py` is authoritative for fields and defaults; configs live in `experiments/configs/` (the production example is `experiments/configs/xgboost_epl_production_tabular_match_stats.yaml`). Every model forbids unknown keys (`extra="forbid"`). A file has `experiment` (name, tags), `training` (`strategy_type`, `data`, `model`, `features`, `feature_selection`, output settings), optional `tuning` and `tracking`, and optional `base:` inheritance (deep-merged). `odds train validate` loads the file and checks strategy, dates and splits; `odds train list-configs` tabulates a directory ([training-pipeline](training-pipeline.md)).

## training.data

- `data_source`: `oddsportal` (`op_` ids), `football_data_uk` (`fduk_`), `oddsapi` (everything else), `all` or null (no filter). `sport_key`, `start_date`/`end_date` (required), `min_snapshots`.
- CV: `cv_method` (`walk_forward`, `timeseries` default, `kfold`), `window_type` (`expanding` default or `sliding`), `min_train_events`, `max_train_events`, `val_step_events`, `n_folds` ([walk-forward-cv](walk-forward-cv.md)). `use_kfold` defaults true and `test_split` to 0.2; set `test_split: 0.0` for walk-forward ([decision-cv-protocol](decision-cv-protocol.md)).

## training.features

- `target_type` (`raw` default, `devigged_bookmaker`), `target_bookmaker` (default `pinnacle`), `sharp_bookmakers` (priority order), `retail_bookmakers`, `closing_source_priority`, `outcome`, `markets` (first entry is the primary market), `feature_groups`, `form_window`, `match_stats_window`, `variance_threshold`, `sampling` ([clv](clv.md), [sharp-benchmark](sharp-benchmark.md), [decision-time](decision-time.md), [feature-groups](feature-groups.md)).

## Choosing data

`target_bookmaker` must belong to the selected source. EPL with bet365 target and hybrid sharp uses `data_source: null`, `sharp_bookmakers: [pinnacle, betfair_exchange]` and `closing_source_priority: [football_data_uk]`. NBA bet365 needs `oddsportal`; NBA Pinnacle needs `oddsapi`. LSTM runs also set `min_snapshots: 5` or more ([decision-hybrid-sharp-reference](decision-hybrid-sharp-reference.md)).

## Hazards

Values the schema accepts without error:

- **Unknown `feature_groups`.** The field is a free `tuple[str, ...]`; a typo is ignored. Its description still lists `trajectory`, which no code handles.
- **`strategy_type: xgboost` / `lstm`.** Legal in the schema, rejected when training and swallowed per trial in tuning ([hyperparameter-tuner](hyperparameter-tuner.md)).
- **Search-space keys that match nothing** are dropped silently.
- **`target_type` defaults to `raw`**, not the documented devigged target.
- **Target bookmaker absent from the source** drops events, counted only in a log line.
- **`n_folds` is ignored for `walk_forward`** and is required (checked at run time, not by the schema) for `kfold` and `timeseries`.
- **One class, many concerns.** `FeatureConfig` carries bookmaker rosters, sampling, LSTM, Polymarket and EPL knobs, so every committed YAML serialises about 30 unrelated fields, and the primary market is `markets[0]` by position (AUD B5, re-verified).

## Related

- [Training pipeline](training-pipeline.md), [Hyperparameter tuner](hyperparameter-tuner.md)
