---
type: component
title: Feature pipeline
summary: The five layers of feature_groups.py - EventDataBundle collection, TierSampler/TimeRangeSampler sampling, targets, XGBoost/LSTM adapters, the orchestrator - and the sampler guards.
sources:
  - packages/odds-analytics/odds_analytics/feature_groups.py::EventDataBundle
  - packages/odds-analytics/odds_analytics/feature_groups.py::collect_event_data
  - packages/odds-analytics/odds_analytics/feature_groups.py::TierSampler
  - packages/odds-analytics/odds_analytics/feature_groups.py::TimeRangeSampler
  - packages/odds-analytics/odds_analytics/feature_groups.py::XGBoostAdapter
  - packages/odds-analytics/odds_analytics/feature_groups.py::LSTMAdapter
  - packages/odds-analytics/odds_analytics/feature_groups.py::FeatureGroupCaches
  - packages/odds-analytics/odds_analytics/feature_groups.py::prepare_training_data
  - packages/odds-analytics/odds_analytics/feature_extraction.py::SequenceFeatureExtractor
  - docs/MODELING.md#guards-against-known-failure-modes
  - docs/DATA_MODELS.md#feature-groups
verified_at: 7887039
---

# Feature pipeline

`packages/odds-analytics/odds_analytics/feature_groups.py` (about 1,300 lines) is the one entry point that turns events into `(X, y)`. Its docstring names five layers:

1. **Collection.** `collect_event_data` bulk-loads one `EventDataBundle` per event: snapshots containing the configured market, the closing snapshot, plus Polymarket, injury, rest, standings, match-stats, schedule, lineup and sequence data, each only if its group is configured. Standings, match stats, ESPN fixtures and lineups come from `FeatureGroupCaches`, filled once by `preload_feature_group_caches` so per-event queries stay constant ([features-epl-context](features-epl-context.md)).
2. **Sampling.** `SnapshotSampler` implementations choose decision snapshots: `TierSampler` (one row per event) or `TimeRangeSampler` (up to `max_samples_per_event`, stratified bins) ([decision-time](decision-time.md)).
3. **Target.** Pure functions from `sequence_loader` ([clv](clv.md)).
4. **Adapters.** `XGBoostAdapter` concatenates the static group arrays (names prefixed `tab_`, `pm_`, `xsrc_`, `inj_`, `rest_`, `stnd_`, `mstat_`, `eplsched_`, `epllu_`) plus `hours_until_event`. `LSTMAdapter` builds a per-timestep sequence and mask with `SequenceFeatureExtractor`, plus an optional static vector.
5. **Orchestrator.** `prepare_training_data` loops over FINAL events and returns `PreparedFeatureData` (`X`, `y`, `feature_names`, `masks`, `event_ids`). Events are skipped when there is no closing snapshot, no target-bookmaker closing data, none of the sharp bookmakers at close ([sharp-benchmark](sharp-benchmark.md)), or no sampled snapshot. Missing optional data NaN-fills and keeps the row. Constant columns are then dropped by `apply_variance_filter` (`variance_threshold` 0.0), which is why `feature_names` can be shorter than the configured groups ([feature-groups](feature-groups.md)).

Row order matters: `event_ids` follow the input event order (`OddsReader.get_events_by_date_range` sorts by `commence_time`), which the walk-forward splitter relies on ([walk-forward-cv](walk-forward-cv.md)).

## Guards against past failures

- **In-play leak.** `TierSampler` skips `IN_PLAY` snapshots unless that is the decision tier ([exp-early-nba-signal-search](exp-early-nba-signal-search.md)).
- **Cross-source contamination (#231).** `TierSampler` takes `required_bookmakers` (any-of) so a snapshot without the sharp book cannot win on timestamp, and features keep NaN instead of being zero-filled; results from before 2026-03-15 are invalid ([exp-epl-sharp-reference](exp-epl-sharp-reference.md)).
- The over-regularised search space is a tuner problem ([exp-principled-search-space](exp-principled-search-space.md)).

## Hazards

- **Adding a group touches several places.** `_STATIC_FEATURE_GROUPS`, `_static_feature_group_names`, `_extract_static_feature_parts`, `collect_event_data` and the preload caches each branch on group names by string; there is no registry (AUD B5, re-verified).
- **Per-group failures NaN-fill silently.** Extraction exceptions are caught and logged at debug level and the block becomes NaN. Only a tabular failure drops the row.
- **Docstring drift.** `LSTMAdapter` says 14 sequence features; `SequenceFeatures` has 15 fields.
- **Standings and schedule caches are EPL-only.** `load_season_events_cache` raises for any other `sport_key`.

## Related

- [Training pipeline](training-pipeline.md), [Look-ahead bias](look-ahead-bias.md)
