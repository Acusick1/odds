---
type: reference
title: Feature pipeline
description: The five layers of odds_analytics.feature_groups, caches, samplers, adapters, orchestrator, NaN-fill and variance policy.
sources:
  - odds_analytics.feature_groups.EventDataBundle
  - odds_analytics.feature_groups.collect_event_data
  - odds_analytics.feature_groups.TierSampler
  - odds_analytics.feature_groups.TimeRangeSampler
  - odds_analytics.feature_groups.XGBoostAdapter
  - odds_analytics.feature_groups.LSTMAdapter
  - odds_analytics.feature_groups.FeatureGroupCaches
  - odds_analytics.feature_groups.prepare_training_data
  - odds_analytics.feature_groups._extract_static_feature_parts
  - odds_analytics.feature_extraction.SequenceFeatureExtractor
  - odds_analytics.feature_extraction.SequenceFeatures
  - odds_analytics.training.feature_selection.apply_variance_filter
status: stable
---

# Feature pipeline

`odds_analytics.feature_groups` (about 1,300 lines) is the one entry point that turns events into `(X, y)`. `odds_analytics.training.config.FeatureConfig.feature_groups` selects which groups to compose, and may be empty only for the `lstm` adapter ([training config](training-config.md)); the groups themselves are catalogued on [Feature groups](feature-groups.md). The module docstring names five layers.

1. **Collection.** `odds_analytics.feature_groups.collect_event_data` bulk-loads one `odds_analytics.feature_groups.EventDataBundle` per event: snapshots containing the configured market, the closing snapshot, plus Polymarket, injury, rest, standings, match-stats, schedule and sequence data, each only if its group is configured. Standings, match stats and ESPN fixtures come from `odds_analytics.feature_groups.FeatureGroupCaches`, filled once by `odds_analytics.feature_groups.preload_feature_group_caches` so per-event queries stay constant ([EPL context features](epl-context-features.md)). The standings and schedule caches are EPL-only.
2. **Sampling.** `odds_analytics.feature_groups.SnapshotSampler` implementations choose decision snapshots: `odds_analytics.feature_groups.TierSampler` (one row per event) or `odds_analytics.feature_groups.TimeRangeSampler` (up to `max_samples_per_event`, stratified bins). The sampler defaults are `time_range` with a 3-12 h window, but the production model samples `tier` with `decision_tier: sharp` ([decision time](look-ahead-bias.md)). `odds_analytics.feature_groups.TierSampler` skips `IN_PLAY` snapshots unless that is the decision tier (commit 2a84d88), and takes `required_bookmakers` (any-of) so a snapshot without the sharp book cannot win on timestamp, leaving features NaN instead of zero-filled (PR #231).
3. **Target.** Pure functions from `odds_analytics.sequence_loader` ([target functions](clv-target-functions.md)). `target_type` has no default; `target_bookmaker` is set only for `devigged_bookmaker`.
4. **Adapters.** `odds_analytics.feature_groups.XGBoostAdapter` concatenates the static group arrays (names prefixed `tab_`, `pm_`, `xsrc_`, `inj_`, `rest_`, `stnd_`, `mstat_`, `eplsched_`) plus `hours_until_event`. `odds_analytics.feature_groups.LSTMAdapter` builds a per-timestep sequence and mask with `odds_analytics.feature_extraction.SequenceFeatureExtractor`, plus an optional static vector ([line-movement models](line-movement-models.md)). The sequence has 15 per-timestep `odds_analytics.feature_extraction.SequenceFeatures` (odds, implied probability, hours to game, time-of-day sin/cos, changes from previous and opening, spread, sharp odds and divergence); the sharp fields are unset when no target outcome is given ([sharp benchmark](sharp-benchmark.md#priority-semantics)).
5. **Orchestrator.** `odds_analytics.feature_groups.prepare_training_data` loops over FINAL events and returns `odds_analytics.feature_groups.PreparedFeatureData` (`X`, `y`, `odds_analytics.feature_groups.PreparedFeatureData.feature_names`, `masks`, `event_ids`). Events are skipped when there is no closing snapshot, no target-bookmaker closing data, none of the sharp bookmakers at close, or no sampled snapshot. The sharp filter is any-of over `sharp_bookmakers`, whose code default is `["pinnacle"]` ([sharp benchmark](sharp-benchmark.md)).

Row order matters: `event_ids` follow the input event order (`odds_lambda.storage.readers.OddsReader.get_events_by_date_range` sorts by `commence_time`), which the walk-forward splitter relies on ([cross-validation](cross-validation.md)).

## NaN-fill and variance policy

- When optional data is missing the block is NaN and the row is kept (`odds_analytics.feature_groups._extract_static_feature_parts`); only a tabular failure drops a row. XGBoost handles NaN natively.
- Extraction exceptions inside a group are caught and logged at debug level and the block becomes NaN, so a per-group failure is silent.
- Strictly constant columns are then dropped by `odds_analytics.training.feature_selection.apply_variance_filter` (`variance_threshold` 0.0), which is why the feature names can be shorter than the configured groups (`rest` adds 0 EPL features).
- Scoring depends on the variance filter: inline scoring aborts unless `odds_analytics.feature_groups.XGBoostAdapter` feature names equal the names saved with the model, so a dropped constant column changes what a model accepts ([prediction scoring](prediction-scoring.md)).

To add a group, see [Add a feature group](howto-add-feature-group.md). Which groups carry signal is on [CLV model findings](../evaluation/clv-model-findings.md#current-conclusions).

## Related

- [Training config](training-config.md), [training pipeline](training-pipeline.md), [CLV target](clv-target.md), [sharp benchmark](sharp-benchmark.md)
