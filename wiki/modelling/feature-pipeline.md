---
type: reference
title: Feature pipeline
description: The five layers of odds_analytics.feature_groups, samplers, adapters, orchestrator, NaN-fill and variance policy.
sources:
  - odds_analytics.feature_groups.EventDataBundle
  - odds_analytics.feature_groups.SnapshotQuotes
  - odds_analytics.feature_groups.collect_event_data
  - odds_analytics.feature_groups.raw_closes
  - odds_analytics.feature_groups.TierSampler
  - odds_analytics.feature_groups.TimeRangeSampler
  - odds_analytics.feature_groups.XGBoostAdapter
  - odds_analytics.feature_groups.prepare_training_data
  - odds_analytics.feature_groups._extract_static_feature_parts
  - odds_analytics.training.feature_selection.apply_variance_filter
status: stable
---

# Feature pipeline

`odds_analytics.feature_groups` (about 900 lines) is the one entry point that turns events into `(X, y)`. `odds_analytics.training.config.FeatureConfig.feature_groups` selects which groups to compose, and may be empty only for `retail_ev` ([training config](training-config.md)); the groups themselves are catalogued on [Feature groups](feature-groups.md). The module docstring names five layers.

1. **Collection.** `odds_analytics.feature_groups.collect_event_data` bulk-loads one `odds_analytics.feature_groups.EventDataBundle` per event: the back quotes of the configured market (`odds_core.quotes.quotes`), one `odds_analytics.feature_groups.SnapshotQuotes` per snapshot, without Polymarket's; each source's closing snapshot (its latest CLOSING snapshot, `odds_analytics.feature_groups.raw_closes`, [closing snapshot](clv-target-functions.md#closing-snapshot-of-a-raw-target)); plus injury and rest data, each only if its group is configured.
2. **Sampling.** `odds_analytics.feature_groups.SnapshotSampler` implementations choose decision snapshots: `odds_analytics.feature_groups.TierSampler` (one row per event) or `odds_analytics.feature_groups.TimeRangeSampler` (up to `max_samples_per_event`, stratified bins). The sampler default is `time_range` with a 3-12 h window; `tier` samples one row per event at `decision_tier` ([decision time](look-ahead-bias.md)), a snapshot's tier derived by `odds_analytics.feature_groups.tiers_of` (`odds_lambda.tier_utils.snapshot_tiers`) from its time, source, time basis and market. `odds_analytics.feature_groups.TierSampler` skips `IN_PLAY` snapshots unless that is the decision tier (commit 2a84d88), and takes `required_bookmakers` (any-of) so a snapshot without the sharp book cannot win on timestamp, leaving features NaN instead of zero-filled (PR #231). Both samplers take `sources`, and `odds_analytics.feature_groups.prepare_training_data` passes the sources whose close has a sharp book, so the sampled snapshot always has a close of its own source to be measured against.
3. **Target.** `odds_analytics.targets.calculate_regression_target` for the `raw` target ([target functions](clv-target-functions.md)). `target_type` has no default. `retail_ev` bypasses layers 1 to 4: `odds_analytics.retail_ev.prepare_retail_ev_training_data` loads its own quotes for every event it is given, whatever its status or scores, and keeps the events that have a close; it builds its rows, its features (with its own as-of-t groups, which load the EPL results and ESPN fixtures themselves rather than through this module's caches), sample weights (the fixed `odds_analytics.retail_ev.LOSS_WEIGHT_SCALE`, 1,013, times the per-event weight over decimal odds squared, passed to XGBoost unchanged) and kick-off times ([target functions](clv-target-functions.md#retail-ev-rows)).
4. **Adapters.** `odds_analytics.feature_groups.XGBoostAdapter` concatenates the static group arrays (names prefixed `tab_`, `inj_`, `rest_`) plus `hours_until_event`.
5. **Orchestrator.** `odds_analytics.feature_groups.prepare_training_data` loops over FINAL events and returns `odds_analytics.feature_groups.PreparedFeatureData` (`X`, `y`, `odds_analytics.feature_groups.PreparedFeatureData.feature_names`, `event_ids`, and `sample_weights` and `commence_times` for `retail_ev` only). Under `retail_ev` it delegates as above and rejects a `sampler`. Events are skipped when they have no close, when no close has any of the sharp bookmakers, or when no snapshot is sampled; a decision snapshot is measured against its own source's close and gives no row when that source has none ([closing snapshot](clv-target-functions.md#closing-snapshot-of-a-raw-target)). The optional `sampler` argument, an `odds_analytics.feature_groups.SnapshotSampler`, replaces the sampler that `config.sampling` specifies; the closing-snapshot and sharp-bookmaker checks do not depend on it. The sharp filter is any-of over `sharp_bookmakers`, whose code default is `["pinnacle"]` ([sharp benchmark](sharp-benchmark.md)).

Row order matters: `event_ids` follow the input event order (`odds_lambda.storage.readers.OddsReader.get_events_by_date_range` sorts by `commence_time`, then event ID, so kick-off ties have a fixed order that does not depend on the query plan; its `min_snapshots` filter counts distinct snapshot times), which the walk-forward splitter relies on ([cross-validation](cross-validation.md)).

## NaN-fill and variance policy

- When optional data is missing the block is NaN and the row is kept (`odds_analytics.feature_groups._extract_static_feature_parts`); only a tabular failure drops a row. The tabular extractor takes each event as an `odds_analytics.feature_extraction.FeatureEvent`, made by `odds_analytics.feature_groups.make_feature_event`. XGBoost handles NaN natively.
- Extraction exceptions inside a group are caught and logged at debug level and the block becomes NaN, so a per-group failure is silent.
- Strictly constant columns of the 2D feature matrix are then dropped by `odds_analytics.training.feature_selection.apply_variance_filter` (`variance_threshold` 0.0), which is why the feature names can be shorter than the configured groups (`rest` adds 0 EPL features).

To add a group, see [Add a feature group](howto-add-feature-group.md).

## Related

- [Training config](training-config.md), [training pipeline](training-pipeline.md), [CLV target](clv-target.md), [sharp benchmark](sharp-benchmark.md)
