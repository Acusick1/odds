---
type: howto
title: Add a feature group
description: Steps to add a static feature group to the pipeline and verify it.
sources:
  - odds_analytics.feature_groups._STATIC_FEATURE_GROUPS
  - odds_analytics.feature_groups._static_feature_group_names
  - odds_analytics.feature_groups._extract_static_feature_parts
  - odds_analytics.feature_groups.collect_event_data
  - odds_analytics.feature_groups.EventDataBundle
  - odds_analytics.training.config.FeatureConfig
  - odds_analytics.training.config.FeatureGroupName
  - odds_analytics.training.config.RetailEvFeatureGroupName
  - odds_analytics.retail_ev.RetailEvFeatures
status: stable
---

# Add a feature group

These steps add a static group for the `raw` target, as `tabular`, `injuries` and `rest` are. Group names are dispatched by string in several places, so a new group touches each of them (see [Feature groups](feature-groups.md)). A `retail_ev` group is computed as of each row's decision time instead: it goes in `odds_analytics.training.config.RetailEvFeatureGroupName` and the group table behind `odds_analytics.retail_ev.RetailEvFeatures`, with no other source's price at t for a row whose decision time is only an upper bound, and into the look-ahead tests of `tests/unit/test_retail_ev.py` (`TestNoLookAhead`), which rebuild each row from price rows truncated at its decision time ([retail-EV features](clv-target-functions.md#retail-ev-features)). Such a group is not available to the [retail-EV model rule](../evaluation/candidate-rules.md#the-retail-ev-model-rule), whose inputs come from one snapshot. A `residual` arm (`odds_analytics.training.config.FeatureConfig.retail_ev_arm`) takes the `odds_analytics.retail_ev.BASELINE_FEATURE` column as its base margin, and `odds_analytics.xgboost_line_movement.XGBoostLineMovementStrategy.train` raises if that column is not among the features.

1. **Write the extractor.** Create a dataclass with array and feature-name methods (as `odds_analytics.schedule_features.RestScheduleFeatures` does), and an extractor that takes only data before the snapshot.
   - The look-ahead rule per layer is on [Decision time and look-ahead bias](look-ahead-bias.md).
2. **Register the name.** Add it to `odds_analytics.training.config.FeatureGroupName` (which `odds_analytics.feature_groups._STATIC_FEATURE_GROUPS` derives from), `odds_analytics.feature_groups._static_feature_group_names` (choose a prefix) and `odds_analytics.feature_groups._extract_static_feature_parts`, which gives the extractors the event as the `odds_analytics.feature_extraction.FeatureEvent` that `odds_analytics.feature_groups.make_feature_event` builds and the decision snapshot's quotes (`odds_analytics.feature_groups.SnapshotQuotes`).
   - The dispatch functions branch on the group name, so a name missed in one of them is dropped without an error.
3. **Load its data.** Add the loading in `odds_analytics.feature_groups.collect_event_data` (which loads the primary market's quotes, one `odds_analytics.feature_groups.SnapshotQuotes` per snapshot, drops Polymarket snapshots and picks each source's raw-target close with `odds_analytics.feature_groups.raw_closes`) and the field, with a default, in `odds_analytics.feature_groups.EventDataBundle`, whose `snapshots` and `closes` hold those quotes.
4. **Describe it in the config.** Mention it in the `feature_groups` description of `odds_analytics.training.config.FeatureConfig`.
   - The field is validated against `FeatureGroupName`, so a name missing from step 2 fails at config load, including as a `feature_groups` tuning choice ([training config](training-config.md)).
   - A config that uses the group also sets `target_type`, which has no default.
   - A `retail_ev` config rejects every static group: they are computed as of kick-off, and its rows need features as of each row's decision time ([CLV and the training target](clv-target.md#the-retail-ev-target)).
5. **Verify.** Run a training config that selects the group and check the verification artifacts for feature count and all-NaN columns ([run an experiment](../evaluation/experiment-scripts.md#verification-artifacts)).
   - Zeroed or all-NaN columns once hid an adapter bug.

Add the group to the catalogue on [Feature groups](feature-groups.md) in the same change.
