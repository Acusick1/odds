---
type: howto
title: Add a feature group
description: Steps to add a static feature group to the pipeline and verify it.
sources:
  - odds_analytics.feature_groups._STATIC_FEATURE_GROUPS
  - odds_analytics.feature_groups._static_feature_group_names
  - odds_analytics.feature_groups._extract_static_feature_parts
  - odds_analytics.feature_groups.collect_event_data
  - odds_analytics.feature_groups.preload_feature_group_caches
  - odds_analytics.feature_groups.EventDataBundle
  - odds_analytics.training.config.FeatureConfig
  - odds_analytics.training.config.FeatureGroupName
status: stable
---

# Add a feature group

Group names are dispatched by string in several places, so a new group touches each of them (see [Feature groups](feature-groups.md)).

1. **Write the extractor.** Create a dataclass with array and feature-name methods (as `odds_analytics.standings_features.StandingsFeatures.to_array` and `odds_analytics.standings_features.StandingsFeatures.get_feature_names`), and an extractor that takes only data before the snapshot.
   - The look-ahead rule per layer is on [Decision time and look-ahead bias](look-ahead-bias.md).
2. **Register the name.** Add it to `odds_analytics.training.config.FeatureGroupName` (which `odds_analytics.feature_groups._STATIC_FEATURE_GROUPS` derives from), `odds_analytics.feature_groups._static_feature_group_names` (choose a prefix) and `odds_analytics.feature_groups._extract_static_feature_parts`.
   - The dispatch functions branch on the group name, so a name missed in one of them is dropped without an error.
3. **Load its data.** Add the loading in `odds_analytics.feature_groups.collect_event_data` (which drops reference-only Polymarket snapshots) (or a cache in `odds_analytics.feature_groups.preload_feature_group_caches`) and the field in `odds_analytics.feature_groups.EventDataBundle`.
4. **Describe it in the config.** Mention it in the `feature_groups` description of `odds_analytics.training.config.FeatureConfig`.
   - The field is validated against `FeatureGroupName`, so a name missing from step 2 fails at config load, including as a `feature_groups` tuning choice ([training config](training-config.md)).
   - A config that uses the group also sets `target_type`, which has no default.
5. **Verify.** Run a training config that selects the group and check the verification artifacts for feature count and all-NaN columns ([run an experiment](../evaluation/experiment-scripts.md#verification-artifacts)).
   - Zeroed or all-NaN columns once hid an adapter bug.

Add the group to the catalogue on [Feature groups](feature-groups.md) in the same change.
