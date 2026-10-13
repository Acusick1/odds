---
type: component
title: Feature pipeline and feature groups
description: How feature_groups.py turns events into (X, y) — collection, sampling, target, adapters, orchestrator — the catalogue of feature groups and their column counts, the NaN-fill and variance policies, and how to add a group.
sources:
  - odds_analytics.feature_groups.EventDataBundle
  - odds_analytics.feature_groups.collect_event_data
  - odds_analytics.feature_groups.TierSampler
  - odds_analytics.feature_groups.TimeRangeSampler
  - odds_analytics.feature_groups.XGBoostAdapter
  - odds_analytics.feature_groups.LSTMAdapter
  - odds_analytics.feature_groups.FeatureGroupCaches
  - odds_analytics.feature_groups.prepare_training_data
  - odds_analytics.feature_groups._STATIC_FEATURE_GROUPS
  - odds_analytics.feature_extraction.SequenceFeatureExtractor
  - odds_analytics.feature_extraction.TabularFeatures
  - odds_analytics.feature_extraction.SequenceFeatures
  - odds_analytics.standings_features.StandingsFeatures
  - odds_analytics.match_stats_features.MatchStatsFeatures
  - odds_analytics.epl_schedule_features.EplScheduleFeatures
  - odds_analytics.epl_lineup_features.EplLineupFeatures
  - odds_analytics.schedule_features.RestScheduleFeatures
  - odds_analytics.injury_features.InjuryFeatures
  - odds_analytics.injury_features.extract_injury_features
  - odds_analytics.polymarket_features.PolymarketTabularFeatures
status: stable
---

# Feature pipeline and feature groups

`odds_analytics.feature_groups` (about 1,300 lines) is the one entry point that turns events into `(X, y)`. `odds_analytics.training.config.FeatureConfig.feature_groups` selects which groups to compose ([training pipeline](training-pipeline.md)). The module docstring names five layers.

1. **Collection.** `odds_analytics.feature_groups.collect_event_data` bulk-loads one `odds_analytics.feature_groups.EventDataBundle` per event: snapshots containing the configured market, the closing snapshot, plus Polymarket, injury, rest, standings, match-stats, schedule, lineup and sequence data, each only if its group is configured. Standings, match stats, ESPN fixtures and lineups come from `odds_analytics.feature_groups.FeatureGroupCaches`, filled once by `odds_analytics.feature_groups.preload_feature_group_caches` so per-event queries stay constant ([EPL context features](epl-context-features.md)).
2. **Sampling.** `odds_analytics.feature_groups.SnapshotSampler` implementations choose decision snapshots: `odds_analytics.feature_groups.TierSampler` (one row per event) or `odds_analytics.feature_groups.TimeRangeSampler` (up to `max_samples_per_event`, stratified bins). The sampler defaults are `time_range` with a 3-12 h window, but the production model samples `tier` with `decision_tier: sharp` ([decision time](decision-time-and-look-ahead.md)).
3. **Target.** Pure functions from `sequence_loader`. `target_type` defaults to `raw` and `target_bookmaker` to `pinnacle`, so configs must set both ([CLV target](clv-target.md)).
4. **Adapters.** `odds_analytics.feature_groups.XGBoostAdapter` concatenates the static group arrays (names prefixed `tab_`, `pm_`, `xsrc_`, `inj_`, `rest_`, `stnd_`, `mstat_`, `eplsched_`, `epllu_`) plus `hours_until_event`. `odds_analytics.feature_groups.LSTMAdapter` builds a per-timestep sequence and mask with `odds_analytics.feature_extraction.SequenceFeatureExtractor`, plus an optional static vector ([line-movement models](line-movement-models.md)).
5. **Orchestrator.** `odds_analytics.feature_groups.prepare_training_data` loops over FINAL events and returns `odds_analytics.feature_groups.PreparedFeatureData` (`X`, `y`, `feature_names`, `masks`, `event_ids`). Events are skipped when there is no closing snapshot, no target-bookmaker closing data, none of the sharp bookmakers at close, or no sampled snapshot. The sharp filter is any-of over `sharp_bookmakers`, whose code default is `["pinnacle"]` ([sharp benchmark](sharp-benchmark.md)). Missing optional data NaN-fills and keeps the row. Constant columns are then dropped by `odds_analytics.training.feature_selection.apply_variance_filter` (`variance_threshold` 0.0), which is why `feature_names` can be shorter than the configured groups.

Row order matters: `event_ids` follow the input event order (`odds_lambda.storage.readers.OddsReader.get_events_by_date_range` sorts by `commence_time`), which the walk-forward splitter relies on ([walk-forward CV](decision-walk-forward-cv.md)).

Scoring depends on the variance filter: inline scoring aborts unless `XGBoostAdapter.feature_names(config)` equals the names saved with the model, so a dropped constant column changes what a model accepts ([publishing and scoring](model-publishing-and-scoring.md)).

## Feature groups

Counts are dataclass fields, checked against code; each ends up as one prefixed column block in `odds_analytics.feature_groups.XGBoostAdapter` output.

| Group | Cols | Sports | Notes |
|---|---|---|---|
| `tabular` | 6 | all | `consensus_prob`, `sharp_prob`, `retail_sharp_diff`, `num_bookmakers`, `is_weekend`, `day_of_week`; outcome-relative. See [sharp benchmark](sharp-benchmark.md) |
| `standings` | 11 | EPL | position, points gap, goal difference, form (`form_window`), goal rates |
| `match_stats` | 14 | EPL | rolling means (`match_stats_window`) of shots, shots on target, corners, fouls, cards, half-time goals |
| `epl_schedule` | 8 | EPL | rest days, midweek, matches in 14 days, European match in 7 days |
| `epl_lineup` | 6 | EPL | starting-XI changes and cumulative starts lost |
| `rest` | 5 | NBA | days rest and back-to-back flags from game logs; all NaN for EPL, so the variance filter removes it |
| `injuries` | 6 | NBA | impact-weighted OUT and GTD burden, report timing |
| `polymarket` | 16 | NBA | 14 Polymarket price/order-book columns plus 2 cross-source divergences |

EPL groups are detailed in [EPL context features](epl-context-features.md); injuries in [NBA injuries and stats](../collection/nba-injuries-and-stats.md) (point-in-time: only reports at or before the snapshot, latest report only; GTD weighted 0.5); Polymarket in [Polymarket](../collection/polymarket.md). The LSTM adapter instead uses 15 per-timestep `odds_analytics.feature_extraction.SequenceFeatures` (odds, implied probability, hours to game, time-of-day sin/cos, changes from previous and opening, spread, sharp odds and divergence), plus optional static groups.

Which groups carry signal: `match_stats` helps on EPL; injuries, rest, FPL and Polymarket add nothing ([EPL ablation](../evaluation/exp-epl-feature-ablation-and-search-space.md), [NBA timing and injuries](../evaluation/exp-nba-timing-and-injuries.md), [early NBA signal search](../evaluation/exp-early-nba-signal-and-lstm.md)).

### NaN-fill and variance policy

When optional data is missing the block is NaN and the row is kept (`odds_analytics.feature_groups._extract_static_feature_parts`); only a tabular failure drops a row. XGBoost handles NaN natively. Strictly constant columns are dropped afterwards, so counts in a trained model can be lower than above (`rest` adds 0 EPL features).

### Adding a group

1. Write a dataclass with `to_array` and `get_feature_names`, and an extractor that takes only data before the snapshot ([look-ahead bias](decision-time-and-look-ahead.md#hazards)).
2. Register the name in `_STATIC_FEATURE_GROUPS`, `odds_analytics.feature_groups._static_feature_group_names` (choose a prefix) and `odds_analytics.feature_groups._extract_static_feature_parts`.
3. Load its data in `odds_analytics.feature_groups.collect_event_data` (or a cache in `odds_analytics.feature_groups.preload_feature_group_caches`) and add the field to `odds_analytics.feature_groups.EventDataBundle`.
4. Mention it in the `feature_groups` description of `odds_analytics.training.config.FeatureConfig` (nothing validates it).

Before trusting a new group, run the experiment verification artifacts, which check feature counts and all-NaN columns ([run an experiment](../evaluation/howto-run-experiment.md)).

## Guards against past failures

- **In-play leak.** `odds_analytics.feature_groups.TierSampler` skips `IN_PLAY` snapshots unless that is the decision tier ([early NBA signal search](../evaluation/exp-early-nba-signal-and-lstm.md); fixed in commit 2a84d88).
- **Cross-source contamination (PR #231).** `odds_analytics.feature_groups.TierSampler` takes `required_bookmakers` (any-of) so a snapshot without the sharp book cannot win on timestamp, and features keep NaN instead of being zero-filled; results from before 2026-03-15 are invalid ([hybrid sharp reference](decision-hybrid-sharp-reference.md)).
- The over-regularised search space is a tuner problem ([principled search space](../evaluation/exp-epl-feature-ablation-and-search-space.md)).

## Hazards

- **Adding a group touches several places.** `_STATIC_FEATURE_GROUPS`, `odds_analytics.feature_groups._static_feature_group_names`, `odds_analytics.feature_groups._extract_static_feature_parts`, `odds_analytics.feature_groups.collect_event_data` and the preload caches each branch on group names by string; there is no registry (AUD B5, re-verified).
- **Per-group failures NaN-fill silently.** Extraction exceptions are caught and logged at debug level and the block becomes NaN. Only a tabular failure drops the row.
- **Docstring drift.** `odds_analytics.feature_groups.LSTMAdapter` says 14 sequence features; `odds_analytics.feature_extraction.SequenceFeatures` has 15 fields.
- **Standings and schedule caches are EPL-only.** `odds_analytics.standings_features.load_season_events_cache` raises for any other `sport_key`.
- **Counts differ from older results.** Experiments before 2026-03-15 used a 7-feature tabular group and `trajectory` and `cross_source` groups, none of which the code defines; Exp 1 and Exp 2 numbers are not reproducible as recorded.
- **`fpl_availability` is not a group.** The 2026-03-20 FPL runs cannot be reproduced.
- **Injury weights are not point-in-time checked.** Impact weights use season-level player stats loaded once per event; nothing in the extractor checks they were known at the snapshot time (read from code).

## Related

- [Training pipeline](training-pipeline.md), [CLV target](clv-target.md), [Sharp benchmark](sharp-benchmark.md)
