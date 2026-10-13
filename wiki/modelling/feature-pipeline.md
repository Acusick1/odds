---
type: component
title: Feature pipeline and feature groups
description: How feature_groups.py turns events into (X, y) — collection, sampling, target, adapters, orchestrator — the catalogue of feature groups and their column counts, the NaN-fill and variance policies, and how to add a group.
sources:
  - packages/odds-analytics/odds_analytics/feature_groups.py::EventDataBundle
  - packages/odds-analytics/odds_analytics/feature_groups.py::collect_event_data
  - packages/odds-analytics/odds_analytics/feature_groups.py::TierSampler
  - packages/odds-analytics/odds_analytics/feature_groups.py::TimeRangeSampler
  - packages/odds-analytics/odds_analytics/feature_groups.py::XGBoostAdapter
  - packages/odds-analytics/odds_analytics/feature_groups.py::LSTMAdapter
  - packages/odds-analytics/odds_analytics/feature_groups.py::FeatureGroupCaches
  - packages/odds-analytics/odds_analytics/feature_groups.py::prepare_training_data
  - packages/odds-analytics/odds_analytics/feature_groups.py::_STATIC_FEATURE_GROUPS
  - packages/odds-analytics/odds_analytics/feature_extraction.py::SequenceFeatureExtractor
  - packages/odds-analytics/odds_analytics/feature_extraction.py::TabularFeatures
  - packages/odds-analytics/odds_analytics/feature_extraction.py::SequenceFeatures
  - packages/odds-analytics/odds_analytics/standings_features.py::StandingsFeatures
  - packages/odds-analytics/odds_analytics/match_stats_features.py::MatchStatsFeatures
  - packages/odds-analytics/odds_analytics/epl_schedule_features.py::EplScheduleFeatures
  - packages/odds-analytics/odds_analytics/epl_lineup_features.py::EplLineupFeatures
  - packages/odds-analytics/odds_analytics/schedule_features.py::RestScheduleFeatures
  - packages/odds-analytics/odds_analytics/injury_features.py::InjuryFeatures
  - packages/odds-analytics/odds_analytics/injury_features.py::extract_injury_features
  - packages/odds-analytics/odds_analytics/polymarket_features.py::PolymarketTabularFeatures
status: stable
---

# Feature pipeline and feature groups

`packages/odds-analytics/odds_analytics/feature_groups.py` (about 1,300 lines) is the one entry point that turns events into `(X, y)`. `FeatureConfig.feature_groups` selects which groups to compose ([training pipeline](training-pipeline.md)). The module docstring names five layers.

1. **Collection.** `packages/odds-analytics/odds_analytics/feature_groups.py::collect_event_data` bulk-loads one `packages/odds-analytics/odds_analytics/feature_groups.py::EventDataBundle` per event: snapshots containing the configured market, the closing snapshot, plus Polymarket, injury, rest, standings, match-stats, schedule, lineup and sequence data, each only if its group is configured. Standings, match stats, ESPN fixtures and lineups come from `packages/odds-analytics/odds_analytics/feature_groups.py::FeatureGroupCaches`, filled once by `packages/odds-analytics/odds_analytics/feature_groups.py::preload_feature_group_caches` so per-event queries stay constant ([EPL context features](epl-context-features.md)).
2. **Sampling.** `packages/odds-analytics/odds_analytics/feature_groups.py::SnapshotSampler` implementations choose decision snapshots: `packages/odds-analytics/odds_analytics/feature_groups.py::TierSampler` (one row per event) or `packages/odds-analytics/odds_analytics/feature_groups.py::TimeRangeSampler` (up to `max_samples_per_event`, stratified bins). The sampler defaults are `time_range` with a 3-12 h window, but the production model samples `tier` with `decision_tier: sharp` ([decision time](decision-time-and-look-ahead.md)).
3. **Target.** Pure functions from `sequence_loader`. `target_type` defaults to `raw` and `target_bookmaker` to `pinnacle`, so configs must set both ([CLV target](clv-target.md)).
4. **Adapters.** `packages/odds-analytics/odds_analytics/feature_groups.py::XGBoostAdapter` concatenates the static group arrays (names prefixed `tab_`, `pm_`, `xsrc_`, `inj_`, `rest_`, `stnd_`, `mstat_`, `eplsched_`, `epllu_`) plus `hours_until_event`. `packages/odds-analytics/odds_analytics/feature_groups.py::LSTMAdapter` builds a per-timestep sequence and mask with `packages/odds-analytics/odds_analytics/feature_extraction.py::SequenceFeatureExtractor`, plus an optional static vector ([line-movement models](line-movement-models.md)).
5. **Orchestrator.** `packages/odds-analytics/odds_analytics/feature_groups.py::prepare_training_data` loops over FINAL events and returns `packages/odds-analytics/odds_analytics/feature_groups.py::PreparedFeatureData` (`X`, `y`, `feature_names`, `masks`, `event_ids`). Events are skipped when there is no closing snapshot, no target-bookmaker closing data, none of the sharp bookmakers at close, or no sampled snapshot. The sharp filter is any-of over `sharp_bookmakers`, whose code default is `["pinnacle"]` ([sharp benchmark](sharp-benchmark.md)). Missing optional data NaN-fills and keeps the row. Constant columns are then dropped by `packages/odds-analytics/odds_analytics/training/feature_selection.py::apply_variance_filter` (`variance_threshold` 0.0), which is why `feature_names` can be shorter than the configured groups.

Row order matters: `event_ids` follow the input event order (`OddsReader.get_events_by_date_range` sorts by `commence_time`), which the walk-forward splitter relies on ([walk-forward CV](decision-walk-forward-cv.md)).

Scoring depends on the variance filter: inline scoring aborts unless `XGBoostAdapter.feature_names(config)` equals the names saved with the model, so a dropped constant column changes what a model accepts ([publishing and scoring](model-publishing-and-scoring.md)).

## Feature groups

Counts are dataclass fields, checked against code; each ends up as one prefixed column block in `packages/odds-analytics/odds_analytics/feature_groups.py::XGBoostAdapter` output.

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

EPL groups are detailed in [EPL context features](epl-context-features.md); injuries in [NBA injuries and stats](../collection/nba-injuries-and-stats.md) (point-in-time: only reports at or before the snapshot, latest report only; GTD weighted 0.5); Polymarket in [Polymarket](../collection/polymarket.md). The LSTM adapter instead uses 15 per-timestep `packages/odds-analytics/odds_analytics/feature_extraction.py::SequenceFeatures` (odds, implied probability, hours to game, time-of-day sin/cos, changes from previous and opening, spread, sharp odds and divergence), plus optional static groups.

Which groups carry signal: `match_stats` helps on EPL; injuries, rest, FPL and Polymarket add nothing ([EPL ablation](../evaluation/exp-epl-feature-ablation-and-search-space.md), [NBA timing and injuries](../evaluation/exp-nba-timing-and-injuries.md), [early NBA signal search](../evaluation/exp-early-nba-signal-and-lstm.md)).

### NaN-fill and variance policy

When optional data is missing the block is NaN and the row is kept (`packages/odds-analytics/odds_analytics/feature_groups.py::_extract_static_feature_parts`); only a tabular failure drops a row. XGBoost handles NaN natively. Strictly constant columns are dropped afterwards, so counts in a trained model can be lower than above (`rest` adds 0 EPL features).

### Adding a group

1. Write a dataclass with `to_array` and `get_feature_names`, and an extractor that takes only data before the snapshot ([look-ahead bias](decision-time-and-look-ahead.md#hazards)).
2. Register the name in `_STATIC_FEATURE_GROUPS`, `packages/odds-analytics/odds_analytics/feature_groups.py::_static_feature_group_names` (choose a prefix) and `packages/odds-analytics/odds_analytics/feature_groups.py::_extract_static_feature_parts`.
3. Load its data in `packages/odds-analytics/odds_analytics/feature_groups.py::collect_event_data` (or a cache in `packages/odds-analytics/odds_analytics/feature_groups.py::preload_feature_group_caches`) and add the field to `packages/odds-analytics/odds_analytics/feature_groups.py::EventDataBundle`.
4. Mention it in the `feature_groups` description of `packages/odds-analytics/odds_analytics/training/config.py::FeatureConfig` (nothing validates it).

Before trusting a new group, run the experiment verification artifacts, which check feature counts and all-NaN columns ([run an experiment](../evaluation/howto-run-experiment.md)).

## Guards against past failures

- **In-play leak.** `packages/odds-analytics/odds_analytics/feature_groups.py::TierSampler` skips `IN_PLAY` snapshots unless that is the decision tier ([early NBA signal search](../evaluation/exp-early-nba-signal-and-lstm.md); fixed in commit 2a84d88).
- **Cross-source contamination (PR #231).** `packages/odds-analytics/odds_analytics/feature_groups.py::TierSampler` takes `required_bookmakers` (any-of) so a snapshot without the sharp book cannot win on timestamp, and features keep NaN instead of being zero-filled; results from before 2026-03-15 are invalid ([hybrid sharp reference](decision-hybrid-sharp-reference.md)).
- The over-regularised search space is a tuner problem ([principled search space](../evaluation/exp-epl-feature-ablation-and-search-space.md)).

## Hazards

- **Adding a group touches several places.** `_STATIC_FEATURE_GROUPS`, `packages/odds-analytics/odds_analytics/feature_groups.py::_static_feature_group_names`, `packages/odds-analytics/odds_analytics/feature_groups.py::_extract_static_feature_parts`, `packages/odds-analytics/odds_analytics/feature_groups.py::collect_event_data` and the preload caches each branch on group names by string; there is no registry (AUD B5, re-verified).
- **Per-group failures NaN-fill silently.** Extraction exceptions are caught and logged at debug level and the block becomes NaN. Only a tabular failure drops the row.
- **Docstring drift.** `packages/odds-analytics/odds_analytics/feature_groups.py::LSTMAdapter` says 14 sequence features; `packages/odds-analytics/odds_analytics/feature_extraction.py::SequenceFeatures` has 15 fields.
- **Standings and schedule caches are EPL-only.** `packages/odds-analytics/odds_analytics/standings_features.py::load_season_events_cache` raises for any other `sport_key`.
- **Counts differ from older results.** Experiments before 2026-03-15 used a 7-feature tabular group and `trajectory` and `cross_source` groups, none of which the code defines; Exp 1 and Exp 2 numbers are not reproducible as recorded.
- **`fpl_availability` is not a group.** The 2026-03-20 FPL runs cannot be reproduced.
- **Injury weights are not point-in-time checked.** Impact weights use season-level player stats loaded once per event; nothing in the extractor checks they were known at the snapshot time (read from code).

## Related

- [Training pipeline](training-pipeline.md), [CLV target](clv-target.md), [Sharp benchmark](sharp-benchmark.md)
