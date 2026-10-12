---
type: concept
title: Feature groups
summary: The catalogue of composable feature groups (tabular, standings, match_stats, epl_schedule, epl_lineup, rest, injuries, polymarket, plus LSTM sequence features), their counts, the NaN-fill policy, and how to add one.
sources:
  - packages/odds-analytics/odds_analytics/feature_extraction.py::TabularFeatures
  - packages/odds-analytics/odds_analytics/feature_extraction.py::SequenceFeatures
  - packages/odds-analytics/odds_analytics/feature_groups.py::_STATIC_FEATURE_GROUPS
  - packages/odds-analytics/odds_analytics/standings_features.py::StandingsFeatures
  - packages/odds-analytics/odds_analytics/match_stats_features.py::MatchStatsFeatures
  - packages/odds-analytics/odds_analytics/epl_schedule_features.py::EplScheduleFeatures
  - packages/odds-analytics/odds_analytics/epl_lineup_features.py::EplLineupFeatures
  - packages/odds-analytics/odds_analytics/schedule_features.py::RestScheduleFeatures
  - packages/odds-analytics/odds_analytics/injury_features.py::InjuryFeatures
  - packages/odds-analytics/odds_analytics/polymarket_features.py::PolymarketTabularFeatures
  - docs/DATA_MODELS.md#feature-groups
  - docs/INJURIES.md#feature-extraction
verified_at: 7887039
---

# Feature groups

`FeatureConfig.feature_groups` selects which groups to compose ([training-config](training-config.md)); the [feature pipeline](feature-pipeline.md) builds them. Counts below are dataclass fields, checked against code; each ends up as one prefixed column block in `XGBoostAdapter` output.

| Group | Cols | Sports | Notes |
|---|---|---|---|
| `tabular` | 6 | all | `consensus_prob`, `sharp_prob`, `retail_sharp_diff`, `num_bookmakers`, `is_weekend`, `day_of_week`; outcome-relative. See [retail-sharp-divergence](retail-sharp-divergence.md) |
| `standings` | 11 | EPL | position, points gap, goal difference, form (`form_window`), goal rates |
| `match_stats` | 14 | EPL | rolling means (`match_stats_window`) of shots, shots on target, corners, fouls, cards, half-time goals |
| `epl_schedule` | 8 | EPL | rest days, midweek, matches in 14 days, European match in 7 days |
| `epl_lineup` | 6 | EPL | starting-XI changes and cumulative starts lost |
| `rest` | 5 | NBA | days rest and back-to-back flags from game logs; all NaN for EPL, so the variance filter removes it |
| `injuries` | 6 | NBA | impact-weighted OUT and GTD burden, report timing |
| `polymarket` | 16 | NBA | 14 Polymarket price/order-book columns plus 2 cross-source divergences |

EPL groups are detailed in [features-epl-context](features-epl-context.md); injuries in [nba-injury-reports](nba-injury-reports.md) (point-in-time: only reports at or before the snapshot, latest report only; GTD weighted 0.5); Polymarket in [polymarket-pipeline](polymarket-pipeline.md). `XGBoostAdapter` always appends `hours_until_event`. The LSTM adapter instead uses 15 per-timestep `SequenceFeatures` (odds, implied probability, hours to game, time-of-day sin/cos, changes from previous and opening, spread, sharp odds and divergence), plus optional static groups ([line-movement-models](line-movement-models.md)).

## NaN-fill policy

When optional data is missing the block is NaN and the row is kept (`AGENTS.md`); only a tabular failure drops a row. XGBoost handles NaN natively. Strictly constant columns are dropped afterwards, so counts in a trained model can be lower than above ([exp-epl-feature-group-ablation](exp-epl-feature-group-ablation.md) shows `rest` adding 0 EPL features).

## Adding a group

1. Write a dataclass with `to_array` and `get_feature_names`, and an extractor that takes only data before the snapshot ([look-ahead-bias](look-ahead-bias.md)).
2. Register the name in `_STATIC_FEATURE_GROUPS`, `_static_feature_group_names` (choose a prefix) and `_extract_static_feature_parts`.
3. Load its data in `collect_event_data` (or a cache in `preload_feature_group_caches`) and add the field to `EventDataBundle`.
4. Mention it in the `feature_groups` description of `FeatureConfig` (nothing validates it).

## Hazards

- **Counts differ from older results.** Experiments before 2026-03-15 used a 7-feature tabular group and a `trajectory` group that no longer exist; Exp 1 and Exp 2 numbers are not reproducible as recorded ([exp-early-nba-signal-search](exp-early-nba-signal-search.md)).
- **`fpl_availability` is not a group.** The 2026-03-20 FPL runs cannot be reproduced ([exp-epl-feature-group-ablation](exp-epl-feature-group-ablation.md)).
- Injury impact weights use season-level player stats loaded once per event; nothing in the extractor checks they were known at the snapshot time (read from code).

## Related

- [CLV](clv.md), [Sharp benchmark](sharp-benchmark.md)
