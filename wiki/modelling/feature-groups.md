---
type: reference
title: Feature groups
description: The catalogue of feature groups and column counts, the tabular extractor and its sharp and retail defaults, and the NBA injury, rest and Polymarket groups.
sources:
  - odds_analytics.feature_groups._STATIC_FEATURE_GROUPS
  - odds_analytics.feature_groups._static_feature_group_names
  - odds_analytics.feature_groups.XGBoostAdapter
  - odds_analytics.feature_extraction.TabularFeatures
  - odds_analytics.feature_extraction.TabularFeatureExtractor
  - odds_analytics.feature_extraction.DEFAULT_RETAIL_BOOKMAKERS
  - odds_analytics.feature_extraction.DEFAULT_SHARP_BOOKMAKERS
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

# Feature groups

Counts are dataclass fields, checked against code; each ends up as one prefixed column block in `odds_analytics.feature_groups.XGBoostAdapter` output. How the blocks are assembled is on [Feature pipeline](feature-pipeline.md).

| Group | Cols | Sports | Notes |
|---|---|---|---|
| `tabular` | 6 | all | `consensus_prob`, `sharp_prob`, `retail_sharp_diff`, `num_bookmakers`, `is_weekend`, `day_of_week`; outcome-relative |
| `standings` | 11 | EPL | position, points gap, goal difference, form (`form_window`), goal rates |
| `match_stats` | 14 | EPL | rolling means (`match_stats_window`) of shots, shots on target, corners, fouls, cards, half-time goals |
| `epl_schedule` | 8 | EPL | rest days, midweek, matches in 14 days, European match in 7 days |
| `epl_lineup` | 6 | EPL | starting-XI changes and cumulative starts lost |
| `rest` | 5 | NBA | days rest and back-to-back flags from game logs; all NaN for EPL, so the variance filter removes it |
| `injuries` | 6 | NBA | impact-weighted OUT and GTD burden, report timing |
| `polymarket` | 16 | NBA | 14 Polymarket price and order-book columns plus 2 cross-source divergences |

The EPL groups are detailed on [EPL context features](epl-context-features.md) and Polymarket data on [Polymarket](../collection/polymarket.md). The group names are dispatched by string in `odds_analytics.feature_groups._STATIC_FEATURE_GROUPS`, `odds_analytics.feature_groups._static_feature_group_names`, `odds_analytics.feature_groups._extract_static_feature_parts`, `odds_analytics.feature_groups.collect_event_data` and the preload caches, with no registry (#475); the places to touch are listed in [Add a feature group](howto-add-feature-group.md).

## Tabular extractor

`odds_analytics.feature_extraction.TabularFeatureExtractor` builds `sharp_prob` and `retail_sharp_diff` from `sharp_bookmakers` and `retail_bookmakers`. The code defaults are `odds_analytics.feature_extraction.DEFAULT_SHARP_BOOKMAKERS` (`["pinnacle"]`) and `odds_analytics.feature_extraction.DEFAULT_RETAIL_BOOKMAKERS` (`fanduel`, `draftkings`, `betmgm`); the other copies of these defaults are compared in [Sharp benchmark](sharp-benchmark.md#code-copies). The extractor stops at the first sharp bookmaker with the market; if that bookmaker lacks the requested outcome, `sharp_prob` is NaN rather than trying the next bookmaker (#477). The implied probabilities are not devigged.

## NBA injuries

`odds_analytics.injury_features.extract_injury_features(reports, event, snapshot_time, player_stats)` takes reports with `report_time <= snapshot_time`, uses only the latest report snapshot, and returns `odds_analytics.injury_features.InjuryFeatures` (6 fields: `impact_out_home/away`, `impact_gtd_home/away`, `report_hours_before_game`, `injury_news_recency`).

- Per-player impact is `(on_off_rtg - on_def_rtg) * (minutes / games_played / 48)` from player stats, falling back to 1.0 when stats, ratings or games are missing.
- QUESTIONABLE and DOUBTFUL count at 0.5x; PROBABLE and AVAILABLE are ignored (GTD weighted 0.5).
- The group is enabled with `feature_groups` containing `injuries`; `odds_analytics.feature_groups.collect_event_data` loads reports and stats, and events without reports get NaN and are not dropped.
- The impact weights use season-level player stats loaded once per event; nothing in the extractor checks they were known at the snapshot time (read from code).

The injury reports and player stats are written by the pipeline on [NBA injuries and stats](../collection/nba-injuries-and-stats.md).

## Related

- [Feature pipeline](feature-pipeline.md), [CLV model findings](../evaluation/clv-model-findings.md#current-conclusions)
