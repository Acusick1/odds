---
type: reference
title: Feature groups
description: The catalogue of static feature groups and column counts, the tabular extractor and its sharp and retail defaults, and the NBA injury, rest and Polymarket groups.
sources:
  - odds_analytics.feature_groups._STATIC_FEATURE_GROUPS
  - odds_analytics.training.config.FeatureGroupName
  - odds_analytics.training.config.FeatureConfig
  - odds_analytics.feature_groups._static_feature_group_names
  - odds_analytics.feature_groups.XGBoostAdapter
  - odds_analytics.feature_extraction.TabularFeatures
  - odds_analytics.feature_extraction.TabularFeatureExtractor
  - odds_analytics.feature_extraction.DEFAULT_RETAIL_BOOKMAKERS
  - odds_analytics.feature_extraction.DEFAULT_SHARP_BOOKMAKERS
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
| `rest` | 5 | NBA | days rest and back-to-back flags from game logs; all NaN for EPL |
| `injuries` | 6 | NBA | impact-weighted OUT and GTD burden, report timing |
| `polymarket` | 16 | NBA | 14 Polymarket columns (price, volume and velocity from price snapshots; best bid, best ask, spread, midpoint and depth from order-book snapshots, taken from the home side's view) plus 2 cross-source divergences |

These groups serve the `raw` target; `retail_ev` takes only its own groups computed as of each row's decision time ([retail-EV features](clv-target-functions.md#retail-ev-features), [EPL standings and schedule features](epl-context-features.md)). Polymarket data is on [Polymarket](../collection/polymarket.md). The valid names are the `odds_analytics.training.config.FeatureGroupName` literal, which `odds_analytics.training.config.FeatureConfig` validates, along with `odds_analytics.training.config.FeatureConfig.retail_ev_arm` (`direct` or `residual`, a `retail_ev` setting; any other target must leave it `direct`) (an `xgboost` config needs at least one group unless its `target_type` is `retail_ev`, which takes none of these; an `lstm` config may have none) and from which `odds_analytics.feature_groups._STATIC_FEATURE_GROUPS` is derived. The groups are dispatched by string in `odds_analytics.feature_groups._static_feature_group_names`, `odds_analytics.feature_groups._extract_static_feature_parts` and `odds_analytics.feature_groups.collect_event_data`, with no registry (#475); `odds_analytics.feature_groups.collect_event_data` also drops snapshots whose only bookmaker is Polymarket (`odds_core.bookmakers.is_reference_only_snapshot`); the places to touch are listed in [Add a feature group](howto-add-feature-group.md).

## Tabular extractor

`odds_analytics.feature_extraction.TabularFeatureExtractor` builds `sharp_prob` and `retail_sharp_diff` from `sharp_bookmakers` and `retail_bookmakers`. The code defaults are `odds_analytics.feature_extraction.DEFAULT_SHARP_BOOKMAKERS` (`["pinnacle"]`) and `odds_analytics.feature_extraction.DEFAULT_RETAIL_BOOKMAKERS` (`fanduel`, `draftkings`, `betmgm`); the other copies of these defaults are compared in [Sharp benchmark](sharp-benchmark.md#code-copies). `sharp_prob` comes from the first sharp bookmaker, in `sharp_bookmakers` order, that quotes the requested outcome in the market; it is NaN only when none does, and `retail_sharp_diff` is then NaN as well. The implied probabilities are `odds_core.models.Odds.implied_probability` (1 over the stored decimal price) and are not devigged. The extractor reads only the two rosters from `odds_analytics.training.config.FeatureConfig`, not its target fields.

## NBA injuries

`odds_analytics.injury_features.extract_injury_features(reports, event, snapshot_time, player_stats)` takes reports with `report_time <= snapshot_time`, uses only the latest report snapshot, and returns `odds_analytics.injury_features.InjuryFeatures` (6 fields: `impact_out_home/away`, `impact_gtd_home/away`, `report_hours_before_game`, `injury_news_recency`).

- Per-player impact is `(on_off_rtg - on_def_rtg) * (minutes / games_played / 48)` from player stats, falling back to 1.0 when stats, ratings or games are missing.
- QUESTIONABLE and DOUBTFUL count at 0.5x; PROBABLE and AVAILABLE are ignored (GTD weighted 0.5).
- Player impact is looked up by exact `player_name`: `odds_lambda.storage.pbpstats_reader.PbpStatsReader.get_players_for_teams` keys stats by name, so a report name that differs from the stats name silently becomes a 1.0 headcount, and two players on the requested teams with the same name collide (the last row wins).
- The group is enabled with `feature_groups` containing `injuries`; `odds_analytics.feature_groups.collect_event_data` loads reports and stats, and events without reports get NaN and are not dropped.
- The impact weights use season-level player stats loaded once per event; nothing in the extractor checks they were known at the snapshot time (read from code).

The injury reports and player stats are written by the pipeline on [NBA injuries and stats](../collection/nba-injuries-and-stats.md).

## Related

- [Feature pipeline](feature-pipeline.md)
