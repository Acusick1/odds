---
type: reference
title: EPL context feature groups
description: The standings, match_stats and epl_schedule extractors with their inputs, windows, point-in-time rules and data dependencies.
sources:
  - odds_analytics.standings_features.extract_standings_features
  - odds_analytics.standings_features.load_season_events_cache
  - odds_analytics.standings_features.get_prior_events_from_cache
  - odds_analytics.match_stats_features.extract_match_stats_features
  - odds_analytics.epl_schedule_features.extract_epl_schedule_features
  - odds_analytics.feature_groups.preload_feature_group_caches
status: stable
---

# EPL context feature groups

Three EPL-only groups in `packages/odds-analytics/odds_analytics/*_features.py`, loaded through caches built once per training run (`odds_analytics.feature_groups.preload_feature_group_caches`, see [feature pipeline](feature-pipeline.md)). Column lists are in the [feature group table](feature-groups.md); ablation results are in the [EPL ablation](../evaluation/exp-epl-feature-ablation-and-search-space.md). The deployed model is tabular plus `match_stats` ([model store](model-store.md)).

Data feeding the groups: `fetch-espn-fixtures` is a local-only daily cron, while FPL is a manual script ([EPL context data](../collection/epl-context-data.md)).

- **standings** (`odds_analytics.standings_features`). Rebuilds the league table for the event's season from completed FINAL events with `commence_time` strictly before a cutoff (`odds_analytics.standings_features.get_prior_events_from_cache`): the event's kick-off for this group, or the last kick-off whose result is known at t for `retail_ev`'s `standings_as_of_t`; season key rolls over in July; table order is points, goal difference, goals for. `form_window` (default 5) sets form. All-None if no prior match. `odds_analytics.standings_features.load_season_events_cache` raises for any `sport_key` other than `soccer_epl`, so the standings and schedule caches are EPL-only. The MCP `odds_mcp.tools.epl.get_team_context` tool derives a second standings table from `espn_fixtures`, and the tiebreak is duplicated in `odds_analytics.standings_features` working on `odds_core.models.Event` rows ([MCP sport context tools](../agent/mcp-sport-context-tools.md)).
- **match_stats** (`odds_analytics.match_stats_features`). Rolling mean over the last `match_stats_window` (default 5) prior matches per team. Stats are read from `raw_data["match_stats"]` of the latest `football_data_uk` snapshot of each completed event, so only events ingested from football-data.co.uk contribute ([historical ingest scripts](../collection/historical-ingest-scripts.md)). football-data.co.uk is compiled by hand twice weekly after matches, so a live event's rolling window omits matches not yet ingested.
- **epl_schedule** (`odds_analytics.epl_schedule_features`). Rest days take the more recent of the previous EPL match and the previous fixture in the ESPN all-competition table; also matches in the last 14 days and a European-competition flag for the last 7 days; midweek means Tuesday or Wednesday. `known_at` limits the matches counted to those that kicked off before it, while rest and the windows are measured back from kick-off; this group passes the kick-off, and `retail_ev`'s `schedule_as_of_t` passes the last kick-off whose result is known at t ([retail-EV features](clv-target-functions.md#retail-ev-features)). With a non-empty fixtures table, a team with no fixture in a window counts 0, whatever the row's status; with no table, or no row before `known_at`, the congestion columns are NaN.

Team strings must equal `odds_core.models.Event.home_team` and `away_team` exactly; mismatches silently produce NaN.

## Related

- [Feature pipeline](feature-pipeline.md), [Feature groups](feature-groups.md)
