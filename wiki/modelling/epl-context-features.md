---
type: reference
title: EPL standings and schedule features
description: The standings and epl_schedule extractors behind the retail-EV as-of-t groups, with their inputs, windows, point-in-time rules and data dependencies.
sources:
  - odds_analytics.standings_features.extract_standings_features
  - odds_analytics.standings_features.load_season_events_cache
  - odds_analytics.standings_features.get_prior_events_from_cache
  - odds_analytics.epl_schedule_features.extract_epl_schedule_features
  - odds_analytics.as_of_team_features.TeamHistory
status: stable
---

# EPL standings and schedule features

Two EPL-only extractors in `packages/odds-analytics/odds_analytics/*_features.py`. `odds_analytics.as_of_team_features.TeamHistory` calls them for the retail-EV `standings_as_of_t` and `schedule_as_of_t` groups, with the results and fixtures loaded once per build ([retail-EV features](clv-target-functions.md#retail-ev-features)). Column lists are in the [feature group table](clv-target-functions.md#retail-ev-features).

Data feeding the groups: `fetch-espn-fixtures` is a local-only daily cron, while FPL is a manual script ([EPL context data](../collection/epl-context-data.md)).

- **standings** (`odds_analytics.standings_features`). Rebuilds the league table for the event's season from completed FINAL events with `commence_time` strictly before a cutoff (`odds_analytics.standings_features.get_prior_events_from_cache`), the last kick-off whose result is known at t; season key rolls over in July; table order is points, goal difference, goals for. `form_window` (default 5) sets form. All-None if no prior match. `odds_analytics.standings_features.load_season_events_cache` loads every completed match, or those kicking off up to an optional `until`, and raises for any `sport_key` other than `soccer_epl`, so the standings and schedule caches are EPL-only. The MCP `odds_mcp.tools.epl.get_team_context` tool derives a second standings table from `espn_fixtures`, and the tiebreak is duplicated in `odds_analytics.standings_features` working on completed `odds_core.models.Event` rows ([MCP sport context tools](../agent/mcp-sport-context-tools.md)).
- **epl_schedule** (`odds_analytics.epl_schedule_features`). Rest days take the more recent of the previous EPL match and the previous fixture in the ESPN all-competition table; also matches in the last 14 days and a European-competition flag for the last 7 days; midweek means Tuesday or Wednesday. `known_at` limits the matches counted to those that kicked off before it, while rest and the windows are measured back from kick-off; `schedule_as_of_t` passes the last kick-off whose result is known at t. With a non-empty fixtures table, a team with no fixture in a window counts 0, whatever the row's status; with no table, or no row before `known_at`, the congestion columns are NaN.

Team strings must equal `odds_core.models.Event.home_team` and `away_team` exactly; mismatches silently produce NaN.

## Related

- [Feature pipeline](feature-pipeline.md), [Target functions](clv-target-functions.md)
