---
type: component
title: EPL context feature groups
description: The standings, match_stats, epl_schedule and epl_lineup extractors — inputs, windows, point-in-time rules and data dependencies — and the undocumented lineup signal check.
sources:
  - odds_analytics.standings_features.extract_standings_features
  - odds_analytics.match_stats_features.extract_match_stats_features
  - odds_analytics.epl_schedule_features.extract_epl_schedule_features
  - odds_analytics.epl_lineup_features.extract_epl_lineup_features
  - experiments/scripts/lineup_signal_check.py
  - experiments/configs/xgboost_epl_lineup.yaml
status: stable
---

# EPL context feature groups

Four EPL-only groups in `packages/odds-analytics/odds_analytics/*_features.py`, loaded through caches built once per training run (`odds_analytics.feature_groups.preload_feature_group_caches`, see [feature pipeline](feature-pipeline.md)). Column lists are in the [feature group table](feature-pipeline.md#feature-groups); ablation results are in the [EPL ablation](../evaluation/exp-epl-feature-ablation-and-search-space.md). In that ablation (PR #361; FPL results in PR #255) `match_stats` adds +0.016 R², `standings` is neutral and `epl_schedule` costs -0.008; the deployed model is tabular plus `match_stats` ([publishing and scoring](model-publishing-and-scoring.md)).

Data feeding the groups: `fetch-espn-fixtures` is a local-only daily cron, while lineups and FPL are manual scripts ([EPL context data](../collection/epl-context-data.md)).

- **standings** (`odds_analytics.standings_features`). Rebuilds the league table for the event's season from completed FINAL events with `commence_time` strictly earlier; season key rolls over in July; table order is points, goal difference, goals for. `form_window` (default 5) sets form. All-None if no prior match. `odds_analytics.standings_features.load_season_events_cache` raises for any `sport_key` other than `soccer_epl`. The MCP `odds_mcp.tools.epl.get_team_context` tool derives a second standings table from `espn_fixtures`, and the tiebreak is duplicated in `odds_analytics.standings_features` working on `odds_core.models.Event` rows ([MCP sport context tools](../agent/mcp-sport-context-tools.md)).
- **match_stats** (`odds_analytics.match_stats_features`). Rolling mean over the last `match_stats_window` (default 5) prior matches per team. Stats are read from `raw_data["match_stats"]` of the latest `football_data_uk` snapshot of each completed event, so only events ingested from football-data.co.uk contribute ([historical odds loading](../collection/howto-load-historical-odds.md)).
- **epl_schedule** (`odds_analytics.epl_schedule_features`). Rest days take the more recent of the previous EPL match and the previous fixture in the ESPN all-competition table; also matches in the last 14 days and a European-competition flag for the last 7 days; midweek means Tuesday or Wednesday.
- **epl_lineup** (`odds_analytics.epl_lineup_features`). Compares each team's current starting XI (from `espn_lineups`) with its previous one: players dropped, and their starts over the previous 38 matches. Needs the current match's own XI in the cache, matched by `commence_time.date()`, and a prior match.

## Lineup signal check (undocumented)

`experiments/scripts/lineup_signal_check.py` correlates lineup metrics with the devigged bet365 CLV target on 1,684-1,709 matched rows; output is `experiments/results/lineup_signal_check/` (`correlations.csv`, `joined_data.csv`). There is no FINDINGS file. From `correlations.csv`:

- Strongest: `diff_cumulative_starts_lost`, r = -0.278 (n = 1,684, univariate R² 0.077); `diff_xi_changes` r = -0.184.
- "Tendency" features that need no lineup announcement are weak: `diff_cumulative_starts_lost_tendency_3` r = -0.074; the xi-change tendencies are not significant (p above 0.08).
- FPL-based `diff_expected_disruption` r = -0.104 (n = 1,498).
- `*_roll3` columns include the current match, so they are look-ahead by construction (the script's own docstring says so).

`experiments/configs/xgboost_epl_lineup.yaml` trains `tabular + standings + match_stats + epl_lineup`; no result for it is recorded in the experiment log.

## Hazards

- **The strong lineup signal is late information.** The match-specific metrics need the starting XI, which the script docstring puts at about 75 minutes before kickoff and the agent's data-source notes at KO-60 ([agent data sources](../agent/agent-data-sources.md)), while training samples at the sharp tier or 3-12 h out ([decision time](decision-time-and-look-ahead.md)). Trained rows therefore use the XI before it is public ([look-ahead bias](decision-time-and-look-ahead.md#hazards)).
- **Match stats lag.** football-data.co.uk is compiled by hand twice weekly after matches ([CLV model findings](../evaluation/clv-model-findings.md)); a live event's rolling window omits matches not yet ingested.
- **Name matching.** Team strings must equal `odds_core.models.Event.home_team`/`away_team` exactly; mismatches silently produce NaN.

## Related

- [Feature pipeline](feature-pipeline.md)
