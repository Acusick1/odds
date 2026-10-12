---
type: component
title: EPL context feature groups
summary: The standings, match_stats, epl_schedule and epl_lineup extractors - inputs, windows and point-in-time rules - plus the undocumented lineup signal check.
sources:
  - packages/odds-analytics/odds_analytics/standings_features.py::extract_standings_features
  - packages/odds-analytics/odds_analytics/match_stats_features.py::extract_match_stats_features
  - packages/odds-analytics/odds_analytics/epl_schedule_features.py::extract_epl_schedule_features
  - packages/odds-analytics/odds_analytics/epl_lineup_features.py::extract_epl_lineup_features
  - experiments/scripts/lineup_signal_check.py
  - experiments/configs/xgboost_epl_lineup.yaml
  - docs/DATA_MODELS.md#feature-groups
  - docs/MODELING.md#epl-feature-group-ablation
verified_at: 7887039
---

# EPL context feature groups

Four EPL-only groups in `packages/odds-analytics/odds_analytics/*_features.py`, loaded through caches built once per training run (`preload_feature_group_caches`, see [feature-pipeline](feature-pipeline.md)). Column lists are in [feature-groups](feature-groups.md); ablation results in [exp-epl-feature-group-ablation](exp-epl-feature-group-ablation.md).

- **standings** (`packages/odds-analytics/odds_analytics/standings_features.py`). Rebuilds the league table for the event's season from completed FINAL events with `commence_time` strictly earlier; season key rolls over in July; table order is points, goal difference, goals for. `form_window` (default 5) sets form. All-None if no prior match. `load_season_events_cache` raises for any `sport_key` other than `soccer_epl`.
- **match_stats** (`packages/odds-analytics/odds_analytics/match_stats_features.py`). Rolling mean over the last `match_stats_window` (default 5) prior matches per team. Stats are read from `raw_data["match_stats"]` of the latest `football_data_uk` snapshot of each completed event, so only events ingested from football-data.co.uk contribute ([historical-odds-ingestion](historical-odds-ingestion.md)).
- **epl_schedule** (`packages/odds-analytics/odds_analytics/epl_schedule_features.py`). Rest days take the more recent of the previous EPL match and the previous fixture in the ESPN all-competition table; also matches in the last 14 days and a European-competition flag for the last 7 days; midweek means Tuesday or Wednesday ([epl-context-data](epl-context-data.md)).
- **epl_lineup** (`packages/odds-analytics/odds_analytics/epl_lineup_features.py`). Compares each team's current starting XI with its previous one: players dropped, and their starts over the previous 38 matches. Needs the current match's own XI in the cache, matched by `commence_time.date()`, and a prior match.

## Lineup signal check (undocumented)

`experiments/scripts/lineup_signal_check.py` correlates lineup metrics with the devigged bet365 CLV target on 1,684-1,709 matched rows; output is `experiments/results/lineup_signal_check/` (`correlations.csv`, `joined_data.csv`). There is no FINDINGS file. From `correlations.csv`:

- Strongest: `diff_cumulative_starts_lost`, r = -0.278 (n = 1,684, univariate R² 0.077); `diff_xi_changes` r = -0.184.
- "Tendency" features that need no lineup announcement are weak: `diff_cumulative_starts_lost_tendency_3` r = -0.074; the xi-change tendencies are not significant (p above 0.08).
- FPL-based `diff_expected_disruption` r = -0.104 (n = 1,498).
- `*_roll3` columns include the current match, so they are look-ahead by construction (the script's own docstring says so).

`experiments/configs/xgboost_epl_lineup.yaml` trains `tabular + standings + match_stats + epl_lineup`; no result for it is recorded in the experiment log.

## Hazards

- **The strong lineup signal is late information.** The match-specific metrics need the starting XI, which the script docstring puts at about 75 minutes before kickoff and the data-source notes at KO-60 ([decision-rejected-data-sources](decision-rejected-data-sources.md)), while training samples at the sharp tier or 3-12 h out ([decision-time](decision-time.md)). Trained rows therefore use the XI before it is public ([look-ahead-bias](look-ahead-bias.md)).
- **Match stats lag.** football-data.co.uk is compiled by hand twice weekly after matches ([clv-model-findings](clv-model-findings.md)); a live event's rolling window omits matches not yet ingested.
- **Name matching.** Team strings must equal `Event.home_team`/`away_team` exactly; mismatches silently produce NaN.

## Related

- [Feature pipeline](feature-pipeline.md)
