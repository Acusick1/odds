---
type: reference
title: MCP sport context tools
description: EPL get_team_context (form, fixtures, standings from ESPN fixtures) and MLB get_probable_pitchers (live read-through with cached fallback), and their data dependencies.
sources:
  - odds_mcp.tools.epl.get_team_context
  - odds_mcp.tools.epl._derive_standings
  - odds_mcp.tools.mlb.get_probable_pitchers
  - odds_mcp.tools.mlb.ProbablePitchersResponse
  - odds_core.mlb_data_models.select_latest_in_window
  - odds_core.team.normalize_team_name
  - odds_analytics.standings_features
  - odds_lambda.scheduling.jobs._JOB_CRON_MAP
status: stable
---

# MCP sport context tools

Two tools in `packages/odds-mcp/odds_mcp/tools/`, each on its own `FastMCP` sub-server mounted without a namespace ([odds-mcp server](mcp-server.md#sport-sub-servers)). They give the [agent](betting-agent.md) structured context that the odds tools do not carry; the agent still researches lineups and injuries on the web ([data sources](agent-data-sources.md)).

## Team context (EPL)

`get_team_context(team, as_of=None, last_n=5, next_n=5, include_standings=True)` reads the `espn_fixtures` table (rows per team, refreshed daily by `fetch-espn-fixtures`, see [EPL context data](../collection/epl-context-data.md)). That job is a daily cron (`0 6 * * *` in `_JOB_CRON_MAP`) on the mini-PC's scheduler, so the tool goes stale whenever the job fails, as it does on ESPN's HTTP 400 (#486). `team` goes through `odds_core.team.normalize_team_name`; `as_of` defaults to now (a naive value is taken as UTC); `last_n` and `next_n` are floored at 1.

| Field | Content |
|---|---|
| `last_results` | Up to `last_n` completed Premier League fixtures before `as_of`, newest first, with scores, W/D/L and home/away. Cup results are excluded so form stays league-only |
| `upcoming_fixtures` | Next `next_n` fixtures after `as_of` in all competitions with `days_until_ko`: the rotation-risk signal |
| `standings` | `team_row` and the full `table`, derived on the fly from the season's completed Premier League rows |

Rows with ESPN `state` `in` (live) are skipped from both lists. A row counts as final when `state == "post"`, or, for older rows with no `state`, when its status text is in a short fallback set. Rows with unparseable scores are dropped from results.

## Probable pitchers (MLB)

`get_probable_pitchers(lookahead_hours=48, refresh=True)` is read-only (clamped to 1-168 hours). With `refresh=True` it fetches the MLB Stats API for the dates covering the window (`odds_lambda.mlb_stats_fetcher.dates_for_window` pads one day each side because MLBAM resolves dates in US Eastern) and returns current state without persisting; `fetch_status` is `live`. On an `httpx.HTTPError` it falls back to the cached `mlb_probable_pitchers` snapshots with `stale_db_only`. `refresh=False` reads the cache only (`db_only`). Both paths use `odds_core.mlb_data_models.select_latest_in_window`: newest `fetched_at` per `game_pk` inside the window, ordered by kickoff.

Each game carries `home_pitcher_name`/`id` and `away_pitcher_name`/`id`, nullable because MLBAM omits the pitcher until announced (null means "not yet announced"), and `hours_until_commence`. The tool cannot tell a starter from an opener; that needs a web search. The cache is written only by the fetch-mlb-probables job, a daily cron on the mini-PC ([MLB probable pitchers](../collection/mlb-probable-pitchers.md)).

## Behaviour to know

- Standings use a simplified tiebreak (points, goal difference, goals for, team name) without the official head-to-head step, so tied positions are approximate (`odds_mcp.tools.epl._derive_standings`). `odds_analytics.standings_features` derives a table with the same chain from `odds_core.models.Event` rows, a second implementation on different data (#463, [EPL context features](../modelling/epl-context-features.md)).
- `odds_core.team.normalize_team_name` returns an unknown name unchanged, so a misspelt team yields empty lists and `team_row: null`, not an error.
- The `team` fixture query is not season-scoped; only standings filter by season label.
- `odds_mcp.tools.epl.get_team_context` exists only for the EPL; MLB has no equivalent form or schedule tool.

## Related

- [Agent prompts](agent-prompts.md), where the agent uses these
- [Betting agent](betting-agent.md)
