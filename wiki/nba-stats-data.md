---
type: component
title: NBA game logs and player stats
summary: stats.nba.com game logs (via Playwright) and PBPStats player season stats - tables, writers and CLI, and what uses them (rest features, score backfill, injury weighting).
sources:
  - packages/odds-lambda/odds_lambda/game_log_fetcher.py::fetch_game_logs
  - packages/odds-lambda/odds_lambda/pbpstats_fetcher.py::fetch_player_season_stats
  - packages/odds-lambda/odds_lambda/storage/game_log_writer.py::GameLogWriter
  - packages/odds-lambda/odds_lambda/storage/game_log_reader.py::GameLogReader
  - packages/odds-lambda/odds_lambda/storage/pbpstats_writer.py::PbpStatsWriter
  - packages/odds-lambda/odds_lambda/storage/pbpstats_reader.py::PbpStatsReader
  - packages/odds-core/odds_core/game_log_models.py::NbaTeamGameLog
  - packages/odds-core/odds_core/player_stats_models.py::NbaPlayerSeasonStats
  - packages/odds-cli/odds_cli/commands/nba_stats.py
  - packages/odds-cli/odds_cli/commands/pbpstats.py
  - docs/ARCHITECTURE.md#game-log-pipeline-nba
  - docs/CLI.md#game-logs
  - docs/CLI.md#player-stats-pbpstats
  - docs/CLI.md#backfill-scores
  - docs/DATABASE.md#sport-context-tables
verified_at: 7887039
---

# NBA game logs and player stats

Two manual-CLI NBA feeds. Neither has a scheduled job.

| | Game logs | Player stats |
|---|---|---|
| Table / model | `nba_team_game_logs` / `NbaTeamGameLog` | `nba_player_season_stats` / `NbaPlayerSeasonStats` |
| Source | stats.nba.com LeagueGameFinder | `api.pbpstats.com/get-totals/nba` |
| Fetcher | `fetch_game_logs(season)` | `fetch_player_season_stats(season)` |
| Writer / key | `GameLogWriter.upsert_game_logs`, `uq_game_log_game_team` (`nba_game_id, team_id`) | `PbpStatsWriter.upsert_player_stats`, `uq_player_season_stats_player_season` (`player_id, season`) |
| CLI | `odds nba-stats fetch\|status` | `odds pbpstats fetch\|backfill\|status` |

## Game logs

stats.nba.com blocks raw HTTP (Akamai), so `fetch_game_logs` is synchronous Playwright: headless Firefox loads nba.com for cookies, then calls the API from the page. Two rows per game, about 2,460 per regular season. Because Playwright's sync API cannot run inside an event loop, the CLI fetches all seasons first, then writes asynchronously.

`GameLogWriter` links each row to an `events` row with the same ET window rule as the injury writer (10:00 ET to next day 06:00 ET, unique match else NULL), mapping `team_abbreviation` through a hardcoded `_ABBREV_TO_CANONICAL`. `GameLogReader.get_game_logs_for_event` and `get_team_previous_game` serve the `rest` features (`packages/odds-analytics/odds_analytics/schedule_features.py::extract_rest_features`, loaded in `collect_event_data`).

`odds backfill scores` reads logs linked to events to fill missing scores (`packages/odds-cli/odds_cli/commands/backfill.py`).

## Player stats

`PlayerSeasonRecord` carries on/off ratings, minutes, games played and box totals, regular season only. `convert_name` turns "First Last" into "Last, First" (suffix-aware) to match injury-report names. Traded players appear once, under the final team. `PbpStatsReader.get_players_for_teams` returns a dict keyed by `player_name` for [injury impact weighting](nba-injury-reports.md).

## Hazards

- Season lists diverge: `packages/odds-cli/odds_cli/commands/nba_stats.py::ALL_SEASONS` ends at 2025-26, `pbpstats.py::ALL_SEASONS` at 2024-25, and `odds pbpstats backfill` is `fetch --all` over that shorter list. Both `--all` help strings say "2021-22 through current"; for PBPStats that is false, so 2025-26 needs `--season`.
- `odds backfill scores` hardcodes `sport_key="basketball_nba"` with no `--sport` option.
- Keying players by name collides if two players share a name on the requested teams (last one wins).
- Missing-name matches in injury weighting fall back silently to 1.0 ([NBA injury reports](nba-injury-reports.md)).

## Related

- [NBA injury reports and features](nba-injury-reports.md)
- [Event matching](event-matching.md)
- [Feature groups](feature-groups.md) - `rest` group
- [CLI overview](cli-overview.md)
