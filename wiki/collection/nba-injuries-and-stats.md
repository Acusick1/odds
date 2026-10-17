---
type: reference
title: NBA injury reports, game logs and player stats
description: "The deprioritized NBA feeds (injury PDFs, stats.nba.com game logs, PBPStats player stats), their manual CLIs, writers and event matching."
sources:
  - odds_core.injury_models.InjuryReport
  - odds_lambda.injury_fetcher.fetch_injury_report
  - odds_lambda.injury_fetcher._to_naive_et
  - odds_lambda.storage.injury_writer.InjuryWriter
  - odds_lambda.storage.injury_reader.InjuryReader
  - odds_lambda.game_log_fetcher.fetch_game_logs
  - odds_lambda.pbpstats_fetcher.fetch_player_season_stats
  - odds_lambda.storage.game_log_writer.GameLogWriter
  - odds_lambda.storage.game_log_reader.GameLogReader
  - odds_lambda.storage.pbpstats_writer.PbpStatsWriter
  - odds_lambda.storage.pbpstats_reader.PbpStatsReader
  - odds_core.game_log_models.NbaTeamGameLog
  - odds_core.player_stats_models.NbaPlayerSeasonStats
  - odds_cli.commands.injuries
  - odds_cli.commands.nba_stats
  - odds_cli.commands.pbpstats
  - odds_cli.commands.backfill
status: stable
---

# NBA injury reports, game logs and player stats

Three manual-CLI NBA feeds. NBA is deprioritized and no scheduled job collects any of them ([project-overview](../foundations/project-overview.md)). The deployed EPL model does not use them; the experiments that found no CLV value in the injury features are in [exp-nba-timing-and-injuries](../evaluation/exp-nba-timing-and-injuries.md), and the feature groups built on these tables are in [feature-groups](../modelling/feature-groups.md).

| | Injury reports | Game logs | Player stats |
|---|---|---|---|
| Table / model | `nba_injury_reports` / `odds_core.injury_models.InjuryReport` | `nba_team_game_logs` / `odds_core.game_log_models.NbaTeamGameLog` | `nba_player_season_stats` / `odds_core.player_stats_models.NbaPlayerSeasonStats` |
| Source | official NBA PDFs via `nbainjuries` | stats.nba.com LeagueGameFinder | `api.pbpstats.com/get-totals/nba` |
| Writer / key | `odds_lambda.storage.injury_writer.InjuryWriter.upsert_injury_reports`, `uq_injury_report_time_team_player_date` (`report_time, team, player_name, game_date`) | `odds_lambda.storage.game_log_writer.GameLogWriter.upsert_game_logs`, `uq_game_log_game_team` (`nba_game_id, team_id`) | `odds_lambda.storage.pbpstats_writer.PbpStatsWriter.upsert_player_stats`, `uq_player_season_stats_player_season` (`player_id, season`) |
| CLI | `odds injuries fetch\|backfill\|status` | `odds nba-stats fetch\|status` | `odds pbpstats fetch\|backfill\|status` |

## Injury reports

League-wide injury report snapshots parsed from the official PDFs by `nbainjuries`; models `odds_core.injury_models.InjuryReport` and `odds_core.injury_models.InjuryStatus` are in `odds_core.injury_models`. `report_time` is the UTC snapshot time; `game_date` is the report's calendar date and `game_time_et` the tip-off string in ET (for example `07:00 PM ET`); `matchup` is `BOS@ORL`-style; `team` is the full name; `player_name` is `Last, First`; `event_id` is a nullable FK set at write time.

- `odds_lambda.injury_fetcher.fetch_injury_report(target_time_utc)` is synchronous. It converts to naive ET (`odds_lambda.injury_fetcher._to_naive_et`, deliberate because the `nbainjuries` API requires it; [timezones](../foundations/timezones.md)), checks `check_reportvalid`, parses via `odds_lambda.injury_fetcher._parse_records`, and drops "NOT YET SUBMITTED" rows and unknown statuses. `nbainjuries` is imported lazily because it starts a JVM (tabula via jpype).
- `odds_lambda.storage.injury_writer.InjuryWriter.upsert_injury_reports` matches each `(team, game_date)` to an `events` row (see [Event matching](#event-matching)).
- `odds injuries fetch|backfill|status` (`odds_cli.commands.injuries`). Backfill is event-driven: for NBA events in the season window it subtracts each `--hours-before` offset (default `8,2`), rounds down to the report slot (hourly before 2025-12-22 09:00 ET, quarter-hourly after, `_NEW_ERA_CUTOFF_ET`), dedupes, and sleeps `--delay-ms` (default 500). Ingestion is manual CLI only.

## Game logs

stats.nba.com blocks raw HTTP (Akamai), so `odds_lambda.game_log_fetcher.fetch_game_logs` is synchronous Playwright: headless Firefox loads nba.com for cookies, then calls the API from the page. Two rows per game, about 2,460 per regular season. Because Playwright's sync API cannot run inside an event loop, the CLI fetches all seasons first, then writes asynchronously.

`odds_lambda.storage.game_log_reader.GameLogReader.get_game_logs_for_event` and `odds_lambda.storage.game_log_reader.GameLogReader.get_team_previous_game` serve the `rest` features, which are all-NaN for EPL and dropped by the variance filter ([feature-groups](../modelling/feature-groups.md)). `odds backfill scores` reads logs linked to events to fill missing scores ([odds-api-backfill](odds-api-backfill.md)).

## Player stats

`odds_lambda.pbpstats_fetcher.PlayerSeasonRecord` carries on/off ratings, minutes, games played and box totals, regular season only. `odds_lambda.pbpstats_fetcher.convert_name` turns "First Last" into "Last, First" (suffix-aware) to match injury-report names. Traded players appear once, under the final team. `odds_lambda.storage.pbpstats_reader.PbpStatsReader.get_players_for_teams` returns a dict keyed by `player_name`, which the injury features use for impact weighting ([feature-groups](../modelling/feature-groups.md)). The NBA CLIs derive the season from `date.today()` (local date, not UTC).

## Event matching

Both writers link rows to `events` with the same ET window rule: events whose `commence_time` is in game_date 10:00 ET to next day 06:00 ET with the team as home or away; exactly one candidate sets `event_id`, otherwise NULL (ambiguous logs a warning). Injury teams are canonicalised with `odds_lambda.polymarket_matching.normalize_team`, not the core normalizer; the game-log writer maps `team_abbreviation` through a hardcoded `_ABBREV_TO_CANONICAL`. Injury reports are league-wide, so non-playing teams stay `event_id = NULL`, and unmatched rows are invisible to per-event readers such as `odds_lambda.storage.injury_reader.InjuryReader.get_injuries_for_event` ([event-matching](../storage/event-matching.md)).

## Scope and known gaps

- NBA-only: backfill and the stats lookup use `Event.sport_key == "basketball_nba"` in the CLI, and `odds_lambda.polymarket_matching.CANONICAL_TO_ABBREV` in `odds_analytics.feature_groups.collect_event_data`. `odds backfill scores` hardcodes `sport_key="basketball_nba"` with no `--sport` option (#448; [odds-api-backfill](odds-api-backfill.md)).
- `odds_cli.commands.nba_stats.ALL_SEASONS` ends at 2025-26 and `odds_cli.commands.pbpstats.ALL_SEASONS` at 2024-25, and `odds pbpstats backfill` is `fetch --all` over that shorter list. 2025-26 player stats need `--season` (#463). Both `--all` help strings say "2021-22 through current", which is false for PBPStats (#466).
- Event-less rows are expected for non-playing teams; an alias gap produces the same NULL `event_id` ([event-matching](../storage/event-matching.md)).

## Related

- [Event matching across sources](../storage/event-matching.md)
- [Feature groups](../modelling/feature-groups.md), the `injuries` and `rest` groups
- [NBA timing and injury experiments](../evaluation/exp-nba-timing-and-injuries.md)
- [The odds CLI](../foundations/odds-cli.md)
