---
type: reference
title: ESPN football schedule
description: How the fetch-espn-games job creates the upcoming events of the Championship, La Liga, Serie A, Bundesliga and Ligue 1 from ESPN's scoreboards, which ESPN team names it drops, and the dry-run command.
sources:
  - odds_lambda.espn_football_schedule
  - odds_lambda.espn_football_schedule.FootballLeague
  - odds_lambda.espn_football_schedule.LEAGUES
  - odds_lambda.espn_football_schedule.parse_scoreboard
  - odds_lambda.espn_football_schedule.fetch_games
  - odds_lambda.espn_football_schedule.sync_games
  - odds_lambda.espn_football_schedule.sync_schedule
  - odds_lambda.espn_football_schedule.sync_upcoming
  - odds_lambda.espn_football_schedule.UPCOMING_DAYS
  - odds_lambda.espn_football_schedule.RECENT_DAYS
  - odds_lambda.jobs.fetch_espn_games.main
  - odds_lambda.jobs.fetch_espn_games._sync_football_schedule
  - odds_lambda.espn_game_fetcher.EspnGameFetcher.fetch_month_scoreboard_json
  - odds_cli.commands.events.sync_football_schedule
  - odds_core.league_teams.league_team
mentions:
  - odds_lambda.live_events.alert_dropped_games
status: stable
---

# ESPN football schedule

OddsPortal and Betfair list only events that already have prices, so a league with no schedule feed leaves their collectors in [dormancy](../scheduling/decision-engine.md#dormancy), and collecting is what would create its events. For the Championship (`eng.2`), La Liga (`esp.1`), Serie A (`ita.1`), the Bundesliga (`ger.1`) and Ligue 1 (`fra.1`) the schedule is ESPN's scoreboard: the events `odds_lambda.espn_football_schedule` creates are what the stored-event lookup of each collector sees, so `odds_lambda.collection_calendar.published_schedule_kickoff` has no table to read for them. The EPL keeps its own game table ([EPL context data](epl-context-data.md)) and the NBA its own scoreboard sync ([NBA schedule](nba-injuries-and-stats.md#nba-schedule-from-espn)).

## Leagues

A league is one `odds_lambda.espn_football_schedule.FootballLeague` in `odds_lambda.espn_football_schedule.LEAGUES`: its sport key, title and ESPN slug. Its clubs are its canonical teams in `odds_core.league_teams.LEAGUE_TEAMS`, of every stored season rather than only this one, the list the [event resolver](../storage/event-resolver.md#canonical-teams) refuses any other name by. Adding a league is one entry there, its teams and aliases, and the steps of [add a league](howto-add-sport.md).

## The job

`odds_lambda.jobs.fetch_espn_games.main` runs `odds_lambda.jobs.fetch_espn_games._sync_football_schedule` for each of those leagues as `fetch-espn-games-<suffix>` (`championship`, `laliga`, `seriea`, `bundesliga`, `ligue1`; daily cron, one heartbeat each, [job deployment](../scheduling/job-deployment-matrix.md)). `odds_lambda.espn_football_schedule.sync_upcoming` covers the next `odds_lambda.espn_football_schedule.UPCOMING_DAYS` and the last `odds_lambda.espn_football_schedule.RECENT_DAYS`, reading one scoreboard per calendar month (`odds_lambda.espn_game_fetcher.EspnGameFetcher.fetch_month_scoreboard_json`; ESPN rejects date ranges).

- A game that has not started goes through `odds_lambda.storage.event_resolver.resolve_event` with source `espn` and ESPN's id: it matches the event of the same teams within two hours (one OddsPortal or Betfair created) or creates a scheduled one, and its kick-off is offered to `odds_lambda.storage.event_kickoff.move_kickoff`. ESPN ranks last in each league's order ([kick-off authority](../storage/kickoff-authority.md#authority-order)), so before kick-off it moves only an event no other source has an id on, and after kick-off a postponement it alone reports moves the event. An event ESPN moves keeps its id, so a rescheduled one is the same event. `odds_lambda.storage.event_kickoff.move_kickoff` also refuses ESPN a start at or before the clock for an event with an `odds_core.models.EventPostponement` row, so a stale ESPN game at the original time cannot undo a postponement.
- An event already started or finished, and any event whose start ESPN marks invalid, creates and moves nothing. ESPN's postponement or cancellation of it, for an event of either kind, attaches ESPN's id to the event it already has (`odds_lambda.storage.event_resolver.attach_event`, under the game lock) so a later reschedule finds the same event, and sets that event POSTPONED or CANCELLED when `odds_core.models.EventStatus.can_become` allows it; with no event, none is created.
- Results are not read here: they come from the results jobs ([results jobs](results-jobs.md)).

## Dropped games

`odds_lambda.espn_football_schedule.parse_scoreboard` drops a game with a team outside the league's canonical teams (`odds_core.league_teams.league_team`): a promoted club or a spelling with no alias (`odds_core.team.normalize_team_name`), which would otherwise store an event under a name OddsPortal and Betfair do not use and split the event in two. Each is a `odds_lambda.live_events.DroppedGame` with reason `odds_lambda.live_events.SkipReason.UNKNOWN_TEAM`, its detail naming the league and the raw name, and the job raises the dropped-games alert for it (`odds_lambda.live_events.alert_dropped_games`, key `games_dropped:fetch-espn-games-<suffix>`), repeating at the stale-reminder interval until the club is added to `odds_core.league_teams.LEAGUE_TEAMS` and its spellings in every source to the alias map ([add a sport](howto-add-sport.md)). Games with no two competitors, or a scheduled game with an invalid start, are skipped silently.

A run fails, alerting, when any event failed after the others are stored, and warns when ESPN lists no event while events are scheduled in the next `odds_lambda.espn_football_schedule.UPCOMING_DAYS` days; the close season is silent.

## Dry run

`odds events sync-football-schedule` (`odds_cli.commands.events.sync_football_schedule`) prints, per league, ESPN's events, how many events it would create, how many match an existing event and the unmatched names. Without `--apply` it reads the database and writes nothing. `--sport` (repeatable) picks leagues and `--days` the window.
