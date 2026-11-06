---
type: reference
title: MLB season load
description: "odds events sync-mlb-seasons: which MLB Stats API games become FINAL events, where postponed, resumed and doubleheader games are placed, how existing events are reused, and what the report lists."
sources:
  - odds_lambda.mlb_season_load
  - odds_cli.commands.events.sync_mlb_seasons
mentions:
  - odds_lambda.mlb_stats_fetcher.MlbStatsFetcher.fetch_results
  - odds_lambda.mlb_game_resolver.played_slots
  - odds_lambda.mlb_game_resolver.final_scores
  - odds_lambda.storage.event_resolver.resolve_event
status: stable
---

# MLB season load

The live jobs create an MLB event only for a game a source prices, so seasons played before collection began hold no events, and the [Odds API backfill planner](odds-api-backfill.md#planning-calls) costs a season from its stored FINAL events. `odds_lambda.mlb_season_load` loads whole calendar seasons from the MLB Stats API schedule instead: one FINAL event per played game, holding its `gamePk`.

## Games

`odds_lambda.mlb_season_load.fetch_season` reads calendar year `season` (1 January to 31 December) with `odds_lambda.mlb_stats_fetcher.MlbStatsFetcher.fetch_results`, in 30-day calls, and `odds_lambda.mlb_season_load.final_games` keeps each Final game once:

- **Game types.** Only the types in `odds_lambda.collection_calendar.MLB_PRICED_GAME_TYPES` load: the regular season and the four postseason rounds. Spring training, exhibitions and the All-Star Game are left out.
- **Final only.** A game loads when a slot of it that is not a resumed continuation is Final with both scores (`odds_lambda.mlb_game_resolver.final_scores`). A cancelled game, or one never played, has none.
- **Postponed games** are listed under each date they were scheduled for, sharing one `gamePk`; the event is placed at the slot played (`odds_lambda.mlb_game_resolver.played_slots`).
- **Resumed games.** A suspended game is placed at the slot it began in, and its score is that slot's; the continuation slot (`resumedFrom`) only repeats it.
- **Doubleheaders.** Each game has its own `gamePk` and event. MLB lists a game 2 whose start was TBD at a placeholder a few minutes after game 1, so that game's event starts at its first pitch (the `gameInfo` hydration's `firstPitch`, `odds_lambda.mlb_stats_fetcher.MlbGameResult.first_pitch`), or at the placeholder when there is none. The other games of the pair on that MLB date are each game's `sibling_starts`.

## Writing

`odds_lambda.mlb_season_load.load_games` resolves each game through `odds_lambda.storage.event_resolver.resolve_event` as a game of source `mlb_stats_api` with its `gamePk` and no `earliest_created_start`, so a played game with no event is created ([resolver](../storage/event-resolver.md)), one transaction per game:

- An event holding the `gamePk` is reused. Otherwise an event of the same teams within ±2 h that holds no `gamePk` is matched and gains it, unless a doubleheader sibling is scheduled nearer to it. Otherwise a SCHEDULED event is created. A rerun therefore matches the events it made, and the live jobs' events are matched rather than duplicated.
- The MLB Stats API is MLB's kick-off authority and may move an event to a past start ([kick-off authority](../storage/kickoff-authority.md)), so a reused event moves to the played slot's scheduled start. A TBD game offers no start, so a reused event keeps its time.
- An event that is not FINAL, or FINAL without both scores, is set FINAL with MLB's score through `odds_lambda.storage.writers.OddsWriter.update_event_status`. An event already FINAL with another score is left as it is and listed as a score conflict for `odds events repair-scores` ([verifying MLB scores](results-jobs.md#verifying-mlb-scores)).

A game that fails is listed with its error and the others go ahead. `odds_lambda.mlb_season_load.LoadReport` counts the games (in total and per season), the events created and matched and those set FINAL.

## Command

`odds events sync-mlb-seasons --season YEAR [--season YEAR ...] [--apply]` (`odds_cli.commands.events.sync_mlb_seasons`) runs `odds_lambda.mlb_season_load.load_seasons` and prints the report. Without `--apply` each game is rolled back, so the counts are what an apply would do.

> [!WARNING]
> `--apply` creates and settles events in the database `DATABASE_URL` names. On the production host that is the production database: run `./scripts/backup-prod-db.sh <label>` first ([back up production](../foundations/howto-back-up-production-database.md)).

## Related

- [Results and status jobs](results-jobs.md#fetch-mlb-results)
- [Assign MLB events their gamePk](../storage/howto-assign-mlb-game-pks.md)
- [Backfill Odds API history](howto-backfill-odds-api-history.md)
