---
type: reference
title: Event matching across sources
description: The matcher each ingest path uses to find its events row (window, team comparison, ambiguity rule) and the duplicates and NULL event ids that follow.
sources:
  - odds_lambda.storage.event_resolver.resolve_event
  - odds_lambda.storage.event_merge.find_match_id
  - odds_lambda.storage.event_merge.find_historical_candidates
  - odds_lambda.storage.event_resolver.MatchWindow
  - odds_lambda.odds_api_events
  - odds_lambda.storage.injury_writer.InjuryWriter._match_event
  - odds_lambda.storage.game_log_writer.GameLogWriter._match_event
  - odds_lambda.polymarket_matching.NBA_ABBREV_MAP
  - odds_lambda.storage.event_resolver.attach_event
  - odds_lambda.polymarket_sports.PolymarketSportConfig
  - odds_core.team.normalize_team_name
  - odds_lambda.mlb_game_resolver
  - odds_lambda.live_events.LiveEventFinder
  - odds_lambda.live_events.alert_dropped_games
  - odds_lambda.storage.event_resolver.resolve_unscheduled_mlb_event
  - odds_lambda.live_events.LiveEventFinder.store_listings
  - odds_lambda.odds_api_events
mentions:
  - odds_lambda.mlb_stats_fetcher.MlbStatsFetcher.fetch_results
  - odds_lambda.mlb_stats_fetcher.MlbGameResult
status: stable
---

# Event matching across sources

There is no shared matcher for deciding which `events` row a piece of incoming data belongs to: each ingest path has its own query, window and ambiguity rule, and exact string equality on team names is the common core. The `events` columns and id prefixes are on [Events table](events.md); this page compares the matchers as implemented (read from the code, not run).

## Matchers

| Matcher | Used by | Window | Teams | Ambiguity |
|---|---|---|---|---|
| `odds_lambda.storage.event_resolver.resolve_event` with `odds_lambda.storage.event_resolver.MatchWindow.HISTORICAL` | historical OddsPortal and FDUK ingest scripts | commence time ±24 h | `odds_core.team.normalize_team_name` applied first; same sport | one candidate matches; several match only through the nearest within ±2 h, else, with none within ±2 h, `odds_lambda.storage.event_resolver.AmbiguousHistoricalMatch` is raised before anything is written and the ingest script skips the row (logged `ambiguous_historical_match`); with no candidate at all the row creates its own event (logged) ([Event resolver](event-resolver.md#steps)) |
| `odds_lambda.storage.event_resolver.resolve_event` | live OddsPortal scrape and Betfair jobs and the Odds API writers, through `odds_lambda.live_events.LiveEventFinder` | MLB: the event holding the row's `gamePk`, else the source's id, else ±2 h among events with none; other sports: the source's id, else ±2 h | `odds_core.team.normalize_team_name` applied first; sport filter | the nearest candidate, ties to the one stored first ([Event resolver](event-resolver.md#steps)) |
| `odds_lambda.storage.injury_writer.InjuryWriter._match_event` | [NBA injury reports](../collection/nba-injuries-and-stats.md) | ET `game_date` 10:00 to next day 06:00, converted to UTC | team is home or away | `None` |
| `odds_lambda.storage.game_log_writer.GameLogWriter._match_event` | [NBA game logs](../collection/nba-injuries-and-stats.md) | same ET window (a verbatim copy) | team is home or away | `None` |
| `odds_lambda.storage.event_resolver.attach_event` | `odds polymarket backfill-odds` and the fetch-polymarket-odds job | the source's id, else ±2 h | the sport's `team_name` (`odds_lambda.polymarket_sports.PolymarketSportConfig`), then `odds_core.team.normalize_team_name`; sport filter | the nearest candidate; with no candidate, or a refused id, it returns `None` and creates no event ([Event resolver](event-resolver.md#attaching-without-creating)) |

## Consequences of the differences

- **Create on miss.** If `odds_lambda.storage.event_resolver.resolve_event` finds nothing it creates an event, so for EPL a name or window mismatch there yields a duplicate event rather than a miss. An MLB event is found by its `gamePk` first, so sources that list one game at different times land on one event ([live MLB events](#live-mlb-events)). A game absent from `events` gets a row from the historical scripts, unless several events match it only within ±24 h, which the scripts skip, so cross-source joins depend on the matchers agreeing.
- **Different windows.** `odds_lambda.storage.event_resolver.resolve_event` uses ±2 h for live listings and ±24 h for historical ones (`odds_lambda.storage.event_resolver.MatchWindow`); both reuse an existing event (including a hex-id one) on a team match.
- **Finders never merge.** `odds_lambda.storage.event_merge.find_match_id` only reads, and returns the candidate a merge would keep ([kept event](event-merge.md#kept-event)). The resolver's historical match takes the nearest candidate within ±2 h when there are several, and none when no candidate is within ±2 h, so neighbouring fixtures of the same teams (an NBA back-to-back) are never conflated. Merging is explicit ([Event merge](event-merge.md)).
- **Odds API writers resolve.** Event sync, ingestion, `odds discover games` and the historical backfill attach a listed game to the event of its match whichever source stored it first, and create one only when none matches ([Odds API events](#odds-api-events)). Twins stored before the resolver stay until they are merged ([Event merge](event-merge.md)).
- **Alias gaps become NULL.** Injury and game-log matching normalise names with `odds_core.team.normalize_team_name`, which returns an unknown name unchanged (the game-log writer tries its abbreviation map first), so an alias gap shows up as a NULL `event_id`. Injury reports are league-wide, so unmatched pairs are expected for teams not playing that day. Unmatched NBA rows keep `event_id` NULL and are invisible to per-event readers.

## MLB gamePk resolver

`odds_lambda.mlb_game_resolver` maps listed MLB games to their MLB Stats API `gamePk`, the identity an event holds as its `mlb_stats_api` external id ([Events table](events.md#external-ids)). The live OddsPortal and Betfair jobs call it once per run ([live MLB events](#live-mlb-events)); events created before they did hold no `gamePk` until they are claimed.

`odds_lambda.mlb_game_resolver.resolve_batch` takes `odds_lambda.mlb_game_resolver.GameRow`s (home, away, start, and Betfair's game number when known) and makes one `odds_lambda.mlb_stats_fetcher.MlbStatsFetcher.fetch_results` range request over the MLB dates they span (split into 30-day calls only for longer spans). `odds_lambda.mlb_game_resolver.resolve_game_pks` returns, in input order, an `odds_lambda.mlb_game_resolver.ResolvedGame` (the `gamePk` and the `odds_lambda.mlb_game_resolver.ResolutionRule` that chose it) or an `odds_lambda.mlb_game_resolver.UnresolvedGame` with an `odds_lambda.mlb_game_resolver.UnresolvedReason`; `odds_lambda.mlb_game_resolver.resolve_batch` logs `mlb_game_unresolved` for each unresolved row. It raises `odds_lambda.mlb_game_resolver.MlbScheduleUnavailable` when the request still fails after the fetcher's retries, the response is not JSON or has no `dates` list, or it lists no games for a non-empty batch.

Rows and schedule slots are grouped by `odds_core.team.normalize_team_name` team pair and MLB date. A row's MLB date is the US Eastern date of its start (`odds_lambda.mlb_game_resolver.mlb_date`), so a 02:10Z start belongs to the previous day. A slot's date is the `dates[].date` it is listed under (`odds_lambda.mlb_stats_fetcher.MlbGameResult.schedule_date`), because a postponed slot stays under its original date while its `officialDate` moves to the day it is played. Every slot counts, postponed ones included, so a rescheduled game is found under both dates by one `gamePk`; the continuation slot of a resumed suspended game does not count. Within a group the first rule that applies wins:

1. **One game**: that game, unless its start is more than `odds_lambda.mlb_game_resolver.MAX_SINGLE_GAME_GAP` from the row's and is not TBD.
2. **Doubleheader**: the game with the row's game number; without one, when the batch lists as many un-numbered rows for the group as there are games, rows and games paired by start order. Order, not time, is the signal, because a TBD game 2 carries MLB's placeholder start a few minutes after game 1 while OddsPortal lists it at an estimated later time (#612).
3. **Lone row**: when every other game is Live or Final and began more than `odds_lambda.mlb_game_resolver.STARTED_SIBLING_MARGIN` (3 min) before the row's start, the remaining game if it is still Scheduled. Cancelled games and states the resolver does not recognise (`odds_lambda.mlb_stats_fetcher.MlbGameState`) are never taken here.
4. **Start time**: the one game with a slot of any state starting within `odds_lambda.mlb_game_resolver.START_MATCH_WINDOW` (15 min) of the row's start. This covers OddsPortal starts listed a minute after MLB's and a lone game-1 row while game 2 is pending. Two games inside the window, or a TBD placeholder start inside it, leave the row unresolved: a TBD game 2's placeholder sits a few minutes after game 1, so a row there may be either game.
5. Otherwise unresolved.

The same module reads a game's slots for the jobs that follow it after resolution: `odds_lambda.mlb_game_resolver.played_slots` gives each game's played slot (`odds_lambda.mlb_game_resolver.PlayedSlot`, neither postponed, cancelled nor a resumed continuation, preferring a set start over a TBD one), `odds_lambda.mlb_game_resolver.PlayedSlot.start_for` where an event moves to it, and `odds_lambda.mlb_game_resolver.final_scores` each Final game's score. The `gamePk` backfill ([Event merge](event-merge.md#assigning-gamepks)) and [fetch-mlb-results](../collection/results-jobs.md#fetch-mlb-results) use them.

## Live MLB events

`odds_lambda.live_events.LiveEventFinder` is how the live [OddsPortal](../collection/fetch-oddsportal-job.md) and [Betfair](../collection/betfair-exchange.md) jobs find each listed game's event. `odds_lambda.live_events.LiveEventFinder.for_run` resolves every distinct listed MLB game of the run with one `odds_lambda.mlb_game_resolver.resolve_batch` call, before the jobs open their database session; Betfair rows carry the doubleheader game number from the event name ([Betfair](../collection/betfair-exchange.md#client-and-adapter)), OddsPortal rows none. Other sports skip the MLB schedule. Each job stores its listings through `odds_lambda.live_events.LiveEventFinder.store_listings`: in one transaction, in fixture-lock order (`odds_lambda.live_events.lock_order`), each game and its snapshot in its own savepoint, so a failed game is rolled back alone and the run's other writes stand; `odds_lambda.live_events.ListingWrites.snapshots_stored` sums the snapshots inserted, not the envelopes stored. Per listed game, `odds_lambda.live_events.LiveEventFinder.resolve` then does one of the following; for other sports it is the resolved case without a `gamePk`. It records the source's id for the game (the OddsPortal event id, the Betfair event id). `odds_lambda.live_events.LiveEventFinder.fixture_key` gives the same lookup without creating anything: an MLB game's resolved `gamePk` (none when the resolver's rule was `odds_lambda.mlb_game_resolver.ResolutionRule.SIBLING_STARTED`), otherwise the event within ±2 h. The Betfair job uses it to match a book it did not price to a calendar fixture ([empty-run check](../collection/betfair-exchange.md#the-fetch-betfair-exchange-job)).

- **Resolved**: `odds_lambda.storage.event_resolver.resolve_event` with the `odds_lambda.mlb_game_resolver.ResolvedGame`, whose start is the game's scheduled start (from a slot neither postponed nor TBD) and whose sibling starts are the listed starts of the pair's other games that day. Its anchor time is that start, or the row's listed start when the schedule gives none.
  1. The event holding that `gamePk`.
  2. Else the event holding the source's id for the game, when it holds no `gamePk`.
  3. Else the nearest same-pair event within ±2 h of the anchor (ties go to the one stored first), which takes the `gamePk` when it holds none and no sibling start is nearer to it. An event holding a `gamePk` is never claimed, and a nearer sibling means it is that game's event, so a TBD game 2 listed a few minutes after game 1 does not take game 1's event.
  4. Else a new event created with the `gamePk` at the anchor time.

  When the event found or claimed has a `commence_time` other than the scheduled start (a postponed game replayed in another slot), it moves to that start and a POSTPONED event becomes SCHEDULED (`odds_lambda.storage.event_kickoff.move_kickoff` on the MLB Stats API's authority at the current time, which may move a game to a past start, [Kick-off authority](kickoff-authority.md#before-and-after-the-stored-kick-off)), and `event_rescheduled` is logged. Tiers are derived against the current kick-off, so the pre-slot snapshots of a postponed game lose their closing tier. A concurrent run resolving the same pair waits on the resolver's fixture lock until this run commits, so it finds the event holding the `gamePk` ([locks and transactions](event-resolver.md#locks-and-transactions)). The MLB Stats API ranks first for MLB, so `odds_lambda.storage.event_kickoff.move_kickoff` never holds its move back for another source's snapshots.
- **Unresolved**: no event and no snapshot. `odds_lambda.mlb_game_resolver.resolve_batch` has logged `mlb_game_unresolved` with the reason, and the job records an `odds_lambda.live_events.DroppedGame` (counted as `games_unresolved`).
- **Schedule unavailable** (`odds_lambda.mlb_game_resolver.MlbScheduleUnavailable`, logged once per run as `mlb_schedule_unavailable`): `odds_lambda.storage.event_resolver.resolve_unscheduled_mlb_event` attaches the game, under the fixture lock and recording the source's id, to the event of its pair within ±2 h only when exactly one exists and the row carries no doubleheader game number, since without the schedule either could be another game. Otherwise the game is dropped (`games_skipped_schedule_unavailable`) and no event is created, since a new one could not carry its `gamePk`. The job adds `MLB schedule unavailable: no new events created` to its errors.

When a run drops any game, the job sends `mlb_games_dropped:<job>` with the count and the matchups (`odds_lambda.live_events.alert_dropped_games`, [Alerts](../operations/alerts.md#alert-history)) and still completes with its heartbeat.

`odds_lambda.storage.event_resolver.ListedGame` raises `ValueError` for an MLB game without its `gamePk` or with the historical window, so the resolver never creates one. The historical loaders (`scripts/ingest_oddsportal.py`, `scripts/ingest_football_data_uk.py`) cover only NBA and EPL, and match with the historical window; their snapshots' tiers follow from each one's time basis and the event's other snapshots ([the tier a snapshot covers](fetch-tier.md#the-tier-a-snapshot-covers)). The Odds API writers resolve an MLB game through its `gamePk` too ([Odds API events](#odds-api-events)); `fetch-odds` is limited to EPL by `odds_lambda.scheduling.jobs._JOB_SPORTS`, so only a manual `odds discover` or `odds fetch` runs them for MLB.

## Odds API events

Event sync (`odds_lambda.event_sync.EventSyncService.sync_sport`), `odds discover games`, ingestion (`odds_lambda.ingestion.OddsIngestionService.ingest_sport`) and the historical backfill (`odds_analytics.backfill_executor.BackfillExecutor`) resolve each game the Odds API lists through `odds_lambda.live_events.LiveEventFinder.resolve` with `odds_lambda.odds_api_events.resolve_odds_api_event`. The listing's id (`odds_core.models.Event.id` of the parsed listing) is the game's `odds_api` external id and the key a created event takes; the listed teams and start are the match. `odds_lambda.odds_api_events.odds_api_finder` makes the finder once per run, so an MLB run reads the MLB schedule once and finds each game by `gamePk` exactly as the live jobs do ([live MLB events](#live-mlb-events)); a game that schedule lacks is dropped, logged as `odds_api_game_dropped` and counted by neither the inserted nor the updated count. Ingestion and the backfill report drops through `odds_lambda.live_events.alert_dropped_games`; a drop is not a failure.

- **Found by id.** The event holding the API's id as an external id, else, for an event stored before ids were recorded, the one keyed by it (`odds_lambda.storage.event_resolver.resolve_event`, step 3). Its sport, teams, status and scores are left as stored, a kick-off the API moves is offered to `odds_lambda.storage.event_kickoff.move_kickoff`, which decides whether the API may move it, and an id the API reissues replaces the recorded one ([Event resolver](event-resolver.md#refused-ids)).
- **Created in play.** The API lists games already under way, so a game with no event creates one however long ago it started (`create_stale`); the live OddsPortal and Betfair jobs refuse such a game (`odds_lambda.live_events.LIVE_EVENT_STALE_THRESHOLD`).
- **One transaction per batch.** `odds_lambda.odds_api_events.resolve_odds_api_events` resolves a batch in fixture-lock order, each game in its own savepoint (`odds discover games` commits every 100 games, so a transaction never holds thousands of locks). Ingestion resolves and stores each event in its own session and commit.
- **Lookups the other way.** `odds_lambda.odds_api_events.find_event_by_odds_api_id` finds the event an Odds API id names, for `fetch-scores` and `odds fetch scores`, and `odds_lambda.odds_api_events.odds_api_id_of` gives the id the API uses for a stored event (the recorded one, else the key of an event the API created, else none), which `odds backfill plan` and `odds backfill gaps` write into each game of the plan as `odds_api_id`.

How twins are merged and detected is on [Event merge](event-merge.md). The resolver's lookup order, lock and the ids it records are on [Event resolver](event-resolver.md).

## Related

- [Event merge](event-merge.md)
- [Events table](events.md)
- [OddsWriter](odds-writer.md)
- [Historical ingest scripts](../collection/historical-ingest-scripts.md)
- [OddsReader](odds-reader.md)
