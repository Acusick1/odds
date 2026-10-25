---
type: reference
title: Event matching across sources
description: The matcher each ingest path uses to find its events row (window, team comparison, ambiguity rule) and the duplicates and NULL event ids that follow.
sources:
  - odds_lambda.storage.event_resolver.resolve_event
  - odds_lambda.storage.event_merge.find_match_id
  - odds_lambda.storage.event_merge.find_historical_candidates
  - odds_lambda.storage.event_resolver.MatchWindow
  - odds_lambda.storage.event_merge.EventMerger.adopt_twins
  - odds_lambda.storage.injury_writer.InjuryWriter._match_event
  - odds_lambda.storage.game_log_writer.GameLogWriter._match_event
  - odds_lambda.polymarket_matching.match_polymarket_event
  - odds_lambda.polymarket_matching.NBA_ABBREV_MAP
  - odds_lambda.polymarket_matching.normalize_team
  - odds_lambda.storage.event_resolver.attach_event
  - odds_lambda.polymarket_odds.canonical_team
  - odds_core.team.normalize_team_name
  - odds_lambda.mlb_game_resolver
  - odds_lambda.live_events.LiveEventFinder
  - odds_lambda.live_events.alert_dropped_games
  - odds_lambda.storage.event_resolver.resolve_unscheduled_mlb_event
  - odds_lambda.live_events.LiveEventFinder.store_listings
  - odds_lambda.live_events.live_event_id
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
| `odds_lambda.storage.event_resolver.resolve_event` with `odds_lambda.storage.event_resolver.MatchWindow.HISTORICAL` | historical OddsPortal and FDUK ingest scripts | commence time ±24 h | `odds_core.team.normalize_team_name` applied first; same sport | one candidate matches; several match only through the nearest within ±2 h, else no match and the row creates its own event (logged) ([Event resolver](event-resolver.md#steps)) |
| `odds_lambda.storage.event_resolver.resolve_event` | live OddsPortal scrape and Betfair jobs, through `odds_lambda.live_events.LiveEventFinder` | MLB: the event holding the row's `gamePk`, else the source's id, else ±2 h among events with none; other sports: the source's id, else ±2 h | `odds_core.team.normalize_team_name` applied first; sport filter | the nearest candidate, ties to the one stored first ([Event resolver](event-resolver.md#steps)) |
| `odds_lambda.storage.injury_writer.InjuryWriter._match_event` | [NBA injury reports](../collection/nba-injuries-and-stats.md) | ET `game_date` 10:00 to next day 06:00, converted to UTC | team is home or away | `None` |
| `odds_lambda.storage.game_log_writer.GameLogWriter._match_event` | [NBA game logs](../collection/nba-injuries-and-stats.md) | same ET window (a verbatim copy) | team is home or away | `None` |
| `odds_lambda.polymarket_matching.match_polymarket_event` | `odds polymarket link` only | ticker date at 12:00 UTC ±24 h (the ticker alone is the input) | exact, via `odds_lambda.polymarket_matching.NBA_ABBREV_MAP` | `None` |
| `odds_lambda.storage.event_resolver.attach_event` | `odds polymarket backfill-odds` and the fetch-polymarket-odds job | the source's id, else ±2 h | `odds_lambda.polymarket_odds.canonical_team`, then `odds_core.team.normalize_team_name`; sport filter | the nearest candidate; with no candidate, or a refused id, it returns `None` and creates no event ([Event resolver](event-resolver.md#attaching-without-creating)) |

## Consequences of the differences

- **Create on miss.** If `odds_lambda.storage.event_resolver.resolve_event` finds nothing it creates an event, so for EPL a name or window mismatch there yields a duplicate event rather than a miss. An MLB event is found by its `gamePk` first, so sources that list one game at different times land on one event ([live MLB events](#live-mlb-events)). A game absent from `events` gets an `op_`/`fduk_` row from the historical scripts, so cross-source joins depend on the matchers agreeing.
- **Different windows.** `odds_lambda.storage.event_resolver.resolve_event` uses ±2 h for live listings and ±24 h for historical ones (`odds_lambda.storage.event_resolver.MatchWindow`); both reuse an existing event (including a hex-id one) on a team match.
- **Finders never merge.** `odds_lambda.storage.event_merge.find_match_id` only reads, and returns the canonical candidate. The resolver's historical match takes the nearest candidate within ±2 h when there are several, and none when no candidate is within ±2 h, so neighbouring fixtures of the same teams (an NBA back-to-back) are never conflated. Merging is explicit ([Event merge](event-merge.md)).
- **Odds API sync merges.** Event sync upserts under the API id without matching, then `odds_lambda.storage.event_merge.EventMerger.adopt_twins` merges every non-Odds-API event for the same sport, teams and ±2 h into it, so a match scraped before the API listed it ends up as one event.
- **Alias gaps become NULL.** Injury and game-log matching fall back to the raw team name when `odds_lambda.polymarket_matching.normalize_team` returns `None` (the game-log writer tries its abbreviation map first), so an alias gap shows up as a NULL `event_id`. Injury reports are league-wide, so unmatched pairs are expected for teams not playing that day. Unmatched NBA rows keep `event_id` NULL and are invisible to per-event readers.
- **Manual Polymarket linking.** Polymarket linking is manual and the fetch job does not call it ([Polymarket pipeline and tables](../collection/polymarket.md)).

## MLB gamePk resolver

`odds_lambda.mlb_game_resolver` maps listed MLB games to their MLB Stats API `gamePk`, the identity stored in `events.mlb_game_pk` ([Events table](events.md#columns)). The live OddsPortal and Betfair jobs call it once per run ([live MLB events](#live-mlb-events)); events created before they did keep a NULL `mlb_game_pk` until they are claimed.

`odds_lambda.mlb_game_resolver.resolve_batch` takes `odds_lambda.mlb_game_resolver.GameRow`s (home, away, start, and Betfair's game number when known) and makes one `odds_lambda.mlb_stats_fetcher.MlbStatsFetcher.fetch_results` range request over the MLB dates they span (split into 30-day calls only for longer spans). `odds_lambda.mlb_game_resolver.resolve_game_pks` returns, in input order, an `odds_lambda.mlb_game_resolver.ResolvedGame` (the `gamePk` and the `odds_lambda.mlb_game_resolver.ResolutionRule` that chose it) or an `odds_lambda.mlb_game_resolver.UnresolvedGame` with an `odds_lambda.mlb_game_resolver.UnresolvedReason`; `odds_lambda.mlb_game_resolver.resolve_batch` logs `mlb_game_unresolved` for each unresolved row. It raises `odds_lambda.mlb_game_resolver.MlbScheduleUnavailable` when the request still fails after the fetcher's retries, the response is not JSON or has no `dates` list, or it lists no games for a non-empty batch.

Rows and schedule slots are grouped by `odds_core.team.normalize_team_name` team pair and MLB date. A row's MLB date is the US Eastern date of its start (`odds_lambda.mlb_game_resolver.mlb_date`), so a 02:10Z start belongs to the previous day. A slot's date is the `dates[].date` it is listed under (`odds_lambda.mlb_stats_fetcher.MlbGameResult.schedule_date`), because a postponed slot stays under its original date while its `officialDate` moves to the day it is played. Every slot counts, postponed ones included, so a rescheduled game is found under both dates by one `gamePk`; the continuation slot of a resumed suspended game does not count. Within a group the first rule that applies wins:

1. **One game**: that game, unless its start is more than `odds_lambda.mlb_game_resolver.MAX_SINGLE_GAME_GAP` from the row's and is not TBD.
2. **Doubleheader**: the game with the row's game number; without one, when the batch lists as many un-numbered rows for the group as there are games, rows and games paired by start order. Order, not time, is the signal, because a TBD game 2 carries MLB's placeholder start a few minutes after game 1 while OddsPortal lists it at an estimated later time (#612).
3. **Lone row**: when every other game is Live or Final and began more than `odds_lambda.mlb_game_resolver.STARTED_SIBLING_MARGIN` (3 min) before the row's start, the remaining game if it is still Scheduled. Cancelled games and states the resolver does not recognise (`odds_lambda.mlb_stats_fetcher.MlbGameState`) are never taken here.
4. **Start time**: the one game with a slot of any state starting within `odds_lambda.mlb_game_resolver.START_MATCH_WINDOW` (15 min) of the row's start. This covers OddsPortal starts listed a minute after MLB's and a lone game-1 row while game 2 is pending. Two games inside the window, or a TBD placeholder start inside it, leave the row unresolved: a TBD game 2's placeholder sits a few minutes after game 1, so a row there may be either game.
5. Otherwise unresolved.

The same module reads a game's slots for the jobs that follow it after resolution: `odds_lambda.mlb_game_resolver.played_slots` gives each game's played slot (`odds_lambda.mlb_game_resolver.PlayedSlot`, neither postponed, cancelled nor a resumed continuation, preferring a set start over a TBD one), `odds_lambda.mlb_game_resolver.PlayedSlot.start_for` where an event moves to it, and `odds_lambda.mlb_game_resolver.final_scores` each Final game's score. The `gamePk` backfill ([Event merge](event-merge.md#assigning-gamepks)) and [fetch-mlb-results](../collection/results-jobs.md#fetch-mlb-results) use them.

## Live MLB events

`odds_lambda.live_events.LiveEventFinder` is how the live [OddsPortal](../collection/fetch-oddsportal-job.md) and [Betfair](../collection/betfair-exchange.md) jobs find each listed game's event. `odds_lambda.live_events.LiveEventFinder.for_run` resolves every distinct listed MLB game of the run with one `odds_lambda.mlb_game_resolver.resolve_batch` call, before the jobs open their database session; Betfair rows carry the doubleheader game number from the event name ([Betfair](../collection/betfair-exchange.md#client-and-adapter)), OddsPortal rows none. Other sports skip the MLB schedule. Each job stores its listings through `odds_lambda.live_events.LiveEventFinder.store_listings`: in one transaction, in fixture-lock order (`odds_lambda.live_events.lock_order`), each game and its snapshot in its own savepoint, so a failed game is rolled back alone and the run's other writes stand. Per listed game, `odds_lambda.live_events.LiveEventFinder.resolve` then does one of the following; for other sports it is the resolved case without a `gamePk`. It records the source's id for the game (the OddsPortal event id, the Betfair event id). `odds_lambda.live_events.LiveEventFinder.fixture_key` gives the same lookup without creating anything: an MLB game's resolved `gamePk` (none when the resolver's rule was `odds_lambda.mlb_game_resolver.ResolutionRule.SIBLING_STARTED`), otherwise the event within ±2 h. The Betfair job uses it to match a book it did not price to a calendar fixture ([empty-run check](../collection/betfair-exchange.md#the-fetch-betfair-exchange-job)).

- **Resolved**: `odds_lambda.storage.event_resolver.resolve_event` with the `odds_lambda.mlb_game_resolver.ResolvedGame`, whose start is the game's scheduled start (from a slot neither postponed nor TBD) and whose sibling starts are the listed starts of the pair's other games that day. Its anchor time is that start, or the row's listed start when the schedule gives none.
  1. The event holding that `gamePk`.
  2. Else the event holding the source's id for the game, when it holds no `gamePk`.
  3. Else the nearest same-pair event within ±2 h of the anchor (ties go to the one stored first), which takes the `gamePk` when it holds none and no sibling start is nearer to it. An event holding a `gamePk` is never claimed, and a nearer sibling means it is that game's event, so a TBD game 2 listed a few minutes after game 1 does not take game 1's event.
  4. Else a new event created with the `gamePk` at the anchor time, with the `op_live_` id that time gives (`odds_lambda.live_events.live_event_id`). When that id belongs to another event, the resolver logs `event_id_collision` and uses the id with `_<gamePk>` appended, which no other game can produce.

  When the event found or claimed has a `commence_time` other than the scheduled start (a postponed game replayed in another slot), it moves to that start, a POSTPONED event becomes SCHEDULED, its stored snapshots' `hours_until_commence` and `fetch_tier` are recomputed against the new start (`odds_lambda.storage.event_kickoff.move_kickoff` on the MLB Stats API's authority at the current time (it may move a game to a past start, [Kick-off authority](kickoff-authority.md#before-and-after-the-stored-kick-off)), through `odds_lambda.storage.event_merge.retime_snapshots`, the helper merges use; [Kick-off authority](kickoff-authority.md)), and `event_rescheduled` is logged; the snapshot stored next is timed against the new start too. So the pre-slot snapshots of a postponed game lose their closing tier. A concurrent run resolving the same pair waits on the resolver's fixture lock until this run commits, so it finds the event holding the `gamePk` ([locks and transactions](event-resolver.md#locks-and-transactions)).
- **Unresolved**: no event and no snapshot. `odds_lambda.mlb_game_resolver.resolve_batch` has logged `mlb_game_unresolved` with the reason, and the job records an `odds_lambda.live_events.DroppedGame` (counted as `games_unresolved`).
- **Schedule unavailable** (`odds_lambda.mlb_game_resolver.MlbScheduleUnavailable`, logged once per run as `mlb_schedule_unavailable`): `odds_lambda.storage.event_resolver.resolve_unscheduled_mlb_event` attaches the game, under the fixture lock and recording the source's id, to the event of its pair within ±2 h only when exactly one exists and the row carries no doubleheader game number, since without the schedule either could be another game. Otherwise the game is dropped (`games_skipped_schedule_unavailable`) and no event is created, since a new one could not carry its `gamePk`. The job adds `MLB schedule unavailable: no new events created` to its errors.

When a run drops any game, the job sends `mlb_games_dropped:<job>` with the count and the matchups (`odds_lambda.live_events.alert_dropped_games`, [Alerts](../operations/alerts.md#alert-history)) and still completes with its heartbeat.

`odds_lambda.storage.event_resolver.ListedGame` raises `ValueError` for an MLB game without its `gamePk` or with the historical window, so the resolver never creates one. The other paths that create events do not run for MLB: Odds API event sync and ingestion (`fetch-odds`) are limited to EPL by `odds_lambda.scheduling.jobs._JOB_SPORTS`, and the historical loaders (`scripts/ingest_oddsportal.py`, `scripts/ingest_football_data_uk.py`) cover only NBA and EPL. A manual `odds discover` or `odds fetch` for MLB would create Odds API events with no `gamePk`; merging a live twin into one carries the twin's `gamePk` over ([Event merge](event-merge.md#merging)).

How twins are merged and detected is on [Event merge](event-merge.md). The resolver's lookup order, lock and the ids it records are on [Event resolver](event-resolver.md).

## Related

- [Event merge](event-merge.md)
- [Events table](events.md)
- [OddsWriter](odds-writer.md)
- [Historical ingest scripts](../collection/historical-ingest-scripts.md)
- [OddsReader](odds-reader.md)
