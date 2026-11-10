---
type: reference
title: Kick-off authority
description: Which source may move an event's commence_time for each sport, the move_kickoff function every move goes through, and what a move does to the event.
sources:
  - odds_lambda.storage.event_kickoff
status: stable
---

# Kick-off authority

Every move of an [event](events.md)'s `commence_time` on behalf of a source goes through `odds_lambda.storage.event_kickoff.move_kickoff(session, event, source, start, now=...)`. It moves the event only when `source` is the sport's schedule authority, and returns whether it moved. The caller injects the clock as `now`.

## Authority order

`odds_lambda.storage.event_kickoff.KICKOFF_AUTHORITY` maps each sport to its ordered sources, best first:

| Sport | Order |
|---|---|
| `baseball_mlb` | MLB Stats API (`mlb_stats_api`) |
| `soccer_epl`, `soccer_efl_champ`, `soccer_spain_la_liga`, `soccer_italy_serie_a`, `soccer_germany_bundesliga`, `soccer_france_ligue_one` | OddsPortal (`oddsportal`), then Odds API (`odds_api`), then football-data.co.uk (`football_data_uk`) |
| `basketball_nba` | ESPN (`espn`), then OddsPortal (`oddsportal`), then Odds API (`odds_api`) |

A source outside its sport's order never moves an event: Betfair and Polymarket move no EPL or NBA event, ESPN moves no EPL event, and football-data.co.uk moves no NBA event. A source in the order moves an event only when no source ranked above it holds an external id on that event ([external ids](events.md#external-ids)), so before kick-off ESPN moves any NBA event to its start, the Odds API moves an EPL event only while it holds no OddsPortal id and an NBA event only while it holds neither an ESPN nor an OddsPortal id, and football-data.co.uk moves an EPL event only while it holds neither an OddsPortal nor an Odds API id and no other source has a snapshot on it (see [Before and after the stored kick-off](#before-and-after-the-stored-kick-off)).

## Basis for the football and NBA order

Measured read-only on a copy of the local database taken 2026-10-25. The copy holds one event per game, the source twins already merged, so sources are compared with ESPN fixtures (team rows, EPL only) and, for NBA, the minute of the hour. Each count is a `SELECT` over `events`, `odds_snapshots.raw_data->>'commence_time'` (Odds API), `polymarket_events.end_date` and `espn_fixtures.date`.

- **EPL, OddsPortal against ESPN:** 1659 of 1663 matched events equal; the four differences are 10 to 30 minutes.
- **EPL, Odds API against ESPN:** 10 of 10 equal, though only 10 of its 83 events fall in ESPN's loaded range. Before kick-off it differed from the stored time on 1 of 167 events, a day early for six days while ESPN and OddsPortal held the original time.
- **NBA, minute of the hour:** the Odds API's first pre-game value falls on :00 or :30 for 148 of 813 events, the rest 10 minutes later; OddsPortal's event rows do for 4972 of 5001 and Polymarket's `end_date` for 891 of 891.
- **NBA, against ESPN's scoreboard (production, 2026-11-06, [#794](https://github.com/Acusick1/odds/issues/794)):** of 940 FINAL events stored off :00 and :30, 487 were exactly 10 minutes after ESPN's start (the Odds API's pre-game listing) and 405 were 11 to 14 minutes after it (the Odds API's in-play start, to the minute); 920 of the 940 equal ESPN's start floored to the half hour. ESPN's own start is on :00 or :30 for 7027 of 7073 games from 2021 to 2026; the rest are staggered slots (Election Day games at :15 and :45, which OddsPortal agrees with) and nationally televised games already played, whose start ESPN rewrites to the actual tip (for example 21:47 ET for a 21:30 ET game).
- **Odds API in-play:** once a game is live it reports the actual start: 78 of 623 NBA events differ from the stored kick-off (up to 14 minutes), 7 of 135 EPL events (up to 2). On production on 2026-11-05, 115 NBA FINAL events stored such a start (seconds past the minute, 1 to 18 minutes after ESPN's scheduled start); all 75 of them with archived Odds API payloads first show that start in a poll taken after it ([#784](https://github.com/Acusick1/odds/issues/784)).

ESPN ranks first for NBA because its scoreboard is the schedule: it lists every game, and the `fetch-espn-fixtures-nba` job reads it each day for three weeks ahead ([NBA schedule](../collection/nba-injuries-and-stats.md#nba-schedule-from-espn)), so it sees each game before it starts. OddsPortal above the Odds API rests on the NBA grid and the one EPL reversal, and OddsPortal's response to a postponement was not measured (inferred). The Championship, La Liga, Serie A, the Bundesliga and Ligue 1 take the EPL's order unmeasured, since no source had collected them when it was set (inferred), with ESPN last: their events are created from ESPN's scoreboards ([ESPN football schedule](../collection/espn-football-schedule.md)), whose kick-offs equalled OddsPortal's and Betfair's when probed. Last, ESPN moves an event before kick-off only while no other source holds an id on it, and moves one whose kick-off has passed to a start at least an hour later and still ahead, so a postponement only ESPN reports is followed.

Left out of both orders:

- **Betfair:** `betfair_meta.market_start_time` is the market's start, which can precede the event's, and the local copy holds none.
- **Polymarket:** the EPL start time is used for matching and not stored; the NBA `end_date` cannot be compared with OddsPortal because the copy's twins are merged and OddsPortal's own kick-off was not kept.
- **ESPN for EPL:** team rows with no event link or id.
- **football-data.co.uk for NBA:** it covers EPL only. For EPL it ranks last: its kick-off is assumed before 2019-20 (`timing.kickoff_time_assumed`) and 588 of 1597 matched events differ from ESPN, so it moves only an event no better source holds an id on.

## Before and after the stored kick-off

A source in `odds_lambda.storage.event_kickoff.IN_PLAY_START_SOURCES`, the Odds API, lists a game already under way at its actual start, so its start at or before `now` is refused whatever the stored kick-off. An event stored at the Odds API's late pre-game time (for example 23:40 for a 23:30 game) therefore keeps it when a poll at 23:37 lists the 23:35:56 tip-off.

For every source except those in `odds_lambda.storage.event_kickoff.PLAYED_START_SOURCES`:

- **Before the stored kick-off** (`now` earlier), the rank rule above applies.
- **Once it has passed**, a start at or before `now` is refused, since a live feed reports the actual start, and so is a start less than `odds_lambda.storage.event_kickoff.POSTPONEMENT_MIN_DELAY` (1 h) after the stored kick-off: a late listing or a delayed start, not a new slot. A later start still ahead is a postponement and is accepted from any source in the order, whatever ids the event holds, so a postponement the top source stopped listing is still followed. So an Odds API poll at 23:00 listing an NBA game at its 23:10 leaves an event stored at 23:00 alone.

The MLB Stats API and football-data.co.uk are the sources in `odds_lambda.storage.event_kickoff.PLAYED_START_SOURCES`. The MLB schedule dates a postponed game by when it was played, and football-data.co.uk records the kick-off of a match already played, so each may move an event to a start in the past and skips the after-kick-off rule. The rank rule still applies to them. A source in that set that the order ranks below another (football-data.co.uk for EPL) moves only an event no other source has observed, because the ids an event holds do not show which sources observed it: every snapshot of the event must have `source` equal to it (`odds_lambda.storage.event_kickoff._observed_by_another`). An event with no snapshots qualifies.

## A move

Nothing happens when no start is given or it equals the stored kick-off. Otherwise:

- `commence_time` becomes the new start and `updated_at` is set.
- A POSTPONED event becomes SCHEDULED, and so does a LIVE event moved to a start after `now` (a postponement).
- The event's snapshots are not touched. Their tiers are derived against the current kick-off ([Fetch tier](fetch-tier.md#the-tier-a-snapshot-covers)), so the pre-move snapshots of a postponed game lose their closing tier.
- `event_rescheduled` is logged with the source and both times.

## Callers

- `odds_lambda.storage.event_resolver.resolve_event` ([Event resolver](event-resolver.md)), for the live OddsPortal and Betfair jobs and the historical ingest scripts (it alone offers the start; `odds_lambda.storage.event_resolver.attach_listing`, which the Odds API backfill uses and `odds_lambda.storage.event_resolver.attach_event` (the Polymarket writers) wraps, offers none; [live MLB events](event-matching.md#live-mlb-events)): the MLB Stats API with the schedule's start for MLB, after recording the `gamePk`, else the listing source with its listed start. An OddsPortal historical listing is always in the past, so it never moves a played event, and an ambiguous historical match raises `odds_lambda.storage.event_resolver.AmbiguousHistoricalMatch` before any start is offered.
- `odds_lambda.jobs.fetch_mlb_results` ([fetch-mlb-results](../collection/results-jobs.md#fetch-mlb-results)), with the MLB Stats API.
- `odds_lambda.mlb_season_load.load_games` ([MLB season load](../collection/mlb-season-load.md)), through `odds_lambda.storage.event_resolver.resolve_event`, with the MLB Stats API and the played slot's start (none for a TBD start).
- `odds_lambda.storage.game_pk_backfill.apply_assignment` ([assigning gamePks](event-merge.md#assigning-gamepks)), with the MLB Stats API, after recording the `gamePk` (`odds_lambda.storage.event_resolver.record_game_pk`).
- `odds_lambda.odds_api_events.resolve_odds_api_event` ([events](events.md#writing-events)), for Odds API ingestion, event sync and `odds discover games` (the API's reissued id replaces the recorded `odds_api` external id), through `odds_lambda.storage.event_resolver.resolve_event`: the Odds API with the listed start. Except for MLB, a game listed under way that matches no event is dropped, so its in-play start never becomes a new event's kick-off ([Odds API events](event-matching.md#odds-api-events)).

## Related

- [Event resolver](event-resolver.md)
- [Event status lifecycle](../collection/event-status-lifecycle.md)
- [Fetch tier](fetch-tier.md)
