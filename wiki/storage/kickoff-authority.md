---
type: reference
title: Kick-off authority
description: Which source may move an event's commence_time for each sport, the move_kickoff function every move goes through, and what a move does to the event and its snapshots.
sources:
  - odds_lambda.storage.event_kickoff
  - odds_lambda.storage.event_merge.retime_snapshots
status: stable
---

# Kick-off authority

Every move of an [event](events.md)'s `commence_time` on behalf of a source goes through `odds_lambda.storage.event_kickoff.move_kickoff(session, event, source, start, now=...)`. It moves the event only when `source` is the sport's schedule authority, and returns whether it moved. The caller injects the clock as `now`.

## Authority order

`odds_lambda.storage.event_kickoff.KICKOFF_AUTHORITY` maps each sport to its ordered sources, best first:

| Sport | Order |
|---|---|
| `baseball_mlb` | MLB Stats API (`mlb_stats_api`) |
| `soccer_epl` | OddsPortal (`oddsportal`), then Odds API (`odds_api`), then football-data.co.uk (`football_data_uk`) |
| `basketball_nba` | OddsPortal (`oddsportal`), then Odds API (`odds_api`) |

A source outside its sport's order never moves an event: Betfair, Polymarket and ESPN move no EPL or NBA event, and football-data.co.uk moves no NBA event. A source in the order moves an event only when no source ranked above it holds an external id on that event ([external ids](events.md#external-ids)), so the Odds API moves an EPL or NBA event only while it holds no OddsPortal id, and football-data.co.uk moves an EPL event only while it holds neither an OddsPortal nor an Odds API id and no other source has a snapshot on it (see [Before and after the stored kick-off](#before-and-after-the-stored-kick-off)).

## Basis for the EPL and NBA order

Measured read-only on a copy of the local database taken 2026-10-25. The copy holds one event per game, the source twins already merged, so sources are compared with ESPN fixtures (team rows, EPL only) and, for NBA, the minute of the hour. Each count is a `SELECT` over `events`, `odds_snapshots.raw_data->>'commence_time'` (Odds API), `polymarket_events.end_date` and `espn_fixtures.date`.

- **EPL, OddsPortal against ESPN:** 1659 of 1663 matched events equal; the four differences are 10 to 30 minutes.
- **EPL, Odds API against ESPN:** 10 of 10 equal, though only 10 of its 83 events fall in ESPN's loaded range. Before kick-off it differed from the stored time on 1 of 167 events, a day early for six days while ESPN and OddsPortal held the original time.
- **NBA, minute of the hour:** the Odds API's first pre-game value falls on :00 or :30 for 148 of 813 events, the rest 10 minutes later; OddsPortal's event rows do for 4972 of 5001 and Polymarket's `end_date` for 891 of 891. That :00 and :30 are the true tip-offs, so the Odds API is the one that is late, is a reading of those three sources (inferred).
- **Odds API in-play:** once a game is live it reports the actual start: 78 of 623 NBA events differ from the stored kick-off (up to 14 minutes), 7 of 135 EPL events (up to 2).

OddsPortal above the Odds API rests on the NBA grid and the one EPL reversal, and OddsPortal's response to a postponement was not measured (inferred).

Left out of both orders:

- **Betfair:** `betfair_meta.market_start_time` is the market's start, which can precede the event's, and the local copy holds none.
- **Polymarket:** the EPL start time is used for matching and not stored; the NBA `end_date` cannot be compared with OddsPortal because the copy's twins are merged and OddsPortal's own kick-off was not kept.
- **ESPN:** team rows with no event link or id.
- **football-data.co.uk for NBA:** it covers EPL only. For EPL it ranks last: its kick-off is assumed before 2019-20 (`timing.kickoff_time_assumed`) and 588 of 1597 matched events differ from ESPN, so it moves only an event no better source holds an id on.

## Before and after the stored kick-off

For every source except those in `odds_lambda.storage.event_kickoff.PLAYED_START_SOURCES`:

- **Before the stored kick-off** (`now` earlier), the rank rule above applies.
- **Once it has passed**, a start at or before `now` is refused, since a live feed reports the actual start. A start still ahead is accepted from any source in the order, whatever ids the event holds, so a postponement the top source stopped listing is still followed.

The MLB Stats API and football-data.co.uk are the sources in `odds_lambda.storage.event_kickoff.PLAYED_START_SOURCES`. The MLB schedule dates a postponed game by when it was played, and football-data.co.uk records the kick-off of a match already played, so each may move an event to a start in the past and skips the after-kick-off rule. The rank rule still applies to them. A source in that set that the order ranks below another (football-data.co.uk for EPL) moves only an event no other source has observed, because the events stored before external ids were recorded hold none: every snapshot of the event must have `source` equal to it, or, for a snapshot with no `source`, a `raw_data` tag equal to it (`odds_lambda.storage.event_kickoff._observed_by_another`; no tag is the Odds API). An event with no snapshots qualifies.

## A move

Nothing happens when no start is given or it equals the stored kick-off. Otherwise:

- `commence_time` becomes the new start and `updated_at` is set.
- A POSTPONED event becomes SCHEDULED, and so does a LIVE event moved to a start after `now` (a delayed kick-off).
- Every stored snapshot of the event has `hours_until_commence` and `fetch_tier` recomputed against the new start (`odds_lambda.storage.event_merge.retime_snapshots`), so the pre-move snapshots of a postponed game lose their closing tier.
- `event_rescheduled` is logged with the source and both times.

## Callers

- `odds_lambda.storage.event_resolver.resolve_event` ([Event resolver](event-resolver.md)), for the live OddsPortal and Betfair jobs and the historical ingest scripts (it alone offers the start; `odds_lambda.storage.event_resolver.attach_event` offers none; [live MLB events](event-matching.md#live-mlb-events)): the MLB Stats API with the schedule's start for MLB, else the listing source with its listed start. An OddsPortal historical listing is always in the past, so it never moves a played event.
- `odds_lambda.jobs.fetch_mlb_results` ([fetch-mlb-results](../collection/results-jobs.md#fetch-mlb-results)), with the MLB Stats API.
- `odds_lambda.storage.game_pk_backfill.apply_assignment` ([assigning gamePks](event-merge.md#assigning-gamepks)), with the MLB Stats API.

- `odds_lambda.storage.writers.OddsWriter.upsert_event` and `odds_lambda.storage.writers.OddsWriter.bulk_upsert_events` ([upsert verbs](events.md#upsert-verbs)), for Odds API ingestion and event sync: the Odds API with the listed start.

## Related

- [Event resolver](event-resolver.md)
- [Event status lifecycle](../collection/event-status-lifecycle.md)
- [Fetch tier](fetch-tier.md)
