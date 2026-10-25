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

Every move of an [event](events.md)'s `commence_time` on behalf of a source goes through `odds_lambda.storage.event_kickoff.move_kickoff(session, event, source, start)`. It moves the event only when `source` is the sport's schedule authority, and returns whether it moved.

## Authority order

`odds_lambda.storage.event_kickoff.KICKOFF_AUTHORITY` maps each sport to its ordered sources, best first:

| Sport | Order |
|---|---|
| `baseball_mlb` | MLB Stats API (`mlb_stats_api`) |
| `soccer_epl` | OddsPortal (`oddsportal`), then Odds API (`odds_api`) |
| `basketball_nba` | OddsPortal (`oddsportal`), then Odds API (`odds_api`) |

A source outside its sport's order never moves an event: Betfair, Polymarket, ESPN and football-data.co.uk move no EPL or NBA event. A source in the order moves an event only when no source ranked above it holds an external id on that event ([external ids](events.md#external-ids)), so the Odds API moves an EPL or NBA event only while it holds no OddsPortal id.

## Basis for the EPL and NBA order

Measured on the local database copy of 2026-10-25, read-only. The sources store kick-offs differently, and the copy holds no pair of events for one game under two sources, so the comparison uses ESPN fixtures (team-level, EPL only) and the minute of the hour.

- **EPL, OddsPortal against ESPN:** 1659 of 1663 matched events equal; the four differences are 10 to 30 minutes.
- **EPL, Odds API against ESPN:** 10 of 10 equal, but only 10 of its 83 events fall in ESPN's loaded range. Its pre-game kick-off differed from the stored one on 1 of 167 events: it listed a day earlier for six days (2026-04-20 to 2026-04-26) while ESPN and OddsPortal held the original time.
- **NBA, minute of the hour:** tip-offs fall on :00 and :30. OddsPortal's event rows use them on 4972 of 5001, Polymarket's `end_date` on 891 of 891, the Odds API's first pre-game value on 148 of 813; the other 664 sit 10 minutes later (:10 and :40).
- **Odds API in-play value:** it reports the actual start once a game is live. An in-play value differed from the stored kick-off on 78 of 623 NBA events (up to 14 minutes) and 7 of 135 EPL events (up to 2 minutes). A kick-off taken from an in-play snapshot moves the event off its scheduled start.

OddsPortal ranks above the Odds API on these differences (NBA grid, the EPL reversal). That the Odds API ranks second, and OddsPortal's response to a real postponement, rest on one EPL event and no postponement data (inferred).

Left out for lack of a stored kick-off or of a comparison:

- **Betfair:** `betfair_meta.market_start_time` is the market's start, which can precede the event's; the local copy holds none to measure.
- **Polymarket:** the EPL start time is used for matching and not stored; the NBA `end_date` has no game shared with OddsPortal to compare against.
- **ESPN:** its fixtures are team rows with no event link or id.
- **football-data.co.uk:** its kick-off is assumed (`timing.kickoff_time_assumed`), and 588 of 1597 matched events differ from ESPN, 549 by an hour or more.

## A move

Nothing happens when no start is given or it equals the stored kick-off. Otherwise:

- `commence_time` becomes the new start and `updated_at` is set.
- A POSTPONED event becomes SCHEDULED.
- Every stored snapshot of the event has `hours_until_commence` and `fetch_tier` recomputed against the new start (`odds_lambda.storage.event_merge.retime_snapshots`), so the pre-move snapshots of a postponed game lose their closing tier.
- `event_rescheduled` is logged with the source and both times.

## Callers

All pass the MLB Stats API as the source:

- `odds_lambda.storage.writers.OddsWriter.find_or_create_event`, for the live OddsPortal and Betfair jobs ([live MLB events](event-matching.md#live-mlb-events)).
- `odds_lambda.jobs.fetch_mlb_results` ([fetch-mlb-results](../collection/results-jobs.md#fetch-mlb-results)).
- `odds_lambda.storage.game_pk_backfill.apply_assignment` ([assigning gamePks](event-merge.md#assigning-gamepks)).
- `odds_lambda.storage.event_resolver.resolve_event` ([Event resolver](event-resolver.md)).

`odds_lambda.storage.writers.OddsWriter.upsert_event` and `odds_lambda.storage.writers.OddsWriter.bulk_upsert_events` overwrite `commence_time` directly, outside this function ([upsert verbs](events.md#upsert-verbs)).

## Related

- [Event resolver](event-resolver.md)
- [Event status lifecycle](../collection/event-status-lifecycle.md)
- [Fetch tier](fetch-tier.md)
