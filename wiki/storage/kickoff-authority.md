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
| `soccer_epl` | none |
| `basketball_nba` | none |

A source outside its sport's order never moves an event, so no source moves EPL or NBA events through this function (#676). A source in the order moves an event only when no source ranked above it holds an external id on that event ([external ids](events.md#external-ids)).

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

`odds_lambda.storage.writers.OddsWriter.upsert_event` and `odds_lambda.storage.writers.OddsWriter.bulk_upsert_events` still overwrite `commence_time` directly ([upsert verbs](events.md#upsert-verbs)).

## Related

- [Event resolver](event-resolver.md)
- [Event status lifecycle](../collection/event-status-lifecycle.md)
- [Fetch tier](fetch-tier.md)
