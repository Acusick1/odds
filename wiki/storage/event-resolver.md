---
type: reference
title: Event resolver
description: How resolve_event finds or creates the one event of a listed game (lock, lookup order, ±2 h match, creation) and records the source's id, and which ids it refuses.
sources:
  - odds_lambda.storage.event_resolver
  - odds_core.models.EventExternalId
  - odds_lambda.storage.event_unmatched
mentions:
  - odds_lambda.storage.event_merge.find_historical_candidates
  - odds_lambda.mlb_game_resolver.ResolvedGame
  - odds_lambda.storage.event_merge.find_match_candidates
status: stable
---

# Event resolver

`odds_lambda.storage.event_resolver.resolve_event(session, game)` returns the one [event](events.md) of a game a source lists, creating it only when none matches, and records the source's id for it in `event_external_ids` ([Events table](events.md#external-ids)). The live OddsPortal and Betfair jobs and the Odds API writers call it through `odds_lambda.live_events.LiveEventFinder.resolve` ([live MLB events](event-matching.md#live-mlb-events)), and the historical OddsPortal and football-data.co.uk ingest scripts call it directly with the historical window ([Historical ingest scripts](../collection/historical-ingest-scripts.md#matching-ids-and-idempotency)); the other writers find events their own way ([Event matching across sources](event-matching.md)). The decision behind it is [ADR-0018](../../docs/adr/0018-surrogate-event-key-with-external-ids.md).

## Input

`odds_lambda.storage.event_resolver.ListedGame` is the game as one source lists it:

- `source` (`odds_core.vocabulary.Source`), `sport_key`, `sport_title`, both teams and `commence_time`.
- `external_id`: the id the source issued for the game. Constructing a game with one from a source outside `odds_core.vocabulary.EXTERNAL_ID_SOURCES` (football-data.co.uk) raises `ValueError`.
- `mlb_game`: an MLB game's `odds_lambda.mlb_game_resolver.ResolvedGame`, required for `baseball_mlb` (`ValueError` without it).
- `legacy_id`: the string id a created event takes while `events.id` is a string (the `op_live_` id for the live jobs, the API's id for an Odds API listing); only needed to create.
- `match_window` (`odds_lambda.storage.event_resolver.MatchWindow`): `live` (the default, ±2 h) or `historical` (±24 h, for rows whose recorded start may be a day off). A historical MLB game raises `ValueError`, since MLB matches by `gamePk`.
- `earliest_created_start`: set by a live OddsPortal or Betfair listing, not by an Odds API one. A game starting before it matches an event but creates none: `odds_lambda.storage.event_resolver.StaleListing` is raised instead, before anything is written. The OddsPortal and Betfair jobs set it an hour (`odds_lambda.live_events.LIVE_EVENT_STALE_THRESHOLD`) before the run.

The result is an `odds_lambda.storage.event_resolver.ResolvedEvent`: the event and an `odds_lambda.storage.event_resolver.Resolution` saying how it was found (`by_external_id`, `by_game_pk`, `matched` or `created`).

## Steps

Team names are normalized with `odds_core.team.normalize_team_name` first.

1. **Lock.** A transaction advisory lock (`pg_advisory_xact_lock`) on `odds_lambda.storage.event_resolver.event_lock_key` of (sport, home, away), held until the caller's transaction ends. Writers resolving the same fixture therefore run one after the other, and the second sees the event the first created. The key leaves out the date, so two sources placing one game on different UTC dates take the same lock. See [locks and transactions](#locks-and-transactions) for the order rule.
2. **MLB by `gamePk`.** The event holding the `gamePk` as an `mlb_stats_api` external id, else the one holding it in `events.mlb_game_pk`.
3. **By source id.** The event holding (`source`, `external_id`); for the Odds API, else the event keyed by the id, which an event stored before ids were recorded is. For MLB an event found this way counts only when it holds no `gamePk`, since the `gamePk` is the game's identity.
4. **Match.** For a live game, the nearest event of the same sport and teams within ±2 h (`odds_lambda.storage.event_merge.find_match_candidates`) of the game's kick-off: for MLB the schedule's start when it has one, else the listed start. Ties go to the event with the lowest `surrogate_id`, the one stored first. For MLB only events holding no `gamePk` count, and none matches when another game of the pair that day (`sibling_starts`) is scheduled nearer to the nearest event, so a doubleheader's second game never takes the first game's event.
5. **Create.** A SCHEDULED event at that kick-off with id `legacy_id`. When that id belongs to another event, the id with the `gamePk` (MLB) or the source id appended after `_` is used and `event_id_collision` is logged. Each id is checked before the insert, so when every choice is taken `odds_lambda.storage.event_resolver.EventIdTaken` is raised with nothing written and the caller's transaction intact.

Then it records the ids and moves the kick-off:

- The source's id, when the game has one.
- For MLB, the `gamePk` as an `mlb_stats_api` id; an event recording it that has no `mlb_game_pk` also takes it there (`odds_core.models.Event.assign_game_pk`).
- `odds_lambda.storage.event_kickoff.move_kickoff` once, with the current time as the clock (the MLB Stats API and football-data.co.uk may move an event to a past start): for MLB with the MLB Stats API and the schedule's start; for other sports with the source and its listed start. `odds_lambda.storage.event_kickoff.move_kickoff` alone decides whether it moves: only a sport's schedule authority moves the event, and an in-play start after the stored kick-off is refused, and football-data.co.uk moves only an event no other source has a snapshot on ([Kick-off authority](kickoff-authority.md)).

It flushes and never commits ([Commit contract](storage-layer.md#commit-contract)).

A historical game (`odds_lambda.storage.event_resolver.MatchWindow.HISTORICAL`) takes the candidates within ±24 h (`odds_lambda.storage.event_merge.find_historical_candidates`) instead: a lone candidate matches whatever its distance; with several, the nearest within ±2 h matches (ties to the lowest `surrogate_id`), and when none is within ±2 h the row is ambiguous, logged as `ambiguous_historical_match`, and creates its own event. Neighbouring fixtures of the same teams (an NBA back-to-back) are therefore never conflated. An OddsPortal historical listing cannot move a played event, because `odds_lambda.storage.event_kickoff.move_kickoff` refuses a past start from a source outside `odds_lambda.storage.event_kickoff.PLAYED_START_SOURCES`; football-data.co.uk is in it and moves an event only when no higher-ranked source holds an id on it.

## Attaching without creating

`odds_lambda.storage.event_resolver.attach_event(session, game)` is for a source that creates no events (Polymarket, through `odds_lambda.polymarket_odds.listed_game`). It takes the fixture lock and looks the game up as `odds_lambda.storage.event_resolver.resolve_event` does (steps 1 to 4), then records the source's id on the event it found. It returns `None`, creating nothing, when no event matches and when the id is [refused](#refused-ids): the caller writes nothing for that game. A refusal is logged once per event and description, not on every run. It does not offer the listed start to `odds_lambda.storage.event_kickoff.move_kickoff`, since such a source is not a schedule authority. `odds_lambda.storage.event_resolver.attach_events` attaches a batch in lock-key order and answers in input order, as `odds_lambda.storage.event_resolver.resolve_events` does. A game needs `legacy_id` (`odds_lambda.storage.event_resolver.ListedGame`) only for `odds_lambda.storage.event_resolver.resolve_event` to create an event; without one that raises `ValueError`. Both flush and never commit, and the [lock order](#locks-and-transactions) applies.

## MLB without the schedule

`odds_lambda.storage.event_resolver.resolve_unscheduled_mlb_event` is the lookup for an MLB game listed while the MLB schedule is unavailable. Under the same fixture lock it matches only the one event of the pair within ±2 h, and only for a row with no doubleheader game number, since without the schedule either game could be another. It records the source's id on that event, never creates one and moves no kick-off; with no match it returns `None`.

## No game in the schedule

`odds_lambda.storage.event_unmatched.mark_unmatched(session, event, reason)` records that an MLB event has no game in MLB's schedule. It sets the status to UNMATCHED and stores `reason` as the event's `missing_mlb_game_pk` acknowledgement (`odds_core.models.HealthCheckAcknowledgement`, [Events table](events.md#health-check-acknowledgements)), replacing an earlier reason. An event whose status cannot become UNMATCHED (`odds_core.models.EventStatus.can_become`, a FINAL one) keeps its status and still gets the acknowledgement. It raises `ValueError` for an empty reason, a non-MLB event or one holding a `gamePk`. `odds_lambda.storage.game_pk_backfill.mark_no_game_pk` (`odds events mark-unassignable`) is its caller and also writes `events.no_game_pk_reason`, the column settlement and the gamePk planning read. `odds_lambda.storage.event_unmatched.clear_unmatched` undoes it when the event gets a `gamePk`, from the resolver or from `odds_lambda.storage.game_pk_backfill.apply_assignment`: the acknowledgement is deleted and an UNMATCHED event becomes SCHEDULED ([Event status lifecycle](../collection/event-status-lifecycle.md)).

## Locks and transactions

The fixture lock is released only when the caller's transaction commits or rolls back. Two transactions that each hold one fixture lock and wait for the other's would deadlock, so within one transaction `odds_lambda.storage.event_resolver.resolve_event` takes fixture locks in ascending key order: a new lock below one the transaction already holds raises `odds_lambda.storage.event_resolver.LockOrderError` before it is requested. Re-taking a held lock is allowed. A caller resolving several fixtures in one transaction passes them to `odds_lambda.storage.event_resolver.resolve_events`, which resolves them in key order and returns the results in input order. A caller that commits after each fixture can use `odds_lambda.storage.event_resolver.resolve_event` in any order.

## Refused ids

An id is recorded only when neither the event nor the id is already taken. The resolver refuses it, logging `external_id_refused` and adding a `data_quality_logs` row through `odds_lambda.storage.data_quality.record_issue`, when:

- the event already holds a different id from that source (except the Odds API, below), or
- another event holds the id.

The Odds API reissues ids, so a different `odds_api` id for an event replaces the stored one: the old row is deleted and `external_id_replaced` is logged with `old_external_id` in the `data_quality_logs` row.

The resolved event is returned either way. The table's unique keys enforce the same rules for any other writer ([Events table](events.md#external-ids)).

## Related

- [Events table](events.md)
- [Event matching across sources](event-matching.md)
- [Kick-off authority](kickoff-authority.md)
