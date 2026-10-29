---
status: accepted
---

# Key events by a surrogate id, hold source ids in a table, and keep one event per match on write

## Context

Before the redesign in #658, an event's id was the id of the source that first stored it ([events table](../../wiki/storage/events.md#legacy-string-ids)):

- The Odds API events carried the API's own id.
- OddsPortal and football-data.co.uk events carried ids built from teams and kick-off (`op_`, `op_live_`, `fduk_`), which the sources never issued.
- `mlb_game_pk` was the only column holding a source's id for an event.

Each ingest path had its own matcher, with its own window and ambiguity rule, and some created an event when theirs found nothing ([event matching](../../wiki/storage/event-matching.md)). One match could therefore become several events. A read-time collapse picked one of them by ranking the sources an id prefix implied (`odds_core.event_source.canonical_event_id`), and twins were merged after the fact ([event merge](../../wiki/storage/event-merge.md)).

MLB and NBA are to be built on the same representation as EPL, so the representation is settled before that work starts (#658). The decision was taken in #660 and amended after the design review of #658.

## Decision

- **Keys.** An event has a surrogate bigint key. `event_external_ids` holds (event, source, external id), unique on (source, external id) and on (event, source): one id per source per event. A second id from one source for an event is refused and logged.
- **Only issued ids.** The table holds only ids a source issued: the Odds API, OddsPortal, Betfair, Polymarket, MLB `gamePk` and ESPN ids. Synthesized ids are not kept. Football-data.co.uk issues none, so its rows match by teams and date.
- **One resolver.** Every writer resolves a game to its event through one function. It takes a transaction advisory lock on (sport, home, away), looks the game up by its source's id, then matches sport, teams and kick-off within ±2 h. MLB matches by `gamePk`, which also tells doubleheader games apart. It records the source's id on the match and creates an event only when nothing matches. Any price source may create an event.
- **One event per match on write.** The read-time collapse and twin merging are deleted once existing twins are merged.
- **Kick-off.** Only a sport's schedule authority moves `commence_time`, through one function. For MLB that is the MLB Stats API. The EPL and NBA order is set after measuring how often sources disagree (#676).
- **No game in MLB's schedule** is an event status, `unmatched`, with its reason in a health-check acknowledgement record rather than a column on `events`.

The lock key leaves out the date. Evening games in the Americas straddle midnight UTC, and Betfair's start can lead `commence_time` by hours, so two writers could take different date keys for one game and each create an event (design review of #658, item 10).

## Consequences

- An event's identity no longer depends on which source stored it first, and joining a source's rows to an event is a lookup by that source's id.
- The schema is expanded first (#671), with the resolver and the external-id table beside the string ids. Every foreign key to `events.id` is re-keyed to the surrogate key, and the string ids are dropped, in the contract step (#677).
- Code that reads the source from an event id prefix stops working after the contract step and has to be ported first (design review of #658, item 8).

## Evidence

- Decision: #660 and its amendment; map #658.
- Design review of #658: `docs/research/design-review-658.md` on the `research/snapshot-payload-inventory` branch, items 8, 9 and 10.
