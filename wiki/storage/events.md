---
type: reference
title: Events table
description: The events columns and statuses, the database-numbered event id, external ids, the legacy string ids kept for the downgrade, team-name normalizers, and the writers that create and update events.
sources:
  - odds_core.models.Event
  - odds_core.models.EventStatus
  - odds_core.models.EventExternalId
  - odds_core.models.HealthCheckAcknowledgement
  - odds_core.sports.SportKey
  - odds_core.team.normalize_team_name
  - odds_lambda.polymarket_matching.normalize_team
  - odds_lambda.polymarket_matching.TEAM_ALIASES
  - odds_lambda.storage.writers.OddsWriter.update_event_status
  - odds_lambda.odds_api_events.resolve_odds_api_events
  - odds_lambda.odds_api_events.find_event_by_odds_api_id
  - odds_lambda.event_sync.EventSyncService.sync_sport
  - odds_core.api_models.ApiEvent
  - migrations/versions/75c92251aa25_rekey_events_to_integer_id.py
status: stable
---

# Events table

`odds_core.models.Event` (table `events`) is the parent of [odds_snapshots](odds-tables.md), `data_quality_logs` and the other per-event tables, all via `event_id` foreign keys to its integer `id`. There is no unique key on teams plus commence time, so duplicates are prevented only by application matching ([Event matching across sources](event-matching.md)) and detected by the check-health job ([Event merge](event-merge.md#detection)).

## Columns

- `odds_core.models.Event.id` (BIGINT identity, primary key): the database numbers each new event, and every table that references an event references it ([ADR-0018](../../docs/adr/0018-surrogate-event-key-with-external-ids.md)). `odds_lambda.storage.event_resolver.resolve_event` leaves it to the database on insert. Ids are numbered per database, so `odds copy from-prod` gives a copied event the local id of the event it matches, or a new one ([what it copies](../foundations/databases-and-environments.md#copying-production-data)). The id says nothing about the source that stored the event: a source's own id for it is an [external id](#external-ids).
- `sport_key`, `sport_title`, `commence_time` (tz-aware), `home_team`, `away_team`: `sport_key`, `commence_time` and both team columns are indexed.
- `odds_core.models.Event.status`: `odds_core.models.EventStatus` = scheduled, live, final, cancelled, postponed, unmatched. Unmatched marks an MLB event with no game in MLB's schedule; `odds_lambda.storage.event_unmatched.mark_unmatched` sets it ([Event resolver](event-resolver.md#no-game-in-the-schedule)). Which job sets which status is on [Event status lifecycle](../collection/event-status-lifecycle.md). `odds_core.models.EventStatus.can_become` defines the allowed moves, FINAL being terminal.
- `home_score`, `away_score`, `completed_at`: set only on completion (`odds_lambda.storage.writers.OddsWriter.update_event_status`).
- `created_at`, `updated_at`.

`odds_core.models.Event.model_post_init` runs `odds_core.team.normalize_team_name` on both team names at construction, so ORM-built events hold canonical aliases. `sport_key` is a plain string column; `odds_core.sports.SportKey` (`soccer_epl`, `baseball_mlb`, `basketball_nba`) is a typing `Literal` used by jobs and CLI, not a DB constraint ([Sport, market and bookmaker keys](../collection/sport-and-market-keys.md)).

## External ids

`odds_core.models.EventExternalId` (table `event_external_ids`) holds the ids sources issued for an event: (`source`, `external_id`) is the primary key, so one id names one event, and `uq_event_external_ids_event_source` allows one id per source per event. `event_id` references `events.id`. `source` is one of `odds_core.vocabulary.EXTERNAL_ID_SOURCES` (the Odds API, OddsPortal, Betfair, Polymarket, the MLB Stats API `gamePk` and ESPN; football-data.co.uk issues none). `odds_lambda.storage.event_resolver.resolve_event` writes it ([Event resolver](event-resolver.md)), called by the live OddsPortal and Betfair jobs and the historical OddsPortal ingest script (FDUK issues no id; the Polymarket writers record theirs through `odds_lambda.storage.event_resolver.attach_event` and `odds_lambda.storage.event_resolver.attach_events`, which create no event and refuse a second Polymarket id for an event), and `odds copy from-prod` copies it ([what it copies](../foundations/databases-and-environments.md#copying-production-data)). An MLB event's `gamePk` is its `mlb_stats_api` id, recorded through `odds_lambda.storage.event_resolver.record_game_pk` and read through `odds_lambda.storage.game_pks`, by event or by `gamePk`; no events column holds it. Why an MLB event has no `gamePk` is its `missing_mlb_game_pk` [acknowledgement](#health-check-acknowledgements). A merge moves a duplicate's ids to the kept event, dropping and logging those from a source the kept event has an id from ([Event merge](event-merge.md#merging)).

## Health-check acknowledgements

`odds_core.models.HealthCheckAcknowledgement` (table `health_check_acknowledgements`) records an event a health check should stop reporting: `event_id` (references `events.id`), `check_name` (`odds_core.vocabulary.HealthCheck`), `reason` and `acknowledged_at`, one row per event and check (`uq_health_check_acknowledgements_event_check`). `odds_lambda.storage.event_unmatched.mark_unmatched` writes the `missing_mlb_game_pk` acknowledgement of an MLB event with no game, and `odds_lambda.storage.event_unmatched.clear_unmatched` deletes it when the event gets a `gamePk`; the `missing_game_pk` warning and the gamePk planning read it (`odds_lambda.storage.event_unmatched.unmatched_reasons`), and settlement reads the UNMATCHED status ([no game in the schedule](event-resolver.md#no-game-in-the-schedule)). A merge drops and logs a duplicate's acknowledgement of a check the kept event is acknowledged for.

## An event's source

No events column names a source. Training's `data_source` filter keeps events with a snapshot from the named source (`odds_snapshots.source`) when `odds_analytics.training.data_preparation.prepare_training_data_from_config` calls `odds_analytics.training.data_preparation.filter_events_by_date_range` to select the events it trains on (FINAL ones, or every status for a `retail_ev` target, which then goes to `odds_analytics.retail_ev.prepare_retail_ev_training_data` (which passes its loss weights, scaled by the fixed `odds_analytics.retail_ev.LOSS_WEIGHT_SCALE`, to the model unchanged) through `odds_analytics.feature_groups.prepare_training_data`, which reads the quotes of every snapshot attached to the event, whatever its source, Polymarket included, and measures a `raw` row, sampled only of the sources whose close has a sharp book, against its own source's close; [Training config](../modelling/training-config.md)).

## Legacy string ids

The table `legacy_event_ids` (`event_id`, `legacy_id`) holds the string key each event stored before `migrations/versions/75c92251aa25_rekey_events_to_integer_id.py` had: the Odds API's id for an event an Odds API listing created, else an id built from teams and kick-off (`op_`, `op_live_` or `fduk_`). It has no model and no foreign key, and only that migration's downgrade reads it, to restore the string keys. Every Odds API key in it is also its event's `odds_api` external id, where the resolver finds it; the built ids were never a source's, and nothing matches on them.

## Team-name normalizers

1. `odds_core.team.normalize_team_name`: whitespace collapse plus a lowercase alias map to the pipeline's canonical short names (EPL and MLB). Unknown names pass through unchanged.
2. `odds_lambda.polymarket_matching.normalize_team`: NBA only (`odds_lambda.polymarket_matching.TEAM_ALIASES`), returns `None` for unknown names. The injury and game-log writers and injury features use it, not the core one.

The two share no data.

## Writing events

- `odds_lambda.storage.writers.OddsWriter.update_event_status`: raises `ValueError` for `FINAL` unless both scores are given, and for a move `odds_core.models.EventStatus.can_become` refuses (a FINAL event never changes status), so a FINAL event written through it always has scores; it then sets the scores and `completed_at`. Scores passed with any other status are ignored.
- `odds_lambda.odds_api_events.resolve_odds_api_events`, for `odds_lambda.event_sync.EventSyncService.sync_sport` (every fetch-odds run) and `odds discover games`: resolves each game the API lists (`odds_core.api_models.ApiEvent`, which carries the API's id as `odds_api_id`) through `odds_lambda.storage.event_resolver.resolve_event` ([Odds API events](event-matching.md#odds-api-events)), in fixture-lock order and one savepoint per game, and returns how many events were created and how many matched. A matched event keeps its sport, teams, status, scores, `gamePk` and key; only its Odds API id is recorded and its kick-off offered to `odds_lambda.storage.event_kickoff.move_kickoff` with the Odds API as the source ([Kick-off authority](kickoff-authority.md)), so a POSTPONED event that moves, or a LIVE one moved to a start after the clock, becomes SCHEDULED. An event the resolver creates is SCHEDULED with null scores and the API's id as its `odds_api` external id.
- `odds_lambda.storage.event_resolver.resolve_event` creates the live jobs' events and the historical scripts' ones, MLB ones with their `gamePk` as an external id, and records each source's id: [Event resolver](event-resolver.md).

## Related

- [Event matching across sources](event-matching.md)
- [Event resolver](event-resolver.md)
- [Odds snapshots table](odds-tables.md)
- [Timezones and time helpers](../foundations/timezones.md)
