---
type: reference
title: Events table
description: The events columns and statuses, id prefixes by source, team-name normalizers, and the writers that create and update events.
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
  - odds_lambda.live_events.live_event_id
  - odds_lambda.event_sync.EventSyncService.sync_sport
  - odds_analytics.training.data_preparation
  - scripts/ingest_oddsportal.py
  - scripts/ingest_football_data_uk.py
status: stable
---

# Events table

`odds_core.models.Event` (table `events`) is the parent of [odds_snapshots and odds](odds-tables.md) and of `data_quality_logs`, all via `event_id` foreign keys. There is no unique key on teams plus commence time, so duplicates are prevented only by application matching ([Event matching across sources](event-matching.md)) and detected by the check-health job ([Event merge](event-merge.md#detection)).

## Columns

- `id` (str PK), `sport_key`, `sport_title`, `commence_time` (tz-aware), `home_team`, `away_team`: `sport_key`, `commence_time` and both team columns are indexed.
- `odds_core.models.Event.surrogate_id` (BIGINT identity, unique by `uq_events_surrogate_id`): the database numbers each new event. Events stored before the column existed were numbered by `created_at`, then `id`, by `migrations/versions/9c822a7a3faf_expand_odds_data_model.py`. The [external ids](#external-ids) and [acknowledgements](#health-check-acknowledgements) reference it; every other table references `id` ([ADR-0018](../../docs/adr/0018-surrogate-event-key-with-external-ids.md)). `odds_lambda.storage.event_resolver.resolve_event` leaves it to the database on insert, and `odds copy from-prod` never writes it.
- `odds_core.models.Event.status`: `odds_core.models.EventStatus` = scheduled, live, final, cancelled, postponed, unmatched. Unmatched marks an MLB event with no game in MLB's schedule; `odds_lambda.storage.event_unmatched.mark_unmatched` sets it ([Event resolver](event-resolver.md#no-game-in-the-schedule)). Which job sets which status is on [Event status lifecycle](../collection/event-status-lifecycle.md). `odds_core.models.EventStatus.can_become` defines the allowed moves, FINAL being terminal.
- `home_score`, `away_score`, `completed_at`: set only on completion (`odds_lambda.storage.writers.OddsWriter.update_event_status`).
- `created_at`, `updated_at`.
- `odds_core.models.Event.mlb_game_pk` (BIGINT, nullable): a copy of the MLB Stats API `gamePk` the event holds as its `mlb_stats_api` [external id](#external-ids), unique where set (partial index `uq_events_mlb_game_pk`). `odds_core.models.Event.assign_game_pk` sets it whenever the id is recorded (`odds_lambda.storage.event_resolver.record_game_pk`) and a merge carries it to the kept event. No reader uses it: the `gamePk` is read from `event_external_ids` (`odds_lambda.storage.game_pks`).
- `odds_core.models.Event.no_game_pk_reason` (TEXT, nullable): written by nothing, and read only by the [shape backfill](shape-backfill.md), which marks each event carrying one UNMATCHED; the reason an MLB event has no game is its `missing_mlb_game_pk` [acknowledgement](#health-check-acknowledgements). `odds_core.models.Event.assign_game_pk` clears it, and a check constraint (`ck_events_game_pk_or_no_game_pk_reason`) keeps it NULL while `mlb_game_pk` is set.

`odds_core.models.Event.model_post_init` runs `odds_core.team.normalize_team_name` on both team names at construction, so ORM-built events hold canonical aliases. `sport_key` is a plain string column; `odds_core.sports.SportKey` (`soccer_epl`, `baseball_mlb`, `basketball_nba`) is a typing `Literal` used by jobs and CLI, not a DB constraint ([Sport, market and bookmaker keys](../collection/sport-and-market-keys.md)).

## External ids

`odds_core.models.EventExternalId` (table `event_external_ids`) holds the ids sources issued for an event: (`source`, `external_id`) is the primary key, so one id names one event, and `uq_event_external_ids_event_source` allows one id per source per event. `event_id` references `events.surrogate_id`. `source` is one of `odds_core.vocabulary.EXTERNAL_ID_SOURCES` (the Odds API, OddsPortal, Betfair, Polymarket, the MLB Stats API `gamePk` and ESPN; football-data.co.uk issues none). `odds_lambda.storage.event_resolver.resolve_event` writes it ([Event resolver](event-resolver.md)), called by the live OddsPortal and Betfair jobs and the historical OddsPortal ingest script (FDUK issues no id; the Polymarket writers record theirs through `odds_lambda.storage.event_resolver.attach_event` and `odds_lambda.storage.event_resolver.attach_events`, which create no event and refuse a second Polymarket id for an event), and `odds copy from-prod` copies it ([what it copies](../foundations/databases-and-environments.md#copying-production-data)). An MLB event's `gamePk` is its `mlb_stats_api` id, which every reader of the `gamePk` uses (`odds_lambda.storage.game_pks`, by event or by `gamePk`). A merge moves a duplicate's ids to the kept event, dropping and logging those from a source the kept event has an id from ([Event merge](event-merge.md#merging)).

## Health-check acknowledgements

`odds_core.models.HealthCheckAcknowledgement` (table `health_check_acknowledgements`) records an event a health check should stop reporting: `event_id` (references `events.surrogate_id`), `check_name` (`odds_core.vocabulary.HealthCheck`), `reason` and `acknowledged_at`, one row per event and check (`uq_health_check_acknowledgements_event_check`). `odds_lambda.storage.event_unmatched.mark_unmatched` writes the `missing_mlb_game_pk` acknowledgement of an MLB event with no game, and `odds_lambda.storage.event_unmatched.clear_unmatched` deletes it when the event gets a `gamePk`; the `missing_game_pk` warning and the gamePk planning read it (`odds_lambda.storage.event_unmatched.unmatched_reasons`), and settlement reads the UNMATCHED status ([no game in the schedule](event-resolver.md#no-game-in-the-schedule)). A merge drops and logs a duplicate's acknowledgement of a check the kept event is acknowledged for.

## Id prefix identifies the source

| Source | Pattern | Built by |
|---|---|---|
| The Odds API | hex id from the API, for an event an Odds API listing creates; the API's id is recorded as an `odds_api` external id on every event the listing matches, and the score jobs find an event by it | API response |
| OddsPortal historical ingest | `op_{season}_{HOME}_{AWAY}_{date}` (only when no event matches within 24 h) | `scripts/ingest_oddsportal.py` (its snapshots take their tier from `odds_lambda.tier_utils.snapshot_tiers`) |
| OddsPortal live scrape | `op_live_{HOME}_{AWAY}_{YYYY-MM-DDTHHMM}` | `odds_lambda.live_events.live_event_id` (prefix `odds_core.event_ids.LIVE_EVENT_ID_PREFIX`) |
| football-data.co.uk | `fduk_{season}_{HOME}_{AWAY}_{date}` (UTC kick-off date; a match found within 24 h in the same sport keeps its existing event, whose `commence_time` FDUK moves only when no higher-ranked source holds an id on it) | `scripts/ingest_football_data_uk.py` (its snapshots take their tier from `odds_lambda.tier_utils.snapshot_tiers`) |

Training's `data_source` filter matches these prefixes (`op_%`, `fduk_%`; `oddsapi` keeps ids that match neither, `odds_core.event_ids.is_odds_api_event_id`) when `odds_analytics.training.data_preparation.prepare_training_data_from_config` calls `odds_analytics.training.data_preparation.filter_events_by_date_range` to select the events it trains on (FINAL ones, or every status for a `retail_ev` target, which then goes to `odds_analytics.retail_ev.prepare_retail_ev_training_data` (which passes its loss weights, scaled by the fixed `odds_analytics.retail_ev.LOSS_WEIGHT_SCALE`, to the model unchanged) through `odds_analytics.feature_groups.prepare_training_data`, which reads every snapshot attached to the event, whatever its source, Polymarket included; [Training config](../modelling/training-config.md)). FDUK rows that match an existing `op_` event get no new event of their own and attach their snapshots, which carry decimal prices ([Odds snapshot and odds tables](odds-tables.md)), to it ([Historical ingest scripts](../collection/historical-ingest-scripts.md)).

## Team-name normalizers

1. `odds_core.team.normalize_team_name`: whitespace collapse plus a lowercase alias map to the pipeline's canonical short names (EPL and MLB). Unknown names pass through unchanged.
2. `odds_lambda.polymarket_matching.normalize_team`: NBA only (`odds_lambda.polymarket_matching.TEAM_ALIASES`), returns `None` for unknown names. The injury and game-log writers and injury features use it, not the core one.

The two share no data.

## Writing events

- `odds_lambda.storage.writers.OddsWriter.update_event_status`: raises `ValueError` for `FINAL` unless both scores are given, and for a move `odds_core.models.EventStatus.can_become` refuses (a FINAL event never changes status), so a FINAL event written through it always has scores; it then sets the scores and `completed_at`. Scores passed with any other status are ignored.
- `odds_lambda.odds_api_events.resolve_odds_api_events`, for `odds_lambda.event_sync.EventSyncService.sync_sport` (every fetch-odds run) and `odds discover games`: resolves each listed game through `odds_lambda.storage.event_resolver.resolve_event` ([Odds API events](event-matching.md#odds-api-events)), in fixture-lock order and one savepoint per game, and returns how many events were created and how many matched. A matched event keeps its sport, teams, status, scores, `mlb_game_pk` and key; only its Odds API id is recorded and its kick-off offered to `odds_lambda.storage.event_kickoff.move_kickoff` with the Odds API as the source ([Kick-off authority](kickoff-authority.md)), so a POSTPONED event that moves, or a LIVE one moved to a start after the clock, becomes SCHEDULED. An event the resolver creates is SCHEDULED with null scores and the API's id as its key.
- `odds_lambda.shape_backfill.events.merge_twins`, in the [shape backfill](shape-backfill.md#steps): merges stored twins and sets the kept event's kick-off through `odds_lambda.storage.event_kickoff.settle_merged_kickoff`.
- `odds_lambda.storage.event_resolver.resolve_event` creates the live jobs' `op_live_` events and the historical scripts' `op_` and `fduk_` ones (from the game's `legacy_id`), MLB ones with their `mlb_game_pk`, and records each source's id: [Event resolver](event-resolver.md).

## Related

- [Event matching across sources](event-matching.md)
- [Event resolver](event-resolver.md)
- [Odds snapshot and odds tables](odds-tables.md)
- [Timezones and time helpers](../foundations/timezones.md)
