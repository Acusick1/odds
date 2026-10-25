---
type: reference
title: Events table
description: The events columns and statuses, id prefixes by source, team-name normalizers, and the upsert verbs that write events.
sources:
  - odds_core.models.Event
  - odds_core.models.EventStatus
  - odds_core.sports.SportKey
  - odds_core.team.normalize_team_name
  - odds_lambda.polymarket_matching.normalize_team
  - odds_lambda.polymarket_matching.TEAM_ALIASES
  - odds_lambda.storage.writers.OddsWriter.upsert_event
  - odds_lambda.storage.writers.OddsWriter.update_event_status
  - odds_lambda.storage.writers.OddsWriter.bulk_upsert_events
  - odds_lambda.storage.event_merge.EventMerger.adopt_twins
  - odds_lambda.storage.writers.OddsWriter._build_event_id
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
- `odds_core.models.Event.status`: `odds_core.models.EventStatus` = scheduled, live, final, cancelled, postponed. Which job sets which status is on [Event status lifecycle](../collection/event-status-lifecycle.md). `odds_core.models.EventStatus.can_become` defines the allowed moves, FINAL being terminal.
- `home_score`, `away_score`, `completed_at`: set only on completion (`odds_lambda.storage.writers.OddsWriter.update_event_status`).
- `created_at`, `updated_at`.
- `odds_core.models.Event.mlb_game_pk` (BIGINT, nullable): the MLB Stats API `gamePk` of an MLB event, unique where set (partial index `uq_events_mlb_game_pk`). The live OddsPortal and Betfair jobs set it when they create or match an MLB event, and a merge carries it to the kept event; events created before then hold NULL until `odds events assign-game-pks` assigns them ([Event merge](event-merge.md#assigning-gamepks)). `odds_core.models.Event.assign_game_pk` sets it and clears `no_game_pk_reason`.
- `odds_core.models.Event.no_game_pk_reason` (TEXT, nullable): why an MLB event has no `gamePk`, set by `odds events mark-unassignable`. A check constraint (`ck_events_game_pk_or_no_game_pk_reason`) keeps it NULL while `mlb_game_pk` is set.

`odds_core.models.Event.model_post_init` runs `odds_core.team.normalize_team_name` on both team names at construction, so ORM-built events hold canonical aliases. `sport_key` is a plain string column; `odds_core.sports.SportKey` (`soccer_epl`, `baseball_mlb`, `basketball_nba`) is a typing `Literal` used by jobs and CLI, not a DB constraint ([Sport, market and bookmaker keys](../collection/sport-and-market-keys.md)).

## Id prefix identifies the source

| Source | Pattern | Built by |
|---|---|---|
| The Odds API | hex id from the API; canonical, because score and status jobs match on it | API response |
| OddsPortal historical ingest | `op_{season}_{HOME}_{AWAY}_{date}` | `scripts/ingest_oddsportal.py` |
| OddsPortal live scrape | `op_live_{HOME}_{AWAY}_{YYYY-MM-DDTHHMM}` | `odds_lambda.storage.writers.OddsWriter._build_event_id` (prefix `odds_core.event_ids.LIVE_EVENT_ID_PREFIX`) |
| football-data.co.uk | `fduk_{season}_{HOME}_{AWAY}_{date}` (UTC kick-off date; a match found within 24 h in the same sport keeps its event, and `op_` events keep their own `commence_time`) | `scripts/ingest_football_data_uk.py` |

Training's `data_source` filter matches these prefixes (`op_%`, `fduk_%`; `oddsapi` keeps ids that match neither, `odds_core.event_ids.is_odds_api_event_id`) when `odds_analytics.training.data_preparation.prepare_training_data_from_config` calls `odds_analytics.training.data_preparation.filter_events_by_date_range` to select the events it trains on (FINAL ones, or every status for a `retail_ev` target, which then goes to `odds_analytics.retail_ev.prepare_retail_ev_training_data` (which passes its loss weights, scaled by the fixed `odds_analytics.retail_ev.LOSS_WEIGHT_SCALE`, to the model unchanged) through `odds_analytics.feature_groups.prepare_training_data`, which reads every snapshot attached to the event, whatever its source, Polymarket included; [Training config](../modelling/training-config.md)). FDUK rows that match an existing `op_` event get no new event of their own and attach their snapshots, which carry decimal prices ([Odds snapshot and odds tables](odds-tables.md)), to it ([Historical ingest scripts](../collection/historical-ingest-scripts.md)).

## Team-name normalizers

1. `odds_core.team.normalize_team_name`: whitespace collapse plus a lowercase alias map to the pipeline's canonical short names (EPL and MLB). Unknown names pass through unchanged.
2. `odds_lambda.polymarket_matching.normalize_team`: NBA only (`odds_lambda.polymarket_matching.TEAM_ALIASES`), returns `None` for unknown names. The injury and game-log writers and injury features use it, not the core one.

The two share no data.

## Upsert verbs

- `odds_lambda.storage.writers.OddsWriter.upsert_event`: select-then-update. On an existing row it overwrites sport, teams, `commence_time` and `updated_at` only; status and scores are untouched.
- `odds_lambda.storage.writers.OddsWriter.update_event_status`: raises `ValueError` for `FINAL` unless both scores are given, and for a move `odds_core.models.EventStatus.can_become` refuses (a FINAL event never changes status), so a FINAL event written through it always has scores; it then sets the scores and `completed_at`. Scores passed with any other status are ignored.
- `odds_lambda.storage.writers.OddsWriter.bulk_upsert_events`: dedupes by id, then `ON CONFLICT (id) DO UPDATE` over every column except the key, `created_at`, `mlb_game_pk`, `no_game_pk_reason`, status, the scores and `completed_at`; returns approximate inserted/updated counts. It is used by `odds discover games` and by `odds_lambda.event_sync.EventSyncService.sync_sport`, i.e. every fetch-odds run, which then runs `odds_lambda.storage.event_merge.EventMerger.adopt_twins` ([Event merge](event-merge.md)). The incoming `odds_core.models.Event` is a discovery record (status `scheduled`, null scores), so a re-discovered event keeps its stored status and scores; the exceptions are a POSTPONED event whose incoming `commence_time` differs and a LIVE event whose incoming `commence_time` is after the `now` passed to the call, both of which become SCHEDULED.
- `odds_lambda.storage.writers.OddsWriter.find_or_create_event` creates `op_live_` events, MLB ones with their `mlb_game_pk`: [OddsWriter](odds-writer.md#find_or_create_event).

## Related

- [Event matching across sources](event-matching.md)
- [Odds snapshot and odds tables](odds-tables.md)
- [Timezones and time helpers](../foundations/timezones.md)
