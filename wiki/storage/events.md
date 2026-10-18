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
  - odds_lambda.storage.writers.OddsWriter.adopt_live_twins
  - odds_lambda.storage.writers.OddsWriter._build_event_id
  - odds_lambda.event_sync.EventSyncService.sync_sport
  - odds_analytics.training.data_preparation
  - scripts/ingest_oddsportal.py
  - scripts/ingest_football_data_uk.py
status: stable
---

# Events table

`odds_core.models.Event` (table `events`) is the parent of [odds_snapshots and odds](odds-tables.md) and of `data_quality_logs`, all via `event_id` foreign keys. There is no unique key on teams plus commence time, so duplicates are prevented only by application matching ([Event matching across sources](event-matching.md)).

## Columns

- `id` (str PK), `sport_key`, `sport_title`, `commence_time` (tz-aware), `home_team`, `away_team`: `sport_key`, `commence_time` and both team columns are indexed.
- `odds_core.models.Event.status`: `odds_core.models.EventStatus` = scheduled, live, final, cancelled, postponed. Which job sets which status is on [Event status lifecycle](../collection/event-status-lifecycle.md). Only LIVE and FINAL are ever written; CANCELLED and POSTPONED are never set.
- `home_score`, `away_score`, `completed_at`: set only on completion (`odds_lambda.storage.writers.OddsWriter.update_event_status`).
- `created_at`, `updated_at`.

`odds_core.models.Event.model_post_init` runs `odds_core.team.normalize_team_name` on both team names at construction, so ORM-built events hold canonical aliases. `sport_key` is a plain string column; `odds_core.sports.SportKey` (`soccer_epl`, `baseball_mlb`, `basketball_nba`) is a typing `Literal` used by jobs and CLI, not a DB constraint ([Sport, market and bookmaker keys](../collection/sport-and-market-keys.md)).

## Id prefix identifies the source

| Source | Pattern | Built by |
|---|---|---|
| The Odds API | hex id from the API; canonical, because score and status jobs match on it | API response |
| OddsPortal historical ingest | `op_{season}_{HOME}_{AWAY}_{date}` | `scripts/ingest_oddsportal.py` |
| OddsPortal live scrape | `op_live_{HOME}_{AWAY}_{YYYY-MM-DDTHHMM}` | `odds_lambda.storage.writers.OddsWriter._build_event_id` (prefix `odds_core.event_ids.LIVE_EVENT_ID_PREFIX`) |
| football-data.co.uk | `fduk_{season}_{HOME}_{AWAY}_{date}` (UTC kick-off date; matched `op_` events keep their own `commence_time`) | `scripts/ingest_football_data_uk.py` |

Training's `data_source` filter matches these prefixes (`op_%`, `fduk_%`; `oddsapi` keeps ids that match neither, `odds_core.event_ids.is_odds_api_event_id`) when `odds_analytics.training.data_preparation.prepare_training_data_from_config` calls `odds_analytics.training.data_preparation.filter_events_by_date_range` to select events for training ([Training config](../modelling/training-config.md)). FDUK rows that match an existing `op_` event get no new event of their own and attach their snapshots, which carry decimal prices ([Odds snapshot and odds tables](odds-tables.md)), to it ([Historical ingest scripts](../collection/historical-ingest-scripts.md)).

## Team-name normalizers

1. `odds_core.team.normalize_team_name`: whitespace collapse plus a lowercase alias map to the pipeline's canonical short names (EPL and MLB). Unknown names pass through unchanged.
2. `odds_lambda.polymarket_matching.normalize_team`: NBA only (`odds_lambda.polymarket_matching.TEAM_ALIASES`), returns `None` for unknown names. The injury and game-log writers and injury features use it, not the core one.

The two share no data.

## Upsert verbs

- `odds_lambda.storage.writers.OddsWriter.upsert_event`: select-then-update. On an existing row it overwrites sport, teams, `commence_time` and `updated_at` only; status and scores are untouched.
- `odds_lambda.storage.writers.OddsWriter.update_event_status`: sets scores and `completed_at` only when status is `FINAL` and both scores are given; scores passed with any other status are ignored.
- `odds_lambda.storage.writers.OddsWriter.bulk_upsert_events`: dedupes by id, then `ON CONFLICT (id) DO UPDATE` over every column except the key and `created_at`; returns approximate inserted/updated counts. It is used by `odds discover games` and by `odds_lambda.event_sync.EventSyncService.sync_sport`, i.e. every fetch-odds run, which then runs `odds_lambda.storage.writers.OddsWriter.adopt_live_twins` ([OddsWriter](odds-writer.md#adopt_live_twins)). The incoming `odds_core.models.Event` has status `scheduled` and null scores, so a re-discovered FINAL event returns to SCHEDULED with null scores (#440).
- `odds_lambda.storage.writers.OddsWriter.find_or_create_event` creates `op_live_` events: [OddsWriter](odds-writer.md#find_or_create_event).

## Related

- [Event matching across sources](event-matching.md)
- [Odds snapshot and odds tables](odds-tables.md)
- [Timezones and time helpers](../foundations/timezones.md)
