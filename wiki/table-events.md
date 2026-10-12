---
type: component
title: events table
summary: Event rows (sport_key, status, scores) and the id-prefix convention that encodes each row's source.
sources:
  - packages/odds-core/odds_core/models.py::Event
  - packages/odds-core/odds_core/models.py::EventStatus
  - packages/odds-core/odds_core/sports.py::SportKey
  - docs/DATABASE.md#core-tables
  - docs/DATA_MODELS.md#sportsbook-odds
verified_at: 7887039
---

# events table

`packages/odds-core/odds_core/models.py::Event` (table `events`) is the parent of [odds_snapshots](table-odds-snapshots.md), [odds](table-odds.md) and `data_quality_logs`, all via `event_id` foreign keys.

## Columns

- `id` (str PK), `sport_key`, `sport_title`, `commence_time` (tz-aware), `home_team`, `away_team`: `sport_key`, `commence_time` and both team columns are indexed.
- `status`: `EventStatus` = scheduled, live, final, cancelled, postponed (`packages/odds-core/odds_core/models.py::EventStatus`); transitions in [event lifecycle](event-lifecycle.md).
- `home_score`, `away_score`, `completed_at`: set only on completion (`OddsWriter.update_event_status`, see [OddsWriter](odds-writer.md)).
- `created_at`, `updated_at`.

`Event.model_post_init` runs `normalize_team_name` on both team names at construction, so stored names are canonical aliases (`packages/odds-core/odds_core/team.py::normalize_team_name`). Name matching elsewhere (e.g. `find_or_create_event`) relies on this.

`sport_key` is a plain string column. `SportKey` (`soccer_epl`, `baseball_mlb`, `basketball_nba`) is a typing `Literal` used by jobs and CLI, not a DB constraint; see [sport and market keys](sport-and-market-keys.md).

## Id prefix identifies the source

| Source | Pattern | Built by |
|---|---|---|
| The Odds API | hex id from the API | API response |
| OddsPortal historical ingest | `op_{season}_{HOME}_{AWAY}_{date}` | `scripts/ingest_oddsportal.py` |
| OddsPortal live scrape | `op_live_{HOME}_{AWAY}_{YYYY-MM-DDTHHMM}` | `OddsWriter._build_event_id` |
| football-data.co.uk | `fduk_{season}_{HOME}_{AWAY}_{date}` | `scripts/ingest_football_data_uk.py` |

Training's `data_source` filter matches these prefixes (`op_%`, `fduk_%`) in `packages/odds-analytics/odds_analytics/training/data_preparation.py`. See [data sources](data-sources.md) and [historical odds ingestion](historical-odds-ingestion.md).

## Hazards

- **Three matchers, three windows.** `OddsWriter.find_or_create_event` matches within plus/minus 2 h, the historical scripts' `packages/odds-lambda/odds_lambda/oddsportal_common.py::find_existing_event` within plus/minus 24 h; both reuse an existing event (including a hex-id one) on a single team match and take no sport filter in the latter. A game absent from `events` gets an `op_`/`fduk_` row, so cross-source joins depend on these matchers agreeing.
- **Live id embeds the start minute.** A rescheduled kickoff creates a new `op_live_` id unless the 2 h window matches an existing event.
- **Upserts differ.** `upsert_event` leaves status and scores alone on update; `bulk_upsert_events` (used by `packages/odds-lambda/odds_lambda/event_sync.py::EventSyncService.sync_sport`) overwrites every non-key column except `created_at` with the incoming `Event`, whose defaults are status `scheduled` and null scores.
- **No unique key on teams plus time.** Duplicates are prevented only by application matching.

## Related

- [Odds ingestion flow](odds-ingestion-flow.md)
- [Timezone handling](timezone-handling.md)
