---
type: component
title: Events table and event matching
description: "The events row and its id-prefix provenance, the five source-specific matchers and two team-name normalizers that decide which event a row attaches to, and why duplicates or NULL event ids appear."
sources:
  - packages/odds-core/odds_core/models.py::Event
  - packages/odds-core/odds_core/models.py::EventStatus
  - packages/odds-core/odds_core/sports.py::SportKey
  - packages/odds-core/odds_core/team.py::normalize_team_name
  - packages/odds-lambda/odds_lambda/oddsportal_common.py::find_existing_event
  - packages/odds-lambda/odds_lambda/storage/writers.py::OddsWriter.find_or_create_event
  - packages/odds-lambda/odds_lambda/storage/injury_writer.py::InjuryWriter._match_event
  - packages/odds-lambda/odds_lambda/storage/game_log_writer.py::GameLogWriter._match_event
  - packages/odds-lambda/odds_lambda/polymarket_matching.py::match_polymarket_event
  - packages/odds-lambda/odds_lambda/polymarket_matching.py::normalize_team
  - packages/odds-lambda/odds_lambda/event_sync.py::EventSyncService
  - packages/odds-analytics/odds_analytics/training/data_preparation.py
status: stable
---

# Events table and event matching

`packages/odds-core/odds_core/models.py::Event` (table `events`) is the parent of [odds_snapshots and odds](odds-storage.md) and of `data_quality_logs`, all via `event_id` foreign keys. There is no shared matcher for deciding which `events` row a piece of incoming data belongs to: each ingest path has its own query, window and ambiguity rule, and exact string equality on team names is the common core.

## Columns

- `id` (str PK), `sport_key`, `sport_title`, `commence_time` (tz-aware), `home_team`, `away_team`: `sport_key`, `commence_time` and both team columns are indexed.
- `packages/odds-core/odds_core/models.py::Event.status`: `packages/odds-core/odds_core/models.py::EventStatus` = scheduled, live, final, cancelled, postponed. Which job sets which status is on [event-lifecycle](../collection/event-lifecycle.md). Only LIVE and FINAL are ever written; CANCELLED and POSTPONED are never set.
- `home_score`, `away_score`, `completed_at`: set only on completion (`packages/odds-lambda/odds_lambda/storage/writers.py::OddsWriter.update_event_status`, see [odds-storage](odds-storage.md#events-written-by-oddswriter)).
- `created_at`, `updated_at`.

`packages/odds-core/odds_core/models.py::Event.model_post_init` runs `packages/odds-core/odds_core/team.py::normalize_team_name` on both team names at construction, so ORM-built events hold canonical aliases (`packages/odds-core/odds_core/team.py::normalize_team_name`). `sport_key` is a plain string column; `SportKey` (`soccer_epl`, `baseball_mlb`, `basketball_nba`) is a typing `Literal` used by jobs and CLI, not a DB constraint ([sources-and-keys](../collection/sources-and-keys.md)).

## Id prefix identifies the source

| Source | Pattern | Built by |
|---|---|---|
| The Odds API | hex id from the API | API response |
| OddsPortal historical ingest | `op_{season}_{HOME}_{AWAY}_{date}` | `scripts/ingest_oddsportal.py` |
| OddsPortal live scrape | `op_live_{HOME}_{AWAY}_{YYYY-MM-DDTHHMM}` | `packages/odds-lambda/odds_lambda/storage/writers.py::OddsWriter._build_event_id` |
| football-data.co.uk | `fduk_{season}_{HOME}_{AWAY}_{date}` | `scripts/ingest_football_data_uk.py` |

Training's `data_source` filter matches these prefixes (`op_%`, `fduk_%`) in `packages/odds-analytics/odds_analytics/training/data_preparation.py` ([training-pipeline](../modelling/training-pipeline.md)). FDUK rows that match an existing `op_` event get no new event of their own ([howto-load-historical-odds](../collection/howto-load-historical-odds.md)).

## Matchers

| Matcher | Used by | Window | Teams | Ambiguity |
|---|---|---|---|---|
| `packages/odds-lambda/odds_lambda/oddsportal_common.py::find_existing_event` (`packages/odds-lambda/odds_lambda/oddsportal_common.py`) | historical OddsPortal and FDUK ingest scripts | commence time ±24 h | exact home and away as stored, no sport filter | logs, returns `None` |
| `packages/odds-lambda/odds_lambda/storage/writers.py::OddsWriter.find_or_create_event` | live OddsPortal scrape and Betfair jobs | ±2 h | `packages/odds-core/odds_core/team.py::normalize_team_name` applied first; sport filter | logs, returns the first candidate |
| `packages/odds-lambda/odds_lambda/storage/injury_writer.py::InjuryWriter._match_event` | [NBA injury reports](../collection/nba-injuries-and-stats.md) | ET `game_date` 10:00 to next day 06:00, converted to UTC | team is home or away | `None` |
| `packages/odds-lambda/odds_lambda/storage/game_log_writer.py::GameLogWriter._match_event` | [NBA game logs](../collection/nba-injuries-and-stats.md) | same ET window (a verbatim copy) | team is home or away | `None` |
| `packages/odds-lambda/odds_lambda/polymarket_matching.py::match_polymarket_event` | `odds polymarket link` only | ticker date at 12:00 UTC ±24 h | exact, via `packages/odds-lambda/odds_lambda/polymarket_matching.py::NBA_ABBREV_MAP` | `None` |

If `packages/odds-lambda/odds_lambda/storage/writers.py::OddsWriter.find_or_create_event` finds nothing it creates an event, so a name or window mismatch there yields a duplicate event rather than a miss. The other matchers return `None` and leave `event_id` NULL. Injury reports are league-wide, so unmatched pairs are expected for teams not playing that day. Unmatched NBA rows keep `event_id` NULL and are invisible to per-event readers. Polymarket linking is manual and is not called by the fetch job ([polymarket](../collection/polymarket.md)).

### Team-name normalizers

1. `packages/odds-core/odds_core/team.py::normalize_team_name`: whitespace collapse plus a lowercase alias map to the pipeline's canonical short names (EPL and MLB). Unknown names pass through unchanged.
2. `packages/odds-lambda/odds_lambda/polymarket_matching.py::normalize_team`: NBA only (`packages/odds-lambda/odds_lambda/polymarket_matching.py::TEAM_ALIASES`), returns `None` for unknown names. The injury and game-log writers and injury features use it, not the core one.

The two share no data. Alias gaps are fixed in the map for the sport in question.

## Hazards

- **Re-discovery resets status.** `bulk_upsert_events` (used by `odds discover games` and by `packages/odds-lambda/odds_lambda/event_sync.py::EventSyncService.sync_sport`, i.e. every fetch-odds run) overwrites every non-key column except `created_at` with the incoming `packages/odds-core/odds_core/models.py::Event`, whose defaults are status `scheduled` and null scores. A FINAL event that is re-discovered goes back to SCHEDULED. `upsert_event` leaves status and scores alone on update.
- **Matcher windows differ.** `packages/odds-lambda/odds_lambda/storage/writers.py::OddsWriter.find_or_create_event` uses ±2 h, `packages/odds-lambda/odds_lambda/oddsportal_common.py::find_existing_event` ±24 h; both reuse an existing event (including a hex-id one) on a team match, and the latter takes no sport filter. A game absent from `events` gets an `op_`/`fduk_` row, so cross-source joins depend on the matchers agreeing.
- **Ambiguity policy differs.** `packages/odds-lambda/odds_lambda/storage/writers.py::OddsWriter.find_or_create_event` picks the first of several candidates; the rest refuse.
- **`packages/odds-lambda/odds_lambda/oddsportal_common.py::find_existing_event` does not normalize.** It compares names as stored, so the OddsPortal path depends on names already being canonical.
- **Live id embeds the start minute.** A rescheduled kickoff creates a new `op_live_` id unless the 2 h window matches an existing event.
- **No unique key on teams plus time.** Duplicates are prevented only by application matching.
- **Polymarket window.** `packages/odds-lambda/odds_lambda/polymarket_matching.py::match_polymarket_event` centres the window on 12:00 UTC of the ticker date to cover US evening games that fall on the next UTC day. `pm_start_date` is accepted and ignored.
- **Silent NULLs.** Injury and game-log matching fall back to the raw team name when `packages/odds-lambda/odds_lambda/polymarket_matching.py::normalize_team` returns `None` (the game-log writer tries its abbreviation map first), so an alias gap shows up as a NULL `event_id`.

## Related

- [Odds snapshots, odds rows and OddsWriter](odds-storage.md)
- [Odds API ingestion](../collection/odds-api-ingestion.md)
- [Timezones](../foundations/conventions.md#timezones)
- [OddsReader and the storage layer](odds-reader-and-storage-layer.md)
