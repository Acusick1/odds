---
type: concept
title: Event matching and team names
summary: How each source attaches rows to an events row (five matcher implementations with different windows and ambiguity policies) and the two team-name normalizers.
sources:
  - packages/odds-lambda/odds_lambda/oddsportal_common.py::find_existing_event
  - packages/odds-lambda/odds_lambda/storage/writers.py::find_or_create_event
  - packages/odds-lambda/odds_lambda/storage/injury_writer.py::_match_event
  - packages/odds-lambda/odds_lambda/storage/game_log_writer.py::_match_event
  - packages/odds-lambda/odds_lambda/polymarket_matching.py::match_polymarket_event
  - packages/odds-lambda/odds_lambda/polymarket_matching.py::normalize_team
  - packages/odds-core/odds_core/team.py::normalize_team_name
  - packages/odds-core/odds_core/models.py::Event
  - docs/INJURIES.md#event-matching
  - docs/POLYMARKET.md#event-matching
verified_at: 7887039
---

# Event matching and team names

There is no shared matcher. Each ingest path resolves "which `events` row is this?" with its own query, window and ambiguity rule. Exact string equality on team names is the common core, so name normalization decides most outcomes.

## Matchers

| Matcher | Used by | Window | Teams | Ambiguity |
|---|---|---|---|---|
| `find_existing_event` (`packages/odds-lambda/odds_lambda/oddsportal_common.py`) | OddsPortal ingest, `scripts/ingest_oddsportal.py` | commence time ±24h | exact home and away, no sport filter | logs, returns `None` |
| `OddsWriter.find_or_create_event` (`packages/odds-lambda/odds_lambda/storage/writers.py::find_or_create_event`) | live scrape, [OddsWriter](odds-writer.md) | ±2h | `normalize_team_name` applied first; sport filter | logs, returns the first candidate |
| `InjuryWriter._match_event` | [NBA injury reports](nba-injury-reports.md) | ET `game_date` 10:00 to next day 06:00, converted to UTC | team is home or away | `None` |
| `GameLogWriter._match_event` | [NBA game logs](nba-stats-data.md) | same ET window (a verbatim copy) | team is home or away | `None` |
| `match_polymarket_event` | `odds polymarket link` only | ticker date at 12:00 UTC ±24h | exact, via `NBA_ABBREV_MAP` | `None` |

If `find_or_create_event` finds nothing it creates an event, so a name or window mismatch there yields a duplicate event rather than a miss. The other matchers return `None` and leave `event_id` NULL. Injury reports are league-wide, so unmatched pairs are expected for teams not playing that day ([NBA injury reports](nba-injury-reports.md)). Polymarket linking is manual and is not called by the fetch job ([Polymarket pipeline](polymarket-pipeline.md)).

## Team-name normalizers

1. `packages/odds-core/odds_core/team.py::normalize_team_name`: whitespace collapse plus a lowercase alias map to the pipeline's canonical short names (EPL and MLB). Unknown names pass through unchanged. `Event.model_post_init` applies it to `home_team` and `away_team` on construction, so ORM-built events are always normalized.
2. `packages/odds-lambda/odds_lambda/polymarket_matching.py::normalize_team`: NBA only (`TEAM_ALIASES`), returns `None` for unknown names. The injury and game-log writers and injury features use it, not the core one.

The two share no data. Alias gaps are fixed in the map for the sport in question.

## Hazards

- `find_existing_event` compares names as stored and does not call `normalize_team_name`; the OddsPortal path depends on names already being canonical.
- Ambiguity policy differs: `find_or_create_event` picks the first of several candidates, the rest refuse.
- `docs/POLYMARKET.md#event-matching` says the window is "±24h date window"; code centres it on 12:00 UTC of the ticker date to cover US evening games that fall on the next UTC day.
- `match_polymarket_event(pm_start_date=...)` is accepted and ignored.
- Injury and game-log matching fall back to the raw team name when `normalize_team` returns `None` (the game-log writer tries its abbreviation map first), so an alias gap shows up as a silent NULL `event_id`.

## Related

- [Events table](table-events.md)
- [Timezone handling](timezone-handling.md)
- [Storage layer](storage-layer.md)
- [Odds ingestion flow](odds-ingestion-flow.md)
