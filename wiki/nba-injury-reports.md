---
type: component
title: NBA injury reports and features
summary: Official NBA injury PDFs into nba_injury_reports - event matching, event-driven backfill and the six injury features, which carry no CLV value.
sources:
  - packages/odds-core/odds_core/injury_models.py::InjuryReport
  - packages/odds-lambda/odds_lambda/injury_fetcher.py::fetch_injury_report
  - packages/odds-lambda/odds_lambda/storage/injury_writer.py::InjuryWriter
  - packages/odds-lambda/odds_lambda/storage/injury_reader.py::InjuryReader
  - packages/odds-analytics/odds_analytics/injury_features.py::extract_injury_features
  - packages/odds-cli/odds_cli/commands/injuries.py
  - docs/INJURIES.md
  - docs/CLI.md#injury-reports
  - docs/DATA_MODELS.md#feature-groups
verified_at: 7887039
---

# NBA injury reports and features

League-wide injury report snapshots parsed from the official NBA PDFs by the `nbainjuries` package, stored in `nba_injury_reports` (`InjuryReport`, `InjuryStatus` in `packages/odds-core/odds_core/injury_models.py`). The deployed model does not use them: injury features add no CLV signal once tuned equally ([NBA injury features experiment](exp-nba-injury-features.md)).

## Ingestion

- `injury_fetcher.py::fetch_injury_report(target_time_utc)` is synchronous. It converts to naive ET, checks `check_reportvalid`, parses via `_parse_records`, and drops "NOT YET SUBMITTED" rows and unknown statuses. `nbainjuries` is imported lazily because it starts a JVM (tabula via jpype).
- `InjuryWriter.upsert_injury_reports` matches each `(team, game_date)` to an `events` row and upserts on `uq_injury_report_time_team_player_date` (`report_time, team, player_name, game_date`).
- Match rule (`InjuryWriter._match_event`): events whose `commence_time` is in game_date 10:00 ET to next day 06:00 ET with that team as home or away; exactly one candidate sets `event_id`, otherwise NULL (ambiguous logs a warning). Teams are canonicalised with `polymarket_matching.normalize_team`. See [Event matching](event-matching.md).
- `odds injuries fetch|backfill|status` (`packages/odds-cli/odds_cli/commands/injuries.py`). Backfill is event-driven: for NBA events in the season window it subtracts each `--hours-before` offset (default `8,2`), rounds down to the report slot (hourly before 2025-12-22 09:00 ET, quarter-hourly after, `_NEW_ERA_CUTOFF_ET`), dedupes, and sleeps `--delay-ms` (default 500). No scheduled job runs it; ingestion is manual CLI only.

## Features

`extract_injury_features(reports, event, snapshot_time, player_stats)` takes reports with `report_time <= snapshot_time` (look-ahead guard, [Look-ahead bias](look-ahead-bias.md)), uses only the latest snapshot, and returns `InjuryFeatures` (6 fields: `impact_out_home/away`, `impact_gtd_home/away`, `report_hours_before_game`, `injury_news_recency`). Per-player impact is `(on_off_rtg - on_def_rtg) * (minutes / games_played / 48)` from [NBA player stats](nba-stats-data.md), falling back to 1.0 when stats, ratings or games are missing. QUESTIONABLE and DOUBTFUL count at 0.5x; PROBABLE and AVAILABLE are ignored. Enabled with `feature_groups` containing `injuries`; `collect_event_data` in `packages/odds-analytics/odds_analytics/feature_groups.py` loads reports and stats, and events without reports get NaN, not dropped ([Feature groups](feature-groups.md)).

## Hazards

- Impact lookup is by exact `player_name` string. Stats names are converted to "Last, First" by `pbpstats_fetcher.convert_name`; any mismatch with the report's name silently becomes 1.0 headcount.
- Backfill and the stats lookup are NBA-only (`Event.sport_key == "basketball_nba"` in the CLI; `CANONICAL_TO_ABBREV` in the loader). No other sport is supported.
- Event-less rows are expected: reports are league-wide, so non-playing teams stay `event_id = NULL`.
- Rows with `event_id` NULL are invisible to `get_injuries_for_event`.

## Related

- [NBA game logs and player stats](nba-stats-data.md)
- [Feature groups](feature-groups.md)
- [NBA injury features experiment](exp-nba-injury-features.md)
