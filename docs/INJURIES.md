# NBA Injury Report Pipeline

Injury data from official NBA injury report PDFs, parsed via the [`nbainjuries`](https://github.com/mcmullarkey/nbainjuries) package.

## Data Source

The NBA publishes injury reports as PDFs on their official site. `nbainjuries` uses tabula-java (via jpype) to parse them. Reports are point-in-time snapshots listing every injured player league-wide, with status and reason.

Report intervals changed on 2025-12-22:
- **Before**: hourly snapshots
- **After**: 15-minute snapshots

## Schema

Table: `nba_injury_reports`

| Column | Type | Description |
|--------|------|-------------|
| report_time | datetime (UTC) | When the report snapshot was published |
| game_date | date | ET calendar date of the game |
| matchup | str | e.g. `BOS@ORL` |
| team | str | Full team name e.g. `Boston Celtics` |
| player_name | str | `Last, First` format |
| status | enum | OUT, QUESTIONABLE, DOUBTFUL, PROBABLE, AVAILABLE |
| reason | str | Injury description |
| event_id | FK → events | Auto-matched at write time via team + game_date |

Unique constraint: `(report_time, team, player_name, game_date)` — upserts are idempotent.

## Event Matching

`InjuryWriter._match_event()` converts the ET `game_date` to a UTC window (10 AM ET → 6 AM ET next day) and looks for exactly one sportsbook Event with that team. Unmatched reports (team didn't play that day, or ambiguous) get `event_id = NULL`.

Reports are league-wide — teams that don't play on a given date still appear. These produce expected unmatched pairs.

## CLI Commands

```bash
# Fetch current injury report
uv run odds injuries fetch

# Backfill historical data for a season
uv run odds injuries backfill --season 2024-25
uv run odds injuries backfill --season 2025-26 --hours-before 12,8,2 --dry-run

# Pipeline health
uv run odds injuries status
```

### Backfill Strategy

Event-driven: computes target report timestamps from game `commence_time` values in the DB, not by crawling every time slot. For each event, fetches reports at `--hours-before` offsets (default: 8h, 2h), rounds to the nearest valid report slot, and deduplicates across events sharing the same slot.

Rate-limited at `--delay-ms` (default 500ms) between fetches. JVM startup adds a few seconds to the first fetch.

## Feature Extraction

`InjuryFeatures` dataclass in `odds_analytics/injury_features.py`. Enabled via `feature_groups=("tabular", "injuries")` in FeatureConfig.

### Pipeline

1. `collect_event_data()` bulk-loads all injury reports for the event (one query)
2. `extract_injury_features()` filters to `report_time <= snapshot_time` (look-ahead prevention), takes the latest report snapshot, counts by status and team
3. NaN-fill when no injury data exists (events without reports are kept, not dropped)

### Features (6)

| Feature | Description |
|---------|-------------|
| impact_out_home | Impact-weighted sum of OUT players on home team |
| impact_out_away | Impact-weighted sum of OUT players on away team |
| impact_gtd_home | Impact-weighted sum of GTD players on home team (0.5x discount) |
| impact_gtd_away | Impact-weighted sum of GTD players on away team (0.5x discount) |
| report_hours_before_game | Hours between latest report and tipoff |
| injury_news_recency | Hours between latest report and snapshot (staleness) |

Impact score per player: `(on_off_rtg - on_def_rtg) * (minutes_per_game / 48)` from `NbaPlayerSeasonStats`. Players without stats fall back to 1.0 (headcount). GTD (QUESTIONABLE + DOUBTFUL) players are discounted by 0.5x. PROBABLE and AVAILABLE statuses are excluded.

## Key Files

| File | Purpose |
|------|---------|
| `odds_core/injury_models.py` | SQLModel schema (`InjuryReport`, `InjuryStatus`) |
| `odds_lambda/injury_fetcher.py` | `nbainjuries` wrapper, PDF parsing, record creation |
| `odds_lambda/storage/injury_writer.py` | Upsert with auto event matching |
| `odds_lambda/storage/injury_reader.py` | Query by event, pipeline stats |
| `odds_analytics/injury_features.py` | Feature extraction (`InjuryFeatures`, `extract_injury_features`) |
| `odds_cli/commands/injuries.py` | CLI: fetch, backfill, status |
