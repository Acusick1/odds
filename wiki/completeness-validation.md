---
type: component
title: Completeness and coverage validation
summary: odds validate, odds quality coverage, TierCoverageValidator, the scripts/check_* tools and the daily-validation workflow, which together check that collected data is complete.
sources:
  - packages/odds-lambda/odds_lambda/storage/tier_validator.py::TierCoverageValidator
  - packages/odds-analytics/odds_analytics/quality_metrics.py::QualityMetrics
  - packages/odds-cli/odds_cli/commands/validate.py
  - packages/odds-cli/odds_cli/commands/quality.py
  - scripts/check_game_tier_coverage.py
  - scripts/check_games_in_database.py
  - scripts/check_tier_distribution.py
  - .github/workflows/daily-validation.yml
  - docs/CLI.md#validate-data
  - docs/CLI.md#data-quality-coverage
  - docs/DEBUGGING_SCHEDULER.md#1-establish-context
  - docs/DEBUGGING_SCHEDULER.md#3-compare-the-database-with-the-real-fixture-list
  - docs/DEBUGGING_SCHEDULER.md#4-check-tier-coverage
verified_at: 7887039
---

# Completeness and coverage validation

Checks that every game was collected across the [fetch tiers](fetch-tier.md), has scores, and exists at all. Separate from per-payload [odds validation](data-quality-validation.md).

## Components

- **`packages/odds-lambda/odds_lambda/storage/tier_validator.py::TierCoverageValidator`**: `validate_game`, `validate_date`, `validate_date_range`. Per game it reads tier counts from [OddsReader](odds-reader.md) and builds a `TierCoverageReport`; per date a `DailyValidationReport`.
  - `CRITICAL_TIERS` = PREGAME, CLOSING: missing makes a game incomplete (`is_complete`). `WARNING_TIERS` = OPENING, EARLY, SHARP: counted in `games_with_warnings`. `DEFAULT_REQUIRED_TIERS` is all five.
  - FINAL games without both scores get the issue `"Missing final scores for completed game"`.
  - `check_discovery=True` calls The Odds API `/scores` and flags `missing_games` (completed upstream, absent from `events`; error) and `missing_scores` (in DB, not FINAL or no score; warning).
- **`odds validate`** (`packages/odds-cli/odds_cli/commands/validate.py`): `daily` (`--date`, `--strict/--no-strict`, `--check-scores`, `--check-discovery`, `--output-json`, `--verbose`), `game <event_id>`, `tier-coverage --start --end`. `--no-strict` requires only OPENING and CLOSING. With both scores and discovery checks on, exit code 1 on incomplete games, missing scores or missing games; otherwise only tier incompleteness fails.
- **`odds quality coverage`** (`packages/odds-cli/odds_cli/commands/quality.py`): `--start`, `--end`, `--sport` (default `basketball_nba`). Uses `packages/odds-analytics/odds_analytics/quality_metrics.py::QualityMetrics` for game counts, odds/score presence, per-tier and per-bookmaker coverage.
- **`.github/workflows/daily-validation.yml`**: 09:00 UTC runs `odds validate daily --check-discovery --verbose --output-json validation_<yesterday>.json` against the `production` environment and uploads it (30 days). The file dated D covers D-1.
- **Read-only diagnostics** (safe against production): `scripts/check_games_in_database.py [N]` (next N scheduled events in the DB, not the league schedule), `scripts/check_game_tier_coverage.py --date|--event`, `scripts/check_tier_distribution.py [days]`.

Workflow for a failing validation: [debugging the AWS scheduler](howto-debug-aws-scheduler.md). Whether the deployed schedule is actually running cannot be verified from the repo; run `odds scheduler list-jobs --backend aws`.

## Hazards

- **NBA-only pieces.** `_fetch_api_games` hardcodes `sport="basketball_nba"` and only works for dates 0 to 3 days back (else skipped); `quality coverage` defaults to NBA; `validate_date` reads `get_games_by_date`, which has no sport filter and uses a noon-to-noon UTC window. EPL/MLB days are validated against NBA discovery.
- **Window mismatch.** `check_game_tier_coverage.py --date` filters on calendar date of `commence_time`, so counts differ from `validate daily` at day edges.
- **Swallowed failures.** `validate_date` logs and skips games whose validation raises, so `total_games` can exceed `complete_games + incomplete_games`. A discovery failure re-raises.
- **`missing_games` meaning.** `docs/DEBUGGING_SCHEDULER.md` describes them as games with zero snapshots; the code flags games absent from `events` altogether.
- **Missing scores overcount.** `QualityMetrics.get_games_missing_scores` has no status filter, so unplayed games in the range count as missing scores.
- **Hard-coded tier ranges.** `QualityMetrics._get_tier_hours_range` hardcodes display strings rather than deriving them from `FetchTier.max_hours`.
- **Name clash.** `quality_metrics.py` has nothing to do with `DataQualityLog`.

## Related

- [Self-scheduling](self-scheduling.md)
- [CI/CD workflows](ci-cd-workflows.md)
- [CLI overview](cli-overview.md)
- [Storage layer](storage-layer.md)
