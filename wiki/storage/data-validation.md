---
type: component
title: Odds validation and coverage checks
description: "The two unrelated \"quality\" systems, per-snapshot OddsValidator checks into data_quality_logs (log-only) and completeness checks (odds validate, odds quality, TierCoverageValidator, the daily-validation workflow, scripts/check_*), with their NBA-only parts."
sources:
  - odds_lambda.storage.validators.OddsValidator
  - odds_lambda.storage.tier_validator.TierCoverageValidator
  - odds_lambda.storage.writers.OddsWriter
  - odds_core.models.DataQualityLog
  - odds_core.config.DataQualityConfig
  - odds_analytics.quality_metrics.QualityMetrics
  - odds_cli.commands.validate
  - odds_cli.commands.quality
  - scripts/check_game_tier_coverage.py
  - scripts/check_games_in_database.py
  - scripts/check_tier_distribution.py
  - .github/workflows/daily-validation.yml
status: stable
---

# Odds validation and coverage checks

Two unrelated systems share the word "quality". **Per-snapshot validation** (`odds_lambda.storage.validators.OddsValidator`) checks one payload for plausibility as it is written and logs findings to `data_quality_logs`; it never blocks a write. **Completeness checks** (`odds validate`, `odds quality`, `odds_lambda.storage.tier_validator.TierCoverageValidator`, the daily-validation workflow) check that every game was collected across the [fetch tiers](fetch-tier.md), has scores, and exists at all. `odds_analytics.quality_metrics` (coverage reporting) has nothing to do with `odds_core.models.DataQualityLog`.

## Per-snapshot validation

`odds_lambda.storage.validators.OddsValidator` is called by [OddsWriter](odds-storage.md) inside `store_odds_snapshot`, and findings go to the `data_quality_logs` table (`odds_core.models.DataQualityLog`).

### Checks

`odds_lambda.storage.validators.OddsValidator.validate_odds_snapshot(data, event_id)` returns `(is_valid, warnings)`:

- Empty list, event missing from a list payload, or no bookmakers: returns invalid immediately.
- Bookmaker `last_update` in the future or unparseable.
- Markets with no outcomes.
- Each outcome price: non-numeric, or outside `MIN_ODDS`..`MAX_ODDS` (-10000 to 10000) via `validate_odds_value`.
- For two-outcome `h2h`/`spreads`/`totals` markets, `validate_vig`: vig above `MAX_VIG_PERCENT` (15), negative, or below `MIN_VIG_PERCENT` (2). The floor is skipped for `EXCHANGE_BOOKMAKERS` (betfair_exchange, betdaq, smarkets, matchbook). Vig warnings do not affect `is_valid`.

### What gets logged

If there are any warnings, `odds_lambda.storage.writers.OddsWriter.store_odds_snapshot` writes one `odds_core.models.DataQualityLog`: `issue_type="validation_warnings"`, `severity` `"warning"` when `is_valid` else `"error"`, `description` with the count, and `raw_data={"warnings": first 10}`. Columns: `event_id` (FK, nullable), `severity`, `issue_type`, `description`, JSON `raw_data`, indexed `created_at`.

`odds_lambda.storage.readers.OddsReader.get_data_quality_logs` feeds `odds status` and the [health monitor](../operations/alerts-and-health.md), which alerts when `error` plus `critical` rows in 24 h reach `odds_core.config.AlertConfig.data_quality_error_threshold` (10). The validator only emits `warning`/`error`, so `critical` never occurs.

### Configuration

`odds_core.config.DataQualityConfig` (env `ENABLE_VALIDATION`, default true; `REJECT_INVALID_ODDS`, default false) lives in `odds_core.config.DataQualityConfig` ([configuration](../foundations/configuration.md)). Only `odds_lambda.ingestion.OddsIngestionService.ingest_sport` reads `enable_validation` (as the `validate` argument).

## Completeness and coverage checks

- **`odds_lambda.storage.tier_validator.TierCoverageValidator`**: `odds_cli.commands.validate.validate_game`, `validate_date`, `validate_date_range`. Per game it reads tier counts from OddsReader and builds a `odds_lambda.storage.tier_validator.TierCoverageReport`; per date a `odds_lambda.storage.tier_validator.DailyValidationReport`.
  - `odds_lambda.storage.tier_validator.CRITICAL_TIERS` = PREGAME, CLOSING: missing makes a game incomplete (`is_complete`). They depend on tiers stored with `<=` boundaries ([fetch-tier](fetch-tier.md#four-boundary-encodings)). `odds_lambda.storage.tier_validator.WARNING_TIERS` = OPENING, EARLY, SHARP: counted in `games_with_warnings`. `DEFAULT_REQUIRED_TIERS` is all five.
  - FINAL games without both scores get the issue `"Missing final scores for completed game"`.
  - `check_discovery=True` calls The Odds API `/scores` and flags `missing_games` (completed upstream, absent from `events`; error) and `missing_scores` (in DB, not FINAL or no score; warning).
- **`odds validate`** (`odds_cli.commands.validate`): `daily` (`--date`, `--strict/--no-strict`, `--check-scores`, `--check-discovery`, `--output-json`, `--verbose`), `game <event_id>`, `tier-coverage --start --end`. `--no-strict` requires only OPENING and CLOSING. With both scores and discovery checks on, exit code 1 on incomplete games, missing scores or missing games; otherwise only tier incompleteness fails.
- **`odds quality coverage`** (`odds_cli.commands.quality`): `--start`, `--end`, `--sport` (default `basketball_nba`). Uses `odds_analytics.quality_metrics.QualityMetrics` for game counts, odds/score presence, per-tier and per-bookmaker coverage.
- **`.github/workflows/daily-validation.yml`**: 09:00 UTC runs `odds validate daily --check-discovery --verbose --output-json validation_<yesterday>.json` against the `production` environment and uploads it (kept 30 days). The file dated D covers D-1 ([ci-cd-workflows](../operations/ci-cd-workflows.md)).
- **Read-only diagnostics** (safe against production): `scripts/check_games_in_database.py [N]` (next N scheduled events in the DB, not the league schedule), `scripts/check_game_tier_coverage.py --date|--event`, `scripts/check_tier_distribution.py [days]`.

Workflow for a failing validation: [howto-debug-aws-scheduler](../scheduling/howto-debug-aws-scheduler.md). Whether the deployed schedule is actually running cannot be verified from the repo; run `odds scheduler list-jobs --backend aws`.

## Hazards

- **`REJECT_INVALID_ODDS` is inert.** `odds_core.config.DataQualityConfig` defines `reject_invalid_odds`, but nothing reads it; `odds_lambda.ingestion.OddsIngestionService` reads only `enable_validation`, so invalid data is always stored.
- **Checks that are not wired.** `odds_lambda.storage.validators.OddsValidator.validate_line_movement` (spread over 10, total over 20 points) and `odds_lambda.storage.validators.OddsValidator.validate_event_data` (required fields) have no caller outside the module.
- **Severities and issue types.** The only writer (`odds_lambda.storage.writers.OddsWriter.store_odds_snapshot`) emits `warning` or `error` with issue type `validation_warnings`; no other severity or issue type is written.
- **`odds_core.models.Odds.is_valid` is always true.** Validation results are not propagated to rows.
- **Third copy of the implied-probability conversion.** `odds_lambda.storage.validators.OddsValidator.american_to_probability` is a separate copy ([odds-math-and-devigging](../foundations/odds-math-and-devigging.md)).
- **NBA-only pieces.** `_fetch_api_games` hardcodes `sport="basketball_nba"` and only works for dates 0 to 3 days back (else skipped); `quality coverage` defaults to NBA; `validate_date` reads `get_games_by_date`, which has no sport filter and uses a noon-to-noon UTC window. EPL and MLB days are validated against NBA discovery.
- **Window mismatch.** `check_game_tier_coverage.py --date` filters on calendar date of `commence_time`, so counts differ from `validate daily` at day edges.
- **Swallowed failures.** `validate_date` logs and skips games whose validation raises, so `total_games` can exceed `complete_games + incomplete_games`. A discovery failure re-raises.
- **`missing_games` meaning.** `odds_lambda.storage.tier_validator.TierCoverageValidator._check_missing_games` flags games absent from `events` altogether, not games with zero snapshots.
- **Missing scores overcount.** `odds_analytics.quality_metrics.QualityMetrics.get_games_missing_scores` has no status filter, so unplayed games in the range count as missing scores.
- **Hard-coded tier ranges.** `odds_analytics.quality_metrics.QualityMetrics._get_tier_hours_range` hardcodes display strings rather than deriving them from `odds_lambda.fetch_tier.FetchTier.max_hours`.

## Related

- [Odds snapshots, odds rows and OddsWriter](odds-storage.md)
- [Self-scheduling](../scheduling/self-scheduling.md)
- [Packages and the odds CLI](../foundations/packages-and-cli.md)
- [OddsReader and the storage layer](odds-reader-and-storage-layer.md)
