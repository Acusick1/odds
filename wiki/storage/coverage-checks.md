---
type: reference
title: Coverage and completeness checks
description: TierCoverageValidator, odds validate, odds quality, QualityMetrics, the daily-validation workflow and the read-only scripts/check_*.py.
sources:
  - odds_lambda.storage.tier_validator.TierCoverageValidator
  - odds_lambda.storage.tier_validator.TierCoverageReport
  - odds_lambda.storage.tier_validator.DailyValidationReport
  - odds_lambda.storage.tier_validator.CRITICAL_TIERS
  - odds_lambda.storage.tier_validator.WARNING_TIERS
  - odds_lambda.storage.tier_validator.TierCoverageValidator._check_missing_games
  - odds_analytics.quality_metrics.QualityMetrics
  - odds_analytics.quality_metrics.QualityMetrics.get_games_missing_scores
  - odds_analytics.quality_metrics.QualityMetrics._get_tier_hours_range
  - odds_lambda.fetch_tier.FetchTier.max_hours
  - odds_cli.commands.validate
  - odds_cli.commands.quality
  - scripts/check_game_tier_coverage.py
  - scripts/check_games_in_database.py
  - scripts/check_tier_distribution.py
  - .github/workflows/daily-validation.yml
status: stable
---

# Coverage and completeness checks

These checks confirm that every game was collected across the [fetch tiers](fetch-tier.md), has scores, and exists at all. They are unrelated to per-snapshot [Odds validation](odds-validator.md); `odds_analytics.quality_metrics` (coverage reporting) has nothing to do with `odds_core.models.DataQualityLog`.

## TierCoverageValidator

`odds_lambda.storage.tier_validator.TierCoverageValidator` is driven by `odds_cli.commands.validate.validate_game` (`odds_lambda.storage.tier_validator.TierCoverageValidator.validate_game`), `odds_cli.commands.validate.validate_daily` (`odds_lambda.storage.tier_validator.TierCoverageValidator.validate_date`) and `odds_cli.commands.validate.validate_tier_coverage` (`odds_lambda.storage.tier_validator.TierCoverageValidator.validate_date_range`). Per game it reads tier counts from [OddsReader](odds-reader.md) and builds a `odds_lambda.storage.tier_validator.TierCoverageReport`; per date a `odds_lambda.storage.tier_validator.DailyValidationReport`.

- `odds_lambda.storage.tier_validator.CRITICAL_TIERS` = PREGAME, CLOSING: missing makes a game incomplete (`is_complete`). They depend on tiers stored with `<=` boundaries ([Boundary encodings](fetch-tier.md#boundary-encodings)). `odds_lambda.storage.tier_validator.WARNING_TIERS` = OPENING, EARLY, SHARP: counted in `games_with_warnings`. `odds_lambda.storage.tier_validator.TierCoverageValidator.DEFAULT_REQUIRED_TIERS` is all five.
- FINAL games without both scores get the issue `"Missing final scores for completed game"`.
- `check_discovery=True` calls The Odds API `/scores` and flags `missing_games` (completed upstream, absent from `events` altogether, not games with zero snapshots; error) and `missing_scores` (in DB, not FINAL or no score; warning). Discovery calls it with `sport='basketball_nba'` only and only for dates 0 to 3 days back; `odds_lambda.storage.tier_validator.TierCoverageValidator.validate_date` reads `odds_lambda.storage.readers.OddsReader.get_games_by_date`, which has no sport filter and uses a noon-to-noon UTC day, so EPL and MLB days are validated against NBA discovery (#465).
- `odds_lambda.storage.tier_validator.TierCoverageValidator.validate_date` logs and skips games whose validation raises, so `total_games` can exceed `complete_games + incomplete_games`. A discovery failure re-raises.

## odds validate

`odds_cli.commands.validate`: `daily` (`--date`, `--strict/--no-strict`, `--check-scores`, `--check-discovery`, `--output-json`, `--verbose`), `game <event_id>`, `tier-coverage --start --end`. `--no-strict` requires only OPENING and CLOSING. With both scores and discovery checks on, exit code 1 on incomplete games, missing scores or missing games; otherwise only tier incompleteness fails.

## odds quality

`odds quality coverage` (`odds_cli.commands.quality`): `--start`, `--end`, `--sport` (default `basketball_nba`). Uses `odds_analytics.quality_metrics.QualityMetrics` for game counts, odds/score presence, per-tier and per-bookmaker coverage.

- `odds_analytics.quality_metrics.QualityMetrics.get_games_missing_scores` has no status filter, so unplayed games in the range count as missing scores (#467).
- `odds_analytics.quality_metrics.QualityMetrics._get_tier_hours_range` hardcodes tier hour ranges as display strings rather than deriving them from `odds_lambda.fetch_tier.FetchTier.max_hours` (#467).

## Daily-validation workflow

`.github/workflows/daily-validation.yml`: 09:00 UTC runs `odds validate daily --check-discovery --verbose --output-json validation_<yesterday>.json` against the `production` environment and uploads it (kept 30 days). The file dated D covers D-1 ([CI/CD workflows](../operations/ci-cd-workflows.md)). Whether the deployed schedule is actually running cannot be verified from the repo; run `odds scheduler list-jobs --backend aws`.

## Read-only diagnostics

Safe against production: `scripts/check_games_in_database.py [N]` (next N scheduled events in the DB, not the league schedule), `scripts/check_game_tier_coverage.py --date|--event` and `scripts/check_tier_distribution.py [days]`. `check_game_tier_coverage.py --date` filters on the calendar date of `commence_time`, so counts differ from `validate daily` at day edges. Workflow for a failing validation: [Debug the AWS self-scheduling jobs](../scheduling/howto-debug-aws-scheduler.md).

## Related

- [Odds validation](odds-validator.md)
- [Fetch tier](fetch-tier.md)
- [Self-scheduling](../scheduling/self-scheduling.md)
- [The odds CLI](../foundations/odds-cli.md)
