---
type: reference
title: Coverage and completeness checks
description: TierCoverageValidator, a sport's league-local game day, odds validate, odds quality, QualityMetrics and the read-only scripts/check_games_in_database.py.
sources:
  - odds_lambda.storage.tier_validator.TierCoverageValidator
  - odds_lambda.storage.tier_validator.TierCoverageReport
  - odds_lambda.storage.tier_validator.DailyValidationReport
  - odds_lambda.storage.tier_validator.CRITICAL_TIERS
  - odds_lambda.storage.tier_validator.WARNING_TIERS
  - odds_lambda.storage.tier_validator.covered_tiers
  - odds_lambda.storage.tier_validator.TierCoverageValidator._check_missing_games
  - odds_lambda.storage.tier_validator.TierCoverageValidator._fetch_api_games
  - odds_lambda.storage.readers.OddsReader.get_games_by_date
  - odds_core.sports.game_day_window
  - odds_core.sports.LEAGUE_TIMEZONES
  - odds_analytics.quality_metrics.QualityMetrics
  - odds_analytics.quality_metrics.QualityMetrics.get_games_missing_scores
  - odds_analytics.quality_metrics.QualityMetrics._get_tier_hours_range
  - odds_lambda.fetch_tier.FetchTier.max_hours
  - odds_cli.commands.validate
  - odds_cli.commands.quality
  - scripts/check_games_in_database.py
status: stable
---

# Coverage and completeness checks

These checks confirm that every game was collected across the [fetch tiers](fetch-tier.md), has scores, and exists at all. They are unrelated to per-snapshot [Odds validation](odds-validator.md); `odds_analytics.quality_metrics` (coverage reporting) has nothing to do with `odds_core.models.DataQualityLog`.

## TierCoverageValidator

`odds_lambda.storage.tier_validator.TierCoverageValidator` is driven by `odds_cli.commands.validate.validate_game` (`odds_lambda.storage.tier_validator.TierCoverageValidator.validate_game`), `odds_cli.commands.validate.validate_daily` (`odds_lambda.storage.tier_validator.TierCoverageValidator.validate_date`) and `odds_cli.commands.validate.validate_tier_coverage` (`odds_lambda.storage.tier_validator.TierCoverageValidator.validate_date_range`). Per game it puts each distinct snapshot time of the event in the tier it covers against the event's current kick-off (`odds_lambda.storage.tier_validator.covered_tiers`): a `close` snapshot and each market's latest OddsPortal history (`source_stamped`) time are CLOSING (the rows it reads carry the market), an `upper_bound` time on a boundary covers the farther tier, and the rest are bucketed by hours ([the tier a snapshot covers](fetch-tier.md#the-tier-a-snapshot-covers)). It builds a `odds_lambda.storage.tier_validator.TierCoverageReport`; per sport and day a `odds_lambda.storage.tier_validator.DailyValidationReport`.

A day is a calendar day in the league's timezone (`odds_core.sports.LEAGUE_TIMEZONES`: London for EPL, New York for MLB and NBA), so an evening NBA tip-off that falls after midnight UTC belongs to the Eastern date it is played on. `odds_core.sports.game_day_window` gives the day's bounds, and `odds_lambda.storage.readers.OddsReader.get_games_by_date` returns the sport's events that commence inside them. `odds_lambda.storage.tier_validator.TierCoverageValidator.validate_date` reads them with every status, so games not yet FINAL are still checked.

- `odds_lambda.storage.tier_validator.CRITICAL_TIERS` = PREGAME, CLOSING: missing makes a game incomplete (`is_complete`). Hours are bucketed by `odds_lambda.tier_utils.calculate_tier`, whose `odds_lambda.fetch_tier.FetchTier.max_hours` boundaries belong to the nearer tier ([the one hours-to-tier rule](fetch-tier.md#the-one-hours-to-tier-rule)), except for an `upper_bound` time, which covers the farther tier on a boundary, so an FDUK opening stamped exactly 3.0 h out is PREGAME, not closing coverage. `odds_lambda.storage.tier_validator.WARNING_TIERS` = OPENING, EARLY, SHARP: counted in `games_with_warnings`. `odds_lambda.storage.tier_validator.TierCoverageValidator.DEFAULT_REQUIRED_TIERS` is all five.
- FINAL games without both scores get the issue `"Missing final scores for completed game"`.
- `check_discovery=True` calls The Odds API `/scores` for the sport (`odds_lambda.storage.tier_validator.TierCoverageValidator._fetch_api_games`, which spends the sport's key), keeps the games that commence on the day, and matches each to the day's event holding its id as the `odds_api` external id. It flags `missing_games` (completed upstream, held by none of the day's events, not games with zero snapshots; error, listed by the API's id) and `missing_scores` (held by an event that is not FINAL or has no score; warning, listed by the event's id). It asks for `daysFrom` as the whole days, rounded up, since the day began; the endpoint accepts 1 to 3, so a day that has not begun or began more than 3 days ago skips discovery with a warning.
- `odds_lambda.storage.tier_validator.TierCoverageValidator.validate_date` logs and skips games whose validation raises, so `total_games` can exceed `complete_games + incomplete_games`. A discovery failure re-raises.

## odds validate

`odds_cli.commands.validate`: `daily` (`--sport`, required, `--date`, ISO `YYYY-MM-DD` on the league's calendar, default yesterday's UTC date, `--strict/--no-strict`, `--check-scores`, `--check-discovery`, `--output-json`, `--verbose`), `game <event_id>` (the integer event id), `tier-coverage --sport --start --end` (ISO `YYYY-MM-DD` dates on the league's calendar). `--sport` takes `soccer_epl`, `baseball_mlb` or `basketball_nba`, and `--output-json` records it as `sport_key`. `--no-strict` requires only OPENING and CLOSING. With both scores and discovery checks on, exit code 1 on incomplete games, missing scores or missing games; otherwise only tier incompleteness fails.

## odds quality

`odds quality coverage` (`odds_cli.commands.quality`): `--sport` (required), `--start`, `--end` (`YYYY-MM-DD`, midnight UTC). Uses `odds_analytics.quality_metrics.QualityMetrics` for game counts, snapshot counts as distinct snapshot times, odds/score presence, per-tier and per-bookmaker coverage. Per-tier coverage counts each event's distinct snapshot times under the tiers `odds_lambda.tier_utils.snapshot_tiers` derives for its snapshots. Per-bookmaker coverage reads the price rows ([prices](price-tables.md#prices)): the events and the snapshots in which each canonical book key has a price.

- `odds_analytics.quality_metrics.QualityMetrics.get_games_missing_scores` has no status filter, so unplayed games in the range count as missing scores (#467).
- `odds_analytics.quality_metrics.QualityMetrics._get_tier_hours_range` hardcodes tier hour ranges as display strings rather than deriving them from `odds_lambda.fetch_tier.FetchTier.max_hours` (#467).

## Read-only diagnostics

Safe against production: `scripts/check_games_in_database.py [N]` (next N scheduled events in the DB, not the league schedule). Workflow for a failing validation: [Debug the AWS self-scheduling jobs](../scheduling/howto-debug-aws-scheduler.md).

## Related

- [Odds validation](odds-validator.md)
- [Fetch tier](fetch-tier.md)
- [Self-scheduling](../scheduling/self-scheduling.md)
- [The odds CLI](../foundations/odds-cli.md)
