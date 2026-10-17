---
type: reference
title: Odds validation
description: OddsValidator checks, the data_quality_logs rows it produces, and DataQualityConfig.
sources:
  - odds_lambda.storage.validators.OddsValidator
  - odds_lambda.storage.validators.OddsValidator.validate_odds_snapshot
  - odds_lambda.storage.validators.OddsValidator.validate_line_movement
  - odds_lambda.storage.validators.OddsValidator.validate_event_data
  - odds_lambda.storage.validators.OddsValidator.american_to_probability
  - odds_lambda.storage.writers.OddsWriter.store_odds_snapshot
  - odds_lambda.storage.readers.OddsReader.get_data_quality_logs
  - odds_lambda.ingestion.OddsIngestionService.ingest_sport
  - odds_core.models.DataQualityLog
  - odds_core.config.DataQualityConfig
  - odds_core.config.AlertConfig.data_quality_error_threshold
status: stable
---

# Odds validation

`odds_lambda.storage.validators.OddsValidator` checks one payload for plausibility as it is written and logs findings to `data_quality_logs` (`odds_core.models.DataQualityLog`); it never blocks a write. It is called by [OddsWriter](odds-writer.md#store_odds_snapshot) inside `odds_lambda.storage.writers.OddsWriter.store_odds_snapshot`. The completeness checks that share the word "quality" are separate ([Coverage and completeness checks](coverage-checks.md)).

## Checks

`odds_lambda.storage.validators.OddsValidator.validate_odds_snapshot(data, event_id)` returns `(is_valid, warnings)`:

- Empty list, event missing from a list payload, or no bookmakers: returns invalid immediately.
- Bookmaker `last_update` in the future or unparseable.
- Markets with no outcomes.
- Each outcome price: non-numeric, or outside `MIN_ODDS`..`MAX_ODDS` (-10000 to 10000) via `odds_lambda.storage.validators.OddsValidator.validate_odds_value`.
- For two-outcome `h2h`/`spreads`/`totals` markets, `odds_lambda.storage.validators.OddsValidator.validate_vig`: vig above `MAX_VIG_PERCENT` (15), negative, or below `MIN_VIG_PERCENT` (2). The floor is skipped for `EXCHANGE_BOOKMAKERS` (betfair_exchange, betdaq, smarkets, matchbook). Vig warnings do not affect `is_valid`.

`odds_lambda.storage.validators.OddsValidator.validate_line_movement` (spread over 10, total over 20 points) and `odds_lambda.storage.validators.OddsValidator.validate_event_data` (required fields) have no caller outside tests (#448). `odds_lambda.storage.validators.OddsValidator.american_to_probability` computes implied probability itself rather than calling the core helper (#463; [Odds arithmetic helpers](../foundations/odds-math.md#conversion-copies)).

## What gets logged

If there are any warnings, `odds_lambda.storage.writers.OddsWriter.store_odds_snapshot` writes one `odds_core.models.DataQualityLog`: `issue_type="validation_warnings"`, `severity` `"warning"` when `is_valid` else `"error"`, `description` with the count, and `raw_data={"warnings": first 10}`. Columns: `event_id` (FK, nullable), `severity`, `issue_type`, `description`, JSON `raw_data`, indexed `created_at`. No other severity or issue type is written, so `critical` never occurs.

`odds_lambda.storage.readers.OddsReader.get_data_quality_logs` feeds `odds status` and the [health monitor](../operations/health-monitor.md), which alerts when `error` plus `critical` rows in 24 h reach `odds_core.config.AlertConfig.data_quality_error_threshold` (10). Validation results are not propagated to `odds` rows: `is_valid` is always true ([Odds snapshot and odds tables](odds-tables.md)).

## Configuration

`odds_core.config.DataQualityConfig` has env `ENABLE_VALIDATION` (default true) and `REJECT_INVALID_ODDS` (default false) ([Configuration](../foundations/configuration.md)). Only `odds_lambda.ingestion.OddsIngestionService.ingest_sport` reads `enable_validation` (as the `validate` argument). `reject_invalid_odds` is read by nothing, so invalid odds are always stored (#448).

## Related

- [OddsWriter](odds-writer.md)
- [Coverage and completeness checks](coverage-checks.md)
- [Odds snapshot and odds tables](odds-tables.md)
