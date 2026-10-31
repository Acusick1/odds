---
type: reference
title: Odds validation
description: OddsValidator checks, the data_quality_logs rows it produces, and DataQualityConfig.
sources:
  - odds_lambda.storage.validators.OddsValidator
  - odds_lambda.storage.validators.OddsValidator.validate_odds_snapshot
  - odds_lambda.storage.validators.OddsValidator.validate_line_movement
  - odds_lambda.storage.validators.OddsValidator.validate_event_data
  - odds_lambda.storage.writers.OddsWriter.store_envelope
  - odds_lambda.storage.writers.OddsWriter.store_envelope
  - odds_lambda.storage.readers.OddsReader.get_data_quality_logs
  - odds_lambda.ingestion.OddsIngestionService.ingest_sport
  - odds_core.models.DataQualityLog
  - odds_core.config.DataQualityConfig
  - odds_core.config.AlertConfig.data_quality_error_threshold
status: stable
---

# Odds validation

`odds_lambda.storage.validators.OddsValidator` checks one payload for plausibility as it is written and logs findings to `data_quality_logs` (`odds_core.models.DataQualityLog`); it never blocks a write. It is called by [OddsWriter](odds-writer.md#store_envelope) inside `odds_lambda.storage.writers.OddsWriter.store_envelope`, after the event is loaded and before the snapshots (one per market) and their price rows are added; a price no price row can take is logged separately, as an `invalid_price` row ([Writing price rows](price-tables.md#reading-an-envelope)). The completeness checks that share the word "quality" are separate ([Coverage and completeness checks](coverage-checks.md)).

## Checks

`odds_lambda.storage.validators.OddsValidator.validate_odds_snapshot(data, event_id)` validates one event's envelope (`event_id`, the event's integer id, only labels the log) and returns `(is_valid, warnings)`:

- No bookmakers: returns invalid immediately.
- Bookmaker `last_update` in the future or unparseable.
- Markets with no outcomes.
- Each outcome price, read as a decimal through `odds_core.odds_math.outcome_price_decimal`: missing or unparseable is an error; outside `MIN_DECIMAL_ODDS`..`MAX_DECIMAL_ODDS` (1.01 to 101.0) is an error via `odds_lambda.storage.validators.OddsValidator.validate_odds_value` (the OddsPortal ingest skips prices at or below 1.0 before they reach it).
- For two-outcome `h2h`/`spreads`/`totals` markets, `odds_lambda.storage.validators.OddsValidator.validate_vig`: vig (from decimal implied probabilities) above `MAX_VIG_PERCENT` (15), negative, or below `MIN_VIG_PERCENT` (2). The floor is skipped for `odds_core.bookmakers.ZERO_MARGIN_BOOKMAKERS`: the exchanges (betfair_exchange, betdaq, smarkets, matchbook) in `odds_core.bookmakers.EXCHANGE_BOOKMAKERS` and `polymarket`. Vig warnings do not affect `is_valid`.

`odds_lambda.storage.validators.OddsValidator.validate_line_movement` (spread over 10, total over 20 points) and `odds_lambda.storage.validators.OddsValidator.validate_event_data` (required fields) have no caller outside tests (#448).

## What gets logged

If an envelope has any warnings, `odds_lambda.storage.writers.OddsWriter.store_envelope` writes one `odds_core.models.DataQualityLog`: `issue_type="validation_warnings"`, `severity` `"warning"` when `is_valid` else `"error"`, `description` with the count, and `raw_data={"warnings": first 10}`. Columns: `event_id` (integer FK to `events.id`, nullable), `severity`, `issue_type`, `description`, JSON `raw_data`, indexed `created_at`. No other severity or issue type is written, so `critical` never occurs.

`odds_lambda.storage.readers.OddsReader.get_data_quality_logs` feeds `odds status` and the [health monitor](../operations/health-monitor.md), which alerts when `error` plus `critical` rows in 24 h reach `odds_core.config.AlertConfig.data_quality_error_threshold` (10). Validation results are not attached to snapshots or price rows.

## Configuration

`odds_core.config.DataQualityConfig` has env `ENABLE_VALIDATION` (default true) and `REJECT_INVALID_ODDS` (default false) ([Configuration](../foundations/configuration.md)). `odds_lambda.ingestion.OddsIngestionService.ingest_sport` is the only reader of `enable_validation`, as the default for its `validate` argument, which it passes to `odds_lambda.storage.writers.OddsWriter.store_envelope` for every event it stores (a game resolved through the event resolver). `reject_invalid_odds` is read by nothing, so invalid odds are always stored (#448).

## Related

- [OddsWriter](odds-writer.md)
- [Coverage and completeness checks](coverage-checks.md)
- [Odds snapshots table](odds-tables.md)
