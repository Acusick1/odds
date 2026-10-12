---
type: component
title: Odds validation and data_quality_logs
summary: OddsValidator checks, the log-only behaviour, the unused REJECT_INVALID_ODDS flag, and the data_quality_logs table they write.
sources:
  - packages/odds-lambda/odds_lambda/storage/validators.py::OddsValidator
  - packages/odds-core/odds_core/models.py::DataQualityLog
  - packages/odds-core/odds_core/config.py::DataQualityConfig
  - packages/odds-lambda/odds_lambda/storage/writers.py::OddsWriter
  - docs/ARCHITECTURE.md#validator
  - docs/DEVELOPMENT.md#data-quality-dataqualityconfig
  - docs/DATABASE.md#core-tables
verified_at: 7887039
---

# Odds validation and data_quality_logs

`packages/odds-lambda/odds_lambda/storage/validators.py::OddsValidator` checks a raw odds payload for plausibility. [OddsWriter](odds-writer.md) calls it inside `store_odds_snapshot` and records findings in the `data_quality_logs` table (`packages/odds-core/odds_core/models.py::DataQualityLog`). Validation never blocks a write.

## What `validate_odds_snapshot` checks

`OddsValidator.validate_odds_snapshot(data, event_id)` returns `(is_valid, warnings)`:

- Empty list, event missing from a list payload, or no bookmakers: returns invalid immediately.
- Bookmaker `last_update` in the future or unparseable.
- Markets with no outcomes.
- Each outcome price: non-numeric, or outside `MIN_ODDS`..`MAX_ODDS` (-10000 to 10000) via `validate_odds_value`.
- For two-outcome `h2h`/`spreads`/`totals` markets, `validate_vig`: vig above `MAX_VIG_PERCENT` (15), negative, or below `MIN_VIG_PERCENT` (2). The floor is skipped for `EXCHANGE_BOOKMAKERS` (betfair_exchange, betdaq, smarkets, matchbook). Vig warnings do not affect `is_valid`.

## What gets logged

If there are any warnings, `OddsWriter.store_odds_snapshot` writes one `DataQualityLog`: `issue_type="validation_warnings"`, `severity` `"warning"` when `is_valid` else `"error"`, `description` with the count, and `raw_data={"warnings": first 10}`. Table columns: `event_id` (FK, nullable), `severity`, `issue_type`, `description`, JSON `raw_data`, indexed `created_at`.

Readers: `OddsReader.get_data_quality_logs` ([OddsReader](odds-reader.md)) feeds `odds status` and the [health monitor](health-monitor.md), which alerts when `error` plus `critical` rows in 24 h reach `AlertConfig.data_quality_error_threshold`.

## Configuration

`DataQualityConfig` (env `ENABLE_VALIDATION`, default true; `REJECT_INVALID_ODDS`, default false) lives in `packages/odds-core/odds_core/config.py::DataQualityConfig`; see [configuration](configuration.md). Only `OddsIngestionService.ingest_sport` reads `enable_validation` (as the `validate` argument).

## Hazards

- **`REJECT_INVALID_ODDS` is inert.** `AGENTS.md`, `docs/ARCHITECTURE.md#validator` and `docs/DEVELOPMENT.md` say strict mode is available. Nothing in the repo reads `reject_invalid_odds`; invalid data is always stored.
- **Documented checks that are not wired.** `validate_line_movement` (spread over 10, total over 20 points) and `validate_event_data` (required fields) have no caller outside the module; docs list line movement and "outcomes match expected format" as active checks. The latter has no implementation.
- **Severities and issue types.** Docs list `critical` and issue types `missing_data`, `suspicious_odds`, `stale_timestamp`. The only writer in the repo emits `warning`/`error` with `validation_warnings`. The health monitor still counts `critical`.
- **`Odds.is_valid` is always true.** Validation results are not propagated to rows ([odds table](table-odds.md)).
- **Name clash.** `packages/odds-analytics/odds_analytics/quality_metrics.py` is coverage reporting, unrelated to `DataQualityLog` ([completeness validation](completeness-validation.md)).

## Related

- [Odds ingestion flow](odds-ingestion-flow.md)
- [Storage layer](storage-layer.md)
- [Odds math](odds-math.md) — `OddsValidator.american_to_probability` is a separate copy of the implied-probability conversion
