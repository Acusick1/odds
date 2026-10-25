---
type: reference
title: OddsWriter
description: store_odds_snapshot (envelope, odds rows and price rows), event upserts, logging helpers, and their edge cases.
sources:
  - odds_lambda.storage.writers.OddsWriter
  - odds_lambda.storage.writers.OddsWriter.store_odds_snapshot
  - odds_lambda.storage.writers.OddsWriter._parse_and_store_odds
  - odds_lambda.storage.shaped_snapshots.SnapshotShape
  - odds_lambda.storage.shaped_snapshots.store_shaped_snapshot
  - odds_lambda.storage.writers.OddsWriter.log_fetch
  - odds_lambda.storage.writers.OddsWriter.log_data_quality_issue
  - odds_lambda.tier_utils.calculate_tier_from_timestamps
  - odds_lambda.ingestion.OddsIngestionService
  - odds_core.config.DataQualityConfig
  - odds_core.models.DataQualityLog
  - odds_core.models.FetchLog
status: stable
---

# OddsWriter

`odds_lambda.storage.writers.OddsWriter` owns an `AsyncSession` and an `odds_lambda.storage.validators.OddsValidator` (its vig floor skips zero-margin books, `odds_core.bookmakers.ZERO_MARGIN_BOOKMAKERS`). It never commits. It flushes each snapshot it stores, to give the snapshot's odds and price rows its id; callers commit ([Commit contract](storage-layer.md#commit-contract)). The tables it writes are on [Odds snapshot and odds tables](odds-tables.md), [Prices and snapshot payloads tables](price-tables.md) and [Events table](events.md). Live jobs find the event a snapshot belongs to through the [event resolver](event-resolver.md) before storing it.

## store_odds_snapshot

`odds_lambda.storage.writers.OddsWriter.store_envelope` does the work and returns an `odds_lambda.storage.writers.StoredEnvelope` (the snapshot, its `odds` rows and whether it was inserted); `odds_lambda.storage.writers.OddsWriter.store_odds_snapshot` calls it and returns the snapshot and rows.

`odds_lambda.storage.writers.OddsWriter.store_odds_snapshot(event_id, raw_data, snapshot_time=None, validate=True, *, shape=None, api_request_id=None)` stores one event's envelope (`raw_data`). `shape` (`odds_lambda.storage.shaped_snapshots.SnapshotShape`) names the snapshot's source, its [time basis](odds-tables.md#odds_snapshots) and the payload it came in; the OddsPortal live, Betfair and live Polymarket jobs pass it, and the other writers pass none. Steps:

1. Loads the event and computes `hours_until_commence` and the fetch tier from `snapshot_time` and the event's own `commence_time` (`odds_lambda.tier_utils.calculate_tier_from_timestamps`). The tier is per event, not per job run. If the event row is missing, both `fetch_tier` and `hours_until_commence` are stored as NULL ([Fetch tier](fetch-tier.md)); with `shape`, a missing event raises `ValueError`, since outcomes are matched to its teams.
2. If `validate`, runs the validator and writes one `odds_core.models.DataQualityLog` row for any warnings; storage proceeds regardless ([Odds validation](odds-validator.md)). `odds_lambda.ingestion.OddsIngestionService.ingest_sport` is the only caller that passes `validate`: the `validate` argument of that method or of `odds_lambda.ingestion.OddsIngestionService.ingest_sports` when given, else `odds_core.config.DataQualityConfig.enable_validation`. Every other caller of `odds_lambda.storage.writers.OddsWriter.store_odds_snapshot` uses the default `True`.
3. Builds an `odds_core.models.OddsSnapshot` ([columns and indexes](odds-tables.md#odds_snapshots)) with the full `raw_data`, `bookmaker_count` and `api_request_id`.
   - Without `shape` it is added with the redesigned columns (`source`, `market`, `time_basis`, the market metadata and `payload_id`) NULL, and flushed to get its id.
   - With `shape` it goes through `odds_lambda.storage.shaped_snapshots.store_shaped_snapshot`, which fills those columns, writes its price rows and, for an envelope quoting several markets, stores one snapshot per market with `raw_data` cut to that market ([Writing price rows](price-tables.md#writing-price-rows)). When a snapshot already holds its key (event, source, market, time), nothing is written, `odds_snapshot_already_stored` is logged and the stored snapshot is returned with no odds rows.
4. `odds_lambda.storage.writers.OddsWriter._parse_and_store_odds` adds one `odds_core.models.Odds` row per bookmaker/market/outcome of the whole envelope, each with `snapshot_id` set to the stored snapshot (the first market's when the envelope was split). Bookmakers without a key are skipped; `last_update` falls back to `snapshot_time` if unparseable; `odds_timestamp` is `snapshot_time`. It stores the decimal price (it must exceed 1.0, `ck_odds_price_decimal_gt_one`) from `odds_core.odds_math.outcome_price_decimal`, which raises `KeyError` for an outcome with neither `price_decimal` nor `price`; the validator flags such an outcome as an error.

Returns `(OddsSnapshot, list[Odds])`.

## Event handling

Event upserts (`odds_lambda.storage.writers.OddsWriter.upsert_event`, `odds_lambda.storage.writers.OddsWriter.update_event_status`, `odds_lambda.storage.writers.OddsWriter.bulk_upsert_events`) are on [Events table](events.md#upsert-verbs); `odds_lambda.storage.writers.OddsWriter.update_event_status` raises `ValueError` for FINAL without both scores and for a move the stored status does not allow ([Event status lifecycle](../collection/event-status-lifecycle.md#allowed-moves)). The live OddsPortal and Betfair jobs find or create events through `odds_lambda.live_events.LiveEventFinder` and the [event resolver](event-resolver.md) ([live MLB events](event-matching.md#live-mlb-events)).

## Log helpers

`odds_lambda.storage.writers.OddsWriter.log_fetch` adds an `odds_core.models.FetchLog` ([fetch_logs](../collection/odds-ingestion-service.md#fetch-logs)) and `odds_lambda.storage.writers.OddsWriter.log_data_quality_issue` adds an `odds_core.models.DataQualityLog`.

## Related

- [Odds snapshot and odds tables](odds-tables.md)
- [How odds are stored](odds-storage-model.md)
- [Storage layer](storage-layer.md)
- [Odds API ingestion service](../collection/odds-ingestion-service.md)
