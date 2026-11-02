---
type: reference
title: OddsWriter
description: store_envelope (validation, one snapshot per market with its price rows), event status updates, logging helpers, and their edge cases.
sources:
  - odds_lambda.storage.writers.OddsWriter
  - odds_lambda.storage.writers.OddsWriter.store_envelope
  - odds_lambda.storage.writers.StoredEnvelope
  - odds_lambda.storage.shaped_snapshots.SnapshotShape
  - odds_lambda.storage.shaped_snapshots.store_shaped_snapshot
  - odds_lambda.storage.writers.OddsWriter.log_fetch
  - odds_lambda.storage.writers.OddsWriter.log_data_quality_issue
  - odds_lambda.ingestion.OddsIngestionService
  - odds_core.config.DataQualityConfig
  - odds_core.models.DataQualityLog
  - odds_core.models.FetchLog
status: stable
---

# OddsWriter

`odds_lambda.storage.writers.OddsWriter` owns an `AsyncSession` and an `odds_lambda.storage.validators.OddsValidator` (its vig floor skips zero-margin books, `odds_core.bookmakers.ZERO_MARGIN_BOOKMAKERS`), which checks the one envelope being stored, logging what it finds, and never blocks the write. It never commits; storing a snapshot flushes it to give its price rows its id, and callers commit ([Commit contract](storage-layer.md#commit-contract)). The tables it writes are on [Odds snapshots table](odds-tables.md), [Prices and snapshot payloads tables](price-tables.md) and [Events table](events.md). Live jobs find the event a snapshot belongs to through the [event resolver](event-resolver.md) before storing it.

## store_envelope

`odds_lambda.storage.writers.OddsWriter.store_envelope(event_id, envelope, snapshot_time=None, validate=True, *, shape)` stores one event's envelope, `event_id` being the event's integer id ([envelope format](odds-tables.md#envelope-format)). `shape` (`odds_lambda.storage.shaped_snapshots.SnapshotShape`) names the snapshot's source, its [time basis](odds-tables.md#odds_snapshots) and the payload it came in. Its callers are the OddsPortal live, Betfair and live Polymarket jobs (through `odds_lambda.live_events.LiveEventFinder.store_listings` for the first two), Odds API ingestion and the Odds API backfill. Steps:

1. Loads the event; a missing event raises `ValueError`, since outcomes are matched to its teams. `snapshot_time` defaults to now.
2. If `validate`, runs the validator and writes one `odds_core.models.DataQualityLog` row for any warnings; storage proceeds regardless ([Odds validation](odds-validator.md)). `odds_lambda.ingestion.OddsIngestionService.ingest_sport` is the only caller that passes `validate`: the `validate` argument of that method or of `odds_lambda.ingestion.OddsIngestionService.ingest_sports` when given, else `odds_core.config.DataQualityConfig.enable_validation`. Every other caller uses the default `True`.
3. Stores the envelope through `odds_lambda.storage.shaped_snapshots.store_shaped_snapshot`: one snapshot per market with its price rows, each skipped when a snapshot already holds its key (event, source, market, time) ([Writing price rows](price-tables.md#writing-price-rows)). When none is new, `odds_snapshot_already_stored` is logged.

It returns an `odds_lambda.storage.writers.StoredEnvelope`: the stored snapshot of the envelope's first market, whether any of the envelope's snapshots was inserted, and `snapshots_written`, the number inserted (one for an envelope quoting no market). No tier or hours to kick-off is stored ([Fetch tier](fetch-tier.md)).

## Event handling

`odds_lambda.storage.writers.OddsWriter.update_event_status` is on [Events table](events.md#writing-events); it raises `ValueError` for FINAL without both scores and for a move the stored status does not allow ([Event status lifecycle](../collection/event-status-lifecycle.md#allowed-moves)). The writer creates no events: the live OddsPortal and Betfair jobs and the Odds API writers find or create them through `odds_lambda.live_events.LiveEventFinder` and the [event resolver](event-resolver.md) ([live MLB events](event-matching.md#live-mlb-events), [Odds API events](event-matching.md#odds-api-events)).

## Log helpers

`odds_lambda.storage.writers.OddsWriter.log_fetch` adds an `odds_core.models.FetchLog` (with its remaining and used quota; [fetch_logs](../collection/odds-ingestion-service.md#fetch-logs)) and `odds_lambda.storage.writers.OddsWriter.log_data_quality_issue` adds an `odds_core.models.DataQualityLog`.

## Related

- [Odds snapshots table](odds-tables.md)
- [How odds are stored](odds-storage-model.md)
- [Storage layer](storage-layer.md)
- [Odds API ingestion service](../collection/odds-ingestion-service.md)
