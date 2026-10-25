---
type: reference
title: Storage layer
description: The odds_lambda.storage module map, the commit and flush contract, and the upsert/insert/store verbs.
sources:
  - odds_lambda.storage
  - odds_lambda.storage.event_merge.EventMerger
  - odds_lambda.storage.writers.OddsWriter.bulk_upsert_events
  - odds_lambda.storage.readers.OddsReader
  - odds_lambda.storage.validators.OddsValidator
  - odds_lambda.storage.tier_validator.TierCoverageValidator
  - odds_lambda.storage.polymarket_writer.PolymarketWriter
  - odds_lambda.storage.injury_writer.InjuryWriter
  - odds_lambda.storage.mlb_pitcher_writer.MlbPitcherWriter
  - odds_lambda.storage.same_time_pairs.delete_same_time_pairs
  - odds_lambda.storage.injury_writer.InjuryWriter.upsert_injury_reports
  - odds_lambda.storage.mlb_pitcher_writer.MlbPitcherWriter.insert_snapshots
  - odds_lambda.ingestion.OddsIngestionService.ingest_sport
status: stable
---

# Storage layer

`packages/odds-lambda/odds_lambda/storage/` holds every DB access class for the pipeline. `__init__.py` is empty, so no contract is written down in the package; this page is the contract. Each module takes an `AsyncSession` in its constructor and issues SQLAlchemy statements against models from `odds-core`.

## Modules

| Module | Class | Notes |
|---|---|---|
| `odds_lambda.storage.writers` / `odds_lambda.storage.readers` | [`odds_lambda.storage.writers.OddsWriter`](odds-writer.md) / [`odds_lambda.storage.readers.OddsReader`](odds-reader.md) | Events (status and scores, live-twin merges, MLB `gamePk`s), snapshots, odds (stored as decimal prices), fetch logs, including their API quota, and data-quality logs |
| `odds_lambda.storage.event_resolver` | functions | [Event resolver](event-resolver.md): finds or creates a listed game's event under an advisory lock taken in key order, and records its external ids; flushes |
| `odds_lambda.storage.event_kickoff` | functions | [Kick-off authority](kickoff-authority.md): the one function that moves `commence_time`, for the sources in the sport's authority order, against a clock the caller passes (the MLB Stats API alone may move to a past start); flushes when it re-times snapshots |
| `odds_lambda.storage.snapshot_store` | functions | [Storing a snapshot under its key](price-tables.md#storing-a-snapshot-under-its-key): inserts a snapshot and its prices, or keeps the stored one; flushes in a savepoint |
| `odds_lambda.storage.data_quality` | `odds_lambda.storage.data_quality.record_issue` | Logs a warning and adds the matching `data_quality_logs` row, without flushing; used by the resolver, `odds_lambda.storage.snapshot_store` and the merger |
| `odds_lambda.storage.validators` | `odds_lambda.storage.validators.OddsValidator` | [Odds validation](odds-validator.md) (zero-margin books skip the vig floor; prices are checked as decimals) |
| `odds_lambda.storage.same_time_pairs` | functions | [Same-time pairs](odds-tables.md#same-time-pairs): finds them and, through `odds_lambda.storage.same_time_pairs.delete_same_time_pairs`, deletes snapshots and odds rows without committing |
| `odds_lambda.storage.tier_validator` | `odds_lambda.storage.tier_validator.TierCoverageValidator` | [Coverage and completeness checks](coverage-checks.md) |
| `odds_lambda.storage.polymarket_writer` / `odds_lambda.storage.polymarket_reader` | `odds_lambda.storage.polymarket_writer.PolymarketWriter` | [Polymarket](../collection/polymarket.md) (price snapshot inserts skip an existing market and time) |
| `espn_*`, `fpl_availability_*` | per-source pairs | [EPL context](../collection/epl-context-data.md) |
| `mlb_pitcher_*` | `odds_lambda.storage.mlb_pitcher_writer.MlbPitcherWriter` | [MLB pitchers](../collection/mlb-probable-pitchers.md) |
| `injury_*` | `odds_lambda.storage.injury_writer.InjuryWriter` | [Injuries](../collection/nba-injuries-and-stats.md) |
| `game_log_*`, `pbpstats_*` | per-source pairs | [NBA stats](../collection/nba-injuries-and-stats.md) |

## Commit contract

- `odds_lambda.storage.writers.OddsWriter` never calls `commit()`. It only adds, deletes or executes statements, except that `odds_lambda.storage.writers.OddsWriter.store_odds_snapshot` flushes each snapshot so its odds rows can carry its id, and `odds_lambda.storage.writers.OddsWriter.find_or_create_event` writes an MLB event's `gamePk` inside a savepoint, which flushes ([OddsWriter](odds-writer.md#find_or_create_event)). `odds_lambda.storage.event_merge.EventMerger` also leaves committing to its caller for its foreign-key and snapshot-timing `UPDATE`s and its deletes (duplicate events, and the clashing candidate decisions with their trades), flushing only to delete a duplicate before the kept event takes its `gamePk` and to re-time snapshots, and it re-points references to `events.surrogate_id` (external ids, acknowledgements) as well as those to `events.id`; the caller commits, as `odds_lambda.ingestion.OddsIngestionService.ingest_sport` does, once per event.
- Every per-source writer except `odds_lambda.storage.writers.OddsWriter` and the Polymarket snapshot/log methods flushes after its bulk upsert (for example `odds_lambda.storage.injury_writer.InjuryWriter.upsert_injury_reports`) but still does not commit.
- Each unit of work gets its own session, never one shared across tasks ([Databases and environments](../foundations/databases-and-environments.md)).

## Verb conventions

- `upsert_*`: Postgres `ON CONFLICT DO UPDATE`, usually with a dynamically built `set_` excluding the key columns (`odds_lambda.storage.writers.OddsWriter.bulk_upsert_events` also leaves `created_at`, `surrogate_id`, `mlb_game_pk`, `no_game_pk_reason`, status, scores and `completed_at` out). Each per-source writer has its own copy of this loop.
- `insert_*` / `bulk_insert_*`: `ON CONFLICT DO NOTHING` (`odds_lambda.storage.mlb_pitcher_writer.MlbPitcherWriter.insert_snapshots`).
- `store_*`: plain `session.add()`.
- The rule is not enforced anywhere; check the method body before relying on conflict behaviour.

## Related

- [Odds API ingestion service](../collection/odds-ingestion-service.md), the main caller of the writers
- [OddsWriter](odds-writer.md)
- [OddsReader](odds-reader.md)
- [Code conventions](../foundations/code-conventions.md)
