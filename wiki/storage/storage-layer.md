---
type: reference
title: Storage layer
description: The odds_lambda.storage module map, the commit and flush contract, and the upsert/insert/store verbs.
sources:
  - odds_lambda.storage
  - odds_lambda.storage.writers.OddsWriter.adopt_live_twins
  - odds_lambda.storage.writers.OddsWriter.bulk_upsert_events
  - odds_lambda.storage.readers.OddsReader
  - odds_lambda.storage.validators.OddsValidator
  - odds_lambda.storage.tier_validator.TierCoverageValidator
  - odds_lambda.storage.polymarket_writer.PolymarketWriter
  - odds_lambda.storage.injury_writer.InjuryWriter
  - odds_lambda.storage.mlb_pitcher_writer.MlbPitcherWriter
  - odds_lambda.storage.writers.OddsWriter.bulk_insert_odds
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
| `odds_lambda.storage.writers` / `odds_lambda.storage.readers` | [`odds_lambda.storage.writers.OddsWriter`](odds-writer.md) / [`odds_lambda.storage.readers.OddsReader`](odds-reader.md) | Events, snapshots, odds, fetch and quality logs |
| `odds_lambda.storage.validators` | `odds_lambda.storage.validators.OddsValidator` | [Odds validation](odds-validator.md) |
| `odds_lambda.storage.tier_validator` | `odds_lambda.storage.tier_validator.TierCoverageValidator` | [Coverage and completeness checks](coverage-checks.md) |
| `odds_lambda.storage.polymarket_writer` / `odds_lambda.storage.polymarket_reader` | `odds_lambda.storage.polymarket_writer.PolymarketWriter` | [Polymarket](../collection/polymarket.md) |
| `espn_*`, `fpl_availability_*` | per-source pairs | [EPL context](../collection/epl-context-data.md) |
| `mlb_pitcher_*` | `odds_lambda.storage.mlb_pitcher_writer.MlbPitcherWriter` | [MLB pitchers](../collection/mlb-probable-pitchers.md) |
| `injury_*` | `odds_lambda.storage.injury_writer.InjuryWriter` | [Injuries](../collection/nba-injuries-and-stats.md) |
| `game_log_*`, `pbpstats_*` | per-source pairs | [NBA stats](../collection/nba-injuries-and-stats.md) |

## Commit contract

- `odds_lambda.storage.writers.OddsWriter` never calls `flush()` or `commit()`. It only adds, deletes or executes statements, including the foreign-key `UPDATE`s of `odds_lambda.storage.writers.OddsWriter.adopt_live_twins`; the caller commits, as `odds_lambda.ingestion.OddsIngestionService.ingest_sport` does once per event.
- Every per-source writer except `odds_lambda.storage.writers.OddsWriter` and the Polymarket snapshot/log methods flushes after its bulk upsert (for example `odds_lambda.storage.injury_writer.InjuryWriter.upsert_injury_reports`) but still does not commit.
- Each unit of work gets its own session, never one shared across tasks ([Databases and environments](../foundations/databases-and-environments.md)).

## Verb conventions

- `upsert_*`: Postgres `ON CONFLICT DO UPDATE`, usually with a dynamically built `set_` excluding the key columns. Each per-source writer has its own copy of this loop.
- `insert_*` / `bulk_insert_*`: `ON CONFLICT DO NOTHING` (`odds_lambda.storage.mlb_pitcher_writer.MlbPitcherWriter.insert_snapshots`, `odds_lambda.storage.writers.OddsWriter.bulk_insert_odds`).
- `store_*`: plain `session.add()`.
- The rule is not enforced anywhere; check the method body before relying on conflict behaviour.

## Related

- [Odds API ingestion service](../collection/odds-ingestion-service.md), the main caller of the writers
- [OddsWriter](odds-writer.md)
- [OddsReader](odds-reader.md)
- [Code conventions](../foundations/code-conventions.md)
