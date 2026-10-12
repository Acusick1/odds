---
type: component
title: Storage layer (readers and writers)
summary: "Map of odds_lambda/storage/: one reader/writer pair per source, the commit/flush contract, upsert verb conventions and known drift between copies."
sources:
  - packages/odds-lambda/odds_lambda/storage/__init__.py
  - packages/odds-lambda/odds_lambda/storage/writers.py::OddsWriter
  - packages/odds-lambda/odds_lambda/storage/readers.py::OddsReader
  - docs/ARCHITECTURE.md#storage-modules
verified_at: 7887039
---

# Storage layer (readers and writers)

`packages/odds-lambda/odds_lambda/storage/` holds every DB access class for the pipeline. `__init__.py` is empty: no contract is written down in the package, so this page is the contract. Each module takes an `AsyncSession` in its constructor and issues SQLAlchemy statements against models from `odds-core`.

## Modules

| Module | Class | Notes |
|---|---|---|
| `writers.py` / `readers.py` | [OddsWriter](odds-writer.md) / [OddsReader](odds-reader.md) | Events, snapshots, odds, fetch and quality logs |
| `validators.py` | `OddsValidator` | [Validation](data-quality-validation.md) |
| `tier_validator.py` | `TierCoverageValidator` | [Completeness](completeness-validation.md) |
| `polymarket_writer.py` / `polymarket_reader.py` | `PolymarketWriter` | [Polymarket](polymarket-pipeline.md) |
| `espn_*`, `fpl_availability_*` | per-source pairs | [EPL context](epl-context-data.md) |
| `mlb_pitcher_*` | `MlbPitcherWriter` | [MLB pitchers](mlb-probable-pitchers.md) |
| `injury_*` | `InjuryWriter` | [Injuries](nba-injury-reports.md) |
| `game_log_*`, `pbpstats_*` | per-source pairs | [NBA stats](nba-stats-data.md) |

## Commit contract

- `OddsWriter` never calls `flush()` or `commit()` (`packages/odds-lambda/odds_lambda/storage/writers.py::OddsWriter`). It only `session.add()`s or executes statements; the caller commits, as `packages/odds-lambda/odds_lambda/ingestion.py::OddsIngestionService.ingest_sport` does after each event.
- Every per-source writer except `OddsWriter` and the Polymarket snapshot/log methods flushes after its bulk upsert (for example `packages/odds-lambda/odds_lambda/storage/injury_writer.py::InjuryWriter.upsert_injury_reports`) but still does not commit.
- Pass a dedicated session per unit of work; never share one across tasks (see [database connections](database-connections.md)).

## Verb conventions (as implemented)

- `upsert_*`: Postgres `ON CONFLICT DO UPDATE`, usually with a dynamically built `set_` excluding the key columns. Each per-source writer has its own copy of this loop.
- `insert_*` / `bulk_insert_*`: `ON CONFLICT DO NOTHING` (`packages/odds-lambda/odds_lambda/storage/mlb_pitcher_writer.py::MlbPitcherWriter.insert_snapshots`, `OddsWriter.bulk_insert_odds`).
- `store_*`: plain `session.add()`.
- The rule is not enforced anywhere; check the method body before relying on conflict behaviour.

## Reader routing

`OddsReader` mixes two data paths: methods over the normalized `odds` table and methods that re-parse `odds_snapshots.raw_data`. See [OddsReader](odds-reader.md#which-path-each-method-uses).

## Hazards

- Event matching is implemented separately in `OddsWriter.find_or_create_event` (plus/minus 2 h, first candidate wins on ambiguity), `InjuryWriter._match_event` and `GameLogWriter._match_event` (`packages/odds-lambda/odds_lambda/storage/game_log_writer.py::GameLogWriter`); windows and ambiguity handling are not shared.
- The raw to `Odds` parser exists twice with different price handling: `OddsWriter._parse_and_store_odds` coerces with `int(outcome.get("price", 0))` (an absent price becomes American odds 0), while `packages/odds-core/odds_core/snapshot_utils.py::extract_odds_from_snapshot` passes the price through unchanged.
- `raw_data["source"]` vocabulary is documented only in [odds_snapshots](table-odds-snapshots.md#source-tags).

## Related

- [Odds ingestion flow](odds-ingestion-flow.md) — the main caller of the writers
- [Code conventions](code-conventions.md)
