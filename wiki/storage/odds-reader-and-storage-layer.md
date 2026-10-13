---
type: component
title: OddsReader and the storage layer
description: "Map of odds_lambda/storage/, the commit/flush contract and upsert verbs, and which OddsReader queries read normalized rows versus re-parsed JSON, including the symmetric get_odds_at_time window."
sources:
  - packages/odds-lambda/odds_lambda/storage/__init__.py
  - packages/odds-lambda/odds_lambda/storage/writers.py::OddsWriter
  - packages/odds-lambda/odds_lambda/storage/readers.py::OddsReader
  - packages/odds-lambda/odds_lambda/storage/injury_writer.py::InjuryWriter
  - packages/odds-lambda/odds_lambda/storage/game_log_writer.py::GameLogWriter
  - packages/odds-lambda/odds_lambda/storage/mlb_pitcher_writer.py::MlbPitcherWriter
  - packages/odds-lambda/odds_lambda/ingestion.py::OddsIngestionService
  - packages/odds-core/odds_core/snapshot_utils.py::extract_odds_from_snapshot
  - packages/odds-analytics/odds_analytics/backtesting/services.py
status: stable
---

# OddsReader and the storage layer

`packages/odds-lambda/odds_lambda/storage/` holds every DB access class for the pipeline. `__init__.py` is empty, so no contract is written down in the package; this page is the contract. Each module takes an `AsyncSession` in its constructor and issues SQLAlchemy statements against models from `odds-core`. `packages/odds-lambda/odds_lambda/storage/readers.py::OddsReader` is the read side for odds.

## Modules

| Module | Class | Notes |
|---|---|---|
| `packages/odds-lambda/odds_lambda/storage/writers.py` / `packages/odds-lambda/odds_lambda/storage/readers.py` | [OddsWriter](odds-storage.md) / `packages/odds-lambda/odds_lambda/storage/readers.py::OddsReader` | Events, snapshots, odds, fetch and quality logs |
| `packages/odds-lambda/odds_lambda/storage/validators.py` | `packages/odds-lambda/odds_lambda/storage/validators.py::OddsValidator` | [Validation](data-validation.md) |
| `packages/odds-lambda/odds_lambda/storage/tier_validator.py` | `packages/odds-lambda/odds_lambda/storage/tier_validator.py::TierCoverageValidator` | [Completeness](data-validation.md#completeness-and-coverage-checks) |
| `packages/odds-lambda/odds_lambda/storage/polymarket_writer.py` / `packages/odds-lambda/odds_lambda/storage/polymarket_reader.py` | `packages/odds-lambda/odds_lambda/storage/polymarket_writer.py::PolymarketWriter` | [Polymarket](../collection/polymarket.md) |
| `espn_*`, `fpl_availability_*` | per-source pairs | [EPL context](../collection/epl-context-data.md) |
| `mlb_pitcher_*` | `packages/odds-lambda/odds_lambda/storage/mlb_pitcher_writer.py::MlbPitcherWriter` | [MLB pitchers](../collection/mlb-probable-pitchers.md) |
| `injury_*` | `packages/odds-lambda/odds_lambda/storage/injury_writer.py::InjuryWriter` | [Injuries](../collection/nba-injuries-and-stats.md) |
| `game_log_*`, `pbpstats_*` | per-source pairs | [NBA stats](../collection/nba-injuries-and-stats.md) |

## Commit contract

- `packages/odds-lambda/odds_lambda/storage/writers.py::OddsWriter` never calls `flush()` or `commit()`. It only `session.add()`s or executes statements; the caller commits, as `packages/odds-lambda/odds_lambda/ingestion.py::OddsIngestionService.ingest_sport` does after each event.
- Every per-source writer except `packages/odds-lambda/odds_lambda/storage/writers.py::OddsWriter` and the Polymarket snapshot/log methods flushes after its bulk upsert (for example `packages/odds-lambda/odds_lambda/storage/injury_writer.py::InjuryWriter.upsert_injury_reports`) but still does not commit.
- Pass a dedicated session per unit of work; never share one across tasks ([databases-and-environments](../foundations/databases-and-environments.md)).

## Verb conventions (as implemented)

- `upsert_*`: Postgres `ON CONFLICT DO UPDATE`, usually with a dynamically built `set_` excluding the key columns. Each per-source writer has its own copy of this loop.
- `insert_*` / `bulk_insert_*`: `ON CONFLICT DO NOTHING` (`packages/odds-lambda/odds_lambda/storage/mlb_pitcher_writer.py::MlbPitcherWriter.insert_snapshots`, `packages/odds-lambda/odds_lambda/storage/writers.py::OddsWriter.bulk_insert_odds`).
- `store_*`: plain `session.add()`.
- The rule is not enforced anywhere; check the method body before relying on conflict behaviour.

## OddsReader

One class, 20 async methods, built on an `AsyncSession`. Also defines `packages/odds-lambda/odds_lambda/storage/readers.py::BookPriceResult`, `packages/odds-lambda/odds_lambda/storage/readers.py::BookPriceEntry` and `packages/odds-lambda/odds_lambda/storage/readers.py::BookPriceMeta` dataclasses.

### Which path each method uses

**Normalized `odds` rows** ([odds-storage](odds-storage.md)): `get_odds_at_time`, `get_line_movement`, `get_best_odds`.

**Re-parsed `raw_data`**, via `packages/odds-core/odds_core/snapshot_utils.py::extract_odds_from_snapshot` and `packages/odds-core/odds_core/utils.py::raw_data_has_market`: `get_sharp_prices`, `get_latest_book_prices`, `get_latest_snapshot(market=...)`. These return transient `packages/odds-core/odds_core/models.py::Odds` objects with `id=None`. Prices pass through unchanged, unlike the writer's `int(...)` coercion.

**Snapshot and event accessors**: `get_snapshots_by_tier`, `get_first_snapshot_in_tier`, `get_last_snapshot_in_tier`, `get_tier_coverage_for_event`, `get_snapshot_count_for_event`, `get_snapshots_for_event`, `snapshot_exists`, `get_events_by_date_range` (optional `event_id_pattern` LIKE and `min_snapshots`), `get_events_by_team`, `get_event_by_id`.

**Operational**: `get_fetch_logs`, `get_data_quality_logs`, `get_database_stats`, `get_games_by_date`.

### Get odds at time

`get_odds_at_time(event_id, timestamp, tolerance_minutes=5)` picks the distinct `odds_timestamp` closest to the target within plus/minus the tolerance and returns all rows at it; empty list if none. The window is symmetric: a snapshot up to `tolerance_minutes` after `timestamp` can win if it is closer.

This method is not a look-ahead guard: the backtester calls it with `tolerance_minutes=30` around a 1 h decision time (`packages/odds-analytics/odds_analytics/backtesting/services.py`), so post-decision odds can win. See [decision-time-and-look-ahead](../modelling/decision-time-and-look-ahead.md#hazards). Events from historical OddsPortal/FDUK ingests have no normalized rows, so it returns nothing for them.

### Other behaviour

- `get_best_odds` orders by `price DESC`; with `timestamp` it filters to a plus/minus 5 minute window and does not pin one snapshot; without it, uses the event's latest `odds_timestamp`.
- `get_sharp_prices(event_id, market, sharp_bookmakers, lookback_hours=2.0, now=None)` walks snapshots newest-first and takes the first sharp bookmaker in priority order per outcome; returns `packages/odds-core/odds_core/match_brief_models.py::SharpPriceResult` with age metadata ([sharp-benchmark](../modelling/sharp-benchmark.md)). Resolution is per outcome, so one response can mix Pinnacle and Betfair.
- `get_latest_book_prices` keeps the newest price per `(bookmaker_key, outcome_name, point)` across the window, so partial-book snapshots (e.g. Betfair-only) do not hide retail books. Its ages are relative to the anchor snapshot, not wall clock. Used by the MCP server ([mcp-market-tools](../agent/mcp-market-tools.md)).
- `get_database_stats()` returns `events_by_status`, `total_events`, `total_odds_records`, `total_snapshots`, `fetch_success_rate_24h`, `api_quota_remaining`; used by `odds status` and the [health monitor](../operations/alerts-and-health.md). `api_quota_remaining` is the newest non-null quota, whereas `get_fetch_logs` does not skip nulls ([odds-api-client-and-quota](../collection/odds-api-client-and-quota.md)).
- `get_games_by_date(target_date, status=FINAL)` uses a noon-to-noon UTC window and has no sport filter, so it drives `validate daily` for every sport ([data-validation](data-validation.md)).

## Hazards

- **Look-ahead.** The symmetric `get_odds_at_time` window above.
- **Event matching is not shared.** `packages/odds-lambda/odds_lambda/storage/writers.py::OddsWriter.find_or_create_event` (±2 h, first candidate wins on ambiguity), `packages/odds-lambda/odds_lambda/storage/injury_writer.py::InjuryWriter._match_event` and `GameLogWriter._match_event` have independent windows and ambiguity handling ([events](events.md#matchers)).
- **Two raw-to-`packages/odds-core/odds_core/models.py::Odds` parsers.** `packages/odds-lambda/odds_lambda/storage/writers.py::OddsWriter._parse_and_store_odds` coerces with `int(outcome.get("price", 0))` (an absent price becomes American odds 0), while `packages/odds-core/odds_core/snapshot_utils.py::extract_odds_from_snapshot` passes the price through unchanged.
- **Naming.** `get_database_stats` is the only `*_stats` on this class; sibling readers use `get_pipeline_stats`. "Snapshot" means an `packages/odds-core/odds_core/models.py::OddsSnapshot` row here, a list of `packages/odds-core/odds_core/models.py::Odds` at a time in `get_odds_at_time`, and something else in Polymarket.
- **`source` vocabulary** is documented only in [odds-storage](odds-storage.md#source-tags).

## Related

- [Odds API ingestion](../collection/odds-api-ingestion.md), the main caller of the writers
- [Odds snapshots, odds rows and OddsWriter](odds-storage.md)
- [Conventions](../foundations/conventions.md)
- [CLV target](../modelling/clv-target.md) and [backtest strategies](../evaluation/backtest-strategies.md) consume these queries
