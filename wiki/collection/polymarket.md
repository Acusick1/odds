---
type: component
title: Polymarket pipeline and tables
description: "The deprioritized, NBA-targeted Polymarket collection: client, ingestion, fetch and backfill jobs, manual event linking, the five polymarket_* tables and order-book metrics."
sources:
  - packages/odds-lambda/odds_lambda/polymarket_fetcher.py::PolymarketClient
  - packages/odds-lambda/odds_lambda/polymarket_fetcher.py::classify_market
  - packages/odds-lambda/odds_lambda/polymarket_fetcher.py::process_order_book
  - packages/odds-lambda/odds_lambda/polymarket_ingestion.py::PolymarketIngestionService
  - packages/odds-lambda/odds_lambda/polymarket_matching.py::match_polymarket_event
  - packages/odds-lambda/odds_lambda/jobs/fetch_polymarket.py::main
  - packages/odds-lambda/odds_lambda/jobs/backfill_polymarket.py::main
  - packages/odds-lambda/odds_lambda/storage/polymarket_writer.py::PolymarketWriter
  - packages/odds-lambda/odds_lambda/storage/polymarket_reader.py::PolymarketReader
  - packages/odds-core/odds_core/config.py::PolymarketConfig
  - packages/odds-core/odds_core/polymarket_models.py
  - packages/odds-cli/odds_cli/commands/polymarket.py
  - experiments/scripts/polymarket_mlb_volume.py
  - deployment/aws/terraform/eventbridge.tf
  - docs/POLYMARKET.md#status
  - docs/POLYMARKET.md#pipeline
  - docs/POLYMARKET.md#api-reference
  - docs/POLYMARKET.md#market-type-classification
  - docs/POLYMARKET.md#configuration
  - docs/POLYMARKET.md#data-model
  - docs/POLYMARKET.md#order-book-processing
  - docs/POLYMARKET.md#storage-estimates
  - docs/DATABASE.md#polymarket-tables
  - docs/CLI.md#polymarket-commands-deprioritized
verified: "7887039"
status: stable
---

# Polymarket pipeline and tables

Collection of Polymarket prices and order books, built for NBA and now deprioritized: UK trading is blocked for all sports. Per sport (from `docs/POLYMARKET.md#status` and the `experiments/scripts/polymarket_mlb_*.py` probes): EPL is thin and AMM-driven and was rejected for the agent ([agent-data-sources](../agent/agent-data-sources.md)); NBA and MLB are deep with about 1 cent spreads; MLB tracks devigged bet365 through arbitrage, so it is a sharp-reference candidate, not a mispricing venue. In the early NBA experiments Polymarket correlated 0.9978 with sportsbook prices and the Polymarket features (a 16-column, NBA-only group) changed nothing ([exp-early-nba-signal-and-lstm](../evaluation/exp-early-nba-signal-and-lstm.md)).

## Components

- `PolymarketClient` (`polymarket_fetcher.py`): aiohttp, tenacity retry (3 attempts), no auth. Gamma `/events` for discovery (`get_nba_events`, uses `PolymarketConfig.nba_series_id` and `game_tag_id`); CLOB `/price`, `/midpoint`, `/book`, `/prices-history`. Batch calls cap at `MAX_CONCURRENT_REQUESTS` (10). The client is sport-agnostic apart from `get_nba_events`; another sport needs its own series/tag IDs and ticker parser.
- `classify_market(question, event_title)`: moneyline when question equals title; else `Spread:` regex, `O/U` regex, player prop (stat keyword after a colon plus over/under), else `OTHER`.
- `PolymarketIngestionService`: `discover_and_upsert_events` then `collect_snapshots`. Tier comes from the single closest active event via `calculate_tier` ([fetch-tier](../storage/fetch-tier.md)); order books (first outcome token only, moneyline only) are fetched when the tier is in `PolymarketConfig.orderbook_tiers`. Market types follow the `collect_*` toggles.
- `fetch_polymarket.py::main`: skipped if `polymarket.enabled` is false; self-schedules after each run at `price_poll_interval` (default 300 s), or 24 h when no events exist.
- `backfill_polymarket.py::main`: pages closed events, skips markets with 10+ snapshots (`get_backfilled_market_ids`), pulls `/prices-history` (`interval=max`, `fidelity=5`) for the first token, and inserts via `PolymarketWriter.bulk_store_price_history`. CLOB history expires after about 30 days, so it must run every few days. Flags `--include-spreads`, `--include-totals`, `--dry-run` arrive through `JobContext`.
- Linking: `match_polymarket_event` parses tickers `nba-{away}-{home}-YYYY-MM-DD`, maps abbreviations via `NBA_ABBREV_MAP`, and matches `events` within ±24 h of noon UTC on the ticker date; single match only, else None. Only `odds polymarket link` calls it ([events](../storage/events.md#matchers)).

CLI: `odds polymarket discover|status|backfill|link|book`. Env prefix `POLYMARKET_`; defaults are in `PolymarketConfig`.

AWS: `enable_polymarket` gates the rules (`deployment/aws/terraform/eventbridge.tf`); the backfill rule is `{rule_prefix}-backfill-polymarket-bootstrap`, `rate(3 days)`. Deployed state is unverifiable from the repo: `odds scheduler list-jobs --backend aws` ([jobs-and-deployment-matrix](../scheduling/jobs-and-deployment-matrix.md)).

## Tables

Five SQLModel tables in `packages/odds-core/odds_core/polymarket_models.py`, written by `PolymarketWriter` and read by `PolymarketReader` (`packages/odds-lambda/odds_lambda/storage/`).

```
PolymarketEvent 1-N PolymarketMarket 1-N PolymarketPriceSnapshot
                                     1-N PolymarketOrderBookSnapshot
PolymarketEvent.event_id -> events.id   (nullable)
PolymarketFetchLog                       (standalone audit)
```

| Model / table | Key columns |
|---|---|
| `PolymarketEvent` / `polymarket_events` | `pm_event_id` (unique string), `ticker`, `event_id` FK to [events](../storage/events.md) (NULL until linked), `start_date`, `active`, `closed` |
| `PolymarketMarket` / `polymarket_markets` | `polymarket_event_id` FK, `pm_market_id` (unique), `clob_token_ids` and `outcomes` (JSON lists), `market_type` (`PolymarketMarketType`), `point` |
| `PolymarketPriceSnapshot` / `polymarket_price_snapshots` | `outcome_0_price`, `outcome_1_price`, `fetch_tier`, `hours_until_commence`, optional `best_bid`/`best_ask`/`spread`/`midpoint`/`volume`/`liquidity` |
| `PolymarketOrderBookSnapshot` / `polymarket_orderbook_snapshots` | `token_id`, `raw_book` (JSON), derived metrics below, `fetch_tier` |
| `PolymarketFetchLog` / `polymarket_fetch_logs` | `job_type`, counts, `success`, `error_message` |

Composite indexes: `ix_pm_event_active_closed`, `ix_pm_market_event_type`, `ix_pm_price_market_time`, `ix_pm_price_market_tier`, `ix_pm_orderbook_market_time`. Several single-column indexes also exist (for example `ticker`, `event_id`, `snapshot_time`, `fetch_tier`).

### Order-book metrics

`process_order_book(raw_book)` sorts bids descending and asks ascending (the CLOB returns them unsorted), and returns None if either side is empty or the book is crossed (best bid >= best ask); `PolymarketWriter.store_orderbook_snapshot` then stores nothing. Otherwise: `spread = ask - bid`, `midpoint`, `imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth)` in [-1, 1], `weighted_mid` (top-of-book sizes cross-weighted), `bid_levels`/`ask_levels`, and total depth per side. `raw_book` is stored as received (unsorted).

Both snapshot writers stamp `fetch_tier` and `hours_until_commence` relative to the event's `start_date`. The tier-gated live path writes books for the first outcome token only. Read helpers `get_price_at_time` and `get_orderbook_at_time` return only snapshots at or before the target, so Polymarket readers are backward-only ([decision-time-and-look-ahead](../modelling/decision-time-and-look-ahead.md)).

### Storage estimates

From the doc, not measured here: per NBA game over its roughly 7-day life, about 6,000 price snapshots (~1.2 MB), ~6 order books (~30 KB) and ~45 market rows (~22 KB); about 1.7 GB per regular season.

## Hazards

- **Two independent gates.** `PolymarketConfig.enabled` defaults to true while Terraform `enable_polymarket` defaults to false ([configuration](../foundations/configuration.md)).
- **No heartbeat, no retry chain.** `fetch-polymarket` and `backfill-polymarket` do not use `job_alert_context` (no heartbeat). `fetch-polymarket` self-schedules only after success and does not use the shared forward-decision/pre-schedule helpers of the other self-scheduling jobs, so an exception skips the reschedule ([self-scheduling](../scheduling/self-scheduling.md), [alerts-and-health](../operations/alerts-and-health.md#heartbeats)).
- **No auto-linking.** Events are not linked during fetch; `event_id` stays NULL until `odds polymarket link` runs, and backfill never links.
- **Doc gap.** `docs/CLI.md` omits `odds polymarket book`.
- **NBA-only in code.** Tickers and the API series IDs are NBA-only; EPL/MLB IDs in the doc are research notes, not config. Docs describe `polymarket_events` as "NBA game mapping"; nothing in the schema is NBA-specific.
- **Id naming.** `pm_event_id` is the Polymarket string id as a column, but `PolymarketWriter.upsert_markets(pm_event_id=...)` takes the integer `polymarket_events.id`; the same file mixes both.
- **Price snapshots are thin.** Live snapshots store only `outcome_0_price`/`outcome_1_price`; `best_bid`, `best_ask`, `spread` and `midpoint` are NULL on every row the current pipeline writes. Backfilled rows set `outcome_1_price = 1 - p` rather than observing it.
- **No uniqueness on `(polymarket_market_id, snapshot_time)`;** `bulk_store_price_history` dedupes by a pre-select, so concurrent writers can duplicate rows.

## Related

- [Sources and keys](sources-and-keys.md)
- [Fetch tier](../storage/fetch-tier.md)
- [Feature pipeline](../modelling/feature-pipeline.md), where the `polymarket` group reads these tables
