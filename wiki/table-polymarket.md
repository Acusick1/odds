---
type: component
title: Polymarket tables
summary: The five polymarket_* tables - relationships, indexes, order-book derived metrics and storage estimates.
sources:
  - packages/odds-core/odds_core/polymarket_models.py
  - packages/odds-lambda/odds_lambda/polymarket_fetcher.py::process_order_book
  - packages/odds-lambda/odds_lambda/storage/polymarket_writer.py::PolymarketWriter
  - packages/odds-lambda/odds_lambda/storage/polymarket_reader.py::PolymarketReader
  - docs/POLYMARKET.md#data-model
  - docs/POLYMARKET.md#order-book-processing
  - docs/POLYMARKET.md#storage-estimates
  - docs/DATABASE.md#polymarket-tables
verified_at: 7887039
---

# Polymarket tables

Five SQLModel tables in `packages/odds-core/odds_core/polymarket_models.py`, written by `PolymarketWriter` and read by `PolymarketReader` (`packages/odds-lambda/odds_lambda/storage/`). Collection logic is in [Polymarket pipeline](polymarket-pipeline.md).

```
PolymarketEvent 1-N PolymarketMarket 1-N PolymarketPriceSnapshot
                                     1-N PolymarketOrderBookSnapshot
PolymarketEvent.event_id -> events.id   (nullable)
PolymarketFetchLog                       (standalone audit)
```

| Model / table | Key columns |
|---|---|
| `PolymarketEvent` / `polymarket_events` | `pm_event_id` (unique string), `ticker`, `event_id` FK to [events](table-events.md) (NULL until linked), `start_date`, `active`, `closed` |
| `PolymarketMarket` / `polymarket_markets` | `polymarket_event_id` FK, `pm_market_id` (unique), `clob_token_ids` and `outcomes` (JSON lists), `market_type` (`PolymarketMarketType`), `point` |
| `PolymarketPriceSnapshot` / `polymarket_price_snapshots` | `outcome_0_price`, `outcome_1_price`, `fetch_tier`, `hours_until_commence`, optional `best_bid`/`best_ask`/`spread`/`midpoint`/`volume`/`liquidity` |
| `PolymarketOrderBookSnapshot` / `polymarket_orderbook_snapshots` | `token_id`, `raw_book` (JSON), derived metrics below, `fetch_tier` |
| `PolymarketFetchLog` / `polymarket_fetch_logs` | `job_type`, counts, `success`, `error_message` |

Composite indexes: `ix_pm_event_active_closed`, `ix_pm_market_event_type`, `ix_pm_price_market_time`, `ix_pm_price_market_tier`, `ix_pm_orderbook_market_time`. Several single-column indexes also exist (for example `ticker`, `event_id`, `snapshot_time`, `fetch_tier`).

## Order-book metrics

`process_order_book(raw_book)` sorts bids descending and asks ascending (the CLOB returns them unsorted), and returns None if either side is empty or the book is crossed (best bid >= best ask); `PolymarketWriter.store_orderbook_snapshot` then stores nothing. Otherwise: `spread = ask - bid`, `midpoint`, `imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth)` in [-1, 1], `weighted_mid` (top-of-book sizes cross-weighted), `bid_levels`/`ask_levels`, and total depth per side. `raw_book` is stored as received (unsorted).

Both snapshot writers stamp `fetch_tier` and `hours_until_commence` relative to the event's `start_date`. The tier-gated live path writes books for the first outcome token only. Read helpers `get_price_at_time` and `get_orderbook_at_time` return only snapshots at or before the target ([Look-ahead bias](look-ahead-bias.md)).

## Storage estimates

From the doc, not measured here: per NBA game over its roughly 7-day life, about 6,000 price snapshots (~1.2 MB), ~6 order books (~30 KB) and ~45 market rows (~22 KB); about 1.7 GB per regular season.

## Hazards

- `pm_event_id` means the Polymarket string id as a column, but `PolymarketWriter.upsert_markets(pm_event_id=...)` takes the integer `polymarket_events.id`; the same file mixes both.
- Live price snapshots store only `outcome_0_price`/`outcome_1_price`; `best_bid`, `best_ask`, `spread` and `midpoint` are NULL on every row the current pipeline writes. Backfilled rows set `outcome_1_price = 1 - p` rather than observing it.
- No unique constraint on `(polymarket_market_id, snapshot_time)`; `bulk_store_price_history` dedupes by a pre-select, so concurrent writers can duplicate rows.
- Docs describe `polymarket_events` as "NBA game mapping"; nothing in the schema is NBA-specific.

## Related

- [Polymarket pipeline](polymarket-pipeline.md)
- [Table: events](table-events.md)
- [Feature groups](feature-groups.md) - the `polymarket` group reads these tables
