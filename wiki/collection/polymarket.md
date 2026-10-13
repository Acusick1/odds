---
type: reference
title: Polymarket pipeline and tables
description: Client, classifier, ingestion, fetch and backfill jobs, manual linking, CLI, AWS gate, the five tables and order-book metrics.
sources:
  - odds_lambda.polymarket_fetcher.PolymarketClient
  - odds_lambda.polymarket_fetcher.classify_market
  - odds_lambda.polymarket_fetcher.process_order_book
  - odds_lambda.polymarket_fetcher.MAX_CONCURRENT_REQUESTS
  - odds_lambda.polymarket_ingestion.PolymarketIngestionService
  - odds_lambda.polymarket_matching.match_polymarket_event
  - odds_lambda.polymarket_matching.parse_ticker
  - odds_lambda.polymarket_matching.NBA_ABBREV_MAP
  - odds_lambda.jobs.fetch_polymarket.main
  - odds_lambda.jobs.backfill_polymarket.main
  - odds_lambda.jobs.backfill_polymarket.CLOB_DELAY_MS
  - odds_lambda.storage.polymarket_writer.PolymarketWriter
  - odds_lambda.storage.polymarket_reader.PolymarketReader
  - odds_core.config.PolymarketConfig
  - odds_core.polymarket_models
  - odds_cli.commands.polymarket
  - deployment/aws/terraform/eventbridge.tf
status: stable
---

# Polymarket pipeline and tables

Collection of Polymarket prices and order books, built for NBA. The pipeline is not being extended ([ADR-0007](../../docs/adr/0007-deprioritise-polymarket.md)); what was observed about each sport's markets is on [polymarket-market-research](polymarket-market-research.md).

## Components

- `odds_lambda.polymarket_fetcher.PolymarketClient`: aiohttp, tenacity retry (3 attempts), no auth. Base URLs are `odds_core.config.PolymarketConfig.gamma_base_url` and `clob_base_url`. Gamma `GET /events` for discovery (`get_nba_events`) sends `series_id` (`odds_core.config.PolymarketConfig.nba_series_id`), `tag_id` (`game_tag_id`), `active`, `closed`, `limit` and `offset`; offset pagination ends when it returns an empty list. CLOB `GET /price` (`token_id`, `side`), `/midpoint` and `/book` (`token_id`), and `/prices-history` (`market` = token id, `interval`, `fidelity`, optional `startTs`/`endTs`) returning `{history: [{t, p}]}`. Batch calls cap at `odds_lambda.polymarket_fetcher.MAX_CONCURRENT_REQUESTS` (10). The client is sport-agnostic apart from `get_nba_events`; another sport needs its own series/tag IDs and ticker parser.
- `odds_lambda.polymarket_fetcher.classify_market`: moneyline when question equals title; else a `Spread:` regex (point is the signed float), an `O/U` regex (point is the total), a player prop (a stat keyword such as points, rebounds, assists, steals, blocks, threes or turnovers after a colon, plus over/under), else `OTHER`. `point` is None except for spreads and totals.
- `odds_lambda.polymarket_ingestion.PolymarketIngestionService`: `discover_and_upsert_events` then `collect_snapshots`. Tier comes from the single closest active event via `odds_lambda.tier_utils.calculate_tier` ([fetch-tier](../storage/fetch-tier.md)); order books (first outcome token only, moneyline only) are fetched when the tier is in `odds_core.config.PolymarketConfig.orderbook_tiers`. Market types follow the `collect_*` toggles.
- `odds_lambda.jobs.fetch_polymarket.main`: skipped if `polymarket.enabled` is false; self-schedules after each run at `price_poll_interval` (default 300 s), or 24 h when no events exist.
- `odds_lambda.jobs.backfill_polymarket.main`: pages closed events, skips markets with 10+ snapshots (`get_backfilled_market_ids`), pulls `/prices-history` (`interval=max`, `fidelity=5`) for the first token, and inserts via `odds_lambda.storage.polymarket_writer.PolymarketWriter.bulk_store_price_history`. CLOB history expires after about 30 days ([data availability](polymarket-market-research.md#data-availability)), so it must run every few days; closed events are paged 100 at a time with a 100 ms pause after each CLOB call (`odds_lambda.jobs.backfill_polymarket.CLOB_DELAY_MS`). Flags `--include-spreads`, `--include-totals`, `--dry-run` arrive through `odds_lambda.scheduling.jobs.JobContext`.
- Neither job uses `odds_core.alerts.job_alert_context`, so neither writes a heartbeat. `fetch-polymarket` schedules only after success and does not use the shared forward-decision helpers, so an exception ends its chain (#438; [self-scheduling](../scheduling/self-scheduling.md)).
- Linking: `odds_lambda.polymarket_matching.match_polymarket_event` parses tickers `nba-{away}-{home}-YYYY-MM-DD`, maps abbreviations via `odds_lambda.polymarket_matching.NBA_ABBREV_MAP`, and matches `events` within ±24 h of 12:00 UTC on the ticker date (chosen to cover US evening games that fall on the next UTC day; `pm_start_date` is accepted and ignored); single match only, else None. Only `odds polymarket link` calls it, so events are linked only manually (#456; [event-matching](../storage/event-matching.md)). `event_id` stays NULL until then, and backfill never links.
- NBA-only in code: tickers and the API series IDs in config are NBA-only. Nothing in the `polymarket_events` schema is NBA-specific (#456).
- `PolymarketWriter.upsert_markets(pm_event_id=...)` takes the integer `polymarket_events.id`, while the `pm_event_id` column holds the Polymarket string id.

CLI: `odds polymarket discover` lists active NBA events currently on Polymarket (`odds_cli.commands.polymarket.discover`); `odds polymarket status` prints event, linked and unlinked counts, market and snapshot totals, date coverage and the last fetch log (`odds_cli.commands.polymarket.status`); `odds polymarket backfill` and `odds polymarket link` run the backfill and linking above (`odds_cli.commands.polymarket.link`); `book <ticker>` prints the live order book of that game's moneyline market, with a ticker such as `nba-dal-mil-2026-01-25` validated by `odds_lambda.polymarket_matching.parse_ticker` (`odds_cli.commands.polymarket.book`). Env prefix `POLYMARKET_`; defaults are in `odds_core.config.PolymarketConfig`.

## Gates and deployment

`odds_core.config.PolymarketConfig.enabled` defaults to true, while Terraform `enable_polymarket` defaults to false ([configuration](../foundations/configuration.md)). On AWS, `enable_polymarket` gates the rules (`deployment/aws/terraform/eventbridge.tf`); the backfill rule is `{rule_prefix}-backfill-polymarket-bootstrap`, `rate(3 days)`. Deployed state is unverifiable from the repo: `odds scheduler list-jobs --backend aws` ([job-deployment-matrix](../scheduling/job-deployment-matrix.md#where-each-job-runs)).

## Tables

Five SQLModel tables in `odds_core.polymarket_models`, written by `odds_lambda.storage.polymarket_writer.PolymarketWriter` and read by `odds_lambda.storage.polymarket_reader.PolymarketReader` (`packages/odds-lambda/odds_lambda/storage/`).

```
PolymarketEvent 1-N PolymarketMarket 1-N PolymarketPriceSnapshot
                                     1-N PolymarketOrderBookSnapshot
PolymarketEvent.event_id -> events.id   (nullable)
PolymarketFetchLog                       (standalone audit)
```

| Model / table | Key columns |
|---|---|
| `odds_core.polymarket_models.PolymarketEvent` / `polymarket_events` | `pm_event_id` (unique string), `ticker`, `event_id` FK to [events](../storage/events.md) (NULL until linked), `start_date`, `active`, `closed` |
| `odds_core.polymarket_models.PolymarketMarket` / `polymarket_markets` | `polymarket_event_id` FK, `pm_market_id` (unique), `clob_token_ids` and `outcomes` (JSON lists), `market_type` (`odds_core.polymarket_models.PolymarketMarketType`), `point` |
| `odds_core.polymarket_models.PolymarketPriceSnapshot` / `polymarket_price_snapshots` | `outcome_0_price`, `outcome_1_price`, `fetch_tier`, `hours_until_commence`, optional `best_bid`/`best_ask`/`spread`/`midpoint`/`volume`/`liquidity` |
| `odds_core.polymarket_models.PolymarketOrderBookSnapshot` / `polymarket_orderbook_snapshots` | `token_id`, `raw_book` (JSON), derived metrics below, `fetch_tier` |
| `odds_core.polymarket_models.PolymarketFetchLog` / `polymarket_fetch_logs` | `job_type`, counts, `success`, `error_message` |

Composite indexes: `ix_pm_event_active_closed`, `ix_pm_market_event_type`, `ix_pm_price_market_time`, `ix_pm_price_market_tier`, `ix_pm_orderbook_market_time`. Several single-column indexes also exist (for example `ticker`, `event_id`, `snapshot_time`, `fetch_tier`).

The live pipeline writes only `outcome_0_price`/`outcome_1_price` on price snapshots; `best_bid`, `best_ask`, `spread` and `midpoint` stay NULL (#451). Backfilled rows set `outcome_1_price = 1 - p` rather than observing it. There is no unique key on `(polymarket_market_id, snapshot_time)`; `bulk_store_price_history` dedupes by a pre-select, so concurrent writers can duplicate rows (#451).

### Order-book metrics

`process_order_book(raw_book)` sorts bids descending and asks ascending (the CLOB returns them unsorted), and returns None if either side is empty or the book is crossed (best bid >= best ask); `odds_lambda.storage.polymarket_writer.PolymarketWriter.store_orderbook_snapshot` then stores nothing. Otherwise: `spread = ask - bid`, `midpoint`, `imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth)` in [-1, 1], `weighted_mid` (top-of-book sizes cross-weighted), `bid_levels`/`ask_levels`, and total depth per side. `raw_book` is stored as received (unsorted).

Both snapshot writers stamp `fetch_tier` and `hours_until_commence` relative to the event's `start_date`. The tier-gated live path writes books for the first outcome token only. Read helpers `get_price_at_time` and `get_orderbook_at_time` return only snapshots at or before the target, so Polymarket readers are backward-only ([look-ahead-bias](../modelling/look-ahead-bias.md)).

## Related

- [Polymarket market character and data availability](polymarket-market-research.md)
- [Data sources](data-sources.md)
- [Fetch tier](../storage/fetch-tier.md)
- [Feature groups](../modelling/feature-groups.md), where the `polymarket` group reads these tables
