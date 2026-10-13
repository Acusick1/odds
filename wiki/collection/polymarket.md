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
  - experiments/scripts/polymarket_mlb_orderbooks.py
  - packages/odds-lambda/odds_lambda/polymarket_matching.py::parse_ticker
  - deployment/aws/terraform/eventbridge.tf
status: stable
---

# Polymarket pipeline and tables

Collection of Polymarket prices and order books, built for NBA and now deprioritized: UK trading is blocked for all sports (inferred; external fact, no repo evidence). In the early NBA experiments Polymarket correlated 0.9978 with sportsbook prices and the Polymarket features (a 16-column, NBA-only group) changed nothing ([exp-early-nba-signal-and-lstm](../evaluation/exp-early-nba-signal-and-lstm.md)). Per-sport market character is below.

## Market character by sport

Figures were observed from live markets, not computed by code in this repo. The MLB figures come from the probe scripts `experiments/scripts/polymarket_mlb_discover.py`, `experiments/scripts/polymarket_mlb_orderbooks.py`, `experiments/scripts/polymarket_mlb_volume.py` and `experiments/scripts/polymarket_mlb_sentiment.py` (recorded in dff040d; no results file exists under `experiments/results/`). The EPL and NBA figures have no probe script and are (inferred).

| Sport | Observed character |
|---|---|
| EPL | (inferred) Thin, $10K-$100K per match, and AMM-driven: the book reflects automated market-maker parameters, not public sentiment. Rejected for the agent ([agent-data-sources](../agent/agent-data-sources.md)). |
| NBA | (inferred) Deep: $178K-$2.5M USDC moneyline volume per game, about 1 cent spreads, 25-28 levels per side, top-of-book size about $1,100-$1,600. |
| MLB | Deep: $152K median per game, up to $5.8M for playoff games, 1 cent spreads, 32-44 levels per side, median $567K total depth. Prices are collinear with devigged bet365 (mean delta about 1 cent on games with real volume) through arbitrage, not AMM seeding. Trading is discrete: 89% of 5-minute ticks show no movement and the rest jump by at least 1 cent. A sharp-reference candidate, not a mispricing venue. |

Series and tag IDs for other sports were recorded during research and are not in config (inferred, dff040d):

| Sport | Series ID | Tag IDs | Ticker |
|---|---|---|---|
| EPL | `10188` | `82`, `306` | `epl-{away3}-{home3}-YYYY-MM-DD` |
| MLB | `3` | `100639` (game tag) | `mlb-{away3}-{home3}-YYYY-MM-DD` |

Config holds only the NBA series (`10345`) and game tag (`100639`) in `packages/odds-core/odds_core/config.py::PolymarketConfig`.

### Market richness

(Inferred; observed on NBA.) Markets per game grow through the season: about 1 early (moneyline only), 10-15 mid-season (moneyline, several spreads, totals) and about 42 late (adds first-half lines and player props). Spread volume was about $418K per line.

## Components

- `packages/odds-lambda/odds_lambda/polymarket_fetcher.py::PolymarketClient` (`packages/odds-lambda/odds_lambda/polymarket_fetcher.py`): aiohttp, tenacity retry (3 attempts), no auth. Base URLs are `packages/odds-core/odds_core/config.py::PolymarketConfig.gamma_base_url` and `clob_base_url`. Gamma `GET /events` for discovery (`get_nba_events`) sends `series_id` (`packages/odds-core/odds_core/config.py::PolymarketConfig.nba_series_id`), `tag_id` (`game_tag_id`), `active`, `closed`, `limit` and `offset`; offset pagination ends when it returns an empty list. CLOB `GET /price` (`token_id`, `side`), `/midpoint` and `/book` (`token_id`), and `/prices-history` (`market` = token id, `interval`, `fidelity`, optional `startTs`/`endTs`) returning `{history: [{t, p}]}`. Batch calls cap at `packages/odds-lambda/odds_lambda/polymarket_fetcher.py::MAX_CONCURRENT_REQUESTS` (10). The client is sport-agnostic apart from `get_nba_events`; another sport needs its own series/tag IDs and ticker parser.
- `packages/odds-lambda/odds_lambda/polymarket_fetcher.py::classify_market`: moneyline when question equals title; else a `Spread:` regex (point is the signed float), an `O/U` regex (point is the total), a player prop (a stat keyword such as points, rebounds, assists, steals, blocks, threes or turnovers after a colon, plus over/under), else `OTHER`. `point` is None except for spreads and totals.
- `packages/odds-lambda/odds_lambda/polymarket_ingestion.py::PolymarketIngestionService`: `discover_and_upsert_events` then `collect_snapshots`. Tier comes from the single closest active event via `packages/odds-lambda/odds_lambda/tier_utils.py::calculate_tier` ([fetch-tier](../storage/fetch-tier.md)); order books (first outcome token only, moneyline only) are fetched when the tier is in `packages/odds-core/odds_core/config.py::PolymarketConfig.orderbook_tiers`. Market types follow the `collect_*` toggles.
- `packages/odds-lambda/odds_lambda/jobs/fetch_polymarket.py::main`: skipped if `polymarket.enabled` is false; self-schedules after each run at `price_poll_interval` (default 300 s), or 24 h when no events exist.
- `packages/odds-lambda/odds_lambda/jobs/backfill_polymarket.py::main`: pages closed events, skips markets with 10+ snapshots (`get_backfilled_market_ids`), pulls `/prices-history` (`interval=max`, `fidelity=5`) for the first token, and inserts via `packages/odds-lambda/odds_lambda/storage/polymarket_writer.py::PolymarketWriter.bulk_store_price_history`. CLOB history expires after about 30 days ([data availability](#data-availability)), so it must run every few days; closed events are paged 100 at a time with a 100 ms pause after each CLOB call (`packages/odds-lambda/odds_lambda/jobs/backfill_polymarket.py::CLOB_DELAY_MS`). Flags `--include-spreads`, `--include-totals`, `--dry-run` arrive through `packages/odds-lambda/odds_lambda/scheduling/jobs.py::JobContext`.
- Linking: `packages/odds-lambda/odds_lambda/polymarket_matching.py::match_polymarket_event` parses tickers `nba-{away}-{home}-YYYY-MM-DD`, maps abbreviations via `packages/odds-lambda/odds_lambda/polymarket_matching.py::NBA_ABBREV_MAP`, and matches `events` within ±24 h of noon UTC on the ticker date; single match only, else None. Only `odds polymarket link` calls it ([events](../storage/events.md#matchers)).

CLI: `odds polymarket discover` lists active NBA events currently on Polymarket (`packages/odds-cli/odds_cli/commands/polymarket.py::discover`); `odds polymarket status` prints event, linked and unlinked counts, market and snapshot totals, date coverage and the last fetch log (`packages/odds-cli/odds_cli/commands/polymarket.py::status`); `odds polymarket backfill` and `odds polymarket link` run the backfill and linking above (`packages/odds-cli/odds_cli/commands/polymarket.py::link`); `book <ticker>` prints the live order book of that game's moneyline market, with a ticker such as `nba-dal-mil-2026-01-25` validated by `packages/odds-lambda/odds_lambda/polymarket_matching.py::parse_ticker` (`packages/odds-cli/odds_cli/commands/polymarket.py::book`). Env prefix `POLYMARKET_`; defaults are in `packages/odds-core/odds_core/config.py::PolymarketConfig`.

AWS: `enable_polymarket` gates the rules (`deployment/aws/terraform/eventbridge.tf`); the backfill rule is `{rule_prefix}-backfill-polymarket-bootstrap`, `rate(3 days)`. Deployed state is unverifiable from the repo: `odds scheduler list-jobs --backend aws` ([jobs-and-deployment-matrix](../scheduling/jobs-and-deployment-matrix.md)).

## Tables

Five SQLModel tables in `packages/odds-core/odds_core/polymarket_models.py`, written by `packages/odds-lambda/odds_lambda/storage/polymarket_writer.py::PolymarketWriter` and read by `packages/odds-lambda/odds_lambda/storage/polymarket_reader.py::PolymarketReader` (`packages/odds-lambda/odds_lambda/storage/`).

```
PolymarketEvent 1-N PolymarketMarket 1-N PolymarketPriceSnapshot
                                     1-N PolymarketOrderBookSnapshot
PolymarketEvent.event_id -> events.id   (nullable)
PolymarketFetchLog                       (standalone audit)
```

| Model / table | Key columns |
|---|---|
| `packages/odds-core/odds_core/polymarket_models.py::PolymarketEvent` / `polymarket_events` | `pm_event_id` (unique string), `ticker`, `event_id` FK to [events](../storage/events.md) (NULL until linked), `start_date`, `active`, `closed` |
| `packages/odds-core/odds_core/polymarket_models.py::PolymarketMarket` / `polymarket_markets` | `polymarket_event_id` FK, `pm_market_id` (unique), `clob_token_ids` and `outcomes` (JSON lists), `market_type` (`packages/odds-core/odds_core/polymarket_models.py::PolymarketMarketType`), `point` |
| `packages/odds-core/odds_core/polymarket_models.py::PolymarketPriceSnapshot` / `polymarket_price_snapshots` | `outcome_0_price`, `outcome_1_price`, `fetch_tier`, `hours_until_commence`, optional `best_bid`/`best_ask`/`spread`/`midpoint`/`volume`/`liquidity` |
| `packages/odds-core/odds_core/polymarket_models.py::PolymarketOrderBookSnapshot` / `polymarket_orderbook_snapshots` | `token_id`, `raw_book` (JSON), derived metrics below, `fetch_tier` |
| `packages/odds-core/odds_core/polymarket_models.py::PolymarketFetchLog` / `polymarket_fetch_logs` | `job_type`, counts, `success`, `error_message` |

Composite indexes: `ix_pm_event_active_closed`, `ix_pm_market_event_type`, `ix_pm_price_market_time`, `ix_pm_price_market_tier`, `ix_pm_orderbook_market_time`. Several single-column indexes also exist (for example `ticker`, `event_id`, `snapshot_time`, `fetch_tier`).

### Order-book metrics

`process_order_book(raw_book)` sorts bids descending and asks ascending (the CLOB returns them unsorted), and returns None if either side is empty or the book is crossed (best bid >= best ask); `packages/odds-lambda/odds_lambda/storage/polymarket_writer.py::PolymarketWriter.store_orderbook_snapshot` then stores nothing. Otherwise: `spread = ask - bid`, `midpoint`, `imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth)` in [-1, 1], `weighted_mid` (top-of-book sizes cross-weighted), `bid_levels`/`ask_levels`, and total depth per side. `raw_book` is stored as received (unsorted).

Both snapshot writers stamp `fetch_tier` and `hours_until_commence` relative to the event's `start_date`. The tier-gated live path writes books for the first outcome token only. Read helpers `get_price_at_time` and `get_orderbook_at_time` return only snapshots at or before the target, so Polymarket readers are backward-only ([decision-time-and-look-ahead](../modelling/decision-time-and-look-ahead.md)).

### Data availability

Observed from the API, not verifiable from the repo:

| Data | Active markets | Resolved markets |
|---|---|---|
| Event and market metadata (Gamma) | yes | yes, indefinitely |
| Price history (CLOB) | yes, accumulating | about 25 days at full 5-minute resolution, partial from 25 to 31 days (a gradual trim, not a hard cutoff), gone after about 31 days |
| Order book (CLOB) | live only | no (404 after close) |
| Current price and midpoint (CLOB) | yes | no |

A full game lifecycle of about 7 days from market creation to resolution yields about 940-960 moneyline price points. Order-book history cannot be backfilled, so only the live path stores books.

## Hazards

- **Two independent gates.** `packages/odds-core/odds_core/config.py::PolymarketConfig.enabled` defaults to true while Terraform `enable_polymarket` defaults to false ([configuration](../foundations/configuration.md)).
- **No heartbeat, no retry chain.** `fetch-polymarket` and `backfill-polymarket` do not use `packages/odds-core/odds_core/alerts.py::job_alert_context` (no heartbeat). `fetch-polymarket` self-schedules only after success and does not use the shared forward-decision/pre-schedule helpers of the other self-scheduling jobs, so an exception skips the reschedule ([self-scheduling](../scheduling/self-scheduling.md), [alerts-and-health](../operations/alerts-and-health.md#heartbeats)).
- **No auto-linking.** Events are not linked during fetch; `event_id` stays NULL until `odds polymarket link` runs, and backfill never links.
- **NBA-only in code.** Tickers and the API series IDs in config are NBA-only; the EPL and MLB IDs above are research notes. Nothing in the `polymarket_events` schema is NBA-specific.
- **Id naming.** `pm_event_id` is the Polymarket string id as a column, but `PolymarketWriter.upsert_markets(pm_event_id=...)` takes the integer `polymarket_events.id`; the same file mixes both.
- **Price snapshots are thin.** Live snapshots store only `outcome_0_price`/`outcome_1_price`; `best_bid`, `best_ask`, `spread` and `midpoint` are NULL on every row the current pipeline writes. Backfilled rows set `outcome_1_price = 1 - p` rather than observing it.
- **No uniqueness on `(polymarket_market_id, snapshot_time)`;** `bulk_store_price_history` dedupes by a pre-select, so concurrent writers can duplicate rows.

## Related

- [Sources and keys](sources-and-keys.md)
- [Fetch tier](../storage/fetch-tier.md)
- [Feature pipeline](../modelling/feature-pipeline.md), where the `polymarket` group reads these tables
