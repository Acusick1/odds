---
type: component
title: Polymarket pipeline
summary: Deprioritized, NBA-targeted Polymarket collection - Gamma/CLOB client, ingestion service, fetch-polymarket and backfill-polymarket jobs, ticker-based event linking, config and market status per sport.
sources:
  - packages/odds-lambda/odds_lambda/polymarket_fetcher.py::PolymarketClient
  - packages/odds-lambda/odds_lambda/polymarket_fetcher.py::classify_market
  - packages/odds-lambda/odds_lambda/polymarket_ingestion.py::PolymarketIngestionService
  - packages/odds-lambda/odds_lambda/polymarket_matching.py::match_polymarket_event
  - packages/odds-lambda/odds_lambda/jobs/fetch_polymarket.py::main
  - packages/odds-lambda/odds_lambda/jobs/backfill_polymarket.py::main
  - packages/odds-core/odds_core/config.py::PolymarketConfig
  - packages/odds-cli/odds_cli/commands/polymarket.py
  - experiments/scripts/polymarket_mlb_volume.py
  - docs/POLYMARKET.md#status
  - docs/POLYMARKET.md#pipeline
  - docs/POLYMARKET.md#api-reference
  - docs/POLYMARKET.md#market-type-classification
  - docs/POLYMARKET.md#configuration
  - docs/CLI.md#polymarket-commands-deprioritized
verified_at: 7887039
---

# Polymarket pipeline

Collection of Polymarket prices and order books, built for NBA and now deprioritized: UK trading is blocked for all sports. Per sport (from [POLYMARKET](../docs/POLYMARKET.md#status) and the `experiments/scripts/polymarket_mlb_*.py` probes): EPL is thin and AMM-driven; NBA and MLB are deep with about 1 cent spreads; MLB tracks devigged bet365 through arbitrage, so it is a sharp-reference candidate, not a mispricing venue. Storage is in [Polymarket tables](table-polymarket.md).

## Components

- `PolymarketClient` (`polymarket_fetcher.py`): aiohttp, tenacity retry (3 attempts), no auth. Gamma `/events` for discovery (`get_nba_events`, uses `PolymarketConfig.nba_series_id` and `game_tag_id`); CLOB `/price`, `/midpoint`, `/book`, `/prices-history`. Batch calls cap at `MAX_CONCURRENT_REQUESTS` (10). The client is sport-agnostic apart from `get_nba_events`; another sport needs its own series/tag IDs and ticker parser.
- `classify_market(question, event_title)`: moneyline when question equals title; else `Spread:` regex, `O/U` regex, player prop (stat keyword after a colon plus over/under), else `OTHER`.
- `PolymarketIngestionService`: `discover_and_upsert_events` then `collect_snapshots`. Tier comes from the single closest active event via `calculate_tier` ([Fetch tier](fetch-tier.md)); order books (first outcome token only, moneyline only) are fetched when the tier is in `PolymarketConfig.orderbook_tiers`. Market types follow the `collect_*` toggles.
- `fetch_polymarket.py::main`: skipped if `polymarket.enabled` is false; self-schedules after each run at `price_poll_interval` (default 300s), or 24h when no events exist.
- `backfill_polymarket.py::main`: pages closed events, skips markets with 10+ snapshots (`get_backfilled_market_ids`), pulls `/prices-history` (`interval=max`, `fidelity=5`) for the first token, and inserts via `PolymarketWriter.bulk_store_price_history`. CLOB history expires after about 30 days, so it must run every few days. Flags `--include-spreads`, `--include-totals`, `--dry-run` arrive through `JobContext`.
- Linking: `match_polymarket_event` parses tickers `nba-{away}-{home}-YYYY-MM-DD`, maps abbreviations via `NBA_ABBREV_MAP`, and matches `events` within +-24h of noon UTC on the ticker date; single match only, else None. Only `odds polymarket link` calls it ([Event matching](event-matching.md)).

CLI: `odds polymarket discover|status|backfill|link|book`. Env prefix `POLYMARKET_`; defaults are in `PolymarketConfig`.

AWS: `enable_polymarket` gates the rules (`deployment/aws/terraform/eventbridge.tf`; backfill is `rate(3 days)`). Deployed state is unverifiable from the repo: `odds scheduler list-jobs --backend aws`.

## Hazards

- `PolymarketConfig.enabled` defaults to True while Terraform `enable_polymarket` defaults to false: two independent gates.
- `fetch-polymarket` self-schedules only after success and does not use the shared forward-decision/pre-schedule helpers of the other self-scheduling jobs, so an exception skips the reschedule ([Self-scheduling](self-scheduling.md)).
- Events are not auto-linked during fetch; `event_id` stays NULL until `odds polymarket link` runs, and backfill never links.
- `docs/CLI.md` omits `odds polymarket book`.
- Tickers and the API series IDs are NBA-only in code; EPL/MLB IDs in the doc are research notes, not config.

## Related

- [Polymarket tables](table-polymarket.md)
- [Data sources](data-sources.md)
- [Fetch tier](fetch-tier.md)
