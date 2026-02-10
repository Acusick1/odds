# Polymarket Integration

> Data model, pipeline architecture, and API reference for Polymarket prediction market data.

For strategic context, key constraints, and quick-reference patterns see the Polymarket Integration section in [CLAUDE.md](../CLAUDE.md).

## Data Model

All tables defined in `packages/odds-core/odds_core/polymarket_models.py`.

### Tables

| Table | Tablename | Purpose | Key Fields |
|-------|-----------|---------|------------|
| `PolymarketEvent` | `polymarket_events` | NBA game mapping | `pm_event_id` (unique), `ticker`, `event_id` (FK → `events.id`, nullable) |
| `PolymarketMarket` | `polymarket_markets` | Tradable market within event | `pm_market_id` (unique), `market_type`, `clob_token_ids` (JSON), `point` |
| `PolymarketPriceSnapshot` | `polymarket_price_snapshots` | Price time series | `outcome_0_price`, `outcome_1_price`, `best_bid`, `best_ask`, `fetch_tier` |
| `PolymarketOrderBookSnapshot` | `polymarket_orderbook_snapshots` | Full depth + derived ML metrics | `raw_book` (JSON), `imbalance`, `weighted_mid`, `token_id` |
| `PolymarketFetchLog` | `polymarket_fetch_logs` | Fetch audit log | `job_type`, `success`, `snapshots_stored` |

### Relationships

```
PolymarketEvent (1) → (N) PolymarketMarket
PolymarketMarket (1) → (N) PolymarketPriceSnapshot
PolymarketMarket (1) → (N) PolymarketOrderBookSnapshot
PolymarketEvent.event_id → events.id (nullable, linked lazily via CLI)
```

### Indexes

| Index | Columns | Table |
|-------|---------|-------|
| `ix_pm_event_active_closed` | `active`, `closed` | `polymarket_events` |
| `ix_pm_market_event_type` | `polymarket_event_id`, `market_type` | `polymarket_markets` |
| `ix_pm_price_market_time` | `polymarket_market_id`, `snapshot_time` | `polymarket_price_snapshots` |
| `ix_pm_price_market_tier` | `polymarket_market_id`, `fetch_tier` | `polymarket_price_snapshots` |
| `ix_pm_orderbook_market_time` | `polymarket_market_id`, `snapshot_time` | `polymarket_orderbook_snapshots` |

### Market Type Classification

`classify_market()` in `packages/odds-lambda/odds_lambda/polymarket_fetcher.py`

| Type | Pattern | Point Extracted |
|------|---------|-----------------|
| `MONEYLINE` | `question == event_title` | None |
| `SPREAD` | Regex `Spread:\s*([+-]?\d+\.?\d*)` | Float (e.g. `-6.5`) |
| `TOTAL` | Regex `O/U\s+(\d+\.?\d*)` (case-insensitive) | Float (e.g. `215.5`) |
| `PLAYER_PROP` | Colon + stat keyword after colon + "over"/"under" | None |
| `OTHER` | Fallback | None |

Player prop stat keywords: `points`, `rebounds`, `assists`, `steals`, `blocks`, `threes`, `turnovers`, `pts`, `reb`, `ast`, `stl`, `blk`

## Pipeline

### Fetch Job (Live Polling)

`packages/odds-lambda/odds_lambda/jobs/fetch_polymarket.py` — entry point: `main()`

Orchestrated by `PolymarketIngestionService` in `packages/odds-lambda/odds_lambda/polymarket_ingestion.py`.

**Phase 1 — Discover + upsert:**
1. Fetch active NBA events from Gamma API
2. Upsert events and markets to DB (ON CONFLICT DO UPDATE)
3. Classify market types via `classify_market()`

**Phase 2 — Collect snapshots:**
1. Load active events from DB
2. Calculate `FetchTier` from closest game's `commence_time`
3. Fetch prices for configured market types via `get_prices_batch()`
4. If tier in `orderbook_tiers` → fetch order books for moneyline markets only
5. Commit + log to `PolymarketFetchLog`

**Self-scheduling:** Uses fixed `price_poll_interval` from config (default 300s). Falls back to 24h daily check when no active events exist.

### Backfill Job

`packages/odds-lambda/odds_lambda/jobs/backfill_polymarket.py` — entry point: `main(include_spreads, include_totals, dry_run)`

1. Fetch all closed NBA events from Gamma API (paginated, 100/page)
2. Query already-backfilled market IDs (markets with ≥10 existing snapshots)
3. For each event: upsert event + markets, fetch `/prices-history` for moneyline (and optionally spreads/totals)
4. Bulk insert via `PolymarketWriter.bulk_store_price_history()` with duplicate detection
5. Log to `PolymarketFetchLog`

**CRITICAL:** Must run every 3–5 days. CLOB price history expires on a ~30-day rolling basis.

### Event Matching

`packages/odds-lambda/odds_lambda/polymarket_matching.py`

**Manual process only** — triggered via `odds polymarket link`, not auto-triggered in fetch job.

1. `parse_ticker(ticker)` extracts away/home abbreviations and date from format `nba-{away}-{home}-{yyyy}-{mm}-{dd}`
2. `NBA_ABBREV_MAP` resolves 3-letter abbreviations to canonical team names (30 NBA teams)
3. Queries `events` table with exact team name match + ±24h date window
4. Returns `event_id` only on unambiguous single match; `None` otherwise (no false positives)

`TEAM_ALIASES` provides a separate canonical-name → aliases mapping for `normalize_team()`.

## API Reference

### Gamma API (Market Discovery)

Base URL: `https://gamma-api.polymarket.com`

| Endpoint | Purpose | Key Params |
|----------|---------|------------|
| `GET /events` | Event/market discovery | `series_id`, `tag_id`, `active`, `closed`, `limit`, `offset` |

Pagination via `offset`; returns `[]` when exhausted. No authentication required.

### CLOB API (Prices and Order Books)

Base URL: `https://clob.polymarket.com`

| Endpoint | Purpose | Key Params |
|----------|---------|------------|
| `GET /price` | Current token price | `token_id`, `side` |
| `GET /midpoint` | Midpoint price | `token_id` |
| `GET /book` | Full order book | `token_id` |
| `GET /prices-history` | Historical prices | `market` (token_id), `interval`, `fidelity`, `startTs`, `endTs` |

Price history params for backfill: `interval=max`, `fidelity=5` (5-minute resolution).

Returns: `{history: [{t: unix_timestamp, p: price_string}]}`

No authentication required. Rate-limit friendly delay of 100ms between calls (configurable).

### Data Availability

| Data | Active Markets | Resolved Markets |
|------|----------------|------------------|
| Event/market metadata (Gamma) | Yes | Yes (persists indefinitely) |
| Price history (CLOB) | Yes (accumulating) | ~25 days full, 25–31 days partial, gone after ~31 days |
| Order book (CLOB) | Yes (live only) | No (404 after close) |
| Current price/midpoint (CLOB) | Yes | No |

Full 5-minute resolution data available for ~25 days from game. Partial data (gradual trim, not a hard cutoff) from 25–31 days. A full game lifecycle (~7 days from market creation to resolution) yields ~940–960 price data points for moneyline.

### Market Richness

Markets per game vary by season phase:
- Early season: ~1 (moneyline only)
- Mid season: ~10–15 (moneyline, multiple spreads, totals)
- Late season: ~42 (above + first half lines, player props)

### Volume and Order Book Characteristics

- Moneyline volume: $178K–$2.5M USDC per game
- Spread volume: ~$418K observed per line
- Order book spread: ~1 cent typical
- Depth: 25–28 levels per side
- Top-of-book size: ~$1,100–$1,600 USDC

**CRITICAL:** CLOB API returns order books unsorted. Must sort bids descending, asks ascending client-side.

## Order Book Processing

`process_order_book()` in `packages/odds-lambda/odds_lambda/polymarket_fetcher.py`

Input: raw book `{bids: [{price, size}], asks: [{price, size}]}`

Returns `None` if either side is empty or book is crossed (best_bid ≥ best_ask).

Derived metrics stored in `PolymarketOrderBookSnapshot`:

| Metric | Formula |
|--------|---------|
| `spread` | `best_ask - best_bid` |
| `midpoint` | `(best_bid + best_ask) / 2` |
| `imbalance` | `(bid_depth - ask_depth) / (bid_depth + ask_depth)`, range [-1, 1] |
| `weighted_mid` | `(best_bid × ask_size_top + best_ask × bid_size_top) / (bid_size_top + ask_size_top)` |
| `bid_levels` / `ask_levels` | Count of price levels per side |
| `bid_depth_total` / `ask_depth_total` | Sum of sizes across all levels |

## Configuration

`PolymarketConfig` in `packages/odds-core/odds_core/config.py`, env prefix `POLYMARKET_`.

| Variable | Type | Default | Purpose |
|----------|------|---------|---------|
| `POLYMARKET_ENABLED` | `bool` | `True` | Master toggle for collection |
| `POLYMARKET_GAMMA_BASE_URL` | `str` | `https://gamma-api.polymarket.com` | Gamma API base URL |
| `POLYMARKET_CLOB_BASE_URL` | `str` | `https://clob.polymarket.com` | CLOB API base URL |
| `POLYMARKET_NBA_SERIES_ID` | `str` | `10345` | NBA series ID |
| `POLYMARKET_GAME_TAG_ID` | `str` | `100639` | Game-day events tag ID |
| `POLYMARKET_PRICE_POLL_INTERVAL` | `int` | `300` | Price snapshot interval (seconds) |
| `POLYMARKET_ORDERBOOK_POLL_INTERVAL` | `int` | `1800` | Order book snapshot interval (seconds) |
| `POLYMARKET_COLLECT_MONEYLINE` | `bool` | `True` | Collect moneyline markets |
| `POLYMARKET_COLLECT_SPREADS` | `bool` | `True` | Collect spread markets |
| `POLYMARKET_COLLECT_TOTALS` | `bool` | `True` | Collect total markets |
| `POLYMARKET_COLLECT_PLAYER_PROPS` | `bool` | `False` | Collect player prop markets |
| `POLYMARKET_ORDERBOOK_TIERS` | `list[str]` | `["closing", "pregame"]` | Tiers that trigger order book collection |

## Storage Estimates

Per NBA game (full 7-day lifecycle):

| Data Type | Records | Size |
|-----------|---------|------|
| Price snapshots (3 market types) | ~6,000 | ~1.2 MB |
| Order book snapshots (closing tier, moneyline) | ~6 | ~30 KB |
| Market metadata | ~45 | ~22 KB |

Per season (~1,230 regular season games): ~1.7 GB total.
