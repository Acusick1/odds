# Polymarket EPL match-market liquidity, 2026-27 (ticket #509)

Measured 2026-10-18 ~20:00 UTC through the public Gamma, CLOB and Data APIs. No authentication, no spend.

## Verdict

**USABLE as a free price source and a near-sharp reference, with caveats.** The "thin, AMM-driven, $10-100K per match" finding does not hold for 2026-27, nor for 2025-26. Pre-kick-off books are thin until about a day out, then deep.

## Facts

### Discovery

- Series `10188` (`premier-league-2025`) and tags `82`/`306` are confirmed. The ticker is `epl-{home3}-{away3}-YYYY-MM-DD`: home first, not away first as the brief had it (`ordering: home`). Source: `GET gamma-api.polymarket.com/sports` (`sport=epl`, `series=10188`, `tags=1,82,306,100639,100350`).
- A match is one event, `epl-nnn-mmm-date`, with three binary Yes/No markets (home win, draw, away win), `negRisk: true`. Prop events share the series with suffixes such as `-halftime-result`, `-exact-score`, `-more-markets`, `-total-corners`. Filter on the exact slug regex to isolate the 1X2 event.
- The event carries `startTime` (kick-off) and `eventMetadata.opticOddsFixtureId`.
- Gamma returns at most 100 events per page whatever `limit` says. Page with `offset`.

### Volume (Gamma `event.volume`, USDC, lifetime including in-play)

- 449 closed 1X2 match events from 2025-08-15 to 2026-10-18. Full list in `polymarket-epl-liquidity-2026/closed_match_volumes.csv`.
- Monthly medians per match (matches that month): 2025-08 $0.42M (30), 2025-12 $1.43M (56), 2026-05 $2.68M (40), 2026-09 $1.76M (30), 2026-10 $1.86M (19).
- 2026-03 shows median 0: Gamma volume is empty or zero for most of that month. Treat as missing data, not zero volume.
- Range per match is roughly $0.1M to $14.4M. Largest: Nottingham Forest v Arsenal 2026-10-18, $13.7M, open interest $5.8M.
- Volume is lumpy: Brighton v Crystal Palace was $7.7M, of which $3.6M traded in the 6h-to-1h window.

### Volume by time to kick-off (Data API `/trades`, 10 matches played 2026-10-12 to 2026-10-18)

Method: all fills across the three markets, one row per fill (`side=BUY`; each fill appears as a BUY and a SELL row), notional = size x price. Sums are within about 3-13% of Gamma `volume`. Buckets in `volume_by_horizon.csv`.

| Match | Pre-KO USD | Pre-KO share of lifetime |
|---|---|---|
| Leeds v Man Utd | 331K | 27% |
| Bournemouth v Sunderland | 479K | 50% |
| Brentford v Liverpool | 402K | 21% |
| Man City v Ipswich | 914K | 33% |
| Everton v Chelsea | 951K | 45% |
| Coventry v Newcastle | 956K | 43% |
| Fulham v Hull | 720K | 68% |
| Newcastle v Aston Villa | 1.47M | 62% |
| Brighton v Crystal Palace | 4.02M | 37% |
| Nottm Forest v Arsenal | at least 5.7M (truncated) | about 33% |

- Median pre-kick-off volume over the nine complete matches is about **$0.91M** (range $0.33M to $4.0M). Only 21-68% of lifetime volume is pre-kick-off; the rest is in-play.
- More than 7 days out, volume is under $12K per match. 72h-24h out: $4K-$211K. Most pre-kick-off volume lands in the final 24h: 24h-6h $3K-$442K, 6h-1h up to $3.6M, last hour $69K-$5.7M.
- Truncation: the trades endpoint returns HTTP 400 beyond offset about 10,500 per query, newest first, so the oldest trades are lost. Querying per market gave complete data for 9 of 10 matches. Forest v Arsenal had one market over the cap, so its early buckets are undercounted and it is excluded from the median.

### Order-book depth and spread (CLOB `GET /book?token_id=<Yes token>`, live snapshot of the 33 open 1X2 markets)

Depth is the sum of price x size, bids plus asks, within +/-1c or +/-5c of mid.

| Horizon | Markets | Median spread | Max spread | Depth within 1c | Depth within 5c |
|---|---|---|---|---|---|
| Tottenham v Coventry, about 23h before KO | 3 | 1c | 1c | median $9.6K (max $36.6K on the favourite) | median $90K (min $63K) |
| 10 other matches, 5-7 days before KO | 30 | 1c | 2c | median $1.3K | median $8.8K (min $4.4K) |

- Tick size is 0.01 (`tick_size` on the book response).
- Event-level `liquidity` is about $75-95K at T-5..7d and $749K at T-23h.
- Finished markets return an empty book, so historical spreads cannot be replayed. Spread against horizon is cross-sectional (one snapshot over different matches), not a time series on one match. Raw rows in `books.json`.
- Spread is about 1c at every horizon seen, but size at the touch grows roughly 7x from T-6d to T-1d.

### Price quality against Pinnacle

Compared the three Polymarket Yes prices (normalised) with Pinnacle h2h from the local dev DB (Odds API snapshots). Polymarket price: CLOB `prices-history` point at or before the horizon (fidelity 10 min). Pinnacle: nearest snapshot at or before the horizon, within 2h, devigged proportionally. Rows in `cmp.json`.

- T-1h, 9 matches: mean absolute probability difference **1.0pp**, mean worst-outcome gap 1.5pp.
- T-6h, 2 matches: 0.7pp. T-24h, 4 matches: 0.6pp.
- Two T-1h outliers, Brentford v Liverpool (3.9pp worst gap) and Bournemouth v Sunderland (3.4pp), may be snapshot staleness of up to 2h; I did not separate that from real divergence.
- Polymarket 1X2 prices sum to 0.985-1.005 (no overround) against 1.03-1.04 for Pinnacle.
- The sample is small. It supports "tracks the sharp price closely", not "adds independent information". It is not a CLV test.

## Reconciling with the 2026-04 finding

I cannot reproduce "$10-100K per match". Closed matches in 2026-04 had median $2.0M, and 2025-08 matches had median $0.42M, already above the old range. The likeliest explanation is that the old figure measured something narrower, such as early pre-game volume; I have no record of its method, so that is **(inferred)**. I found no evidence on the "AMM-driven" claim: books are multi-level with real depth near kick-off, but I did not inspect maker identity.

## Cost and limits

- Odds API: 0 units. Money: $0.
- The trades cap means trade-level history of the busiest markets cannot be fully reconstructed after the fact; a collector should poll live.
- No historical order books exist, only price history and trades. A forward collector could snapshot the book. `PolymarketClient` in `packages/odds-lambda/odds_lambda/polymarket_fetcher.py` already wraps Gamma and CLOB, but is wired to the NBA series ID and game tag (`nba_series_id`, `game_tag_id`) from `packages/odds-core/odds_core/config.py`.

## What the map can build on

1. Polymarket EPL 1X2 can enter experiments as a feature source, a second sharp reference, or an extra price. It is free, covers every match, and has 449 matches of Yes-token price history back to 2025-08 (`prices-history`).
2. Horizon design: the market is only liquid inside the final 24h, so a model using Polymarket should use the 24h-to-1h horizons. Earlier prices are thin, lightly traded quotes.
3. Not tested: whether Polymarket leads or lags Pinnacle (a lead-lag question), a price-history backfill run for the 449 matches, and a forward book-snapshot collector for the last 24h.

## Data in `polymarket-epl-liquidity-2026/`

- `closed_match_volumes.csv`: slug, end date, lifetime volume for 449 matches.
- `volume_by_horizon.csv`: per-match volume buckets by time to kick-off.
- `books.json`: live book summary for 33 markets.
- `cmp.json`: Polymarket vs Pinnacle rows (slug, horizon hours, bookmaker, PM sum, Pinnacle sum, max diff, mean diff). Trader identities from the trades API were not kept.
