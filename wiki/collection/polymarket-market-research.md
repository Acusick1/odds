---
type: explanation
title: Polymarket market character and data availability
description: Observed market depth by sport, series and tag ids from research, market richness, and how long the API keeps data.
sources:
  - odds_core.config.PolymarketConfig
  - experiments/scripts/polymarket_mlb_discover.py
  - experiments/scripts/polymarket_mlb_orderbooks.py
  - experiments/scripts/polymarket_mlb_volume.py
  - experiments/scripts/polymarket_mlb_sentiment.py
status: stable
---

# Polymarket market character and data availability

Polymarket was collected for NBA and is not a trading venue for us ([ADR-0013](../../docs/adr/0013-collect-polymarket-epl-as-sharp-reference.md)). UK trading is blocked for all sports (an external fact recorded on 2026-03-09 and 2026-04-17, which the repo cannot verify). In a February 2026 NBA cross-source check over 228 events, Polymarket correlated 0.9978 with sportsbook probabilities (#134), and at 230 events the Polymarket features (a 16-column, NBA-only group) showed no measurable effect (#136, [exp-early-nba-signal-and-lstm](../evaluation/exp-early-nba-signal-and-lstm.md)). The pipeline that exists is described on [polymarket](polymarket.md).

## Market character by sport

Figures were observed from live markets, not computed by code in this repo. The MLB figures come from the probe scripts `experiments/scripts/polymarket_mlb_discover.py`, `experiments/scripts/polymarket_mlb_orderbooks.py`, `experiments/scripts/polymarket_mlb_volume.py` and `experiments/scripts/polymarket_mlb_sentiment.py` (recorded in dff040d; no results file exists under `experiments/results/`). The EPL and NBA figures have no probe script and are (inferred).

| Sport | Observed character |
|---|---|
| EPL | Since 2025-08 the median match trades about $0.9M before kick-off (range $0.33M-$4.0M over nine complete matches), mostly in the last 24 hours, with 1-2 cent spreads and prices within about 1 pp of Pinnacle ([#509](https://github.com/Acusick1/odds/issues/509), [findings](https://github.com/Acusick1/odds/blob/research/polymarket-epl-liquidity-2026/research/polymarket-epl-liquidity-2026.md)). The earlier "thin, $10K-$100K, AMM-driven" figure has no record of its method and could not be reproduced. Prices stay within 0.35-0.5 pp of Betfair Exchange from 3 days out ([#518](https://github.com/Acusick1/odds/issues/518)). |
| NBA | (inferred) Deep: $178K-$2.5M USDC moneyline volume per game, about 1 cent spreads, 25-28 levels per side, top-of-book size about $1,100-$1,600. |
| MLB | Deep: $152K median per game, up to $5.8M for playoff games, 1 cent spreads, 32-44 levels per side, median $567K total depth. Prices are collinear with devigged bet365 (mean delta about 1 cent on games with real volume) through arbitrage, not AMM seeding. Trading is discrete: 89% of 5-minute ticks show no movement and the rest jump by at least 1 cent. A sharp-reference candidate, not a mispricing venue. |

Use of Polymarket as an MLB sharp reference is untested and not part of the ADR.

## Series and tag ids

`odds_lambda.polymarket_sports.POLYMARKET_SPORTS` holds the series and tag IDs for NBA and EPL. The MLB IDs were recorded during research and are not in config (inferred, dff040d):

| Sport | Series ID | Tag IDs | Ticker |
|---|---|---|---|
| EPL | `10188` | `82`, `306` (the series ID alone selects the same events) | `epl-{home3}-{away3}-YYYY-MM-DD` (home first) |
| MLB | `3` | `100639` (game tag) | `mlb-{away3}-{home3}-YYYY-MM-DD` |

NBA (`10345`, game tag `100639`) and EPL (`10188`) are in `odds_lambda.polymarket_sports.POLYMARKET_SPORTS`, not in `odds_core.config.PolymarketConfig`.

## Market richness

(Inferred; observed on NBA.) Markets per game grow through the season: about 1 early (moneyline only), 10-15 mid-season (moneyline, several spreads, totals) and about 42 late (adds first-half lines and player props). Spread volume was about $418K per line.

## Data availability

Observed from the API, not verifiable from the repo:

| Data | Active markets | Resolved markets |
|---|---|---|
| Event and market metadata (Gamma) | yes | yes, indefinitely |
| Price history (CLOB) | yes, accumulating | NBA with `interval=max`: about 25 days at full 5-minute resolution, partial from 25 to 31 days, gone after about 31 days. EPL with explicit `startTs`/`endTs` windows of at most 14 days: complete one-minute series for all 449 finished matches back to 2025-08-15, while `interval=max` is empty for old markets ([#518](https://github.com/Acusick1/odds/issues/518)) |
| Order book (CLOB) | live only | no (404 after close) |
| Current price and midpoint (CLOB) | yes | no |

A full game lifecycle of about 7 days from market creation to resolution yields about 940-960 moneyline price points. Order-book history cannot be backfilled, so only the live path stores books.

## Related

- [Polymarket pipeline and tables](polymarket.md)
- [Data sources](data-sources.md)
- [Agent data sources](../agent/agent-data-sources.md)
