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

Polymarket was collected for NBA and is deprioritised as a trading venue ([ADR-0007](../../docs/adr/0007-deprioritise-polymarket.md)). UK trading is blocked for all sports (inferred; external fact, no repo evidence). In the early NBA experiments Polymarket correlated 0.9978 with sportsbook prices and the Polymarket features (a 16-column, NBA-only group) changed nothing ([exp-early-nba-signal-and-lstm](../evaluation/exp-early-nba-signal-and-lstm.md)). The pipeline that exists is described on [polymarket](polymarket.md).

## Market character by sport

Figures were observed from live markets, not computed by code in this repo. The MLB figures come from the probe scripts `experiments/scripts/polymarket_mlb_discover.py`, `experiments/scripts/polymarket_mlb_orderbooks.py`, `experiments/scripts/polymarket_mlb_volume.py` and `experiments/scripts/polymarket_mlb_sentiment.py` (recorded in dff040d; no results file exists under `experiments/results/`). The EPL and NBA figures have no probe script and are (inferred).

| Sport | Observed character |
|---|---|
| EPL | (inferred) Thin, $10K-$100K per match, and AMM-driven: the book reflects automated market-maker parameters, not public sentiment. Rejected for the agent ([agent-data-sources](../agent/agent-data-sources.md#rejected-sources)). |
| NBA | (inferred) Deep: $178K-$2.5M USDC moneyline volume per game, about 1 cent spreads, 25-28 levels per side, top-of-book size about $1,100-$1,600. |
| MLB | Deep: $152K median per game, up to $5.8M for playoff games, 1 cent spreads, 32-44 levels per side, median $567K total depth. Prices are collinear with devigged bet365 (mean delta about 1 cent on games with real volume) through arbitrage, not AMM seeding. Trading is discrete: 89% of 5-minute ticks show no movement and the rest jump by at least 1 cent. A sharp-reference candidate, not a mispricing venue. |

Whether Polymarket works as an MLB sharp reference is an open question (#456), not part of the ADR.

## Series and tag ids

Series and tag IDs for other sports were recorded during research and are not in config (inferred, dff040d):

| Sport | Series ID | Tag IDs | Ticker |
|---|---|---|---|
| EPL | `10188` | `82`, `306` | `epl-{away3}-{home3}-YYYY-MM-DD` |
| MLB | `3` | `100639` (game tag) | `mlb-{away3}-{home3}-YYYY-MM-DD` |

Config holds only the NBA series (`10345`) and game tag (`100639`) in `odds_core.config.PolymarketConfig`.

## Market richness

(Inferred; observed on NBA.) Markets per game grow through the season: about 1 early (moneyline only), 10-15 mid-season (moneyline, several spreads, totals) and about 42 late (adds first-half lines and player props). Spread volume was about $418K per line.

## Data availability

Observed from the API, not verifiable from the repo:

| Data | Active markets | Resolved markets |
|---|---|---|
| Event and market metadata (Gamma) | yes | yes, indefinitely |
| Price history (CLOB) | yes, accumulating | about 25 days at full 5-minute resolution, partial from 25 to 31 days (a gradual trim, not a hard cutoff), gone after about 31 days |
| Order book (CLOB) | live only | no (404 after close) |
| Current price and midpoint (CLOB) | yes | no |

A full game lifecycle of about 7 days from market creation to resolution yields about 940-960 moneyline price points. Order-book history cannot be backfilled, so only the live path stores books.

## Related

- [Polymarket pipeline and tables](polymarket.md)
- [Data sources](data-sources.md)
- [Agent data sources](../agent/agent-data-sources.md)
