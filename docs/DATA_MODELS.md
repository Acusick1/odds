# Cross-Source Data Model & Training Pipeline

## Data Sources

### Sportsbook Odds

Collected via The Odds API on a polling schedule. Each fetch stores an `OddsSnapshot` tagged
with a `FetchTier` indicating how far before the game it was collected:

| Tier | Window before game |
|------|--------------------|
| `opening` | >72h |
| `early` | 24–72h |
| `sharp` | 12–24h |
| `pregame` | 3–12h |
| `closing` | 0–3h |

Each snapshot contains one `Odds` row per bookmaker per market per outcome. Bookmakers are
classified as **sharp** (Pinnacle — efficient, signal-bearing) or **retail** (FanDuel,
DraftKings, BetMGM — slower to move). American odds are converted to implied probabilities
internally.

**What this gives us:** A time series of market-consensus probabilities across bookmakers,
from well before the game to just before tip-off.

---

### Polymarket Prices

Collected via two APIs — Gamma (metadata/discovery) and CLOB (prices/order books). Each NBA
game has a Polymarket moneyline market with two binary outcome tokens. Prices are already
implied probabilities (0.0–1.0).

Two snapshot types are stored:
- **`PolymarketPriceSnapshot`** — mid price, best bid/ask, spread, volume, liquidity; polled every 5 min
- **`PolymarketOrderBookSnapshot`** — full depth (bid/ask depth totals, imbalance, weighted mid); stored when available

**What this gives us:** A high-frequency time series of a prediction market's view of the same game.

Linkage between the two sources is done by matching the PM event ticker
(`nba-{away}-{home}-{date}`) to the sportsbook `Event` record, stored as
`PolymarketEvent.event_id`.

---

## How We Combine Them (Cross-Source Training)

Training uses three **feature groups**, all evaluated at the same logical **decision point**
(currently 7.5h before game = midpoint of the `pregame` tier). The target is the line movement
from the opening snapshot to the closing snapshot.

### Feature Group 1: `tabular` — Sportsbook snapshot at opening

Point-in-time features from the earliest available sportsbook snapshot (configured as
`opening_tier`, currently `early` = 24–72h before game). Captures the market's initial line.

Key features (all relative to the configured outcome — home team by default):
- `consensus_prob` — average implied probability across all bookmakers
- `sharp_prob` — Pinnacle's implied probability
- `retail_prob` — average retail bookmaker implied probability
- `diff` — retail − sharp (sharp money signal at opening)
- `avg_market_hold` — average overround across bookmakers (market efficiency proxy)
- `best_available_odds` — best odds available for the outcome

### Feature Group 2: `trajectory` — Sportsbook line movement to decision point

Aggregate features across all snapshots from opening up to the `pregame` tier. Captures
*how* the line moved, not just where it started.

Key features:
- **Momentum:** total prob change, avg rate of change, max single-step increase/decrease, net direction
- **Volatility:** std dev of probabilities, total range, movement count
- **Trend:** linear regression slope + R², reversal count, acceleration (2nd derivative)
- **Sharp signal:** total sharp prob change, trend in sharp–retail divergence

### Feature Group 3: `polymarket` — PM price + cross-source divergence at decision point

Two sub-groups, both computed at the same 7.5h decision time:

**PM tabular (14 features):**
- `pm_home_prob`, `pm_away_prob` — PM implied probabilities
- `pm_spread`, `pm_midpoint`, `pm_best_bid`, `pm_best_ask`
- `pm_volume`, `pm_liquidity`
- `pm_bid_depth`, `pm_ask_depth`, `pm_imbalance`, `pm_weighted_mid` (from order book, when available)
- `pm_price_velocity` — prob change per hour over the prior 2h
- `pm_price_acceleration` — change in velocity (first vs second half of window)

**Cross-source (8 features):**
- `pm_sb_prob_divergence` — `pm_home_prob − sportsbook_consensus_prob` (at same timestamp ±30min)
- `pm_sb_divergence_abs`, `pm_sb_divergence_direction`
- `pm_spread_vs_sb_hold` — PM spread minus SB market hold (relative liquidity cost)
- `pm_sharp_divergence`, `pm_sharp_divergence_abs` — PM vs Pinnacle specifically
- `pm_mid_vs_sb_consensus` — order book midpoint vs sportsbook consensus

---

## Training Target

For each event, the **regression target** is the closing line movement from opening:

```
target = closing_prob − opening_prob
```

This is the CLV direction: how far and which way the line moved from the opening snapshot to
close. A model that predicts this can identify games where the current market price is likely
to shift, which drives the bet-vs-pass decision at prediction time.

---
