# Cross-Source Data Model & Training Pipeline

## Data Sources

### Sportsbook Odds

Collected via The Odds API (live polling) and OddsPortal (headless scraping). Each fetch stores
an `OddsSnapshot` tagged with a `FetchTier` indicating how far before the game it was collected:

| Tier | Window before game |
|------|--------------------|
| `opening` | >72h |
| `early` | 24–72h |
| `sharp` | 12–24h |
| `pregame` | 3–12h |
| `closing` | 0–3h |

Two bookmaker sets with non-overlapping coverage:

| Source | Bookmakers | Events |
|--------|-----------|--------|
| Odds API | Pinnacle, FanDuel, DraftKings, BetMGM, Bovada, etc. (US) | ~1K (NBA, dense snapshots) |
| OddsPortal | bet365, Betway, Betfred, bwin (UK) | ~5K NBA + ~1.8K EPL (opening + closing only) |

Each snapshot contains one `Odds` row per bookmaker per market per outcome. American odds are
converted to implied probabilities internally.

**What this gives us:** A time series of market-consensus probabilities across bookmakers,
from well before the game to just before tip-off.

---

### Polymarket Prices (deprioritized)

Collected via two APIs — Gamma (metadata/discovery) and CLOB (prices/order books). Each NBA
game has a Polymarket moneyline market with two binary outcome tokens. Prices are already
implied probabilities (0.0–1.0). Pipeline exists but is inactive — not accessible from UK,
data likely collinear with sportsbook odds.

---

## Training Target

The **regression target** is the devigged bookmaker CLV delta at each snapshot:

```
target = devigged_fair_close - devigged_fair_at_snapshot
```

This measures how far and which way the line moved from the current snapshot to close. A model
that predicts this can identify games where the current market price is likely to shift, which
drives the bet-vs-pass decision.

The target bookmaker is configurable (`target_bookmaker` in training config). bet365 is the
primary target for EPL; Pinnacle was tested for NBA but yielded no signal (see MODELING.md).

Why delta (not absolute close):
- We don't need to predict what the line will be — we need to predict how much it will move
- Delta is stationary and mean-zero, easier to learn than absolute levels
- Directly maps to CLV: positive delta = current price is too low, line will move up

---

## Available Feature Groups

### Sportsbook Tabular (28 features)
Point-in-time snapshot at decision time:
- **Consensus**: avg/std odds, implied probs (home/away)
- **Sharp vs retail**: sharp prob vs retail avg, differential
- **Market efficiency**: num bookmakers, avg/std market hold
- **Line shopping**: best/worst odds, range across books

### Trajectory (23 features)
Aggregate statistics from the full odds sequence up to decision time:
- **Momentum**: prob change to decision, avg change rate, max increase/decrease
- **Volatility**: prob range, odds volatility, movement count
- **Trend**: slope, strength, reversals, acceleration
- **Sharp money**: sharp prob trajectory, sharp-retail divergence trend, sharp leads retail (binary)
- **Timing**: early vs recent movement distribution

### Polymarket (14 features) — deprioritized
PM prices and order book microstructure. Not validated at scale (tested with only 230 events).

### Cross-Source (7 features) — deprioritized
PM vs sportsbook divergence.

### Injury (6 features) — no predictive value
Impact-weighted injury burden per team. Extensively tested (Exp 6, 6b) — adds zero signal
for CLV prediction at any decision tier when properly tuned. See MODELING.md for details.

### Rest/Schedule (5 features)
Game context from NBA game logs:
- **Days rest**: home/away days since previous game
- **Rest advantage**: home days rest minus away days rest
- **Back-to-back**: home/away boolean flags

### Sequence (13 features per timestep, for LSTM) — no predictive value
Time-series per snapshot. LSTM conclusively ruled out — sequential modeling adds no value
over cross-sectional features (see MODELING.md).

---

## Configuration

Training uses YAML config files in `experiments/configs/`. Key fields:

- `data_source`: `"oddsportal"`, `"oddsapi"`, `"all"`, or `null` (no filter)
- `target_type`: `"devigged_bookmaker"`
- `target_bookmaker`: e.g. `"bet365"`, `"pinnacle"`
- `feature_groups`: list of groups to include (e.g. `["tabular"]`)
- `outcome`: `"home"` or `"away"` (which side to predict CLV for)
- `start_date` / `end_date`: date range filter
- `min_snapshots`: minimum snapshots per event (for sequence data)
- `sport_key`: sport filter (e.g. `"soccer_epl"`, `"basketball_nba"`)
