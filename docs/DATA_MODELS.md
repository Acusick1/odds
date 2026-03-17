# Cross-Source Data Model & Training Pipeline

## Data Sources

### Sportsbook Odds

Three sources with complementary bookmaker coverage. Each fetch stores an `OddsSnapshot` tagged
with a `FetchTier` indicating how far before the game it was collected:

| Tier | Window before game |
|------|--------------------|
| `opening` | >72h |
| `early` | 24–72h |
| `sharp` | 12–24h |
| `pregame` | 3–12h |
| `closing` | 0–3h |

| Source | Bookmakers | Events | Key value |
|--------|-----------|--------|-----------|
| Odds API | Pinnacle, FanDuel, DraftKings, BetMGM, Bovada (US) | ~1K NBA (dense snapshots) | Dense snapshot sequences, US bookmakers |
| OddsPortal | bet365, Betway, Betfred, bwin (UK) | ~5K NBA + ~1.8K EPL (opening + closing only) | UK retail bookmakers |
| football-data.co.uk | Pinnacle, bet365, bwin, William Hill, Betvictor, Interwetten, Ladbrokes | ~4K EPL (11 seasons, 2015–2026) | Pinnacle + Betfair Exchange closing odds (sharp reference) |

Event ID patterns: `op_` = OddsPortal, `fduk_` = football-data.co.uk, hex UUID = Odds API.
1,810 EPL events matched between OddsPortal and FDUK.

Each snapshot contains one `Odds` row per bookmaker per market per outcome. American odds are
converted to implied probabilities internally.

### Polymarket Prices (deprioritized)

Pipeline exists but is inactive — not accessible from UK, data likely collinear with sportsbook
odds. See [POLYMARKET.md](POLYMARKET.md) for technical details.

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

### Standings (11 features, EPL only)
League table context at decision time:
- **Position**: home/away league position, position difference
- **Form**: home/away points per game, recent form (last 5)
- **Goal difference**: home/away GD, GD difference
- **Promotion/relegation**: home/away flags for top 4 / bottom 3

### Match Stats (14 features, EPL only)
Rolling averages of prior match statistics from football-data.co.uk:
- **Shots**: home/away total shots, shots on target (rolling avg, configurable `match_stats_window`, default 5)
- **Set pieces**: home/away corners
- **Discipline**: home/away fouls, yellow cards, red cards
- **Half-time**: home/away half-time goals scored
- Strict time filtering: only uses completed matches prior to `commence_time` (no look-ahead bias)

### EPL Schedule (8 features, EPL only)
Rest and fixture congestion from ESPN all-competition fixture data:
- **Days rest**: home/away days since previous match (any competition)
- **Rest advantage**: home minus away days rest
- **Fixture congestion**: home/away matches in trailing 14-day window
- **European competition**: home/away flags for midweek European fixtures

### Rest/Schedule (5 features, NBA only)
Game context from NBA game logs:
- **Days rest**: home/away days since previous game
- **Rest advantage**: home days rest minus away days rest
- **Back-to-back**: home/away boolean flags

### Injury (6 features, NBA only) — no predictive value
Impact-weighted injury burden per team. Extensively tested (Exp 6, 6b) — adds zero signal
for CLV prediction at any decision tier when properly tuned. See MODELING.md for details.

### Polymarket (14 features) — deprioritized
PM prices and order book microstructure. Not validated at scale.

### Cross-Source (7 features) — deprioritized
PM vs sportsbook divergence.

### Sequence (13 features per timestep, for LSTM) — no predictive value
Time-series per snapshot. LSTM conclusively ruled out — sequential modeling adds no value
over cross-sectional features (see MODELING.md).

---

## Cross-Validation Protocol

Walk-forward CV with a **sliding 1-season window** and **~5 matchday validation steps**:

```yaml
cv_method: walk_forward
window_type: sliding
min_train_events: 380    # ~1 EPL season
max_train_events: 380
val_step_events: 50      # ~5 matchdays
```

This yields ~26 folds for ~1.8K EPL events. Protocol chosen via grid search over window type,
window size, and validation step size — see MODELING.md for the full grid results and rationale.

Key design choices:
- **Sliding window** prevents stale data from degrading the model
- **val_step=50** gives the tuner enough per-fold signal to select different hyperparameters per feature set (smaller steps force identical params across all feature sets)
- **`test_split: 0.0`** — no held-out test set; walk-forward CV simulates the production retrain cycle

---

## Configuration

Training uses YAML config files in `experiments/configs/`. Key fields:

### Data selection
- `data_source`: `"oddsportal"`, `"oddsapi"`, `"football_data_uk"`, `"all"`, or `null` (no filter)
- `sport_key`: sport filter (e.g. `"soccer_epl"`, `"basketball_nba"`)
- `start_date` / `end_date`: date range filter
- `min_snapshots`: minimum snapshots per event (for sequence data)
- `closing_source_priority`: ordered list of preferred sources for closing snapshot selection (e.g. `[football_data_uk]`)

### Target
- `target_type`: `"devigged_bookmaker"`
- `target_bookmaker`: e.g. `"bet365"`, `"pinnacle"`
- `sharp_bookmakers`: list of sharp references for cross-source features (e.g. `[pinnacle]`)

### Features
- `feature_groups`: list of groups to include (e.g. `["tabular", "standings"]`)
- `outcome`: `"home"` or `"away"` (which side to predict CLV for)

### Cross-validation
- `cv_method`: `"walk_forward"`, `"timeseries"`, or `"kfold"`
- `window_type`: `"sliding"` or `"expanding"` (walk-forward only)
- `min_train_events` / `max_train_events`: training window size bounds
- `val_step_events`: number of events per validation fold
- `validation_split`: set to `0.0` when using CV
- `test_split`: set to `0.0` for walk-forward CV
