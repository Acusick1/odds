# EPL Feature Research

## Current State

The EPL CLV prediction pipeline uses **purely odds-based features** plus two calendar features:

| Feature | Source | Description |
|---------|--------|-------------|
| `consensus_prob` | Odds snapshot | Average implied probability across bookmakers |
| `sharp_prob` | Odds snapshot | Sharp bookmaker implied probability |
| `retail_sharp_diff` | Odds snapshot | Retail minus sharp probability |
| `num_bookmakers` | Odds snapshot | Count of bookmakers in snapshot |
| `is_weekend` | Match time | Weekend flag |
| `day_of_week` | Match time | Day of week (0-6) |
| `hours_until_event` | Snapshot time | Hours between snapshot and kickoff |

Baseline results: XGBoost CV R²=0.016 (OddsPortal-only), R²=0.031 with Pinnacle as sharp reference (FDUK combined). See MODELING.md for full experiment history.

## Data Tracks

**Track 1 — Historical (backtesting & training)**
- OddsPortal: ~1,800 EPL events, 2 snapshots/event (opening + closing), UK bookmakers
- football-data.co.uk: 4,091 events (2015-2026), Pinnacle + Betfair Exchange closing odds, match stats, referee
- Combined: enables cross-source features (Pinnacle sharp vs bet365 retail)

**Track 2 — Live collection (accumulating)**
- OddsPortal Lambda scrape: hourly, EPL 1x2 + O/U 2.5, same UK bookmaker set
- Not yet used for training — insufficient history
- Enables trajectory features (momentum, volatility, trend) once mature

Track 2 shares the same bookmaker set as Track 1. The key difference is snapshot density: 2 per event (Track 1) vs many per event (Track 2). Trajectory features (23 in DATA_MODELS.md) require dense snapshots and are only viable with Track 2 data.

## Candidate Feature Groups

### Tier 1 — Derived from existing data

No new data sources needed. Full coverage 2015-2026 (11 seasons).

#### Standings & Form (derived from FDUK match results)

Reconstruct the league table state before each match from chronologically ordered results.

| Feature | Description |
|---------|-------------|
| `league_position` | Team's league position entering the match |
| `points` | Points accumulated |
| `points_gap_to_opponent` | Points difference between teams |
| `goal_difference` | Goals scored minus conceded |
| `form_last_5` | Points from last 5 matches (0-15) |
| `home_form_last_5` | Points from last 5 home/away matches |
| `goals_scored_rate` | Goals per game (rolling) |
| `goals_conceded_rate` | Goals conceded per game (rolling) |
| `h2h_record` | Head-to-head points in recent meetings |

#### Match Stats (from FDUK CSVs, currently not ingested)

Post-match stats from prior games become rolling pre-match features. These columns exist in the FDUK CSVs but are currently ignored during ingestion.

| CSV Column | Description |
|------------|-------------|
| `HS` / `AS` | Shots (home/away) |
| `HST` / `AST` | Shots on target |
| `HC` / `AC` | Corners |
| `HF` / `AF` | Fouls |
| `HY` / `AY` | Yellow cards |
| `HR` / `AR` | Red cards |
| `HTHG` / `HTAG` | Half-time goals |

These would be used as rolling averages (e.g., team's shots on target per game over last N matches) to proxy team quality and style.

#### Schedule (derived from FDUK match dates)

| Feature | Description |
|---------|-------------|
| `rest_days_home` | Days since home team's last match |
| `rest_days_away` | Days since away team's last match |
| `rest_advantage` | Home rest minus away rest |
| `is_midweek` | Tuesday/Wednesday fixture flag |

European competition fixture data would be needed for congestion features (not in FDUK).

#### Referee (from FDUK CSVs, currently not ingested)

The `Referee` column exists in FDUK CSVs. Could derive rolling referee tendencies (cards/game, fouls/game). Likely low value for CLV — the market prices referee assignment efficiently.

### Tier 2 — External sources, partial coverage

#### xG / Advanced Stats

| Source | Package | Coverage | Key Data |
|--------|---------|----------|----------|
| FBref | `soccerdata` | 2017-2026 | xG, xAG, possession, progressive passes, GCA/SCA |
| Understat | `understat` | 2014-2026 | Independent xG model, shot-level data |

Rolling xG for/against is the best public proxy for team quality beyond raw goals. FBref uses StatsBomb xG; Understat uses their own model — having both gives independent views.

Features: rolling xG for/against, xG overperformance (goals minus xG), shot quality metrics.

#### Injury / Squad Availability

| Source | Access | Coverage | Key Data |
|--------|--------|----------|----------|
| FPL GitHub archive (`vaastav/Fantasy-Premier-League`) | CSV download | 2016-2026 | Player status, injury flags, expected return |
| FBref match reports | `soccerdata` | 2017-2026 | Starting XI, minutes played |

Injury news arriving between opening and closing odds could drive the line movement we're predicting. Features: count of unavailable regular starters, total minutes-share of missing players.

### Tier 3 — Likely noise for CLV

#### Weather

Open-Meteo provides free historical weather by coordinates. Would need a static stadium coordinate mapping (20 entries). Temperature, wind, precipitation at kickoff. Probably priced in by the market.

#### Venue Details

Stadium capacity, pitch dimensions, altitude. Static data, trivially compiled. Almost certainly no CLV signal.

## Key Question

The same issue that killed NBA injury features applies here: **does public information help predict line movement, or is it already priced into the opening line?**

Features most likely to add signal are those capturing information that arrives *between* snapshots — injury news, form changes after recent results. Static team quality metrics (xG, league position) are already known to the market when it sets opening prices.

Counter-argument: bet365 is a retail book that may be slower to incorporate public information than Pinnacle. The R²=0.031 result with Pinnacle as sharp reference shows genuine cross-source signal. Non-odds features might improve the model's ability to predict *which direction* bet365 will adjust toward Pinnacle.

## Implementation Plan

1. **Standings & form features** — derive from existing FDUK match results, no ingestion needed
2. **FDUK match stats** — extend ingestion script to capture shots, corners, cards, referee
3. **Schedule features** — derive rest days from match dates
4. Run tuned XGBoost experiments comparing tabular-only vs tabular+derived features
5. If signal exists, evaluate Tier 2 sources (FBref xG, FPL injuries)
