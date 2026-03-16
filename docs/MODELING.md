# Modeling

## Goal

Predict line movement: the delta between current fair price and closing fair price. A positive signal here means we can identify mispriced lines before they correct, enabling +EV execution on either sportsbook or Betfair exchange.

We have confirmed a weak but real signal (~2-3% R² on EPL with Pinnacle as sharp reference, ~3.6% R² on NBA with bet365 self-reference). Current focus: EPL with cross-source features (Pinnacle sharp vs bet365 retail) and Betfair exchange execution.

## Data Sources

Three data sources with complementary bookmaker coverage:

| | OddsPortal | Odds API | football-data.co.uk |
|--|-----------|----------|---------------------|
| **Events** | ~5K NBA (5 seasons) + ~1.8K EPL (5 seasons) | ~1K NBA (Mar–Apr 2025 + Oct 2025–Feb 2026) | ~4K EPL (11 seasons, 2015-2026) |
| **Snapshots/event** | 2 (opening + closing) | 15+ avg (all tiers) | 2 (opening + closing) |
| **Bookmakers** | UK: bet365, betway, betfred, bwin | US: pinnacle, fanduel, draftkings, betmgm, bovada | pinnacle, bet365, bwin, williamhill, betvictor, interwetten, ladbrokes |
| **Event ID pattern** | `op_YYYY-YYYY_AWAY_HOME_DATE` | hex UUID | `fduk_YYYY-YYYY_HOME_AWAY_DATE` |
| **Key value** | UK retail bookmakers | Dense snapshots, US bookmakers | Pinnacle closing odds (sharp reference) |
| **Overlap** | 1,810 EPL events matched to FDUK | 141 NBA events matched to OddsPortal | 1,810 EPL events matched to OddsPortal |

football-data.co.uk is collected manually by Joseph Buchdahl (twice weekly, post-match). Pinnacle odds sourced from Pinnacle's public API, which became unreliable after July 2025 — 81 events in 2025-26 season lack Pinnacle closing data.

### Config fields

- `data_source`: `"oddsportal"`, `"oddsapi"`, `"football_data_uk"`, `"all"`, or `null` (no filter). Filters by event ID prefix (`op_` = OddsPortal, `fduk_` = football-data.co.uk).
- `closing_source_priority`: ordered list of preferred sources for closing snapshot selection (by `api_request_id`). When multiple sources provide closing snapshots for the same event, prefer sources in this order.
- `min_snapshots`: Minimum snapshots per event (e.g., `min_snapshots: 5` for dense sequence data).
- `start_date` / `end_date`: Date range filter (always required).

### Which source for which experiment

- **EPL bet365 target with Pinnacle sharp** → `data_source: null` (combined OddsPortal + FDUK), `sharp_bookmakers: [pinnacle]`, `closing_source_priority: [football_data_uk]`. Best current EPL setup — Pinnacle as genuine sharp reference, FDUK closing preferred for target.
- **NBA bet365 target** → `data_source: oddsportal` — bet365 only available from OddsPortal. ~5K events, 5 seasons.
- **Pinnacle target** → `data_source: oddsapi` — Pinnacle only available from Odds API for NBA. ~1K events but dense snapshots.
- **LSTM experiments** → also set `min_snapshots: 5` (or higher) to ensure sufficient sequence data.
- `target_bookmaker` must belong to the corresponding source's bookmaker set.

## Target Definition

**Devigged bookmaker CLV delta**: `fair_close - fair_at_snapshot` for a configured `target_bookmaker`.

The target bookmaker must belong to the data source's bookmaker set (see Data Sources above). Pinnacle is the default for Odds API data; bet365 for OddsPortal data.

Why Pinnacle (when available):
- Sharpest bookmaker — closest to true market probability
- Proportional devigging removes ~2% vig cleanly
- 32% lower variance than raw consensus target (multi-book average)
- Single source of truth avoids averaging noise across books with different vig structures

Why delta (not absolute close):
- We don't need to predict what the line will be — we need to predict how much it will move
- Delta is stationary and mean-zero, easier to learn than absolute levels
- Directly maps to CLV: positive delta = current price is too low, line will move up

## Available Features

### Sportsbook Tabular (28 features)
Point-in-time snapshot at decision time:
- **Consensus**: avg/std odds, implied probs (home/away)
- **Sharp vs retail**: Pinnacle prob vs FanDuel/DraftKings/BetMGM avg, differential
- **Market efficiency**: num bookmakers, avg/std market hold
- **Line shopping**: best/worst odds, range across books

### Trajectory (23 features)
Aggregate statistics from the full odds sequence up to decision time:
- **Momentum**: prob change to decision, avg change rate, max increase/decrease
- **Volatility**: prob range, odds volatility, movement count
- **Trend**: slope, strength, reversals, acceleration
- **Sharp money**: sharp prob trajectory, sharp-retail divergence trend, sharp leads retail (binary)
- **Timing**: early vs recent movement distribution

### Polymarket (14 features)
PM prices and order book microstructure:
- **Implied probs**: home/away
- **Order book**: spread, midpoint, best bid/ask, bid/ask depth, imbalance, weighted mid
- **Liquidity**: volume, total liquidity
- **Velocity**: 2-hour price velocity and acceleration

### Cross-Source (7 features)
PM vs sportsbook divergence:
- **Divergence**: PM-SB prob difference (signed, absolute, direction)
- **Spread vs hold**: PM spread compared to SB market hold
- **Sharp divergence**: PM vs Pinnacle specifically

### Injury (6 features)
Impact-weighted injury burden per team at decision time:
- **Impact OUT**: home/away sum of `(on_off_rtg - on_def_rtg) * (mpg/48)` for OUT players
- **Impact GTD**: same formula for QUESTIONABLE/DOUBTFUL players, discounted 0.5x
- **Timing**: hours between latest report and game, hours between latest report and snapshot (staleness)
- Players without PBPStats data fall back to 1.0 (headcount behavior)

### Rest/Schedule (5 features)
Game context from NBA game logs:
- **Days rest**: home/away days since previous game
- **Rest advantage**: home days rest minus away days rest
- **Back-to-back**: home/away boolean flags

### Match Stats (14 features)
Rolling averages of prior match statistics from football-data.co.uk:
- **Shots**: home/away total shots, shots on target (rolling avg over last N matches, configurable `match_stats_window`, default 5)
- **Set pieces**: home/away corners
- **Discipline**: home/away fouls, yellow cards, red cards
- **Half-time**: home/away half-time goals scored
- Strict time filtering: only uses completed matches prior to the event's `commence_time` (no look-ahead bias)

### Sequence (13 features per timestep, for LSTM)
Time-series per snapshot:
- american/decimal odds, implied prob, num bookmakers
- hours to game, time of day (sin/cos encoding)
- odds/prob change from previous and from opening
- odds std, sharp odds, sharp prob, sharp-retail diff

## What We Know

### Cross-source validation (Feb 2026, 228 events)
- PM-SB probability correlation: 0.9978 — tracking the same underlying market
- PM prices systematically ~2.4pp lower than SB (liquidity premium / vig difference)
- Only 92/228 events had time-matched snapshots within 30min — sparse alignment limits cross-source features

### XGBoost v1 (Feb 2026, 193 events)
- 656 samples, 75 features, multi-horizon sampling (3.4 rows/event)
- Group timeseries CV (event-level splits)
- R² ≈ 0 — no signal beyond predicting the mean
- Likely causes: insufficient data (193 events), tabular feature bug (21/75 features zeroed), orderbook data gaps, heavy regularization

### Feature-target correlation analysis (Feb 2026, 229 events)
- 719 samples, 75 features — raw Pearson/Spearman correlations with devigged Pinnacle target
- **Strongest signal: sharp-retail divergence** (tab_retail_sharp_diff_home r=0.12, traj_sharp_retail_divergence_trend ρ=0.13)
- 12/75 features significant uncorrected (p<0.05), 0/75 after BH correction
- Bug fix: 21 tabular features were zeroed due to outcome pre-filtering in XGBoostAdapter — now resolved
- 15/75 features remain sparse (PM orderbook features require live polling, not yet deployed)
- 140 collinear feature pairs — heavy redundancy from home/away duplication and line shopping overlap
- Target variance halves closer to game (std: 0.049 far → 0.025 close); sharp-retail signal peaks 5-8h out
- Full results: [experiments/results/exp1_feature_correlations/FINDINGS.md](../experiments/results/exp1_feature_correlations/FINDINGS.md)

### Feature group isolation (Feb 2026, 230 events)
- Trained Ridge and shallow XGBoost independently on each feature group (tabular, trajectory, PM+cross-source, sharp-retail subset, all)
- Walk-forward group timeseries CV (TimeSeriesSplit on event boundaries)
- **All groups R² < 0** — no group, architecture, or decision time outperforms predicting the training mean
- Smaller groups (sharp_retail: 3 features, pm_cross_source: 9) are more stable than larger (tabular, all) — adding features makes CV worse due to collinearity
- Adding PM features to sportsbook-only features makes no difference (`all` ≈ `all_no_pm` in MSE)
- Phase 2 controlled comparison (2×2: architecture × decision time):

| | XGBoost | LSTM |
|--|---------|------|
| **time_range (3-12h)** | R²≈-0.011 (n=538) | R²≈-0.009 (n=717) |
| **tier (pregame)** | R²≈-0.024 (n=229) | R²≈-0.015 (n=228) |

- Neither architecture nor decision time matters — all four cells R²≈0
- LSTM adds no value over XGBoost at either decision time
- **TierSampler bug found and fixed**: `IN_PLAY` snapshots were incorrectly included as candidates for pregame tier — since in-play snapshots occur *after* the closing snapshot, this was look-ahead bias (features from the future "predicting" past closing prices); earlier LSTM pregame R²≈+0.02 was contaminated; corrected result is R²≈-0.015
- Full results: [experiments/results/exp2_feature_group_isolation/FINDINGS.md](../experiments/results/exp2_feature_group_isolation/FINDINGS.md)

### XGBoost with injury + rest features (Feb 2026, 800 events)
- 18 features (tabular 4 + injury 6 + rest 5 + timing 3), tier sampling (pregame), devigged Pinnacle target
- 100-trial Optuna tuning with 5-fold timeseries CV
- **First positive out-of-sample signal**: validation R²=0.050, CV mean R²=0.020 ± 0.025
- Best params: `max_depth=2, min_child_weight=20, lr=0.295` — heavy regularization prevents overfitting
- Feature importance: **injuries 55%** (`impact_out_away` 17%, `injury_news_recency` 12%), tabular 28%, rest 9%
- Dead features (zero importance): `away_is_b2b`, `away_days_rest`, `rest_advantage`, `home_is_b2b`, `is_weekend`, `num_bookmakers`
- Config: `experiments/xgboost_injuries_rest_tuning_best.yaml` (gitignored)

### LSTM tuning (Feb 2026, 800 events)
- Two-branch architecture: LSTM processes 15-feature sequences (24 timesteps), optional static feature branch (tabular + injury + rest) concatenated with final hidden state
- 100-trial Optuna tuning with 5-fold timeseries CV, same date range as XGBoost experiment (204 events skipped due to missing snapshots/sequences)
- Controlled comparison: sequence-only vs sequence + static features

| Variant | Features | CV R² | CV MSE | Best params |
|---------|----------|-------|--------|-------------|
| **Sequence only** | 15 seq features × 24 timesteps | -0.010 ± 0.039 | 0.000471 ± 0.000139 | hidden=48, layers=2, dropout=0.2, lr=0.00289 |
| **Sequence + static** | 15 seq + 17 static (tab 6 + inj 6 + rest 5) | -0.122 ± 0.115 | — | hidden=112, layers=3, dropout=0.0, lr=0.00229 |

- **Both variants R² < 0** — LSTM does not outperform predicting the mean, regardless of static features
- Adding static features made things *worse* (R² -0.122 vs -0.010) with much higher variance (±0.115 vs ±0.039), suggesting the larger model overfits
- XGBoost R²=+0.020 remains the best architecture — injury signal is captured by tabular features, not temporal patterns
- Hypothesis disproven: temporal patterns in line movement sequences do not improve prediction beyond aggregate features
- Configs: `experiments/lstm_tuning_seq_only_best.yaml`, `experiments/lstm_tuning_best.yaml`

### XGBoost bet365 at scale (Feb 2026, ~5K events OddsPortal)
- 11-fold walk-forward CV, 100-trial Optuna tuning, devigged bet365 target
- **Tabular + injuries**: CV R²=0.036 ± 0.033, MSE=0.002883 — best params: n_est=350, max_depth=3, lr=0.054, min_child_weight=37
- **Tabular-only baseline**: CV R²=0.036 ± 0.028, MSE=0.002884 — best params: n_est=250, max_depth=3, lr=0.280, min_child_weight=36
- **Identical performance** — injuries add zero signal when both variants are properly tuned under the same CV scheme
- Earlier comparison (untuned baseline vs tuned injuries) was misleading; the apparent "injury signal" was a tuning artifact
- ~3.6% R² appears to be the ceiling for public sportsbook features
- Configs: `experiments/configs/xgboost_bet365_tuning_best.yaml`, `experiments/configs/xgboost_bet365_baseline_tuning_best.yaml`

### Hours-to-game effect (Feb 2026, 477 events Odds API)
- 1,593 samples across 1-48h before game, time_range sampling with up to 10 snapshots per event
- **Target variance increases with decision distance**: mean |CLV delta| = 1.4pp at 0-3h → 3.0pp at 24-36h. Market efficiency peaks at 3-6h (std=0.022, lowest).
- **Sharp-retail divergence peaks at 3-6h** (r=+0.097) — confirming the pregame window as the sweet spot for this feature
- **GTD injury at 0-3h is the strongest individual signal found** (r=+0.28, p=0.0003) — late injury news not yet priced in. Sign flips negative at 18-24h (early reports already reflected).
- **Prob level correlation flips**: high prob → negative CLV close to game (reversion), positive CLV far from game (momentum)
- **Per-bin models all R²<0**: 477 Odds API events too small for per-hour model evaluation. OddsPortal's 5K events only have 2 snapshots, preventing hour stratification.
- Current pregame tier (3-12h) is a reasonable default; no evidence to change
- Full results: [experiments/results/exp4_hours_to_game/FINDINGS.md](../experiments/results/exp4_hours_to_game/FINDINGS.md)

### LSTM Pinnacle tuning (Feb 2026, ~1K events Odds API)
- 5-fold walk-forward CV, 50-trial Optuna tuning, devigged Pinnacle target
- `data_source: oddsapi`, `min_valid_timesteps: 3`, pregame decision tier (3-12h), packed sequences for correct masking
- **CV R²=-0.075 ± 0.113** — worse than a constant predictor
- Best params: hidden_size=128, num_layers=3, dropout=0.5, lr=0.00738, batch_size=32, weight_decay=0.000541, patience=20
- LSTM conclusively ruled out for CLV prediction with current features — sequential modeling adds no value over cross-sectional features
- Config: `experiments/configs/lstm_pinnacle_tuning_best.yaml`

### XGBoost Pinnacle tuning (Mar 2026, ~800 events Odds API)
- 4-fold walk-forward CV, 100-trial Optuna tuning, devigged Pinnacle target
- **CV R²=-0.017 ± 0.015**, MSE=0.000431 — tuner converged to max regularization (no signal)
- Best params: n_est=400, max_depth=2, lr=0.145, min_child_weight=47, subsample=0.5
- Pinnacle MSE (0.000431) is 6.7× lower than bet365 MSE (0.002883) — Pinnacle lines move less, leaving less variance to predict
- **Conclusion**: bet365 is the viable target for CLV prediction with public features. Pinnacle would need substantially more history or non-public features (order flow, bettor identity) to revisit.
- Config: `experiments/configs/xgboost_pinnacle_tuning_best.yaml`

### Data volume learning curve (Mar 2026, 500–4,524 events OddsPortal)
- Walk-forward CV with tuned bet365 baseline params (tabular-only) on chronological subsets of 500 → 4,524 events
- **R² plateaus at ~1,500 events**: oscillates between 0 and +0.035 from 1.5K onwards, with no upward trend
- Log-fit: R² = 0.0301·ln(N) − 0.2302, marginal return dR²/dN = 6.7e-06 at N=4,524
- Extrapolated: R²≈0.047 at 10K events, R²≈0.068 at 20K events — diminishing returns
- **Injury timing diagnostic (2×2)**: tabular-only vs tabular+injuries at sharp (12-24h) vs pregame (3-12h) tier — injuries add nothing at either tier
- Pregame tier (3-12h) yields lower R² than sharp (0.001 vs 0.018) — consistent with OddsPortal snapshot timing centered at ~19h
- **Caveat resolved by Exp 6b**: OddsPortal had zero snapshots in the 0-3h closing window. Exp 6b tested this on 479 Odds API events with bet365 closing-tier snapshots (avg 0.2h) — injuries add exactly zero. The closing-tier target is near-zero (std=0.0016) because the line has already priced in GTD designations.
- Full results: [experiments/results/exp6_learning_curve/FINDINGS.md](../experiments/results/exp6_learning_curve/FINDINGS.md)

### Walk-forward betting simulation (Mar 2026, ~4,500 events OddsPortal)
- Walk-forward CV (12 folds, expanding window), flat $100 bets on bet365 vigged American odds, threshold sweep [0.005–0.05]
- **Not profitable**: all thresholds from 0.005 to 0.03 produce negative ROI (-2.4% to -11.8%)
- Only threshold=0.05 shows +3.36% ROI (56 bets), but **NOT significant** (p=0.260, 1K permutations) — likely noise from small sample
- Avg CLV captured is positive at all thresholds (+0.010 to +0.047), confirming the model identifies directional line movement, but the edge is consumed by the vig
- **Always-home baseline**: 54.2% win rate, -7.55% ROI — confirms the ~4.5% house edge the model must overcome
- Away bias: model predicts more away bets than home bets at all thresholds (e.g., 812 away vs 495 home at 0.005), suggesting systematic away-side line movement in bet365 pricing
- **Conclusion**: ~3.6% R² CLV signal is directionally correct but too weak to overcome bet365's vig (~4.5%) for flat betting. Would need either stronger signal (non-public features) or lower-cost execution venues
- Full results: [experiments/results/exp7_backtest_sim/FINDINGS.md](../experiments/results/exp7_backtest_sim/FINDINGS.md)

### Line shopping across bookmakers (Mar 2026, ~4,500 events OddsPortal)
- Same walk-forward simulation as Exp 7, but selects the best available price across 4 UK bookmakers (bet365, betway, betfred, bwin) instead of bet365-only
- **Vig reduction**: effective overround drops from 4.13% (bet365) to 1.59% (best-available), a 2.55pp reduction
- **ROI improvement at every threshold**: +0.3pp to +1.3pp improvement, with largest gains at lower thresholds (1.3pp at 0.005, 0.9pp at 0.01)
- **Still not profitable**: all thresholds 0.005–0.03 remain negative ROI. Best threshold=0.05 improves from +3.36% to +3.68% ROI but remains insignificant (p=0.334)
- **bet365 provides best price 67% of the time**; betfred 13.1%, betway 12.4%, bwin 7.5% — bet365 is already competitive, limiting the shopping upside
- **Conclusion**: line shopping is directionally correct but insufficient — the ~2.5pp vig reduction is smaller than the ~4% gap between signal strength and breakeven. Would need either exchange-level execution costs or stronger signal
- Full results: [experiments/results/exp7b_line_shopping/FINDINGS.md](../experiments/results/exp7b_line_shopping/FINDINGS.md)

### EPL with Pinnacle sharp reference (Mar 2026, ~1.7K events combined)

Three-way comparison isolating the effect of Pinnacle as sharp reference vs more data. **Note**: the original Mar 10 results (R²=0.031) were inflated by a cross-source snapshot contamination bug (#231, fixed Mar 15) — OddsPortal snapshots without Pinnacle were selected over FDUK snapshots with Pinnacle, then `nan_to_num` silently filled missing Pinnacle probs with 0.0. Post-fix results below are the trustworthy baseline.

**Post-fix results (walk-forward, 100 trials each):**

| Experiment | Sharp Ref | Data | Features | CV R² | CV MSE |
|---|---|---|---|---|---|
| OddsPortal-only baseline | bet365 | OP only | 7 tabular | 0.016 ± 0.045 | 0.002410 |
| Combined + Pinnacle sharp | pinnacle | OP + FDUK | 7 tabular | 0.019 ± 0.043 | 0.000793 |
| Combined + Pinnacle + match_stats | pinnacle | OP + FDUK | 21 (7 tab + 14 match_stats) | 0.025 ± 0.042 | 0.000782 |
| Combined + Pinnacle + standings | pinnacle | OP + FDUK | 18 (7 tab + 11 standings) | 0.030 ± 0.040 | 0.000774 |
| Combined + Pinnacle + standings + match_stats | pinnacle | OP + FDUK | 32 (7 tab + 11 stnd + 14 mstat) | **0.052 ± 0.045** | 0.000767 |

- **Pinnacle as sharp reference** improves R² modestly (0.016 → 0.019) with much lower MSE (0.0024 → 0.0008) because Pinnacle closing prices are tighter reference points
- **Standings features add ~1% absolute R²** (0.019 → 0.030) — league position, points gap, GD, last-5 form, goal rates. Std overlap means not conclusive.
- **Match stats add incremental signal** (0.019 → 0.025 alone, 0.030 → 0.052 on top of standings) — rolling shots, corners, fouls, cards. The stacking pattern suggests match stats and standings capture complementary information.
- **Combined standings + match_stats is the best EPL result** (R²=0.052) — nearly 3× the tabular-only baseline. MSE improves monotonically across all configs.
- **More data alone adds no signal** — the control (combined data, bet365 sharp) gave R²=-0.005 ± 0.008
- MSE drop (0.0024 → 0.0008) reflects lower target variance with Pinnacle closing, not necessarily better prediction
- Tuner still converges to high regularization — weak signal regime
- 81/1,810 events (Jan-Mar 2026) lack Pinnacle closing data due to Pinnacle API shutdown (July 2025); small impact
- Configs: `experiments/configs/xgboost_epl_combined_tuning_best.yaml`, `experiments/configs/xgboost_epl_combined_standings_tuning_best.yaml`, `experiments/configs/xgboost_epl_combined_match_stats_tuning_best.yaml`, `experiments/configs/xgboost_epl_combined_all_tuning_best.yaml`, `experiments/configs/xgboost_epl_combined_bet365sharp_tuning_best.yaml`

**Cross-source contamination bug (#231):** TierSampler selected decision snapshots by latest wall-clock time from eligible tiers. OddsPortal `sharp`-tier snapshots (no Pinnacle) won over FDUK `early`-tier snapshots (has Pinnacle) for ~37/150 events in the final walk-forward fold (Apr-Oct 2025). Combined with `np.nan_to_num(X, nan=0.0)` converting missing Pinnacle probs to 0.0, this corrupted cross-source features and distorted tuner optimization across all folds. Fix: TierSampler now accepts `required_bookmakers` to filter candidates; all `nan_to_num` calls removed (XGBoost handles NaN natively).

## Open Questions

### Signal
- ~~~3.6% R² is the ceiling for public sportsbook features at 5K events, with the learning curve plateaued. Can Polymarket cross-source features push it higher?~~ — **Partially answered**: Pinnacle cross-source features push EPL R² to ~2% (from 1.6% OddsPortal-only baseline); adding standings reaches ~3%, and standings + match stats reaches ~5.2%. Non-odds public features (league context + match performance) nearly triple the tabular-only baseline. PM features remain untested at scale.
- ~~Injury features dominate importance~~ — **Answered** (Exp 6 + 6b): injuries add zero signal at all decision tiers when properly tuned. GTD injury r=0.28 (Exp 4, Pinnacle) does not transfer to bet365 target (r=-0.01, Exp 6b). At closing tier, the line has already priced in GTD designations. Earlier apparent importance was a tuning artifact.
- ~~Does the signal generalize to other sports, or is it NBA-specific?~~ — **Answered**: EPL CLV is predictable (R²≈2-3% with Pinnacle sharp + standings), confirming the signal generalizes beyond NBA. Cross-source sharp-retail divergence appears to be the key feature across sports.
- Do PM features add signal? Untested at scale — only tested with 230 events (insufficient data). PM order flow from CLOB snapshots remains untapped.
- ~~Does sharp-retail divergence (Pinnacle vs DraftKings/FanDuel) contain more signal than cross-source (PM vs SB)?~~ — **Partially answered**: Pinnacle vs bet365 divergence is the strongest signal found (R²≈2-3% on EPL). PM features remain untested at scale but are deprioritized (UK inaccessible).

### Execution
- ~~At what hours-before-game does the model's edge peak?~~ — **Answered** (Exp 4 + 6b): sharp-retail diff peaks at 3-6h (Exp 4). GTD injury r=0.28 at 0-3h (Exp 4, Pinnacle target) does NOT transfer to bet365 target (r=-0.01, Exp 6b) — the closing-tier catch-22 means the line has already priced in GTD designations by the time they're visible.
- Can cross-venue execution (sportsbook vs Polymarket) extract more value than single-venue?
- ~~What is the optimal bet sizing given the weak but real signal?~~ — **Moot** (Exp 7): flat betting is unprofitable; Kelly sizing cannot fix an insufficient edge. Would need stronger signal first.

### Data
- ~~Is more OddsPortal data worth collecting?~~ — **No** (Exp 6): learning curve plateaued at ~1.5K events. More events of the same type won't help.
- **Pinnacle data continuity risk**: football-data.co.uk sources Pinnacle odds from Pinnacle's public API, which became unreliable after July 2025. 81 EPL events in 2025-26 already lack Pinnacle closing. Need an alternative Pinnacle source (direct API access requires commercial partnership application to `api@pinnacle.com`; or via The Odds API which carries Pinnacle).
- PM order flow features from existing CLOB snapshots — untapped data source (deprioritized — UK inaccessible)
- Betfair Exchange historical data (historicdata.betfair.com) offers 1-min to 50ms tick data — could replace Pinnacle as sharp reference with even finer granularity

### Methodology
- Is devigged Pinnacle the right target, or should we explore market-wide targets?
- Multi-horizon sampling: does it genuinely increase effective sample size, or just add correlated noise?

## Experiment Plan

### Completed
- **~~1. Feature-target correlation analysis~~** — max |r|=0.12, sharp-retail diff strongest, 0/75 significant after BH correction
- **~~2. Feature group isolation~~** — all groups R²<0 at 230 events; TierSampler IN_PLAY bug found
- **~~3. Minimal feature models~~** — subsumed by XGBoost bet365 at scale: 4 tabular features match 10-feature model
- **~~4. Hours-to-game effect~~** — target variance grows with decision distance; sharp-retail peaks at 3-6h; GTD injury r=0.28 at 0-3h. Per-bin models inconclusive (478 events). Pregame window confirmed. [Full results](../experiments/results/exp4_hours_to_game/FINDINGS.md).
- **~~5. LSTM evaluation~~** — LSTM R²<0 across all configurations (800-event Pinnacle, 1K-event Pinnacle with masking fix). Conclusively worse than XGBoost.

- **~~6. Data volume learning curve~~** — R² plateaued at ~1.5K events, oscillating 0–0.035 thereafter (log-fit dR²/dN = 6.7e-06). More OddsPortal data will not meaningfully improve the model. Injury timing diagnostic (2×2): injuries add nothing at sharp tier; pregame comparison diluted (~93% of events fall back to sharp-tier snapshots, only ~335 sample at actual 3-12h). The 0-3h closing window where GTD signal is strongest remains untested. [Full results](../experiments/results/exp6_learning_curve/FINDINGS.md).

- **~~6b. Injury closing tier~~** — Tested GTD hypothesis on Odds API data with bet365 closing-tier snapshots (avg 0.2h before game). Closing-tier R²=0.596 is a measurement artifact: target std=0.0016 (line has barely moved at 0.2h), so the model explains near-zero residuals. Injuries add exactly zero at closing tier (identical R² to tabular-only). `inj_impact_gtd_away` r=-0.011 at closing — the Exp 4 r=0.28 does not transfer to bet365 target. Injury signal is conclusively uninformative for CLV prediction at any tier. [Full results](../experiments/results/exp6b_injury_closing_tier/FINDINGS.md).

- **~~7. Walk-forward betting simulation~~** — Flat $100 bets on bet365 vigged odds across 6 thresholds (0.005–0.05). All thresholds 0.005–0.03 produce negative ROI (-2.4% to -11.8%). Only threshold=0.05 shows +3.36% ROI (56 bets) but p=0.260 (not significant). Model captures positive CLV directionally but the ~3.6% R² signal is too weak to overcome bet365's ~4.5% vig. [Full results](../experiments/results/exp7_backtest_sim/FINDINGS.md).

- **~~7b. Line shopping across bookmakers~~** — Re-ran Exp 7 simulation selecting best available price across 4 UK bookmakers (bet365, betway, betfred, bwin). Effective vig drops from 4.13% to 1.59% (2.55pp reduction), ROI improves +0.3–1.3pp at every threshold, but still not profitable at thresholds ≤0.03. bet365 provides the best price 67% of the time, limiting shopping upside. [Full results](../experiments/results/exp7b_line_shopping/FINDINGS.md).

### Active

### 8. Cross-venue execution analysis (deprioritized)
~~Compare execution opportunity across sportsbook and Polymarket for the same events.~~
**Deprioritized:** Polymarket not accessible from UK; focus shifted to EPL football with Betfair exchange as the cross-venue execution target (deep liquidity confirmed — 0.5-0.7% spreads on EPL favourites).

### ~~9. Position sizing / Kelly criterion~~
**Deprioritized** (Exp 7): flat betting is unprofitable — Kelly sizing cannot fix an insufficient edge. Revisit only if signal improves via non-public features or cross-venue execution.

## Running Experiments

Experiments live in `experiments/scripts/` as standalone Python scripts. Each script writes outputs to `experiments/results/<experiment_name>/`.

### Required outputs

Every experiment must produce:
- **`FINDINGS.md`** — the primary artifact future agents read. Must follow this structure:
  - **Setup**: date, git SHA, dataset size, sampling method, target definition, exact command to reproduce
  - **Key Results**: numbers, tables, significance — no vague summaries
  - **Interpretation**: what the results mean, caveats, alternative explanations
  - **Implications**: what to do next, go/no-go recommendation for downstream experiments
- **Plots** saved as PNGs in the same directory.
- **Data artifacts** (CSVs, etc.) for downstream analysis.

### After running

1. Update the **Experiment Log** table below with the headline result and decision.
2. Update the **What We Know** section if the experiment changes our understanding, linking to the FINDINGS.md. If results contradict a prior entry, amend the prior entry inline with the correction and a link to the new experiment (do not delete — preserve the history of what we believed and why it changed).

### Conventions

- Scripts use the existing training pipeline (`prepare_training_data`, `MLTrainingConfig`, etc.) to load data.
- Config reuse: reference existing YAML configs in `experiments/configs/` where possible.
- Scripts must be runnable via `uv run python experiments/scripts/<script>.py`.
- FINDINGS.md must record the git SHA of the commit containing the experiment code (commit the script first, then run, then commit results).

## Experiment Log

| Date | Experiment | Features | Target | Samples | Result | Decision | Notes |
|------|-----------|----------|--------|---------|--------|----------|-------|
| 2026-02-14 | XGBoost v1 | tabular + trajectory + PM + cross-source | devigged pinnacle | 656 (193 events) | R² ≈ 0 | — | Multi-horizon, group CV; 21 tab features zeroed (bug) |
| 2026-02-17 | Exp 1: correlations | 60 testable / 75 total | devigged pinnacle | 719 (229 events) | max \|r\|=0.12 | Proceed to Exp 2 | Sharp-retail diff strongest; 12/60 uncorrected, 0/60 BH |
| 2026-02-18 | Exp 2: feature groups | 47 features, 6 groups | devigged pinnacle | 538–719 (230 events) | All R²<0 | No signal at 230 events | 2×2 (arch × time): all cells R²≈0; TierSampler IN_PLAY bug fixed |
| 2026-02-20 | XGBoost + injuries/rest | tabular 4 + injury 6 + rest 5 + timing 3 | devigged pinnacle | 800 events | val R²=0.050, CV R²=0.020±0.025 | First positive signal | 100-trial Optuna; injuries 55% importance; 6 dead features |
| 2026-02-21 | LSTM seq-only | 15 seq features × 24 timesteps | devigged pinnacle | 800 events | CV R²=-0.010±0.039 | No signal | 100-trial Optuna; best: hidden=48, layers=2 |
| 2026-02-21 | LSTM + static branch | 15 seq + 17 static (tab+inj+rest) | devigged pinnacle | 800 events | CV R²=-0.122±0.115 | Worse than seq-only | Static features increase overfitting; XGBoost remains best |
| 2026-02-27 | XGBoost bet365 tuned | tabular 4 + injury 6 | devigged bet365 | ~5K events (OddsPortal) | CV R²=0.036±0.033 | Plateau at ~3.6% | 11-fold walk-forward; injuries add zero over tabular-only |
| 2026-02-27 | XGBoost bet365 baseline tuned | tabular 4 | devigged bet365 | ~5K events (OddsPortal) | CV R²=0.036±0.028 | Same as +injuries | Confirms injuries are noise; public features plateau |
| 2026-02-27 | LSTM mask fix | 15 seq features × 8 timesteps | devigged pinnacle | ~1K events (Odds API) | — | Bug fix | Packed sequences for correct mask application (#162) |
| 2026-02-28 | LSTM Pinnacle tuned | 15 seq × 8 timesteps | devigged pinnacle | ~1K events (Odds API) | CV R²=-0.075±0.113 | LSTM ruled out | 50-trial Optuna, 5-fold walk-forward; packed sequences; worse than constant predictor |
| 2026-02-28 | Exp 4: hours-to-game | tabular 6 + injury 6 | devigged pinnacle | 1,593 (477 events, Odds API) | All bins R²<0; sharp-retail peaks 3-6h | Pregame window confirmed | CLV delta grows with hours; GTD injury r=0.28 at 0-3h; dataset too small for per-bin models |
| 2026-03-01 | XGBoost Pinnacle tuned | tabular 4 + injury 6 | devigged pinnacle | ~800 events (Odds API) | CV R²=-0.017±0.015 | No signal | 100-trial walk-forward; max regularization; Pinnacle CLV unpredictable with public features |
| 2026-03-01 | Exp 6: learning curve | tabular 4 | devigged bet365 | 500–4,524 events (OddsPortal) | Plateau at ~1.5K; R²≈0.02 | More data won't help | Log-fit dR²/dN=6.7e-06; injuries add nothing at sharp; pregame tier diluted (93% fallback to sharp) |
| 2026-03-01 | Exp 6b: injury closing tier | tabular 4 ± injury 6 | devigged bet365 | 479 (closing) / 826 (sharp) events (Odds API) | Closing R²=0.596 (artifact); sharp R²=-0.02 | GTD hypothesis falsified | Closing-tier target near-zero (std=0.0016); injuries add zero at both tiers; inj_impact_gtd_away r=-0.01 (not r=0.28 from Exp 4) |
| 2026-03-01 | Exp 7: backtest sim | tabular 4 | devigged bet365 | 2,088 bets (4,524 events OddsPortal) | Best ROI +3.36% (p=0.260) | Not profitable | Flat $100, 12-fold walk-forward; all thresholds ≤0.03 negative ROI; CLV signal too weak to overcome ~4.5% vig |
| 2026-03-01 | Exp 7b: line shopping | tabular 4 | devigged bet365 | 2,088 bets (4 bookmakers) | Best ROI +3.68% (p=0.334) | Still not profitable | Vig drops 4.13%→1.59% (2.55pp); ROI +0.3–1.3pp at all thresholds; bet365 best price 67% of time |
| 2026-03-10 | EPL combined + Pinnacle sharp | tabular 7, pinnacle sharp | devigged bet365 | ~1.8K EPL (OP+FDUK) | ~~CV R²=0.031±0.014~~ | ~~Best EPL result~~ | **Inflated by #231 bug** — pre-fix snapshot contamination + nan_to_num; see post-fix row below |
| 2026-03-10 | EPL combined + bet365 sharp (control) | tabular 7, bet365 sharp | devigged bet365 | ~1.8K EPL (OP+FDUK) | CV R²=-0.005±0.008 | More data ≠ signal | Control proves improvement from Pinnacle features, not data volume |
| 2026-03-15 | #231 bug fix | — | — | — | — | Bug fix | Cross-source snapshot contamination: TierSampler picked OP (no Pinnacle) over FDUK; nan_to_num filled NaN→0.0. Fix: required_bookmakers filter + remove nan_to_num |
| 2026-03-15 | EPL combined + Pinnacle (post-fix) | tabular 7, pinnacle sharp | devigged bet365 | ~1.7K EPL (OP+FDUK) | CV R²=0.019±0.043 | Trustworthy baseline | 6-fold walk-forward, 100 trials; post-fix clean results |
| 2026-03-15 | EPL combined + standings | tabular 7 + standings 11, pinnacle sharp | devigged bet365 | ~1.7K EPL (OP+FDUK) | CV R²=0.030±0.040 | +1% R² from standings | 6-fold walk-forward, 100 trials; league position, form, GD features |
| 2026-03-16 | EPL combined + match_stats | tabular 7 + match_stats 14, pinnacle sharp | devigged bet365 | ~1.7K EPL (OP+FDUK) | CV R²=0.025±0.042 | +0.6% R² over tabular | 5-fold walk-forward, 100 trials; rolling shots, corners, fouls, cards |
| 2026-03-16 | EPL combined + standings + match_stats | tabular 7 + standings 11 + match_stats 14, pinnacle sharp | devigged bet365 | ~1.7K EPL (OP+FDUK) | CV R²=0.052±0.045 | Best EPL result | 5-fold walk-forward, 100 trials; match stats add incremental signal on top of standings |
