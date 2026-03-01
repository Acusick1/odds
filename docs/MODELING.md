# Modeling

## Goal

Predict line movement: the delta between current fair price and closing fair price. A positive signal here means we can identify mispriced lines before they correct, enabling +EV execution on either sportsbook or Polymarket.

We are in the **signal discovery** phase. No execution strategy until we have evidence of predictive signal.

## Data Sources

Two data sources with **non-overlapping bookmaker sets**:

| | OddsPortal | Odds API |
|--|-----------|----------|
| **Events** | ~5,000 (5 NBA seasons, 2021-2026) | ~1,000 (Mar–Apr 2025 + Oct 2025 – Feb 2026) |
| **Snapshots/event** | 2 (opening + closing) | 15+ avg (all tiers) |
| **Bookmakers** | UK: bet365, betway, betfred, bwin | US: pinnacle, fanduel, draftkings, betmgm, bovada |
| **Event ID pattern** | `op_YYYY-YYYY_AWAY_HOME_DATE` | hex UUID |
| **Overlap** | 141 events matched to Odds API events (Oct 2025+) |

### Config fields

- `data_source`: `"oddsportal"`, `"oddsapi"`, `"all"`, or `null` (no filter). Filters by event ID prefix (`op_` = OddsPortal).
- `min_snapshots`: Minimum snapshots per event (e.g., `min_snapshots: 5` for dense sequence data).
- `start_date` / `end_date`: Date range filter (always required).

### Which source for which experiment

- **bet365 target** → `data_source: oddsportal` — bet365 only available from OddsPortal. ~5K events, 5 seasons. Best for tabular/injury features with 2-snapshot coverage.
- **Pinnacle target** → `data_source: oddsapi` — Pinnacle only available from Odds API. ~1K events but dense snapshots. Best for LSTM/sequence features.
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
- Walk-forward CV (11 folds, expanding window), 100-trial Optuna
- **Tabular + injuries**: CV R²=0.036 ± 0.033, test R²=0.079 (744 held-out)
- **Tabular-only baseline**: CV R²=0.036 ± 0.028 — injuries add zero signal once both are tuned
- Public features plateau at ~3.6% explained variance on bet365 CLV
- Config: `experiments/configs/xgboost_bet365_tuning_best.yaml`

### Pinnacle CLV: XGBoost + LSTM (Feb–Mar 2026, ~800 events Odds API)
- Both architectures yield **negative R²** on devigged Pinnacle target:

| Architecture | CV R² | CV MSE | Notes |
|-------------|-------|--------|-------|
| **XGBoost** (100 trials, 4-fold walk-forward) | -0.017 ± 0.015 | 0.000431 | Max regularization; no signal |
| **LSTM** (50 trials, 5-fold) | -0.075 ± 0.113 | 0.000437 | Negative R²; ruled out |

- Pinnacle MSE (0.000431) is 6.7× lower than bet365 MSE (0.002883) — Pinnacle lines move less, leaving less variance to predict
- Pinnacle is the sharpest book; less pricing inefficiency to exploit vs retail books like bet365
- **Conclusion**: bet365 is the viable target for CLV prediction with public features. Pinnacle would need substantially more history or non-public features (order flow, bettor identity) to revisit.
- Configs: `experiments/configs/xgboost_pinnacle_tuning_best.yaml`, `experiments/configs/lstm_pinnacle_tuning_best.yaml`

## Open Questions

### Signal
- R²=0.05 is positive but small — is this the ceiling for public features, or can more data / PM order flow push it higher?
- Injury features dominate importance — is the model primarily learning "star player OUT → line moves"? If so, how robust is this across seasons?
- Does the signal generalize to other sports, or is it NBA-specific (injury report timing)?
- Do PM features add signal, or is the sparse time-alignment too limiting? (untested with 803-event dataset)
- Does sharp-retail divergence (Pinnacle vs DraftKings/FanDuel) contain more signal than cross-source (PM vs SB)?

### Data
- **Historical odds backfill**: 297 events from 2024-25 have injuries + player stats but no odds snapshots. Worth the API cost (10 units/region/market)?
- PM order flow features from existing CLOB snapshots — untapped data source
- Is order book microstructure (depth, imbalance) informative at our snapshot frequency (5min)?
- Would higher-frequency PM data (sub-minute) improve velocity/acceleration features?

### Methodology
- Should we prune the 6 dead features, or keep them for future tuning runs with more data?
- Is devigged Pinnacle the right target, or should we explore market-wide targets?
- Multi-horizon sampling: does it genuinely increase effective sample size, or just add correlated noise?

## Experiment Plan

Ordered by priority. Each experiment should inform whether to proceed with the next.

### 1. Feature-target correlation analysis
No model — compute raw Pearson/Spearman correlations between each feature and the devigged Pinnacle target. If nothing correlates individually, no model will extract signal from combinations. Directly addresses the feature relevance question.

### 2. Feature group isolation
Train simple models (ridge regression or shallow XGBoost) on each feature group independently:
- Tabular only (28 features)
- Trajectory only (23 features)
- PM + cross-source only (21 features)
- Sharp-retail divergence only (3-5 features)

Sharp-retail divergence is our strongest theoretical prior — if Pinnacle moves before retail, the divergence at decision time should predict further movement.

### 3. Minimal feature models
Take the top 5-10 correlated features from experiment 1 and train a simple model. With 193 events, 75 features is almost certainly overparameterized. Fewer features may perform better with the same data.

### 4. Hours-to-game effect
Plot target variance and feature correlations as a function of hours before game. Markets closer to game time may be more efficient (harder to predict) or less (sharp money arriving late). Informs optimal decision time.

### 5. LSTM evaluation
Compare LSTM on sequence data against trajectory-only XGBoost. The hypothesis: aggregate trajectory features (slope, momentum) lose timing information a sequence model could capture. Only worth running if experiments 1-2 show trajectory/temporal features carry signal.

### 6. Data volume learning curve
Train on increasing subsets (50, 100, 150, 193 events). If performance improves at 193, the bottleneck is data volume and we should prioritize collection over feature engineering. If flat, more data won't help.

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
| 2026-02-27 | LSTM Pinnacle tuned | 15 seq × 8 timesteps + tabular + injury | devigged pinnacle | ~1K events (Odds API) | CV R²=-0.075±0.113 | No signal | 50-trial Optuna; negative R²; LSTM ruled out for CLV |
| 2026-03-01 | XGBoost Pinnacle tuned | tabular 4 + injury 6 | devigged pinnacle | ~800 events (Odds API) | CV R²=-0.017±0.015 | No signal | 100-trial walk-forward; max regularization; Pinnacle CLV unpredictable with public features |
