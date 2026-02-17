# Modeling

## Goal

Predict line movement: the delta between current fair price and closing fair price. A positive signal here means we can identify mispriced lines before they correct, enabling +EV execution on either sportsbook or Polymarket.

We are in the **signal discovery** phase. No execution strategy until we have evidence of predictive signal.

## Target Definition

**Devigged Pinnacle CLV delta**: `pinnacle_fair_close - pinnacle_fair_at_snapshot`

Why Pinnacle:
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

### LSTM v1 (Feb 2026)
- Implemented but not yet trained/evaluated at scale
- Hypothesis: temporal patterns in line movement (momentum, sharp money timing) may be captured better by sequence models than aggregate trajectory features

## Open Questions

### Feature Relevance
The current feature set was largely designed for a backtesting/execution system before the goal shifted to CLV delta prediction. Not all features may be relevant to predicting line movement:
- **Line shopping features** (best/worst odds, range across books) — useful for execution, but do they predict movement?
- **Market hold features** (avg/std hold) — describe market efficiency, but unclear if they predict directional change
- **Consensus probability** — may be redundant with sharp probability for a delta target
- A systematic audit of feature relevance to the delta prediction task is needed before adding more features

### Signal
- Is 193 events fundamentally too few, or is the feature set not capturing the right information?
- Do PM features add signal, or is the sparse time-alignment too limiting?
- Does sharp-retail divergence (Pinnacle vs DraftKings/FanDuel) contain more signal than cross-source (PM vs SB)?

### Features
- Should we add game context features (team records, rest days, home/away)?
- Is order book microstructure (depth, imbalance) informative at our snapshot frequency (5min)?
- Would higher-frequency PM data (sub-minute) improve velocity/acceleration features?

### Methodology
- Multi-horizon sampling: does it genuinely increase effective sample size, or just add correlated noise?
- Is devigged Pinnacle the right target, or should we explore market-wide targets (e.g., devigged consensus)?
- At what sample size should we expect to see signal if it exists?

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
