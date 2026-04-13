# Evaluation Framework

How to measure whether this system can make money, and what must be true for it to work.

## The Bet

The system predicts CLV (closing line value): how much current odds will move before the market closes. A positive predicted CLV means the current price is too generous — the market will tighten. If we bet now and the prediction is correct, we got a better price than the closing line, which is positive expected value.

The profit from a single bet is:

```
profit = actual_price_improvement - execution_cost
```

Where:
- **actual_price_improvement**: the gap between the odds we bet at and the closing fair odds
- **execution_cost**: venue-dependent (Betfair spread + commission, or sportsbook vig)

## What We Need to Measure

### 1. Does the model rank correctly?

The most fundamental question. When the model predicts higher CLV, does actual CLV tend to be higher?

- **Metric**: Spearman rank correlation between predicted and actual CLV
- **Visualisation**: Scatter plot of predicted vs actual CLV, binned calibration plot
- **Why it matters**: Even with low R², if the model correctly identifies which games have the most CLV, we can set a threshold and only bet the top predictions

### 2. Are tail predictions profitable?

Overall R² (~1.6–3.6%) is low, but we don't bet on every game. We only care whether the high-confidence predictions (top decile by predicted CLV) are accurate enough to overcome execution costs.

- **Metric**: Mean actual CLV for predictions above various thresholds (1%, 2%, 3%+)
- **Why it matters**: A model with 3% overall R² but well-calibrated tails can still be profitable. A model with higher R² but poor tail calibration cannot.

### 3. P&L simulation

The bottom line. Simulate flat-stake betting on all predictions above a threshold, accounting for execution costs.

- **Betfair execution cost**: ~2–5% commission on net winnings + ~0.5% spread. Effective cost per bet roughly 1–2% depending on odds.
- **Sportsbook execution cost**: 3–5% vig embedded in price. Higher cost but no commission structure.
- **Metric**: Cumulative P&L, ROI (profit / total staked), yield per bet, by threshold
- **Sweep thresholds**: Find the threshold that maximises P&L — this tells us the minimum signal the model needs to produce

### 4. How often is it right directionally?

When the model predicts positive home CLV, is actual home CLV positive?

- **Metric**: Directional accuracy (% of positive predictions that have positive actual CLV)
- **Why it matters**: Below 50% means the model is worse than random for bet selection. Above ~55% is likely profitable after costs on Betfair.

## Current Gaps

### Only home CLV is predicted

The model predicts CLV for the home outcome only. This means:

- We can identify "+EV home" bets when predicted CLV is positive and large
- We **cannot** identify "+EV away" or "+EV draw" bets — a negative home CLV suggests away/draw may be underpriced, but we don't know by how much
- **Impact**: We're ignoring ~2/3 of potential betting opportunities

### No Pinnacle in live data

The best research result (EPL R²=3.1%) used Pinnacle closing odds as the sharp reference. Pinnacle isn't available in the live OddsPortal scrape (UK IP). Without Pinnacle, the OddsPortal-only EPL baseline was R²=1.6%.

Available sharp references in live data: betfair_exchange, 1xbetir, betinasia. Betfair Exchange is the most credible sharp reference — deep liquidity, low overround — but hasn't been tested as a sharp reference in the training pipeline.

**This is the most important open question**: can Betfair Exchange serve as the sharp reference in the live pipeline and recover the signal lost from not having Pinnacle?

### No execution price tracking

The system records predictions but not the price available at prediction time. Without this, we can't measure:

- What odds we would have actually gotten
- Slippage between prediction time and execution
- Whether the best venue (Betfair vs sportsbook) was actually available

### Snapshot-to-prediction timing

Predictions are scored hourly on every new snapshot. But the most valuable predictions are those made when:

1. There's enough time to execute (not 5 minutes before kickoff)
2. The line hasn't already moved (early predictions on stale lines may be too late)

We need to track hours-to-kickoff at prediction time and evaluate whether earlier or later predictions are more profitable.

## Alignment with Strategic Goal

The CLAUDE.md strategic goal is:

> Predict line movement (closing line value) using cross-source market data. Execute on whichever venue offers the best price relative to the predicted close.

The current system **partially implements this**:

- **Line movement prediction**: Implemented. Model runs hourly, predictions stored.
- **Cross-source data**: Partially. We have multiple bookmakers from OddsPortal, but missing Pinnacle (the sharpest reference). Betfair Exchange is available and untested as sharp reference.
- **Execute on best venue**: Not implemented. No cross-venue price comparison, no execution tracking.

### What must be true for viability

1. **The model must identify +EV bets above execution cost.** At minimum, mean actual CLV for top predictions must exceed ~2% (Betfair) or ~4% (sportsbook). With current R² of 1.6%, this is uncertain.
2. **We need a sharp reference in live data.** Pinnacle is unavailable. Betfair Exchange is the best candidate — must validate in training pipeline.
3. **Execution must be feasible.** Betfair has deep EPL liquidity (confirmed: £6K–£40K+ matched per game). Sportsbook accounts get limited if they detect sharp betting.

### What could change the calculus

- **Betfair as sharp reference**: If training with Betfair Exchange as the sharp reference produces R² comparable to Pinnacle (~3%), the live pipeline becomes much more viable.
- **Away/draw predictions**: Tripling the opportunity set (all three outcomes) could make a marginal model viable through volume.
- **More snapshot density**: 20-35 snapshots per event (current) vs 2 historically — the model may perform differently with richer temporal data than what it was trained on.
- **Cross-venue arbitrage**: Even without CLV prediction, systematically betting where a bookmaker's price exceeds the Betfair-implied fair price is a strategy that doesn't require ML at all. The data pipeline already supports this.

## Training Objective Alignment

The model currently trains with MSE (mean squared error) on CLV prediction. The evaluation metrics above (Spearman correlation, tail calibration, P&L) measure different things. Whether the training objective should align with the evaluation metric is an open question.

### Why they might diverge

MSE optimises for accurate point predictions across the **entire** CLV distribution. But we only bet the tail — games with the highest predicted CLV. A model that spends optimisation budget getting ~0% CLV predictions slightly more accurate is wasting capacity on predictions we'd never act on.

### Options to explore

1. **Add Spearman as a secondary CV metric alongside MSE.** Cheapest change. Track whether MSE improvements during tuning also improve ranking. If they diverge, that's a signal to change objectives.
2. **XGBoost ranking objectives** (`rank:pairwise`, `rank:ndcg`). Directly optimise for ordering rather than point prediction. Natural fit for "which games should I bet?" but needs grouping (by matchweek) and more data to converge than regression.
3. **Asymmetric loss.** The real cost is asymmetric: predicting +3% CLV when actual is -1% costs money (bad bet). Missing a +EV bet is just a missed opportunity. Weighting the loss to penalise over-prediction more heavily aligns training with the actual risk profile.
4. **Quantile regression.** Instead of predicting mean CLV, predict a lower bound (e.g., 25th percentile). If the pessimistic estimate still clears execution cost, the bet is robust. Reframes the problem from "what's the exact CLV?" to "am I confident CLV exceeds my threshold?"

### Current recommendation

With R² of 1.6–3.6% and ~1,800 EPL samples, the bottleneck is signal (features, sharp reference), not objective function. Fancy losses won't extract signal that isn't in the features. Start with option 1 (Spearman as secondary metric) — it's free and diagnostic. Revisit objective alignment once the feature set is stronger (Betfair as sharp reference, richer snapshot data).

## Evaluation Process

Once weekend games (Mar 14–16) complete:

1. `fetch-oddsportal-results` captures closing odds (runs daily 08:00 UTC)
2. Compute actual CLV: compare each prediction's snapshot odds against closing fair odds
3. Run metrics 1–4 above
4. Determine minimum viable threshold for Betfair execution

This is a one-weekend sample (~10 games). Statistically insufficient for conclusions, but enough to sanity-check the pipeline and identify obvious calibration issues. Meaningful evaluation needs 50+ games (5–6 matchweeks).
