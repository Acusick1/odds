---
type: concept
title: What the CLV model can and cannot do
description: "Hub for the CLV model: the deployed model, current conclusions with links to each decision and experiment, the data used, and the betting simulation showing the signal does not survive the vig."
sources:
  - docs/MODELING.md#what-the-model-predicts
  - docs/MODELING.md#deployed-model
  - docs/MODELING.md#current-conclusions
  - docs/MODELING.md#data-used
  - docs/MODELING.md#experiment-log
  - docs/MODELING.md#betting-simulation-and-line-shopping
  - docs/BETTING_AGENT.md#role-of-the-xgboost-model
  - experiments/configs/xgboost_epl_production_tabular_match_stats.yaml
  - experiments/results/exp7_backtest_sim/FINDINGS.md
  - experiments/results/exp7b_line_shopping/FINDINGS.md
  - experiments/scripts/exp7_backtest_simulation.py
  - experiments/scripts/exp7b_line_shopping.py
verified: "7887039"
status: stable
---

# What the CLV model can and cannot do

The model predicts line movement: the devigged bet365 CLV delta between a snapshot and the close ([CLV target](../modelling/clv-target.md)). The signal is weak but real: best EPL CV R² about 0.058, NBA plateau about 0.036. It cannot be bet alone: flat betting on NBA loses at every useful threshold ([betting simulation](#betting-simulation-exp-7-7b)). The agent treats it as a supplementary sanity check; its strongest feature is the sharp-retail spread the agent already sees through `find_retail_edges` ([sharp benchmark](../modelling/sharp-benchmark.md), [pivot to information edge](../agent/decision-information-edge.md)).

## Deployed model

`epl-clv-home` version `v308058c`, published 2026-04-23: XGBoost on tabular + match_stats, sliding-380:50 walk-forward CV, principled search space, hybrid sharp reference; CV val_R² +0.058, val_MSE 0.000716, replacing a tabular-only model (+0.035). Source config: `experiments/configs/xgboost_epl_production_tabular_match_stats.yaml`. The live S3 version is the authority and cannot be verified from the repo; check with `aws s3 ls s3://<model-bucket>/epl-clv-home/` ([publishing and scoring](../modelling/model-publishing-and-scoring.md)). Predictions exist only where inline scoring runs, which is a local scheduler, one row per snapshot.

## Current conclusions

- bet365, not Pinnacle, is the target: [bet365 target](../modelling/decision-bet365-target.md).
- Sharp reference is Pinnacle with Betfair fallback: [hybrid sharp reference](../modelling/decision-hybrid-sharp-reference.md).
- Sharp-vs-retail divergence is the core feature: [sharp benchmark](../modelling/sharp-benchmark.md).
- On EPL, match_stats helps and the rest do not: [EPL ablation](exp-epl-feature-ablation-and-search-space.md).
- Sliding-380:50 is the protocol: [walk-forward CV](../modelling/decision-walk-forward-cv.md); it only works with the principled search space ([same experiment page](exp-epl-feature-ablation-and-search-space.md#principled-search-space-re-test)).
- NBA injury features add nothing: [NBA timing and injuries](exp-nba-timing-and-injuries.md#nba-injury-features-exp-6-6b) (untested on EPL).
- LSTM adds nothing: [early NBA signal and LSTM](exp-early-nba-signal-and-lstm.md#lstm-sequence-models).
- More NBA data will not help (plateau about 1.5K events); EPL has about 1.8K events and a plateau is likely but unconfirmed.
- Pregame (3-12 h) is the default window: [NBA timing](exp-nba-timing-and-injuries.md#hours-to-game-exp-4), [decision time](../modelling/decision-time-and-look-ahead.md). Timing on EPL is untested.
- Polymarket features are untested at scale (230 events, no effect): [Polymarket](../collection/polymarket.md).
- NBA bookmakers release lines at different times: [line release timing](exp-nba-timing-and-injuries.md#bookmaker-line-release-timing).

## Data used

| | OddsPortal | Odds API | football-data.co.uk |
|---|---|---|---|
| Events | about 5K NBA + 1.8K EPL | about 1K NBA | about 4K EPL (2015-2026) |
| Snapshots per event | 2 (open, close) | 15+ | 2 |

1,810 EPL events match between OddsPortal and FDUK; 141 NBA events match between the Odds API and OddsPortal. FDUK is compiled by hand twice weekly, its Pinnacle odds became unreliable after July 2025 (81 events in 2025-26 lack Pinnacle closing), and Betfair Exchange closing odds exist from 2024-25 ([historical loads](../collection/howto-load-historical-odds.md), [sources](../collection/sources-and-keys.md)).

## Betting simulation (Exp 7, 7b)

Question: does the roughly 3.6% R² CLV signal survive the vig? Answer: no. The model finds direction (average CLV captured is positive at every threshold) but the vig consumes it.

**Exp 7 (2026-03-01).** About 4,500 OddsPortal NBA events (4,524), bet365 target, tuned tabular-only XGBoost, 12-fold expanding walk-forward (splits as Exp 6), 2,400 validation predictions of which 2,088 had betting context. Flat $100 stakes on bet365 odds; a prediction above `+threshold` bets home, below `-threshold` bets away. Significance: 1,000 permutation shuffles per threshold.

| Threshold | Bets | Win rate | ROI | p |
|---|---|---|---|---|
| 0.005 | 1,307 | 43.9% | -4.21% | 0.589 |
| 0.010 | 792 | 46.0% | -2.42% | 0.412 |
| 0.015 | 509 | 45.8% | -6.01% | 0.666 |
| 0.020 | 339 | 45.4% | -9.31% | 0.813 |
| 0.030 | 154 | 44.8% | -11.75% | 0.766 |
| 0.050 | 56 | 48.2% | +3.36% | 0.260 |

The +3.36% at 0.05 is not significant. Always-home baseline: 2,088 bets, 54.2% wins, ROI -7.55%.

**Exp 7b, line shopping (2026-03-01).** Same simulation, best price across bet365, Betway, Betfred and bwin. Effective overround falls from 4.13% (median 4.22%) to 1.59% (median 2.21%), 2.55 pp, over 3,992 events. ROI improves 0.3-1.3 pp at every threshold (0.005: -4.21% to -2.88%; 0.05: +3.36% to +3.68%, p = 0.334), but thresholds up to 0.03 still lose. bet365 already has the best price 67.0% of the time (Betway 12.4%, Betfred 13.1%, bwin 7.5%).

**Conclusion.** Public sportsbook features alone are insufficient for profitable betting. Line shopping helps but is not enough; the source suggests cross-venue execution (Polymarket) as a further vig reduction, and the signal may suit settings with lower transaction costs. Findings: `experiments/results/exp7_backtest_sim/FINDINGS.md`, `experiments/results/exp7b_line_shopping/FINDINGS.md`.

## Dated index

- 2026-02-14 to 02-18: [early NBA signal search](exp-early-nba-signal-and-lstm.md).
- 2026-02-20 and 03-01 (6b): [NBA injury features](exp-nba-timing-and-injuries.md#nba-injury-features-exp-6-6b).
- 2026-02-21 to 02-28: [LSTM](exp-early-nba-signal-and-lstm.md#lstm-sequence-models).
- 2026-02-27, 03-01: [bet365 target and data volume](../modelling/decision-bet365-target.md).
- 2026-02-28: [hours to game](exp-nba-timing-and-injuries.md#hours-to-game-exp-4).
- 2026-03-01: betting simulation (above).
- 2026-03-10 to 03-19: [EPL sharp reference](../modelling/decision-hybrid-sharp-reference.md).
- 2026-03-16, 03-19: [CV protocol grid](../modelling/decision-walk-forward-cv.md).
- 2026-03-20, 04-23: [EPL feature ablation](exp-epl-feature-ablation-and-search-space.md).
- 2026-04-23: [principled search space](exp-epl-feature-ablation-and-search-space.md#principled-search-space-re-test).

## Hazards

- Results are as recorded at the time; several used feature sets that no longer exist ([feature pipeline](../modelling/feature-pipeline.md)).
- **Exp 7 is not the backtest engine.** It is a standalone script (`experiments/scripts/exp7_backtest_simulation.py`) with its own settlement on raw vigged odds, so numbers are not reproducible with `odds backtest` ([backtesting](backtesting.md)).
- The 56-bet cell has no power; the p-values are permutation-based on few bets.
- The simulation is NBA only; EPL has not been simulated.

## Related

- [Walk-forward CV](../modelling/decision-walk-forward-cv.md), [Training pipeline](../modelling/training-pipeline.md), [Run an experiment](howto-run-experiment.md)
