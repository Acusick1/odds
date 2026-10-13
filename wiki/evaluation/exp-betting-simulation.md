---
type: explanation
title: Betting simulation (Exp 7, 7b)
description: Whether the CLV signal survives the vig with flat stakes and with line shopping, and how to read backtest-engine metrics against it.
sources:
  - experiments/results/exp7_backtest_sim/FINDINGS.md
  - experiments/results/exp7b_line_shopping/FINDINGS.md
  - experiments/scripts/exp7_backtest_simulation.py
  - experiments/scripts/exp7b_line_shopping.py
status: stable
---

# Betting simulation (Exp 7, 7b)

Question: does the roughly 3.6% R² CLV signal survive the vig? Answer: no. The model finds direction (average CLV captured is positive at every threshold) but the vig consumes it. The simulation is NBA only; EPL has not been simulated. The numbers come from a standalone script (`experiments/scripts/exp7_backtest_simulation.py`) with its own settlement on raw vigged odds, not from the [backtest engine](backtest-engine.md), so they are not reproducible with `odds backtest`.

## Exp 7 (2026-03-01)

About 4,500 OddsPortal NBA events (4,524), bet365 target, tuned tabular-only XGBoost, 12-fold expanding walk-forward (splits as Exp 6), 2,400 validation predictions of which 2,088 had betting context. Flat $100 stakes on bet365 odds; a prediction above `+threshold` bets home, below `-threshold` bets away. Significance: 1,000 permutation shuffles per threshold.

| Threshold | Bets | Win rate | ROI | p |
|---|---|---|---|---|
| 0.005 | 1,307 | 43.9% | -4.21% | 0.589 |
| 0.010 | 792 | 46.0% | -2.42% | 0.412 |
| 0.015 | 509 | 45.8% | -6.01% | 0.666 |
| 0.020 | 339 | 45.4% | -9.31% | 0.813 |
| 0.030 | 154 | 44.8% | -11.75% | 0.766 |
| 0.050 | 56 | 48.2% | +3.36% | 0.260 |

The +3.36% at 0.05 is not significant: the 56-bet cell has no power, and the p-values are permutation-based on few bets. Always-home baseline: 2,088 bets, 54.2% wins, ROI -7.55%.

## Exp 7b, line shopping (2026-03-01)

Same simulation, best price across bet365, Betway, Betfred and bwin. Effective overround falls from 4.13% (median 4.22%) to 1.59% (median 2.21%), 2.55 pp, over 3,992 events. ROI improves 0.3-1.3 pp at every threshold (0.005: -4.21% to -2.88%; 0.05: +3.36% to +3.68%, p = 0.334), but thresholds up to 0.03 still lose. bet365 already has the best price 67.0% of the time (Betway 12.4%, Betfred 13.1%, bwin 7.5%).

## Conclusion

Public sportsbook features alone are insufficient for profitable betting. Line shopping helps but is not enough; the source suggests cross-venue execution (Polymarket) as a further vig reduction, and the signal may suit settings with lower transaction costs. Findings: `experiments/results/exp7_backtest_sim/FINDINGS.md` (#170), `experiments/results/exp7b_line_shopping/FINDINGS.md` (#171). This result is part of the context of [ADR-0005](../../docs/adr/0005-information-edge.md).

## Reading backtest-engine results

The conventional readings come from the original backtest guide (2c0bc80) and were not derived from this engine: Sharpe above 1.0 good, above 2.0 excellent, above 3.0 exceptional; profit factor above 1.0 profitable, above 2.0 very good. Sortino reads higher than Sharpe for a profitable strategy because only downside is penalised. The Sharpe thresholds assume a return series, but the engine feeds dollar P&L per game date, so they do not transfer ([metrics as computed](backtest-engine.md#metrics)). The numbers on this page do not come from the engine either, so neither can be read against the other.

## Related

- [CLV model findings](clv-model-findings.md), [Target bookmaker and data volume](exp-target-and-data-volume.md)
