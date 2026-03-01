# Experiment 7: Walk-Forward Betting Simulation

## Setup

- **Date**: 2026-03-01
- **Dataset**: OddsPortal, 4524 events, bet365 target
- **Model**: XGBoost with tuned baseline params (tabular-only)
- **CV**: Walk-forward expanding window (same splits as Exp 6)
- **Predictions**: 2400 val-fold predictions, 2088 with betting context
- **Sizing**: Flat $100 per bet
- **Significance**: 1,000 permutation shuffles per threshold

## Key Results

### Threshold Sweep

| Threshold | N Bets | Home | Away | Win Rate | P&L | ROI | Avg CLV | p-value |
|-----------|--------|------|------|----------|-----|-----|---------|---------|
| 0.005 | 1307.0 | 495.0 | 812.0 | 43.9% | $-5,509 | -4.21% | +0.0105 | 0.589 |
| 0.010 | 792.0 | 283.0 | 509.0 | 46.0% | $-1,915 | -2.42% | +0.0151 | 0.412 |
| 0.015 | 509.0 | 173.0 | 336.0 | 45.8% | $-3,060 | -6.01% | +0.0186 | 0.666 |
| 0.020 | 339.0 | 128.0 | 211.0 | 45.4% | $-3,156 | -9.31% | +0.0240 | 0.813 |
| 0.030 | 154.0 | 68.0 | 86.0 | 44.8% | $-1,809 | -11.75% | +0.0310 | 0.766 |
| 0.050 | 56.0 | 41.0 | 15.0 | 48.2% | $+188 | +3.36% | +0.0472 | 0.260 |

### Baselines

- **Always home**: 2088 bets, win rate 54.2%, ROI -7.55%, P&L $-15,774
- **Permutation mean ROI** (at best threshold 0.05): -5.79%

### Best Threshold

- **Threshold**: 0.05
- **ROI**: +3.36%
- **p-value**: 0.260

## Interpretation

The model achieves positive ROI of +3.36% at threshold=0.05, but this is NOT statistically significant (p=0.260). The result could be due to chance — the CLV signal may not overcome the vig.

The always-home baseline ROI of -7.55% confirms the house edge the model must overcome.

## Implications

- Public sportsbook features alone are insufficient for profitable betting
- Consider: non-public features (order flow, bettor identity), cross-venue execution, or accepting the edge is too thin
- The signal may still have value for market-making or spread strategies where transaction costs are lower

