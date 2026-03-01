# Experiment 7b: Line Shopping Across Bookmakers

## Setup

- **Date**: 2026-03-01
- **Dataset**: OddsPortal, 4524 events, bet365 target (devigged CLV)
- **Bookmakers**: bet365, betway, betfred, bwin (UK OddsPortal set)
- **Model**: XGBoost with tuned baseline params (tabular-only, same as Exp 7)
- **CV**: Walk-forward expanding window (same splits as Exp 7)
- **Predictions**: 2400 val-fold predictions, 2088 with betting context
- **Sizing**: Flat $100 per bet
- **Significance**: 1,000 permutation shuffles per threshold per mode

## Vig Reduction

- **bet365 mean overround**: 4.13% (median 4.22%)
- **Best-available mean overround**: 1.59% (median 2.21%)
- **Vig reduction**: 2.55 percentage points
- **Events**: 3992

## Key Results

### Threshold Sweep — bet365 Only

| Threshold | N Bets | Win Rate | P&L | ROI | Avg CLV | p-value |
|-----------|--------|----------|-----|-----|---------|---------|
| 0.005 | 1307 | 43.9% | $-5,509 | -4.21% | +0.0105 | 0.589 |
| 0.010 | 792 | 46.0% | $-1,915 | -2.42% | +0.0151 | 0.412 |
| 0.015 | 509 | 45.8% | $-3,060 | -6.01% | +0.0186 | 0.666 |
| 0.020 | 339 | 45.4% | $-3,156 | -9.31% | +0.0240 | 0.813 |
| 0.030 | 154 | 44.8% | $-1,809 | -11.75% | +0.0310 | 0.766 |
| 0.050 | 56 | 48.2% | $+188 | +3.36% | +0.0472 | 0.260 |

### Threshold Sweep — Best Available (Line Shopping)

| Threshold | N Bets | Win Rate | P&L | ROI | Avg CLV | p-value |
|-----------|--------|----------|-----|-----|---------|---------|
| 0.005 | 1307 | 43.9% | $-3,761 | -2.88% | +0.0105 | 0.796 |
| 0.010 | 792 | 46.0% | $-1,179 | -1.49% | +0.0151 | 0.616 |
| 0.015 | 509 | 45.8% | $-2,805 | -5.51% | +0.0186 | 0.818 |
| 0.020 | 339 | 45.4% | $-2,981 | -8.79% | +0.0240 | 0.884 |
| 0.030 | 154 | 44.8% | $-1,737 | -11.28% | +0.0310 | 0.841 |
| 0.050 | 56 | 48.2% | $+206 | +3.68% | +0.0472 | 0.334 |

### Side-by-Side Comparison

| Threshold | bet365 ROI | Best ROI | ROI Improvement |
|-----------|-----------|----------|-----------------|
| 0.005 | -4.21% | -2.88% | +1.34% |
| 0.010 | -2.42% | -1.49% | +0.93% |
| 0.015 | -6.01% | -5.51% | +0.50% |
| 0.020 | -9.31% | -8.79% | +0.52% |
| 0.030 | -11.75% | -11.28% | +0.47% |
| 0.050 | +3.36% | +3.68% | +0.32% |

### Bookmaker Selection Frequency (Best Available, lowest threshold)

| Bookmaker | Home | Away | Total |
|-----------|------|------|-------|
| bet365 | 67.5% | 66.7% | 67.0% |
| betway | 12.9% | 12.1% | 12.4% |
| betfred | 12.1% | 13.7% | 13.1% |
| bwin | 7.5% | 7.5% | 7.5% |

### Best Thresholds

- **bet365 only**: threshold=0.05, ROI=+3.36%, p=0.260
- **Best available**: threshold=0.05, ROI=+3.68%, p=0.334

## Interpretation

Line shopping improves ROI at the best threshold from +3.36% (bet365) to +3.68% (best available), but this is NOT statistically significant (p=0.334).

Line shopping improves ROI at every threshold, confirming that best-price execution can only help (or tie) versus single-book execution.

Effective vig drops from 4.13% (bet365) to 1.59% (best-available), a 2.55pp reduction.

## Implications

- Line shopping is a meaningful improvement but insufficient alone
- The effective vig reduction may become significant with a larger sample
- Consider combining with cross-venue execution (Polymarket) for additional vig reduction

