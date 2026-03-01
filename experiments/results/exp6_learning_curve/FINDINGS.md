# Experiment 6: Data Volume Learning Curve + Injury Timing Diagnostic

## Part 1: Learning Curve

Train XGBoost (tuned bet365 baseline params, tabular-only) on chronologically
increasing subsets of OddsPortal data. Walk-forward CV with expanding window.

### Results

| N Events | N Rows | R² Mean | R² Std | MSE Mean | Folds | Hours Mean |
|----------|--------|---------|--------|----------|-------|------------|
| 500 | 500 | -0.0664 | 0.0960 | 0.003089 | 6 | 21.7h |
| 1,000 | 1,000 | -0.0396 | 0.0567 | 0.003667 | 12 | 25.3h |
| 1,500 | 1,500 | +0.0356 | 0.0793 | 0.003932 | 12 | 27.1h |
| 2,000 | 2,000 | +0.0313 | 0.0600 | 0.003610 | 12 | 25.2h |
| 2,500 | 2,500 | +0.0190 | 0.0598 | 0.003522 | 12 | 25.3h |
| 3,000 | 3,000 | +0.0140 | 0.0462 | 0.003027 | 12 | 24.6h |
| 3,500 | 3,500 | -0.0025 | 0.0533 | 0.003114 | 12 | 24.9h |
| 4,000 | 4,000 | -0.0062 | 0.0454 | 0.002986 | 12 | 24.6h |
| 4,500 | 4,500 | +0.0192 | 0.0461 | 0.003068 | 12 | 25.0h |
| 4,524 | 4,524 | +0.0182 | 0.0547 | 0.003057 | 12 | 24.9h |

### Log-Fit Extrapolation

- **Model**: R² = 0.0301 · ln(N) + (-0.2302)
- **Marginal return at N=4,524**: dR²/dN = 6.66e-06
- **Extrapolated R² at 10K events**: 0.0473
- **Extrapolated R² at 20K events**: 0.0682

### Hours-to-Event Distribution

All subsets sample from the sharp tier (~12-24h before game). OddsPortal
snapshots average ~19h before game, with zero coverage in the 0-3h closing window.

| N Events | Hours Mean | Hours Median | Hours P25 | Hours P75 |
|----------|-----------|-------------|-----------|-----------|
| 500 | 21.7 | 17.6 | 15.4 | 20.3 |
| 1,000 | 25.3 | 18.0 | 16.0 | 20.3 |
| 1,500 | 27.1 | 18.6 | 16.5 | 21.0 |
| 2,000 | 25.2 | 19.0 | 16.8 | 21.0 |
| 2,500 | 25.3 | 19.2 | 17.0 | 21.3 |
| 3,000 | 24.6 | 19.4 | 17.4 | 21.6 |
| 3,500 | 24.9 | 19.7 | 17.5 | 22.0 |
| 4,000 | 24.6 | 19.9 | 17.7 | 22.1 |
| 4,500 | 25.0 | 19.9 | 17.8 | 22.2 |
| 4,524 | 24.9 | 19.9 | 17.8 | 22.2 |

## Part 2: Injury Timing Diagnostic

Compare tabular-only vs tabular+injuries at sharp (12-24h) vs pregame (3-12h) tier.
Tests whether the "injuries add nothing" conclusion is a timing artifact of
OddsPortal's ~19h average decision time.

### Results

| Feature Group | Tier | N Events | N Rows | R² Mean | R² Std | MSE Mean | Note |
|---------------|------|----------|--------|---------|--------|----------|------|
| tabular | sharp | 4,524 | 4,524 | +0.0182 | 0.0547 | 0.003057 |  |
| tabular | pregame | 4,857 | 4,857 | +0.0011 | 0.0489 | 0.003226 |  |
| tabular+injuries | sharp | 4,524 | 4,524 | +0.0064 | 0.0701 | 0.003092 |  |
| tabular+injuries | pregame | 4,857 | 4,857 | +0.0015 | 0.0607 | 0.003223 |  |

### Interpretation

**Pregame tier is diluted**: `TierSampler(decision_tier="pregame")` accepts
pregame, sharp, early, and opening tiers, picking the most recent. Since most
OddsPortal events only have opening (~19h) and closing (~0.1h) snapshots, ~93%
of "pregame" events fall back to the same sharp-tier snapshot as the sharp
experiment. Only ~335 events actually sample at 3-12h pregame timing. The R²
difference (0.001 vs 0.018) may reflect those 335 events dragging down the
average rather than a clean timing effect.

**Important caveat**: OddsPortal has zero snapshots in the 0-3h closing window
where GTD injury signal is theoretically strongest (r=0.28, p=0.0003 in Exp 4).
The pregame tier (3-12h) partially probes closer-to-game timing, but the true
test requires Odds API data with dense snapshot coverage at < 3h before game.

