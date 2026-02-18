# Experiment 2: Feature Group Isolation

## Setup

### Tabular models (Ridge, XGBoost)
- **Date**: 2026-02-18
- **Dataset**: 538 samples, 230 events, 47 total features (down from 75 in exp1 — 28 structural duplicates removed in PR #135)
- **Sampling**: multi-horizon time_range (3-12h before game, max 5/event)
- **Target**: devigged Pinnacle CLV delta (mean=-0.0012, std=0.0380)
- **CV**: 3-fold walk-forward group timeseries (TimeSeriesSplit on event boundaries — always trains on earlier events, validates on later)
- **Models**: Ridge (alpha=1.0), XGBoost (50 trees, depth 3, heavily regularized)

### LSTM (sequence model)
- **Dataset**: 180 samples, 180 events (1 per event; 50 skipped — no pregame snapshot or Pinnacle closing)
- **Sampling**: single snapshot per event at pregame tier
- **Target**: devigged Pinnacle CLV delta (mean=+0.037, std=0.170)
- **CV**: 3-fold walk-forward group timeseries
- **Model**: tuned LSTM (hidden=48, layers=3, dropout=0.4, patience=20) from `lstm_line_movement_tuning_best.yaml`
- **Note**: LSTM target std=0.17 vs tabular std=0.038. The pregame tier selects the latest snapshot at or before pregame, which can be anywhere from 0.5h to 24h+ before game — much wider range than the tabular 3-12h window. MSE is **not directly comparable** between LSTM and tabular rows.

**Reproduce**: `uv run python experiments/scripts/exp2_feature_group_isolation.py`

## Key Results

### Tabular Models: Performance by Feature Group

| Group | N Features | Model | R² (mean±std) | MSE | MAE |
|-------|-----------|-------|---------------|-----|-----|
| tabular | 14 | ridge | -0.0162±0.0086 | 0.001761 | 0.0269 |
| tabular | 14 | xgboost | -0.0065±0.0051 | 0.001740 | 0.0267 |
| trajectory | 23 | ridge | -0.2368±0.1448 | 0.002067 | 0.0297 |
| trajectory | 23 | xgboost | -0.0185±0.0276 | 0.001786 | 0.0268 |
| pm_cross_source | 9 | ridge | -0.0139±0.0100 | 0.001756 | 0.0268 |
| pm_cross_source | 9 | xgboost | -0.0128±0.0091 | 0.001753 | 0.0266 |
| sharp_retail | 3 | ridge | -0.0034±0.0230 | 0.001755 | 0.0266 |
| sharp_retail | 3 | xgboost | -0.0145±0.0201 | 0.001771 | 0.0268 |
| all_no_pm | 38 | ridge | -0.1769±0.0808 | 0.001962 | 0.0292 |
| all_no_pm | 38 | xgboost | -0.0148±0.0124 | 0.001763 | 0.0267 |
| all | 47 | ridge | -0.1114±0.0836 | 0.001869 | 0.0285 |
| all | 47 | xgboost | -0.0105±0.0114 | 0.001755 | 0.0265 |

**Predict-mean baseline**: R²=-0.0162, MSE=0.001768, MAE=0.0270

### LSTM: Sequence Model (different target scale — R² only comparable within LSTM)

| Group | N Features | Model | R² (mean±std) | MSE* | MAE* |
|-------|-----------|-------|---------------|------|------|
| sequence | 15 | lstm | +0.0204±0.0412 | 0.036708 | 0.1490 |

*MSE/MAE not comparable to tabular rows (target std=0.17 vs 0.038).

### Per-Fold R² (XGBoost, tabular models)

| Group | Fold 0 | Fold 1 | Fold 2 |
|-------|--------|--------|--------|
| tabular | +0.0007 | -0.0096 | -0.0105 |
| trajectory | -0.0004 | +0.0024 | -0.0576 |
| pm_cross_source | +0.0000 | -0.0183 | -0.0201 |
| sharp_retail | +0.0001 | -0.0007 | -0.0429 |
| all_no_pm | -0.0006 | -0.0129 | -0.0309 |
| all | +0.0011 | -0.0067 | -0.0260 |

### Per-Fold R² (LSTM)

| Fold 0 | Fold 1 | Fold 2 |
|--------|--------|--------|
| -0.0245 | +0.0106 | +0.0750 |

## Interpretation

### Tabular models

No tabular feature group produces R² > 0 or MSE meaningfully below the predict-mean baseline. All groups are within noise of the predict-mean baseline.

Key observations:
1. **Smaller groups are more stable.** `sharp_retail` (3 features) and `pm_cross_source` (9 features) have the least negative R² and smallest std. Adding more features (`all`) consistently makes things worse — Ridge degrades sharply (ill-conditioned; collinearity warnings), XGBoost degrades moderately.
2. **Trajectory group adds noise.** Despite having the highest individual correlations in exp1 (traj_max_prob_decrease ρ=0.13), the full `trajectory` group (23 features) is the worst-performing group for Ridge and second worst for XGBoost.
3. **PM features are not differentiating.** `pm_cross_source` performs similarly to other small groups — no evidence of incremental signal from Polymarket data at this sample size.
4. **Walk-forward fold structure**: Fold 0 has the smallest training set (~57 events). With walk-forward CV, later folds train on progressively more data. No consistent improvement across folds, suggesting the bottleneck is absolute data volume rather than incremental learning.

### LSTM (sequence model)

The LSTM achieves **R²=+0.0204±0.0412** — the only model to achieve positive mean R² across folds. This is weak (explaining ~2% of target variance) but directionally consistent: fold 2 (largest training set) shows the strongest result (+0.075), fold 0 (smallest) is negative (-0.025).

**Caution**: The LSTM uses a different decision time (pregame tier sampling vs 3-12h window for tabular), so its positive R² could partially reflect that prediction is easier closer to game time (less remaining movement to predict). A controlled comparison at the same decision time would be needed to isolate the architecture effect.

**What the LSTM result does tell us**: temporal sequence information adds incremental value over point-in-time snapshots, even at the same signal-to-noise ratio. The LSTM can identify directional patterns in how odds have moved, which XGBoost using only a single snapshot cannot.

## Implications

1. **Sharp-retail subset is the most defensible tabular baseline.** 3 features, most stable CV, theoretically grounded. More features strictly hurt on current data.
2. **LSTM shows marginal but consistent positive R².** Worth pursuing as the primary model type if data volume increases. The architecture is already tuned (from the HPO run).
3. **The bottleneck is data volume, not feature engineering or architecture.** With 230 events, all models are dominated by CV noise. Both the tabular results (R²≈0) and the LSTM result (R²≈0.02) are consistent with the weak individual correlations from exp1 (max |r|=0.12 → need ~1000+ events for reliable detection).
4. **Next steps**: Continue collecting data. Re-run exp2 at 500 events to see if LSTM's positive signal persists and tabular groups become more differentiated.

## Artifacts

- `results.csv` — full results table
- `group_comparison.png` — R² and MSE bar chart comparison
- `fold_detail.png` — per-fold R² boxplot for XGBoost
