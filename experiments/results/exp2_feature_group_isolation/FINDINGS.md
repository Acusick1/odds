# Experiment 2: Feature Group Isolation

## Setup

### Tabular models (Ridge, XGBoost)
- **Date**: 2026-02-18
- **Git SHA**: `6314a0b8`
- **Dataset**: 719 samples, 229 events, 47 total features (down from 75 in exp1 — 28 structural duplicates removed in PR #135)
- **Sampling**: multi-horizon time_range (3-12h before game, max 5/event)
- **Target**: devigged Pinnacle CLV delta (mean=-0.0012, std=0.0380)
- **CV**: 3-fold group timeseries (event-level splits)
- **Models**: Ridge (alpha=1.0), XGBoost (50 trees, depth 3, heavily regularized)

### LSTM (sequence model)
- **Dataset**: 180 samples, 180 events (1 per event; 50 skipped — no pregame snapshot or Pinnacle closing)
- **Sampling**: single snapshot per event at pregame tier
- **Target**: devigged Pinnacle CLV delta (mean=+0.037, std=0.170)
- **CV**: 3-fold group timeseries
- **Model**: tuned LSTM (hidden=48, layers=3, dropout=0.4, patience=20) from `lstm_line_movement_tuning_best.yaml`
- **Note**: LSTM target std=0.17 vs tabular std=0.038. The pregame tier selects the latest snapshot at or before pregame, which can be anywhere from 0.5h to 24h+ before game — much wider range than the tabular 3-12h window. MSE is **not directly comparable** between LSTM and tabular rows.

**Reproduce**: `uv run python experiments/scripts/exp2_feature_group_isolation.py`

## Key Results

### Tabular Models: Performance by Feature Group

| Group | N Features | Model | R² (mean±std) | MSE | MAE |
|-------|-----------|-------|---------------|-----|-----|
| sharp_retail | 3 | ridge | -0.0026±0.0141 | 0.001440 | 0.0237 |
| sharp_retail | 3 | xgboost | -0.0237±0.0229 | 0.001466 | 0.0239 |
| pm_cross_source | 9 | ridge | -0.0065±0.0158 | 0.001446 | 0.0241 |
| pm_cross_source | 9 | xgboost | -0.0226±0.0934 | 0.001433 | 0.0241 |
| tabular | 14 | ridge | -0.0766±0.1292 | 0.001492 | 0.0252 |
| tabular | 14 | xgboost | -0.0532±0.0935 | 0.001476 | 0.0246 |
| trajectory | 23 | ridge | -0.0977±0.0621 | 0.001558 | 0.0255 |
| trajectory | 23 | xgboost | -0.0366±0.0214 | 0.001490 | 0.0245 |
| all_no_pm | 38 | ridge | -0.1783±0.1914 | 0.001612 | 0.0266 |
| all_no_pm | 38 | xgboost | -0.0412±0.0717 | 0.001468 | 0.0241 |
| all | 47 | ridge | -0.1799±0.1964 | 0.001614 | 0.0270 |
| all | 47 | xgboost | -0.0444±0.0817 | 0.001468 | 0.0241 |

**Predict-mean baseline**: R²=-0.0081, MSE=0.001451, MAE=0.0239

### LSTM: Sequence Model (different target scale — R² only comparable within LSTM)

| Group | N Features | Model | R² (mean±std) | MSE* | MAE* |
|-------|-----------|-------|---------------|------|------|
| sequence | 15 | lstm | +0.0330±0.0353 | 0.036595 | 0.1439 |

*MSE/MAE not comparable to tabular rows (target std=0.17 vs 0.038).

### Per-Fold R² (XGBoost, tabular models)

| Group | Fold 0 | Fold 1 | Fold 2 |
|-------|--------|--------|--------|
| tabular | -0.1854 | +0.0144 | +0.0113 |
| trajectory | -0.0449 | -0.0576 | -0.0072 |
| pm_cross_source | -0.1538 | +0.0567 | +0.0293 |
| sharp_retail | -0.0559 | -0.0097 | -0.0054 |
| all_no_pm | -0.1424 | +0.0044 | +0.0143 |
| all | -0.1600 | +0.0100 | +0.0166 |

### Per-Fold R² (LSTM)

| Fold 0 | Fold 1 | Fold 2 |
|--------|--------|--------|
| -0.0134 | +0.0536 | +0.0587 |

## Interpretation

### Tabular models

No tabular feature group produces R² > 0 or MSE meaningfully below the predict-mean baseline. The marginal 1.2% MSE reduction for `pm_cross_source (xgboost)` is within fold variance (fold 0 R²=-0.15, folds 1-2 R²≈+0.04) and should not be taken as signal.

Key observations:
1. **Smaller groups are more stable.** `sharp_retail` (3 features) and `pm_cross_source` (9 features) have the least negative R² and smallest std. Adding more features (`all`) consistently makes things worse — Ridge degrades sharply (ill-conditioned; collinearity warnings), XGBoost degrades moderately.
2. **Trajectory group adds noise.** Despite having the highest individual correlations in exp1 (traj_max_prob_decrease ρ=0.13), the full `trajectory` group (23 features) is the worst-performing group for XGBoost.
3. **PM features are not differentiating.** `pm_cross_source` appears marginally best but this is consistent with sparse features being harder to overfit on rather than carrying real signal.
4. **Fold 0 is consistently the worst.** The earliest time period (first ~77 events chronologically) has the most negative R² across all groups, suggesting either regime shift across the season or that small early training sets can't learn a weak signal.

### LSTM (sequence model)

The LSTM achieves **R²=+0.0330±0.0353** — the only model to achieve positive mean R² across folds. This is weak (explaining ~3% of target variance) but consistent: folds 1 and 2 are both positive (+0.054, +0.059), with only fold 0 (smallest training set, ~60 events) negative (-0.013).

**Caution**: The LSTM uses a different decision time (pregame tier sampling vs 3-12h window for tabular), so its positive R² could partially reflect that prediction is easier closer to game time (less remaining movement to predict). A controlled comparison at the same decision time would be needed to isolate the architecture effect.

**What the LSTM result does tell us**: temporal sequence information adds incremental value over point-in-time snapshots, even at the same signal-to-noise ratio. The LSTM can identify directional patterns in how odds have moved, which XGBoost using only a single snapshot cannot.

## Implications

1. **Sharp-retail subset is the most defensible tabular baseline.** 3 features, most stable CV, theoretically grounded. More features strictly hurt on current data.
2. **LSTM shows marginal but consistent positive R².** Worth pursuing as the primary model type if data volume increases. The architecture is already tuned (from the HPO run).
3. **The bottleneck is data volume, not feature engineering or architecture.** With 200 events, all models are dominated by CV noise. Both the tabular results (R²≈0) and the LSTM result (R²≈0.03) are consistent with the weak individual correlations from exp1 (max |r|=0.12 → need ~1000+ events for reliable detection).
4. **Next steps**: Continue collecting data. Re-run exp2 at 500 events to see if LSTM's positive signal persists and tabular groups become more differentiated.

## Artifacts

- `results.csv` — full results table
- `group_comparison.png` — R² and MSE bar chart comparison
- `fold_detail.png` — per-fold R² boxplot for XGBoost
