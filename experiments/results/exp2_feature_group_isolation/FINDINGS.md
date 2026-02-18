# Experiment 2: Feature Group Isolation

## Setup

### Tabular models (Ridge, XGBoost)
- **Date**: 2026-02-18
- **Dataset**: 538 samples, 230 events, 47 total features (down from 75 in exp1 — 28 structural duplicates removed in PR #135)
- **Sampling**: multi-horizon time_range (3-12h before game, max 5/event)
- **Target**: devigged Pinnacle CLV delta (mean=-0.0012, std=0.0380)
- **CV**: 3-fold walk-forward group timeseries (TimeSeriesSplit on event boundaries — always trains on earlier events, validates on later)
- **Models**: Ridge (alpha=1.0), XGBoost (50 trees, depth 3, heavily regularized)

### LSTM (sequence model, pregame tier)
- **Dataset**: 228 samples, 228 events (1 per event; pregame tier)
- **Sampling**: single snapshot per event at pregame tier (3+ hours before game)
- **Target**: devigged Pinnacle CLV delta (mean≈0, std=0.028)
- **CV**: 3-fold walk-forward group timeseries
- **Model**: tuned LSTM (hidden=48, layers=3, dropout=0.4, patience=20) from `lstm_line_movement_tuning_best.yaml`

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

### Per-Fold R² (XGBoost, tabular models — time_range)

| Group | Fold 0 | Fold 1 | Fold 2 |
|-------|--------|--------|--------|
| tabular | +0.0007 | -0.0096 | -0.0105 |
| trajectory | -0.0004 | +0.0024 | -0.0576 |
| pm_cross_source | +0.0000 | -0.0183 | -0.0201 |
| sharp_retail | +0.0001 | -0.0007 | -0.0429 |
| all_no_pm | -0.0006 | -0.0129 | -0.0309 |
| all | +0.0011 | -0.0067 | -0.0260 |

## Phase 2: Controlled Architecture Comparison

### Motivation

Earlier runs (before TierSampler bugfix) showed LSTM @ pregame tier as the only model with positive R². This raised the question: is the signal from the LSTM architecture, or from the different decision time? Phase 2 adds a 2×2 comparison (architecture × decision time) to isolate both effects.

### TierSampler bug (fixed in this run)

The initial Phase 2 run showed XGBoost @ tier with R²=+0.45 — suspiciously high. Investigation revealed a bug in `TierSampler`: `IN_PLAY` snapshots (taken *during* the game) were included as candidates for pregame tier requests. `IN_PLAY` is listed last in `FetchTier.get_priority_order()` (lowest priority, index 5), so the `tier_idx >= decision_idx` filter incorrectly included them. Since `TierSampler` picks the *latest* candidate, in-play snapshots — which occur after game start — were always preferred over genuine pregame snapshots.

**Impact**: 135/180 events used in-play snapshots as "pregame" decisions. Since the closing snapshot is the last pre-game snapshot (~0.5h before game) and the "decision" snapshot was taken *during* the game (1-2h after start), the features were computed from a time point **after** the target's reference point. This is textbook look-ahead bias: the model used future (in-game) odds — which reflect score, injuries, and momentum — to "predict" the pre-game closing price, which was already in the past. The R²=+0.45 reflected this temporal inversion, not genuine predictive signal.

**Fix**: `TierSampler.sample()` now excludes `IN_PLAY` snapshots unless the decision tier is explicitly `IN_PLAY`.

### 2×2 Results: Architecture × Decision Time (post-fix)

| | XGBoost | LSTM |
|--|---------|------|
| **time_range (3-12h)** | R²=-0.011±0.011 (n=538) | R²=-0.009±0.026 (n=717) |
| **tier (pregame)** | R²=-0.024±0.013 (n=229) | R²=-0.015±0.026 (n=228) |

All four cells are R² ≈ 0. No model beats predict-mean at either decision time.

### Phase 2 Detail

| Group | Model | N Features | Samples | R² (mean±std) | MSE | MAE |
|-------|-------|-----------|---------|---------------|-----|-----|
| sequence_pregame | lstm | 15 | 228 | -0.0149±0.0259 | 0.000828 | 0.0201 |
| sequence_time_range | lstm | 15 | 717 | -0.0088±0.0260 | 0.001666 | 0.0267 |
| all_tier | xgboost | 47 | 229 | -0.0236±0.0131 | 0.000851 | 0.0196 |

### Per-Fold R² (Phase 2)

| Condition | Fold 0 | Fold 1 | Fold 2 |
|-----------|--------|--------|--------|
| LSTM pregame | -0.0485 | +0.0146 | -0.0107 |
| LSTM time_range | -0.0222 | -0.0317 | +0.0276 |
| XGBoost tier | -0.0191 | -0.0103 | -0.0414 |

## Interpretation

### Tabular models (Phase 1)

No tabular feature group produces R² > 0 or MSE meaningfully below the predict-mean baseline. All groups are within noise of the predict-mean baseline.

Key observations:
1. **Smaller groups are more stable.** `sharp_retail` (3 features) and `pm_cross_source` (9 features) have the least negative R² and smallest std. Adding more features (`all`) consistently makes things worse — Ridge degrades sharply (ill-conditioned; collinearity warnings), XGBoost degrades moderately.
2. **Trajectory group adds noise.** Despite having the highest individual correlations in exp1 (traj_max_prob_decrease ρ=0.13), the full `trajectory` group (23 features) is the worst-performing group for Ridge and second worst for XGBoost.
3. **PM features are not differentiating.** `pm_cross_source` performs similarly to other small groups — no evidence of incremental signal from Polymarket data at this sample size.
4. **Walk-forward fold structure**: Fold 0 has the smallest training set (~57 events). With walk-forward CV, later folds train on progressively more data. No consistent improvement across folds, suggesting the bottleneck is absolute data volume rather than incremental learning.

### Phase 2: Neither architecture nor decision time matters

The 2×2 comparison is unambiguous: **all four cells produce R² ≈ 0**.

- LSTM adds no value over XGBoost at either decision time.
- Pregame tier (3+ hours) produces the same R² as time_range (3-12h) — the decision time window doesn't matter within this range.
- The tier-sampled data has lower MSE (~0.0008 vs ~0.0017) simply because pregame tier selects one snapshot per event near the boundary (3-5h before game), where less line movement remains. The target has lower variance, but models can't exploit it.

### TierSampler bug implications

The bug had no effect on Phase 1 results (tabular models use `TimeRangeSampler`, not `TierSampler`). It affected LSTM pregame tier results in the initial run and the XGBoost @ tier comparison. The corrected LSTM pregame result (R²=-0.015) replaces the previously reported R²≈+0.02.

Any prior analysis that used `TierSampler` with non-closing tiers on events with in-play snapshots was subject to look-ahead bias. The LSTM v1 results (which used pregame tier) should be re-evaluated.

## Implications

1. **No detectable signal at 230 events.** Neither architecture (tabular vs sequence), decision time (3-12h vs pregame), nor feature group selection produces R² > 0. All models are indistinguishable from predict-mean.
2. **Sharp-retail subset is the most stable tabular configuration.** 3 features, smallest negative R², theoretically grounded. More features strictly hurt.
3. **LSTM does not add value.** At both decision times, LSTM ≈ XGBoost ≈ predict-mean. The sequence architecture is not worth pursuing at current data volume.
4. **The bottleneck is data volume, not feature engineering or architecture.** With 230 events and weak individual correlations (max |r|=0.12 from exp1), all models are dominated by noise.
5. **Next steps**: Continue collecting data. Re-run at 500+ events. If still R²≈0, consider whether the feature set captures the right information for CLV prediction, or whether the target itself (devigged Pinnacle delta) is too noisy at available sample frequencies.

## Artifacts

- `results.csv` — full results table (Phase 1 + Phase 2)
- `group_comparison.png` — R² and MSE bar chart comparison
- `fold_detail.png` — per-fold R² boxplot for XGBoost
