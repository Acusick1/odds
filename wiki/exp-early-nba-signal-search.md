---
type: experiment
title: Early NBA signal search (XGBoost v1, Exp 1, Exp 2)
summary: February 2026 NBA runs on about 230 events - no signal, sharp-retail divergence the strongest correlate, an IN_PLAY leak found, and a Polymarket vs sportsbook cross-check.
sources:
  - docs/MODELING.md#early-nba-experiments
  - docs/MODELING.md#experiment-log
  - experiments/results/exp1_feature_correlations/FINDINGS.md
  - experiments/results/exp2_feature_group_isolation/FINDINGS.md
  - experiments/scripts/exp1_feature_correlations.py
  - experiments/scripts/exp2_feature_group_isolation.py
  - packages/odds-analytics/odds_analytics/feature_groups.py::TierSampler
verified_at: 7887039
---

# Early NBA signal search (XGBoost v1, Exp 1, Exp 2)

All February 2026, devigged Pinnacle target, 193-230 events. Result: no signal at this size, but two lasting findings, sharp-retail divergence as the best correlate and the `IN_PLAY` look-ahead bug.

## Runs

- **XGBoost v1** (2026-02-14, 193 events, 656 rows, 75 features): R² about 0. Likely causes at the time: too little data, 21 tabular features zeroed by a bug in `XGBoostAdapter` (it pre-filtered odds to the target outcome, starving the bilateral features), order-book gaps.
- **Exp 1, feature-target correlations** (2026-02-17, 229 events, 719 rows): maximum |r| = 0.12, from `tab_retail_sharp_diff_home` (+0.120). Of 60 testable features 12 pass uncorrected p < 0.05 and 0 survive Benjamini-Hochberg or Bonferroni; 15 of 75 were constant. After the adapter fix the sparse-feature count fell from 40 to 15. Target std shrinks toward the game (far 0.049, close 0.025). Source: `experiments/results/exp1_feature_correlations/FINDINGS.md`.
- **Exp 2, feature group isolation** (2026-02-18, 230 events, 538 rows, 47 features, 3-fold walk-forward): every group and both architectures at R² < 0; the predict-mean baseline is R² -0.0162 and MSE 0.001768; XGBoost on all features is -0.0105. Smaller groups are steadier and adding features hurt (Ridge sharply). Source: `experiments/results/exp2_feature_group_isolation/FINDINGS.md`.
- **Polymarket vs sportsbook cross-check** (Feb 2026, 228 events): probability correlation 0.9978; Polymarket prices about 2.4 pp below the sportsbook (liquidity premium or vig difference); only 92 of 228 events had time-matched snapshots within 30 minutes. Adding Polymarket features changed nothing ([polymarket-pipeline](polymarket-pipeline.md)).

## The IN_PLAY leak

`TierSampler` treated `IN_PLAY` as lowest priority, so for pregame requests it accepted snapshots taken during the game and, choosing the latest, preferred them: 135 of 180 events used in-play snapshots. The inflated results were XGBoost at the tier R² +0.45 and LSTM about +0.02; after the fix the 2x2 (architecture x decision time) is XGBoost time-range -0.011, XGBoost tier -0.024, LSTM time-range -0.009, LSTM pregame -0.015 (corrected from +0.02). Fix: `TierSampler` excludes `IN_PLAY` unless requested ([feature-pipeline](feature-pipeline.md), [look-ahead-bias](look-ahead-bias.md)).

## Conclusion

The bottleneck was data volume, not architecture or decision time; next steps were more data and re-running at 500+ events ([exp-nba-injury-features](exp-nba-injury-features.md), [exp-nba-target-and-data-volume](exp-nba-target-and-data-volume.md), [exp-lstm-sequence-models](exp-lstm-sequence-models.md)).

## Hazards

- **Not reproducible as recorded.** The scripts group features as `trajectory` and `cross_source`, which no longer exist in `packages/odds-analytics/odds_analytics/feature_groups.py`; the tabular group now has 6 columns ([feature-groups](feature-groups.md)).
- `docs/MODELING.md` says 0 of 75 features survived correction; `FINDINGS.md` says 0 of the 60 testable ones.

## Related

- [CLV model findings](clv-model-findings.md), [Sharp benchmark](sharp-benchmark.md)
