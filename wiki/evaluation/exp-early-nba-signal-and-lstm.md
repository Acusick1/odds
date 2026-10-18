---
type: explanation
title: Early NBA signal search and LSTM
description: "February 2026 NBA runs (XGBoost v1, Exp 1, Exp 2, LSTM variants): no signal at about 230-1K events, sharp-retail divergence as the best correlate, the IN_PLAY leak, and why LSTM was ruled out."
sources:
  - experiments/results/exp1_feature_correlations/FINDINGS.md
  - experiments/results/exp2_feature_group_isolation/FINDINGS.md
  - experiments/scripts/exp1_feature_correlations.py
  - experiments/scripts/exp2_feature_group_isolation.py
  - experiments/configs/lstm_pinnacle_tuning.yaml
  - experiments/configs/lstm_line_movement_tuning.yaml
  - odds_analytics.feature_groups.TierSampler
  - odds_analytics.lstm_line_movement.LSTMLineMovementStrategy
status: stable
---

# Early NBA signal search and LSTM

All February 2026, NBA, Pinnacle target. Result: no signal at 193-230 events (and none for LSTM at about 1K), but two lasting findings: sharp-retail divergence is the best correlate, and `odds_analytics.feature_groups.TierSampler` had an `IN_PLAY` look-ahead bug. The bottleneck was data volume, not architecture or decision time.

## Early runs (193-230 events, devigged Pinnacle target)

- **XGBoost v1** (2026-02-14, #134, 193 events, 656 rows, 75 features): R² about 0. Likely causes at the time: too little data, 21 tabular features zeroed by a bug in `odds_analytics.feature_groups.XGBoostAdapter` (it pre-filtered odds to the target outcome, starving the bilateral features), order-book gaps.
- **Exp 1, feature-target correlations** (2026-02-17, 229 events, 719 rows): maximum |r| = 0.12, from `tab_retail_sharp_diff_home` (+0.120). Of 60 testable features 12 pass uncorrected p < 0.05 and 0 survive Benjamini-Hochberg or Bonferroni; 15 of 75 were constant. After the adapter fix the sparse-feature count fell from 40 to 15. Target std shrinks toward the game (far 0.049, close 0.025). Source: `experiments/results/exp1_feature_correlations/FINDINGS.md`.
- **Exp 2, feature group isolation** (2026-02-18, #136, 230 events, 538 rows, 47 features, 3-fold walk-forward): every group and both architectures at R² < 0; the predict-mean baseline is R² -0.0162 and MSE 0.001768; XGBoost on all features is -0.0105. Smaller groups are steadier and adding features hurt (Ridge sharply). Source: `experiments/results/exp2_feature_group_isolation/FINDINGS.md`.
- **Polymarket vs sportsbook cross-check** (Feb 2026, 228 events; no results file, recorded in #134): probability correlation 0.9978; Polymarket prices about 2.4 pp below the sportsbook (liquidity premium or vig difference); only 92 of 228 events had time-matched snapshots within 30 minutes. Adding Polymarket features changed nothing. Polymarket collection is deprioritized ([Polymarket market research](../collection/polymarket-market-research.md), [ADR-0007](../../docs/adr/0007-deprioritise-polymarket.md)).

## The IN_PLAY leak

`odds_analytics.feature_groups.TierSampler` treated `IN_PLAY` as lowest priority, so for pregame requests it accepted snapshots taken during the game and, choosing the latest, preferred them: 135 of 180 events used in-play snapshots. The inflated results were XGBoost at the tier R² +0.45 and LSTM about +0.02; after the fix the 2x2 (architecture x decision time) is XGBoost time-range -0.011, XGBoost tier -0.024, LSTM time-range -0.009, LSTM pregame -0.015 (corrected from +0.02). Fix: `odds_analytics.feature_groups.TierSampler` excludes `IN_PLAY` unless requested ([feature pipeline](../modelling/feature-pipeline.md), [look-ahead bias](../modelling/look-ahead-bias.md)). The same experiment also showed why [verification artifacts](experiment-scripts.md#verification-artifacts) matter: zeroed features had hidden the adapter bug.

## LSTM sequence models

Conclusion: sequence models add nothing; no LSTM variant beat predicting the mean, and LSTM was ruled out. XGBoost remains the model ([line-movement models](../modelling/line-movement-models.md), [model hub](clv-model-findings.md)).

Rows recorded in #160 (2026-02-21) and #166 (2026-02-28); no results directory holds the runs.

| Date | Variant | Data | CV R² |
|---|---|---|---|
| 2026-02-21 | Sequence only (15 features x 24 timesteps) | 800 events, Pinnacle | -0.010 ± 0.039 |
| 2026-02-21 | Sequence + 17 static features | 800 events, Pinnacle | -0.122 ± 0.115 |
| 2026-02-28 | Sequence only, packed sequences, 15 x 8 timesteps, 50 trials, 5-fold walk-forward | about 1K Odds API events, Pinnacle | -0.075 ± 0.113 |

Config for the Pinnacle runs: `experiments/configs/lstm_pinnacle_tuning.yaml` (also `experiments/configs/lstm_line_movement_tuning.yaml`). Best of the first tuning: hidden 48, 2 layers; the static branch increased overfitting. The 2026-02-28 result is worse than a constant predictor. The 2026-02-28 row has only the config behind it, no results file, so it is unverified.

- **2026-02-27 mask fix (#163).** The log row says packed sequences were introduced so the mask is applied correctly (code: `odds_analytics.lstm_line_movement._compact_sequences` in `odds_analytics.lstm_line_movement`). The experiment log files it as a bug fix with no result.
- **Exp 2 cross-check:** LSTM at both decision times is R² about 0 once the `IN_PLAY` leak is fixed; the earlier LSTM +0.02 was contaminated.

## Conclusion

More data and a re-run at 500+ events were the next steps ([injury features and timing](exp-nba-timing-and-injuries.md), [bet365 target and data volume](exp-target-and-data-volume.md)).

## Caveats

- **Not reproducible as recorded.** The scripts group features as `trajectory` and `cross_source`, which are not group names defined in `odds_analytics.feature_groups`, so `odds_analytics.training.config.FeatureConfig` rejects them as unknown names; the tabular group has 6 columns ([feature groups](../modelling/feature-groups.md)).
- **Feature counts differ by denominator.** 0 of the 60 testable features survive correction; 15 of the 75 are constant (`experiments/results/exp1_feature_correlations/FINDINGS.md`).
- **LSTM is still selectable.** Sequence features and the LSTM adapter remain in code behind `strategy_type: lstm_line_movement`; nothing prevents using them ([line-movement models](../modelling/line-movement-models.md)).
- All LSTM runs used the Pinnacle target on NBA; no LSTM run with the bet365 target or on EPL is recorded, and neither LSTM config sets `target_bookmaker`, so the code default (Pinnacle) applies.

## Related

- [CLV model findings](clv-model-findings.md), [Sharp benchmark](../modelling/sharp-benchmark.md)
