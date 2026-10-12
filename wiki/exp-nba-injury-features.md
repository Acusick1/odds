---
type: experiment
title: NBA injury features
summary: Injury and rest features looked positive until both arms were tuned the same way; they add nothing at the sharp and closing tiers.
sources:
  - docs/MODELING.md#injury-features-nba
  - docs/MODELING.md#experiment-log
  - docs/INJURIES.md
  - experiments/results/exp6b_injury_closing_tier/FINDINGS.md
  - experiments/results/exp6_learning_curve/FINDINGS.md
  - experiments/configs/xgboost_bet365_injuries.yaml
  - packages/odds-analytics/odds_analytics/injury_features.py::extract_injury_features
verified_at: 7887039
---

# NBA injury features

Conclusion: injury features add no predictive signal for CLV at any decision tier once models are tuned the same way. The pipeline still collects the data and the feature group still exists, but the deployed model does not use it ([nba-injury-reports](nba-injury-reports.md), [feature-groups](feature-groups.md)). Not tested on EPL.

## Results

- **First positive result** (2026-02-20, 800 events, Pinnacle target, tabular + injury + rest + timing, 100-trial Optuna): val R² 0.050, CV R² = 0.020 ± 0.025, injuries 55% of feature importance, 6 dead features.
- **bet365 at scale** (2026-02-27, about 5K OddsPortal events, 11-fold walk-forward): tabular + injuries CV R² 0.036 ± 0.033 against tabular-only 0.036 ± 0.028 once both were tuned. The earlier signal came from comparing a tuned model with an untuned baseline. Configs: `experiments/configs/xgboost_bet365_tuning.yaml`, `experiments/configs/xgboost_bet365_baseline_tuning.yaml`, `experiments/configs/xgboost_bet365_injuries.yaml`.
- **Exp 6, timing diagnostic** (2026-03-01): sharp tier tabular R² +0.018 vs +0.006 with injuries; pregame +0.001 vs +0.0015. About 93% of "pregame" events fell back to the sharp snapshot, and OddsPortal has no 0-3 h snapshots ([exp-nba-target-and-data-volume](exp-nba-target-and-data-volume.md), [decision-time](decision-time.md)).
- **Exp 6b, closing tier** (2026-03-01, Odds API, bet365 target, 479 closing and 826 sharp events): at closing both arms give R² 0.5959 identical to 16 decimals, a measurement artefact because the target is almost zero (std 0.0016 against 0.054 at the sharp tier; MSE 0.000005). At the sharp tier injuries slightly hurt (-0.031 vs -0.020). The Exp 4 game-time-decision correlation (`inj_impact_gtd_away` r = 0.28, Pinnacle target) does not transfer: r = -0.01 (p = 0.81) on bet365. At the sharp tier `tab_retail_sharp_diff` dominates (r = 0.356). Source: `experiments/results/exp6b_injury_closing_tier/FINDINGS.md`.

## Interpretation

By the time game-time injury designations are public (0-3 h) the closing line has priced them, so the closing-tier target is near zero: a catch-22 for testing the hypothesis. The GTD hypothesis is falsified for bet365; the signal in this data is cross-sectional market state ([retail-sharp-divergence](retail-sharp-divergence.md)). The injury pipeline can be deprioritised.

## Hazards

- The injury weights come from season-level player stats and the group is NBA-only; EPL has no equivalent ([feature-groups](feature-groups.md)).
- Closing-tier R² near 0.6 is an artefact and must not be quoted as skill.

## Related

- [Exp: hours-to-game](exp-hours-to-game.md), [CLV model findings](clv-model-findings.md)
