---
type: decision
title: bet365 as the CLV target
description: Why the model predicts bet365 rather than Pinnacle line movement, with the NBA target and learning-curve evidence and what the choice requires of the data source.
sources:
  - odds_analytics.training.config.FeatureConfig
  - odds_analytics.feature_groups._compute_target
  - odds_analytics.sequence_loader.calculate_devigged_bookmaker_target
  - experiments/configs/xgboost_epl_production_tabular_match_stats.yaml
  - experiments/results/exp6_learning_curve/FINDINGS.md
  - experiments/configs/xgboost_bet365_tuning.yaml
  - experiments/configs/xgboost_pinnacle_tuning.yaml
  - experiments/scripts/exp6_learning_curve.py
status: stable
---

# bet365 as the CLV target

**Decision.** The CLV model predicts the devigged bet365 close-minus-snapshot delta (`target_bookmaker: bet365`), not Pinnacle's ([CLV target](clv-target.md)). No ADR exists; the evidence is PRs #164, #167, #168 and #361, the production config (PR #362) and `experiments/results/exp6_learning_curve/`.

## Context

The target bookmaker is configurable. Pinnacle was the first target (Feb 2026 NBA runs); bet365 became the target for NBA at scale and for EPL.

## Evidence

- Pinnacle CLV was unpredictable with public features: XGBoost CV R² -0.017 ± 0.015 (about 800 NBA events, tuner converged to maximum regularisation; PR #167) and LSTM -0.075 ([line-movement models](line-movement-models.md), [early NBA signal and LSTM](../evaluation/exp-early-nba-signal-and-lstm.md)).
- Pinnacle lines move less: its MSE (0.000431) is 6.7x lower than bet365's (0.002883), leaving little variance to explain. The MSE figures were recorded in PR #167 with no results artefact, so they are unverified.
- bet365 reached CV R² 0.036 on about 5K NBA events and 0.058 on EPL, the deployed model (PR #361 and PR #362; [search space and ablation](../evaluation/exp-epl-feature-ablation-and-search-space.md), [CLV model findings](../evaluation/clv-model-findings.md)).

### NBA target and data-volume runs

**bet365 at scale (2026-02-27).** About 5K OddsPortal NBA events, 11-fold walk-forward CV, 100 Optuna trials, devigged bet365 target. Tabular-only CV R² = 0.036 ± 0.028; adding injuries gave 0.036 ± 0.033 ([NBA timing and injuries](../evaluation/exp-nba-timing-and-injuries.md)). Configs: `experiments/configs/xgboost_bet365_tuning.yaml`, `experiments/configs/xgboost_bet365_baseline_tuning.yaml`. First recorded in PR #164; no results directory holds the 11-fold numbers.

**Pinnacle as target (2026-03-01).** About 800 Odds API NBA events, 4-fold walk-forward, 100 trials: CV R² = -0.017 ± 0.015. Config: `experiments/configs/xgboost_pinnacle_tuning.yaml`. Introduced in PR #167.

**Learning curve, Exp 6 (2026-03-01).** Tuned tabular-only bet365 model on 500-4,524 OddsPortal events, expanding walk-forward CV (`experiments/scripts/exp6_learning_curve.py`, PR #168, results in `experiments/results/exp6_learning_curve/FINDINGS.md`):

| Events | 500 | 1,000 | 1,500 | 2,000 | 3,000 | 4,000 | 4,524 |
|---|---|---|---|---|---|---|---|
| R² mean | -0.066 | -0.040 | +0.036 | +0.031 | +0.014 | -0.006 | +0.018 |

R² plateaus at about 1.5K events and oscillates between about 0 and +0.035 afterwards (`FINDINGS.md`: -0.006 to +0.036). A log fit gives R² = 0.0301 ln(N) - 0.2302, marginal dR²/dN = 6.66e-06 at N = 4,524, and extrapolates 0.047 at 10K and 0.068 at 20K events; the recorded decision was that more data alone will not help. EPL has about 1.8K events, so a similar plateau is likely but not confirmed. All subsets sample the sharp tier (mean about 19 h before the game) and OddsPortal has no 0-3 h snapshots.

## Consequences

- `target_bookmaker` must belong to the selected data source: bet365 exists only in OddsPortal data for NBA, and for EPL the combined OddsPortal + FDUK set with `data_source: null` ([training pipeline](training-pipeline.md)).
- Pinnacle (with Betfair fallback) serves as the sharp reference feature instead ([hybrid sharp reference](decision-hybrid-sharp-reference.md)).
- That bet365 is a retail book a bettor can actually use is a plausible reason but is not stated in the source `(inferred)`.

## Hazards

- The code default is `target_bookmaker: pinnacle` (and `target_type: raw`), so a config that omits them trains the rejected target (`odds_analytics.training.config.FeatureConfig`).
- The production YAML lists bet365 in `retail_bookmakers` as well, so the target bookmaker's snapshot price is also an input to `retail_sharp_diff` ([sharp benchmark](sharp-benchmark.md)).
- The extrapolation to 20K events (+0.07) sits uneasily with "more data will not help"; the decision rests on the small marginal slope.
- Tuned CV R² is optimistically biased (best of N trials); comparisons between equally tuned runs are fair ([walk-forward CV](decision-walk-forward-cv.md)).
- Pinnacle and bet365 runs use different datasets (Odds API vs OddsPortal), so the comparison also changes the data source.

## Related

- [CLV model findings](../evaluation/clv-model-findings.md), [NBA timing and injuries](../evaluation/exp-nba-timing-and-injuries.md)
