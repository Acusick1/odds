---
status: proposed
---

# Predict bet365 line movement as the CLV target

The CLV model trains on the devigged bet365 close-minus-snapshot delta, not Pinnacle's. The experiments behind it are on [Target bookmaker and data volume experiments](../../wiki/evaluation/exp-target-and-data-volume.md).

## Context

The target bookmaker is configurable (`odds_analytics.training.config.FeatureConfig.target_bookmaker`). Pinnacle was the first target (February 2026, NBA). Pinnacle CLV was unpredictable from public features: XGBoost CV R² -0.017 ± 0.015 on about 800 NBA events, with the tuner converging to maximum regularisation (PR #167), and LSTM -0.075 (#166). Pinnacle lines move less: MSE 0.000431 against bet365's 0.002883, 6.7x lower (PR #167; no results artefact, unverified).

## Decision

Train on the devigged bet365 close-minus-snapshot delta: `target_type: devigged_bookmaker` and `target_bookmaker: bet365`, set explicitly in each training YAML. Adopted from PR #164 for NBA at scale; the EPL production config is PR #362.

## Considered options

- **Pinnacle as target.** Rejected on the CV R² and MSE figures above.
- **More data with the same target.** The learning curve plateaus at about 1.5K events, and the recorded reading was that more data alone will not help (`experiments/results/exp6_learning_curve/FINDINGS.md`, PR #168).

## Consequences

- bet365 reached CV R² 0.036 on about 5K NBA events (PR #164) and 0.058 on EPL, the deployed model (PRs #361, #362).
- The target bookmaker must exist in the selected data source: OddsPortal for NBA; OddsPortal plus FDUK with `data_source: null` for EPL ([training config](../../wiki/modelling/training-config.md#choosing-data)).
- Pinnacle, with Betfair as fallback, becomes the sharp reference feature instead ([ADR-0003](0003-hybrid-sharp-reference.md)).
- The code defaults stay `raw` and `pinnacle`, so a YAML that omits them trains the rejected target (#434).
- The Pinnacle and bet365 runs used different datasets (Odds API versus OddsPortal), so the comparison also changed the data source.

## Evidence

PRs #164, #167, #168, #361, #362; `experiments/results/exp6_learning_curve/FINDINGS.md`; `experiments/configs/xgboost_bet365_tuning.yaml`, `experiments/configs/xgboost_bet365_baseline_tuning.yaml`, `experiments/configs/xgboost_pinnacle_tuning.yaml`; `experiments/scripts/exp6_learning_curve.py`. Wiki: [CLV and the training target](../../wiki/modelling/clv-target.md), [CLV model findings](../../wiki/evaluation/clv-model-findings.md).
