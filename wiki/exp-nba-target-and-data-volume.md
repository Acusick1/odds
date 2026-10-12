---
type: experiment
title: NBA target bookmaker and data volume
summary: bet365 at about 5K events plateaus at R² near 0.036, Pinnacle CLV is unpredictable, and the learning curve is flat after about 1.5K events.
sources:
  - docs/MODELING.md#xgboost-bet365-at-scale
  - docs/MODELING.md#pinnacle-as-target
  - docs/MODELING.md#data-volume-learning-curve
  - docs/MODELING.md#experiment-log
  - experiments/results/exp6_learning_curve/FINDINGS.md
  - experiments/configs/xgboost_bet365_tuning.yaml
  - experiments/configs/xgboost_pinnacle_tuning.yaml
  - experiments/scripts/exp6_learning_curve.py
verified_at: 7887039
---

# NBA target bookmaker and data volume

Three linked NBA results: bet365 is predictable only weakly, Pinnacle not at all, and more data does not change that. They justify [decision-bet365-target](decision-bet365-target.md).

## bet365 at scale (2026-02-27)

About 5K OddsPortal NBA events, 11-fold walk-forward CV, 100 Optuna trials, devigged bet365 target. Tabular-only CV R² = 0.036 ± 0.028; adding injuries gave 0.036 ± 0.033 ([exp-nba-injury-features](exp-nba-injury-features.md)). Configs: `experiments/configs/xgboost_bet365_tuning.yaml`, `experiments/configs/xgboost_bet365_baseline_tuning.yaml`.

## Pinnacle as target (2026-03-01)

About 800 Odds API NBA events, 4-fold walk-forward, 100 trials: CV R² = -0.017 ± 0.015, with the tuner converging to maximum regularisation. Pinnacle MSE (0.000431) is 6.7x lower than bet365's (0.002883), so there is little target variance to explain: Pinnacle lines move less. The LSTM reached -0.075 on the same target ([exp-lstm-sequence-models](exp-lstm-sequence-models.md)). Config: `experiments/configs/xgboost_pinnacle_tuning.yaml`.

## Learning curve, Exp 6 (2026-03-01)

Tuned tabular-only bet365 model on 500-4,524 OddsPortal events, expanding walk-forward CV (`experiments/scripts/exp6_learning_curve.py`):

| Events | 500 | 1,000 | 1,500 | 2,000 | 3,000 | 4,000 | 4,524 |
|---|---|---|---|---|---|---|---|
| R² mean | -0.066 | -0.040 | +0.036 | +0.031 | +0.014 | -0.006 | +0.018 |

R² plateaus at about 1.5K events and oscillates between about 0 and +0.035 afterwards (`FINDINGS.md`: -0.006 to +0.036). A log fit gives R² = 0.0301 ln(N) - 0.2302, marginal dR²/dN = 6.66e-06 at N = 4,524, and extrapolates 0.047 at 10K and 0.068 at 20K events; the recorded decision was that more data alone will not help. EPL has about 1.8K events, so a similar plateau is likely but not confirmed. All subsets sample the sharp tier (mean about 19 h before the game) and OddsPortal has no 0-3 h snapshots.

## Hazards

- The extrapolation to 20K events (+0.07) sits uneasily with "more data will not help"; the decision rests on the small marginal slope.
- Tuned CV R² is optimistically biased (best of N trials); comparisons between equally tuned runs are fair ([walk-forward-cv](walk-forward-cv.md)).
- Pinnacle and bet365 runs use different datasets (Odds API vs OddsPortal), so the comparison also changes the data source ([training-config](training-config.md)).

## Related

- [Hours-to-game](exp-hours-to-game.md), [CLV model findings](clv-model-findings.md)
