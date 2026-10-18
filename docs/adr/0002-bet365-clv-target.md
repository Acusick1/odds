---
status: superseded by ADR-0012
---

# Predict bet365 line movement as the CLV target

The CLV model trains on the devigged bet365 close-minus-snapshot delta, not Pinnacle's. The experiments behind it are on [Target bookmaker and data volume experiments](../../wiki/evaluation/exp-target-and-data-volume.md).

## Context

The target bookmaker became configurable in PR #161 (`odds_analytics.training.config.FeatureConfig.target_bookmaker`). The February 2026 NBA experiments used a devigged Pinnacle target on 193 to 800 events (`docs/MODELING.md` experiment log, commit 2840086).

- **bet365 at scale** (2026-02-27): about 5K OddsPortal NBA events, 11-fold walk-forward CV, devigged bet365 target, CV R² 0.036 ± 0.028 tabular-only (commit 2840086, in PR #164). Its CV MSE was 0.002883 (commit b950dc5, in PR #166).
- **Pinnacle as target** (2026-03-01): about 800 Odds API NBA events, 4-fold walk-forward CV, XGBoost CV R² −0.017 ± 0.015 with the tuner converging to maximum regularisation, and CV MSE 0.000431, 6.7 times lower than bet365's 0.002883 (commit d7c6621, in PR #167). An LSTM on about 1K Odds API events scored CV R² −0.075 ± 0.113 (commit b950dc5, in PR #166).
- **More data with the same target.** The Exp 6 learning curve on the bet365 target plateaus at about 1.5K events, and PR #168 records that more OddsPortal data will not meaningfully improve the model (`experiments/results/exp6_learning_curve/FINDINGS.md`).

## Decision

Train on the devigged bet365 close-minus-snapshot delta: `target_type: devigged_bookmaker` and `target_bookmaker: bet365`, set explicitly in each training YAML. Commit d7c6621 (2026-03-01) records the Pinnacle result as confirming bet365 as the viable target, with the reasons that Pinnacle lines move less, leaving less variance to predict, and that Pinnacle, as the sharpest book, has less pricing inefficiency to exploit than retail books such as bet365. It records that Pinnacle would need substantially more history or non-public features to revisit. The EPL production config uses the same target (PR #362). No reason is recorded for why bet365 movement, rather than Pinnacle movement, is the quantity to bet on.

## Considered options

- **Pinnacle as target.** Rejected on the CV R² and MSE figures above.

## Consequences

- bet365 reached CV R² 0.058 on EPL, the deployed model (PR #362; deployed as `epl-clv-home` `v308058c`, PR #361).
- The target bookmaker must exist in the selected data source: OddsPortal for NBA; OddsPortal plus FDUK with `data_source: null` for EPL ([training config](../../wiki/modelling/howto-train-and-publish-model.md#choose-data)).
- Pinnacle, with Betfair as fallback, becomes the sharp reference feature instead ([ADR-0003](0003-hybrid-sharp-reference.md)).
- The code defaults stay `target_type: raw` and `target_bookmaker: pinnacle`. A YAML that omits both trains the raw average-implied-probability delta (#434); a YAML that sets `devigged_bookmaker` without a bookmaker trains the rejected Pinnacle target.
- The bet365 and Pinnacle runs differ in source (OddsPortal against the Odds API), size (about 5K against about 800 events) and folds (11 against 4). In the project's learning curve, 500 events gave R² −0.066 and 1,500 events +0.036 (`experiments/results/exp6_learning_curve/FINDINGS.md`). No run compares the two targets on the same source and event count (#482).

## Evidence

- `docs/MODELING.md` commits 2840086 (PR #164), b950dc5 (PR #166) and d7c6621 (PR #167); text at `git show 7887039:docs/MODELING.md`.
- PRs #161, #168, #361, #362; issues #434, #482.
- `experiments/results/exp6_learning_curve/FINDINGS.md`; `experiments/scripts/exp6_learning_curve.py`.
- Configs: `experiments/configs/xgboost_bet365_tuning.yaml`, `experiments/configs/xgboost_bet365_baseline_tuning.yaml`, `experiments/configs/xgboost_pinnacle_tuning.yaml`, `experiments/configs/xgboost_epl_production_tabular_match_stats.yaml`.
- Wiki: [CLV and the training target](../../wiki/modelling/clv-target.md), [CLV model findings](../../wiki/evaluation/clv-model-findings.md).
