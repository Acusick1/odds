---
status: superseded by ADR-0012
---

# Keep bet365 over Pinnacle as the CLV target, on a like-for-like EPL test

If the EPL CLV model is kept in any role, it keeps the devigged bet365 close-minus-snapshot delta as its target, not Pinnacle's. This supersedes [ADR-0002](0002-bet365-clv-target.md) once accepted. ADR-0002 made the same choice on NBA runs that the map #494 voids. The evidence is on [Pinnacle against bet365 as the EPL CLV target](../../wiki/evaluation/exp-epl-clv-pinnacle-retest.md).

## Context

ADR-0002 rejected Pinnacle on two NBA runs that differ in source, size and folds: bet365 on about 5K OddsPortal events over 11 folds, and Pinnacle on about 800 Odds API events over 4 folds. Its stated reason was that Pinnacle lines move less, with an MSE 6.7 times lower than bet365's. Those runs also predate the training and CV validity fixes (#474, #434, #477, #437, #475, #481), and #494 voids every pre-fix result.

FDUK EPL data quotes both bookmakers, so the two targets can be compared on the same events. The tabular bet365 model failed the keep test of #496 ([EPL CLV re-baseline](../../wiki/evaluation/exp-epl-clv-rebaseline.md)). Under #495, a retarget means another target passing the same test. #482 pre-registered that test for Pinnacle, and the bet365 comparison alongside it.

## Decision

Proposed 2026-10-18 (#482, under the map #494):

- The target stays `target_type: devigged_bookmaker` with `target_bookmaker: bet365`.
- Pinnacle is not adopted as the target. On the pre-registered test it fails both keep-test conditions against its own OLS `retail_sharp_diff` baseline. On the same 550 evaluation-range events and 11 folds, it has no out-of-sample skill: R² against the training-fold mean is -0.002 and -0.001 over the two seeds. bet365's is 0.023 for both seeds, although its gain over the training-fold mean has a CI that includes zero.
- ADR-0002's reason, that Pinnacle lines move too little, does not hold on EPL. Pinnacle's target variance is 91% of bet365's, and the two targets correlate at 0.91.

Whether the model is kept at all is a separate decision, on #494. No tested target passes the keep test, which on the #495 rules means retire.

## Considered options

Both targets were scored with tabular features, the hybrid sharp reference and the ADR-0010 protocol. Each was scored on the same day-before rows, folds and training events. Each target was compared against its own per-fold OLS baseline. The figures are pooled over the 550 scored events, with 95% paired bootstrap CIs by event (`experiments/results/epl_clv_pinnacle_retest/summary.json`):

- **bet365:** the gain over OLS is 6.6 [-6.0, 20.0] × 10⁻⁶ for seed A and 6.9 [-6.7, 20.6] × 10⁻⁶ for seed B, with a seed spread of 0.33 × 10⁻⁶. That passes condition 1 and fails condition 2, as on #496's 650 events.
- **Pinnacle:** the gain over OLS is 0.08 [-7.6, 7.6] × 10⁻⁶ for seed A and 1.1 [-4.8, 6.7] × 10⁻⁶ for seed B, with a seed spread of 0.98 × 10⁻⁶. That fails both conditions. Both seeds tune to the fewest trees and close to the lowest learning rate and highest `min_child_weight` the search space allows.
- **Skill, Pinnacle minus bet365:** R² is -0.025 [-0.051, 0.001] for seed A and -0.024 [-0.049, 0.002] for seed B. The gain over OLS relative to MSE(OLS) is -0.81% [-2.1%, 0.42%] and -0.72% [-2.0%, 0.59%]. Raw MSE is not compared, because the targets' variances differ.

## Consequences

- The evidence for bet365 over Pinnacle is weak. bet365 leads on every point estimate, but each CI on the skill difference just includes zero. The choice rests on Pinnacle showing no skill at all, not on bet365 showing significant skill.
- The comparison covers FDUK day-before rows with a Pinnacle sharp price, from 2024-08-16 to 2025-12-22. From 2026-01-17, FDUK has no Pinnacle prices. From May 2026 the Odds API quotes Pinnacle only under the `h2h` market key, which a `1x2` config does not read. So no later EPL event can be scored for a Pinnacle target without a change to how the 3-way market is keyed.
- The target stays a config choice with no code default. `target_type` is required, and `target_bookmaker` is required for a devigged target (`odds_analytics.training.config.FeatureConfig`, #434). So each training YAML must name bet365.
- The test covers the tabular group. The deployed model's tabular and match_stats features were not re-tested on either target.

## Evidence

- Issues #482, #494, #495, #496.
- `experiments/scripts/epl_clv_pinnacle_retest.py`; `experiments/results/epl_clv_pinnacle_retest/` (`summary.json`, `results.csv`, `event_counts.csv`, `lost_events.csv`, `folds.csv`, `oof_predictions.csv`).
- `experiments/scripts/epl_clv_rebaseline.py`; `experiments/results/epl_clv_rebaseline/summary.json`.
- Wiki: [Pinnacle against bet365 as the EPL CLV target](../../wiki/evaluation/exp-epl-clv-pinnacle-retest.md), [EPL CLV re-baseline](../../wiki/evaluation/exp-epl-clv-rebaseline.md), [CLV target](../../wiki/modelling/clv-target.md).
