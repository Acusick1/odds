---
type: explanation
title: Pinnacle against bet365 as the EPL CLV target
description: The October 2026 test of the devigged Pinnacle CLV target against bet365 on the same EPL events, folds and features, with the event counts, the pre-registered retarget test and its result, and the skill comparison behind the target ADR.
sources:
  - experiments/scripts/epl_clv_pinnacle_retest.py
  - experiments/scripts/epl_clv_rebaseline.py
  - experiments/configs/xgboost_epl_clv_rebaseline.yaml
  - experiments/configs/xgboost_epl_cv_protocol_expanding.yaml
  - experiments/results/epl_clv_pinnacle_retest/summary.json
  - experiments/results/epl_clv_pinnacle_retest/results.csv
  - experiments/results/epl_clv_pinnacle_retest/event_counts.csv
  - experiments/results/epl_clv_pinnacle_retest/lost_events.csv
  - experiments/results/epl_clv_pinnacle_retest/decision_rows.csv
  - experiments/results/epl_clv_pinnacle_retest/folds.csv
  - experiments/results/epl_clv_pinnacle_retest/oof_predictions.csv
  - experiments/results/epl_clv_pinnacle_retest/trials_pinnacle_seed_a.csv
  - experiments/results/epl_clv_pinnacle_retest/trials_pinnacle_seed_b.csv
  - experiments/results/epl_clv_pinnacle_retest/pinnacle_market_keys.csv
status: stable
---

# Pinnacle against bet365 as the EPL CLV target

This experiment applies the retarget test pre-registered on #482 to the tabular EPL CLV model, on the map #494. It trains the model on the devigged Pinnacle CLV delta instead of bet365's ([CLV target](../modelling/clv-target.md)), and scores both targets on the same events, folds and features. **The Pinnacle target fails both keep-test conditions.** bet365 had already failed ([EPL CLV re-baseline](exp-epl-clv-rebaseline.md)), so on the #495 rules no tested target passes. The comparison is the evidence for [ADR-0011](../../docs/adr/0011-bet365-clv-target-on-like-for-like-evidence.md).

## Setup

- **Script:** `experiments/scripts/epl_clv_pinnacle_retest.py`, with outputs in `experiments/results/epl_clv_pinnacle_retest/`. It reuses the row loading, fold scoring, comparison and keep-test functions of `experiments/scripts/epl_clv_rebaseline.py`, which take the target bookmaker from the config. With those functions shared, the re-baseline script still reproduces every #496 output byte for byte; only trial timings differ.
- **Config:** the re-baseline config `experiments/configs/xgboost_epl_clv_rebaseline.yaml` and the #452 expanding cell `experiments/configs/xgboost_epl_cv_protocol_expanding.yaml`. The script derives each Pinnacle config by changing only `target_bookmaker` to `pinnacle`. Everything else is as in the re-baseline: tabular features, the `[pinnacle, betfair_exchange]` sharp reference, expanding walk-forward at 50-event steps ([ADR-0010](../../docs/adr/0010-expanding-walk-forward-and-evaluation-range.md)) and one day-before row per event.
- **Decision rows:** the re-baseline's day-before rule, applied per target. The row is the snapshot nearest 24 h before kick-off, within 1 h, that quotes the target bookmaker and a sharp bookmaker ([decision rows](exp-epl-clv-rebaseline.md#decision-rows)). The script fails unless the rule reproduces every production tuning-range row for each target.
- **Common events:** both targets are scored on the evaluation-range events that have a row for both, with one row per match by the re-baseline's twin rule. The script fails unless every common event has the same decision snapshot and identical features under both targets, so only the target differs.
- **Training rule:** each fold trains on every common event before its validation block, so both targets train on identical events and features. The other rule, where each target trains on every earlier event it has a row for, gives the same folds here. No event outside the common set kicks off before the last validation block (`summary.json` `training_rule`).
- **Pinnacle hyperparameters:** tuned on the tuning range with the #452 expanding cell's 100 TPE trials, search space and folds. Seed A uses Optuna sampler seed 42 and XGBoost `random_state` 42, and seed B uses 7 and 7. Each seed's best trial is then frozen.
- **bet365 hyperparameters:** both #496 seeds, frozen as recorded in `experiments/results/epl_clv_rebaseline/summary.json`.
- **Baselines, metric and bootstrap:** as in the re-baseline. Each target has its own per-fold OLS fit on `tab_retail_sharp_diff` and its own training-fold mean. MSE is pooled over every scored row. The paired bootstrap resamples events, with 2,000 resamples and seed 42, and uses the same resamples for both targets.

## Event counts

- **Tuning range:** 1,900 events for each target, and the same 1,900. Every tuning-range event has a Pinnacle target, because every FDUK closing and day-before snapshot from 2019-20 to 2023-24 quotes Pinnacle.
- **Evaluation range:** 685 events have a bet365 row (as in #496) and 591 a Pinnacle row. 590 have both. They run from 2024-08-16 to 2026-01-08.
- **Scored:** 11 folds of 50 events, 550 events from 2024-08-16 to 2025-12-22. The last 40 common events form the partial last block, which is dropped as in #496.
- **Every scored row is an FDUK row, with Pinnacle as the sharp price.** A common event's decision snapshot quotes Pinnacle, so the hybrid fallback to Betfair Exchange never applies.

`experiments/results/epl_clv_pinnacle_retest/lost_events.csv` gives each evaluation-range event without a row, by target:

| Season | Target | Events | Rows | Common | No closing snapshot with the target and a sharp book | No day-before snapshot | No final score |
|---|---|---|---|---|---|---|---|
| 2024-25 | bet365 | 380 | 380 | 380 | 0 | 0 | 0 |
| 2024-25 | Pinnacle | 380 | 380 | 380 | 0 | 0 | 0 |
| 2025-26 | bet365 | 387 | 305 | 210 | 28 | 21 | 33 |
| 2025-26 | Pinnacle | 387 | 211 | 210 | 139 | 4 | 33 |
| 2026-27 | bet365 | 43 | 0 | 0 | 43 | 0 | 0 |
| 2026-27 | Pinnacle | 43 | 0 | 0 | 43 | 0 | 0 |

- **Where the common set ends.** From 2026-01-17 the FDUK snapshots carry no Pinnacle price. After that date, the only `1x2` Pinnacle closing prices are in Odds API snapshots of 13 April 2026 events. One of those events, on 2026-04-12, yields a Pinnacle row, and it has no bet365 row. Every other event from 2026-01-17 has no Pinnacle row: 139 have no closing snapshot that quotes Pinnacle and 4 have no day-before one. That drops 95 events that have a bet365 row, 60 of which #496 scored.
- **Pinnacle prices exist outside the `1x2` market key.** From May 2026, and for every 2026-27 event, the Odds API snapshots quote Pinnacle only under the `h2h` key, which the `1x2` config does not read. So the 2026-27 events have no Pinnacle target here, although Pinnacle's prices for them are in the database. `experiments/results/epl_clv_pinnacle_retest/pinnacle_market_keys.csv` counts the events with a Pinnacle quote by month, source, market key and tier. The 2026-27 events have no bet365 target either ([event counts](exp-epl-clv-rebaseline.md#event-counts)).

## Hyperparameters

| Parameter | Pinnacle seed A | Pinnacle seed B | bet365 seed A (#452) | bet365 seed B (#496) |
|---|---|---|---|---|
| `n_estimators` | 100 | 100 | 150 | 250 |
| `max_depth` | 3 | 2 | 2 | 2 |
| `learning_rate` | 0.01074 | 0.01053 | 0.01654 | 0.01106 |
| `min_child_weight` | 14 | 15 | 11 | 14 |
| `subsample` | 0.6 | 0.5 | 0.5 | 0.5 |
| `colsample_bytree` | 0.7 | 0.6 | 1.0 | 0.9 |
| `reg_lambda` | 0.7603 | 0.1020 | 3.164 | 0.01108 |
| Best trial | 62 | 85 | | 92 |
| Tuning-range CV MSE | 0.000759 | 0.000759 | 0.000708 | 0.000709 |

Both Pinnacle seeds settle at the lower bound of `n_estimators` (100), near the lower bound of `learning_rate` (0.01) and at or near the upper bound of `min_child_weight` (15). Those three are at or near the end of the search space that gives the smallest model; seed A still takes depth 3 (`summary.json` `pinnacle`).

## Results

MSE pooled over the 550 scored events (`experiments/results/epl_clv_pinnacle_retest/results.csv`), in units of 10⁻⁶, with 95% paired bootstrap CIs:

| | bet365 | Pinnacle |
|---|---|---|
| Target variance | 817.3 | 742.0 |
| Seed A | 800.4 | 744.6 |
| Seed B | 800.1 | 743.6 |
| OLS baseline | 807.0 | 744.7 |
| Training-fold mean (floor) | 819.3 | 743.0 |
| OLS minus seed A | 6.6 [-6.0, 20.0] | 0.08 [-7.6, 7.6] |
| OLS minus seed B | 6.9 [-6.7, 20.6] | 1.1 [-4.8, 6.7] |
| Seed A minus seed B | 0.33 | 0.98 |
| Gain over OLS relative to MSE(OLS), seed A / seed B | 0.82% / 0.86% | 0.01% / 0.14% |
| R² against the floor, seed A / seed B | 0.023 / 0.023 | -0.002 / -0.001 |
| R² of the OLS baseline | 0.015 | -0.002 |

## Retarget test

The two conditions pre-registered on #482, from #496, against Pinnacle's own OLS baseline (`summary.json` `keep_test_pinnacle`):

1. **FAIL.** The mean over the seeds of MSE(baseline) − MSE(model) is 0.57 × 10⁻⁶. That is smaller than |MSE(seed A) − MSE(seed B)| = 0.98 × 10⁻⁶.
2. **FAIL.** Seed A's CI is [-7.6, 7.6] × 10⁻⁶ and seed B's [-4.8, 6.7] × 10⁻⁶. Both include zero.

**The Pinnacle target fails the retarget test.** The pre-registered decision is therefore retire. On the same 550 events, bet365 passes condition 1 and fails condition 2, as it did on #496's 650 (`summary.json` `keep_test_bet365_on_common_set`).

## Comparing the targets

The targets have different variances, so raw MSE does not compare their skill. The comparison uses R² against each target's own training-fold mean, and the gain over each target's own OLS baseline relative to that baseline's MSE. `summary.json` `skill_pinnacle_minus_bet365` gives Pinnacle minus bet365, with paired bootstrap CIs:

| | Seed A | Seed B |
|---|---|---|
| R² | -0.025 [-0.051, 0.001] | -0.024 [-0.049, 0.002] |
| Relative gain over OLS | -0.81% [-2.1%, 0.42%] | -0.72% [-2.0%, 0.59%] |

- **Pinnacle has no out-of-sample skill.** Neither the model nor the OLS baseline beats the training-fold mean. The bet365 model's R² is 0.023, but its gain over the floor is not established either: 18.9 [-6.1, 43.8] × 10⁻⁶ for seed A (`results.csv` `mean_minus_seed_a`).
- **The difference between the targets is not established.** Each CI on the difference just includes zero. bet365 is ahead on every point estimate, but 550 events cannot separate the targets at 95%.
- **Pinnacle lines move about as much as bet365's on EPL.** Pinnacle's target variance is 91% of bet365's, and the two targets correlate at 0.91 (`summary.json` `target_correlation`). The NBA finding that Pinnacle's MSE was 6.7 times lower than bet365's ([target and data volume](exp-target-and-data-volume.md)) does not hold here. So a smaller target does not explain Pinnacle's lower point estimates of skill. Those estimates are within bootstrap noise of bet365's.
- **By season,** in units of 10⁻⁶ (informational):

| Season | Events | bet365 OLS minus seed A | Pinnacle OLS minus seed A | bet365 R² seed A | Pinnacle R² seed A |
|---|---|---|---|---|---|
| 2024-25 | 380 | -2.5 [-18.6, 13.5] | -1.1 [-10.6, 8.3] | 0.032 | 0.000 |
| 2025-26 | 170 | 27.1 [4.8, 52.5] | 2.7 [-6.3, 11.9] | 0.000 | -0.011 |

In 2025-26 the bet365 model's gain over OLS has a CI that excludes zero, because that season's OLS fit scores below the floor (R² -0.036). The keep test is pooled over the whole range, and a season slice does not decide it.

## Caveats

- **No live-collected rows.** Every scored row is an FDUK day-before snapshot with a Pinnacle sharp price, so the test says nothing about Betfair Exchange sharp rows or live-collected snapshots. Those are what live scoring reads (inferred; [prediction scoring](../modelling/prediction-scoring.md)).
- **The common set is shorter than #496's.** It ends at the FDUK Pinnacle gap, so it has 550 scored events against #496's 650. The bet365 numbers above are on the common set, not #496's.
- **The `retail_sharp_diff` feature uses Pinnacle as the sharp price on every scored row.** For the Pinnacle target, the feature's sharp leg is the target bookmaker's own decision-time price. The retail leg includes bet365, the other target bookmaker ([sharp benchmark](../modelling/sharp-benchmark.md)).
- **Tabular features only.** As in #496, the test covers the tabular group under the ADR-0010 protocol, not the deployed tabular and match_stats model.

## Related

- [EPL CLV re-baseline](exp-epl-clv-rebaseline.md), [CLV model findings](clv-model-findings.md), [Target bookmaker and data volume experiments](exp-target-and-data-volume.md), [CLV target](../modelling/clv-target.md)
