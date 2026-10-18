---
type: explanation
title: EPL CLV re-baseline against the simple baseline
description: The October 2026 out-of-sample test of the tabular EPL bet365 CLV model against a one-feature OLS baseline over two seeds on the evaluation range, with the decision-row rule, event counts, the pre-registered keep test and its result.
sources:
  - experiments/scripts/epl_clv_rebaseline.py
  - experiments/configs/xgboost_epl_clv_rebaseline.yaml
  - experiments/configs/xgboost_epl_cv_protocol_expanding.yaml
  - experiments/results/epl_clv_rebaseline/results.csv
  - experiments/results/epl_clv_rebaseline/summary.json
  - experiments/results/epl_clv_rebaseline/event_counts.csv
  - experiments/results/epl_clv_rebaseline/sampling_comparison.csv
  - experiments/results/epl_clv_rebaseline/lost_events.csv
  - experiments/results/epl_clv_rebaseline/decision_rows.csv
  - experiments/results/epl_clv_rebaseline/folds.csv
  - experiments/results/epl_clv_rebaseline/oof_predictions.csv
  - experiments/results/epl_clv_rebaseline/tail_predictions.csv
  - experiments/results/epl_clv_rebaseline/trials_seed_b.csv
status: stable
---

# EPL CLV re-baseline against the simple baseline

This experiment applies the keep test pre-registered on #496 to the tabular EPL CLV model, on the map #494. The model predicts the devigged bet365 CLV delta ([CLV target](../modelling/clv-target.md)). It is scored out of sample on the evaluation range of [ADR-0010](../../docs/adr/0010-expanding-walk-forward-and-evaluation-range.md), against the baseline defined on #495. **The model fails the keep test.** It beats the baseline by more than the seed spread, but neither seed's bootstrap CI excludes zero.

## Setup

- **Script:** `experiments/scripts/epl_clv_rebaseline.py`, with outputs in `experiments/results/epl_clv_rebaseline/`. It reuses the tuning, refit, baseline and bootstrap functions of `experiments/scripts/cv_protocol_common_range.py` ([CV protocol on a common validation range](exp-cv-protocol-common-range.md)). A second run reproduced `results.csv`, `folds.csv`, `oof_predictions.csv`, `tail_predictions.csv` and `decision_rows.csv` byte for byte. `trials_seed_b.csv` differed only in trial timestamps.
- **Config:** `experiments/configs/xgboost_epl_clv_rebaseline.yaml` inherits the #452 expanding cell `experiments/configs/xgboost_epl_cv_protocol_expanding.yaml`. It extends `end_date` to 2026-07-31, which covers the last FINAL EPL event in the database (2026-05-24), and sets `min_train_events` to 1,900. It holds seed A's hyperparameters. The script fails unless the config differs from the cell only in those fields and its model fields equal the #452 best trial in `experiments/results/cv_protocol_common_range/summary.json`. The features are tabular only, the target is devigged bet365, and the sharp reference is `[pinnacle, betfair_exchange]`.
- **Ranges:** the tuning range is the 1,900 events before 2024-08-01. The evaluation range is every event on or after that date. Nothing is tuned on the evaluation range.
- **Folds:** expanding walk-forward through `odds_analytics.training.cross_validation.make_walk_forward_splits`, with 50-event validation blocks. The first fold trains on the 1,900 tuning-range events. Each later fold trains on every earlier event. The partial last block is dropped, as ADR-0010 specifies.
- **Seeds:**
  - **Seed A** uses the frozen #452 expanding best trial, with XGBoost `random_state` 42 and Optuna sampler seed 42, as in #452.
  - **Seed B** re-tunes the #452 expanding cell on the tuning range: the same 100 TPE trials, search space and folds (`min_train_events` 380), but Optuna sampler seed 7 (`odds_analytics.training.tuner.OptunaTuner` `sampler_seed`) and `random_state` 7. Its best trial (92) is then frozen.
- **Baselines:** both are fitted on each fold's training rows. The gating baseline is a one-feature OLS fit with intercept on `tab_retail_sharp_diff`, the column the model receives. The training-fold mean is the floor.
- **Metric:** MSE pooled over every scored evaluation-range row. The model and the baselines are scored on identical rows, and the script fails if the two seeds' rows, targets or baseline predictions differ.
- **Uncertainty:** a paired bootstrap resamples events, with 2,000 resamples and seed 42. Every comparison uses the same resampled events.

## Decision rows

Each event contributes one decision row, its day-before snapshot. That is the row type of every tuning-range row.

- **The rule.** `DayBeforeSampler` in the script takes the snapshot nearest 24 h before kick-off. The snapshot must lie within 1 h of that point and quote bet365 and a sharp bookmaker for the home outcome. Ties go to the earlier snapshot. The sampler is passed to `odds_analytics.training.data_preparation.prepare_training_data_from_config` in place of the config's.
- **Why ±1 h.** football-data.co.uk (FDUK) stamps its day-before snapshot 23 h before kick-off during British Summer Time and 24 h otherwise. The 23 h stamps occur only in March to October kick-offs. Tuning-range rows run from 23.0 to 24.25 h. A strict "at or before 24 h" rule would drop 263 FDUK evaluation rows that sit at 23.0 h.
- **Why the rule was needed.** The production `odds_analytics.feature_groups.TierSampler` also yields one row per event. For live-collected events, though, it picks the latest snapshot at least 12 h out that quotes a sharp bookmaker, whatever its horizon. When that snapshot lacks bet365, the event is dropped instead of falling back to an earlier snapshot.
- **The check.** The script fails unless the day-before sampler reproduces every production tuning-range row exactly: events, features and targets.

`experiments/results/epl_clv_rebaseline/sampling_comparison.csv` compares the production sampler's row with the day-before row, by event:

| Season | Production row | Day-before row | Events |
|---|---|---|---|
| 2024-25 | 23 to 25 h | yes | 380 |
| 2025-26 | 23 to 25 h | yes | 281 |
| 2025-26 | under 23 h (15 to 22 h) | yes | 16 |
| 2025-26 | over 25 h (27 to 44 h) | no | 18 |
| 2025-26 | under 23 h | no | 2 |
| 2025-26 | none | yes | 7 |
| 2025-26 | none | no | 63 |

Every FDUK event gets the same row from both samplers. The two disagree only on live-collected events (`oddsportal_live`, April and May 2026). There the production sampler would mix horizons from 15 h to 44 h, or drop the event when its chosen snapshot lacks bet365.

## Event counts

`experiments/results/epl_clv_rebaseline/event_counts.csv` counts FINAL EPL events in the evaluation range:

| Season | Events | Distinct matches | Scored | With a day-before row | Rows per event |
|---|---|---|---|---|---|
| 2024-25 | 380 | 380 | 380 | 380 | 1 |
| 2025-26 | 387 | 379 | 354 | 304 | 1 |

- **Twins.** In 2025-26, 8 matches in April 2026 exist under two event IDs: an Odds API ID and an `op_live_` twin. The script fails if one match yields rows under two IDs. None does.
- **Lost events in 2025-26** (`experiments/results/epl_clv_rebaseline/lost_events.csv`), 83 of 387:
  - 33 have no final score: the 28 `op_live_` events of 3 to 22 March 2026, and 5 April events. Four of the April ones are `op_live_` twins of scored Odds API events.
  - 28 have no closing snapshot that quotes both bet365 and a sharp bookmaker. All kick off from 11 April 2026, and 7 of them are Odds API twins.
  - 22 have no snapshot within 1 h of the day-before point that quotes bet365 and a sharp bookmaker. All kick off from 12 April 2026 and are live-collected.
- **Pinnacle gap.** In 71 FDUK events from 2026-01-17 to 2026-03-01, the day-before snapshot has no Pinnacle price. These are not lost: the hybrid reference uses Betfair Exchange instead ([sharp benchmark](../modelling/sharp-benchmark.md)).
- **Tuning-range recount.** The tuning range has 1,900 events, the last kicking off at 2024-05-19 15:00 UTC, as ADR-0010 requires. Ten events share that kick-off. With ties ordered by event ID (`odds_lambda.storage.readers.OddsReader.get_events_by_date_range`), the 1,900th is `op_2023-2024_SHEUTD_TOT_2024-05-19`, not the `op_2023-2024_ARS_EVE_2024-05-19` that the #452 run's query-plan order gave. Either way, the first validation event is the first evaluation-range event, `op_2024-2025_MANUTD_FUL_2024-08-16`.
- **Scored rows.** The 684 evaluation rows give 13 folds, 650 events from 2024-08-16 to 2026-02-22 (`summary.json` `evaluation_range`). The last 34 events form the dropped partial block: 11 FDUK events up to 2026-03-01 and all 23 live-collected rows. So **no live-collected row enters the keep test.**

## Hyperparameters

| Parameter | Seed A (#452) | Seed B |
|---|---|---|
| `n_estimators` | 150 | 250 |
| `max_depth` | 2 | 2 |
| `learning_rate` | 0.01654 | 0.01106 |
| `min_child_weight` | 11 | 14 |
| `subsample` | 0.5 | 0.5 |
| `colsample_bytree` | 1.0 | 0.9 |
| `reg_lambda` | 3.164 | 0.01108 |
| Tuning-range CV MSE | 0.000708 | 0.000709 |

The tuning-range CV MSE is the mean fold MSE over the #452 expanding cell's folds, with ties ordered by event ID. #452 recorded 0.000706 for seed A, under its query-plan tie order (`summary.json` `seed_a`).

## Results

MSE pooled over the 650 scored rows (`experiments/results/epl_clv_rebaseline/results.csv`), in units of 10⁻⁶. Each difference has its 95% paired bootstrap CI.

| | Pooled MSE | Baseline minus this |
|---|---|---|
| Seed A | 809.0 | 5.9 [-6.7, 18.5] |
| Seed B | 808.9 | 6.0 [-6.7, 18.8] |
| OLS baseline | 814.9 | 0 |
| Training-fold mean (floor) | 830.3 | -15.3 [-39.8, 9.7] |

Seed A minus seed B is 0.13 [-2.7, 2.8].

## Keep test

These are the two conditions pre-registered on #496 (`summary.json` `keep_test`):

1. **PASS.** The mean over the seeds of MSE(baseline) − MSE(model) is 5.95 × 10⁻⁶. That is larger than |MSE(seed A) − MSE(seed B)| = 0.13 × 10⁻⁶.
2. **FAIL.** The test requires each seed's 95% CI on MSE(baseline) − MSE(model) to exclude zero, with the model better. Seed A's CI is [-6.7, 18.5] × 10⁻⁶ and seed B's is [-6.7, 18.8] × 10⁻⁶. Both include zero.

**The bet365 target fails the keep test.** On the #495 rules, keep is ruled out. Retarget or retire depends on whether another target passes under the same rules. No other target was tested here.

## Informational breakdowns

These do not decide anything. All values are in units of 10⁻⁶, from `results.csv`.

| Slice | Events | Seed A | Seed B | OLS | Floor | OLS minus seed A | OLS minus seed B |
|---|---|---|---|---|---|---|---|
| 2024-25 | 380 | 823.4 | 822.6 | 820.8 | 850.7 | -2.5 [-18.6, 13.5] | -1.8 [-18.6, 15.0] |
| 2025-26 | 270 | 788.8 | 789.6 | 806.6 | 801.5 | 17.7 [-1.8, 37.0] | 17.0 [-1.9, 35.8] |
| Pinnacle sharp | 590 | 802.1 | 801.8 | 807.6 | 820.2 | 5.5 [-7.5, 18.8] | 5.8 [-7.9, 19.6] |
| Betfair Exchange sharp | 60 | 877.0 | 878.6 | 887.2 | 929.4 | 10.2 [-24.0, 46.7] | 8.6 [-25.8, 44.9] |
| Partial last block (not a fold) | 34 | 604.6 | 605.8 | 597.4 | 595.1 | -7.2 [-56.0, 46.3] | -8.4 [-64.2, 53.4] |

- **Pooled R² against the training-fold mean:** 0.0256 for seed A, 0.0257 for seed B and 0.0185 for the OLS baseline.
- **By season:** in 2024-25 the model is no better than the OLS baseline (R² 0.032 and 0.033, against 0.035 for OLS). In 2025-26 it is about 17 × 10⁻⁶ better, with CIs that just include zero. There the OLS baseline scores worse than the floor (R² −0.006).
- **Pinnacle to Betfair Exchange switch:** the 60 scored rows whose decision snapshot uses Betfair Exchange as the sharp price fall from 2026-01-17 to 2026-02-22. They have higher MSE than the Pinnacle rows, and the model's gain over OLS there is similar to the Pinnacle rows' gain, with a CI about three times wider. Sixty events cannot separate the regimes.
- **Partial last block:** this block is the 34 events after the last full fold, scored once with training on every earlier event. On it, the model is worse than both the OLS baseline and the floor. The CIs are very wide, and it holds all 23 live-collected rows (`tail_predictions.csv`).

## Caveats

- **Single target.** Only bet365 was tested. A retarget, for example to Pinnacle, needs its own run under the same rules (#495).
- **No live-collected rows are scored.** The 2025-26 rows in the folds are all FDUK day-before snapshots. The live-collected regime (`oddsportal_live` and Betfair API snapshots, from April 2026) appears only in the dropped partial block. Live scoring reads live-collected snapshots, so this test says little about the model's live behaviour (inferred; [prediction scoring](../modelling/prediction-scoring.md)).
- **Local database coverage.** The run uses the local development database. In it, the 28 `op_live_` events of March 2026 have no scores, and FDUK coverage of 2025-26 ends on 2026-03-01. A fuller copy would add events, and with them new folds.
- **The deployed model is not this model.** `epl-clv-home` uses tabular and match_stats features with sliding-380 folds ([CLV model findings](clv-model-findings.md#deployed-model)). This test covers the tabular group under the ADR-0010 protocol only.
- **Seed B lands close to seed A.** The two seeds' tuned hyperparameters differ, but their pooled MSEs differ by 0.13 × 10⁻⁶. Condition 1 is therefore easy to pass. What decides the result is the event-level noise in condition 2.

## Related

- [CV protocol on a common validation range](exp-cv-protocol-common-range.md), [CLV model findings](clv-model-findings.md), [Cross-validation](../modelling/cross-validation.md), [Hyperparameter tuner](../modelling/hyperparameter-tuner.md)
