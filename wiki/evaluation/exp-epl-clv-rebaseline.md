---
type: explanation
title: EPL CLV re-baseline against the simple baseline
description: The October 2026 out-of-sample test of the tabular EPL bet365 CLV model against a one-feature OLS baseline over two seeds on the evaluation range, with its two runs, the decision-row rule, event counts, the pre-registered keep test and its result.
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
  - experiments/results/epl_clv_rebaseline/run1_local_copy/results.csv
  - experiments/results/epl_clv_rebaseline/run1_local_copy/summary.json
  - experiments/results/epl_clv_rebaseline/run1_local_copy/event_counts.csv
status: stable
---

# EPL CLV re-baseline against the simple baseline

This experiment applies the keep test pre-registered on #496 to the tabular EPL CLV model, on the map #494. The model predicts the devigged bet365 CLV delta ([CLV target](../modelling/clv-target.md)). It is scored out of sample on the evaluation range of [ADR-0010](../../docs/adr/0010-expanding-walk-forward-and-evaluation-range.md), against the baseline defined on #495. **The model fails the keep test.** It beats the baseline by more than the seed spread, but neither seed's bootstrap CI excludes zero.

## Two runs

The test ran twice, on two copies of the data. Both are recorded, and the second is the decision result.

1. **Run 1, the earlier local copy** (`experiments/results/epl_clv_rebaseline/run1_local_copy/`). The local development database then held EPL events up to 2026-05-24. Its config `end_date` was 2026-07-31. It scored 650 events and failed condition 2.
2. **Run 2, the full range** (`experiments/results/epl_clv_rebaseline/`). After seeing run 1's verdict, the user chose to refresh the local database from production for 2025-08-01 to 2026-10-18 and re-run with everything else unchanged. Because that choice came after a result was seen, run 2 is the decision result and run 1 is kept as recorded. The config `end_date` is 2026-10-18, which covers every FINAL, scored EPL event in the database (the last kicks off at 2026-10-17 16:30).

**The verdict did not change, and nor did any scored number.** The refresh added one 2025-26 row and the 43 events of 2026-27, but none of them falls in a full 50-event block. The 13 folds, the 650 scored events and every keep-test number are identical in the two runs (`folds.csv` is byte-identical). Only the partial last block (34 events in run 1, 35 in run 2), the counts and the sampling comparison differ. The figures below are run 2's.

## Setup

- **Script:** `experiments/scripts/epl_clv_rebaseline.py`, with outputs in `experiments/results/epl_clv_rebaseline/`. It reuses the tuning, refit, baseline and bootstrap functions of `experiments/scripts/cv_protocol_common_range.py` ([CV protocol on a common validation range](exp-cv-protocol-common-range.md)). On run 1's data, a repeat run reproduced `results.csv`, `folds.csv`, `oof_predictions.csv`, `tail_predictions.csv` and `decision_rows.csv` byte for byte. `trials_seed_b.csv` differed only in trial timestamps.
- **Config:** `experiments/configs/xgboost_epl_clv_rebaseline.yaml` inherits the #452 expanding cell `experiments/configs/xgboost_epl_cv_protocol_expanding.yaml`. It extends `end_date` to 2026-10-18 and sets `min_train_events` to 1,900. It holds seed A's hyperparameters. The script fails unless the config differs from the cell only in those fields and its model fields equal the #452 best trial in `experiments/results/cv_protocol_common_range/summary.json`. The features are tabular only, the target is devigged bet365, and the sharp reference is `[pinnacle, betfair_exchange]`.
- **Ranges:** the tuning range is the 1,900 events before 2024-08-01. The evaluation range is every event on or after that date. Nothing is tuned on the evaluation range.
- **Folds:** expanding walk-forward through `odds_analytics.training.cross_validation.make_walk_forward_splits`, with 50-event validation blocks. The first fold trains on the 1,900 tuning-range events. Each later fold trains on every earlier event. `odds_analytics.training.cross_validation.make_walk_forward_splits` drops the partial last block ([CV protocol on a common validation range](exp-cv-protocol-common-range.md#fold-layout-for-the-re-baseline)).
- **Seeds:**
  - **Seed A** uses the frozen #452 expanding best trial, with XGBoost `random_state` 42 and Optuna sampler seed 42, as in #452.
  - **Seed B** re-tunes the #452 expanding cell on the tuning range: the same 100 TPE trials, search space and folds (`min_train_events` 380), but Optuna sampler seed 7 (`odds_analytics.training.tuner.OptunaTuner` `sampler_seed`) and `random_state` 7. Its best trial (92) is then frozen.
- **Baselines:** both are fitted on each fold's training rows. The gating baseline is a one-feature OLS fit with intercept on `tab_retail_sharp_diff`, the column the model receives. The training-fold mean is the floor.
- **Metric:** MSE pooled over every scored evaluation-range row. The model and the baselines are scored on identical rows, and the script fails if the two seeds' rows, targets or baseline predictions differ.
- **Uncertainty:** a paired bootstrap resamples events, with 2,000 resamples and seed 42. Every comparison uses the same resampled events.
- **One row per match.** A live-collected match can exist under two event IDs, an Odds API ID and an `op_live_` twin. When both yield rows, the script keeps the Odds API ID's row, the ID that `odds_lambda.storage.writers.OddsWriter.adopt_live_twins` keeps when it merges twins, and drops the other. The script fails if a match still has two rows. Neither run dropped a twin, because no match had rows under both IDs (`summary.json` `evaluation_range.twins_dropped`).

## Decision rows

Each event contributes one decision row, its day-before snapshot. That is the row type of every tuning-range row.

- **The rule.** `DayBeforeSampler` in the script takes the snapshot nearest 24 h before kick-off. The snapshot must lie within 1 h of that point and quote bet365 and a sharp bookmaker for the home outcome. Ties go to the earlier snapshot. The rule does not prefer snapshots at or before the 24 h mark: 12 of the 24 live-collected rows sit between 23.0 and 23.9 h. All of those are in the unscored partial block. The sampler is passed to `odds_analytics.training.data_preparation.prepare_training_data_from_config` in place of the config's.
- **Why ±1 h.** football-data.co.uk (FDUK) stamps its day-before snapshot 23 h before kick-off during British Summer Time and 24 h otherwise. The 23 h stamps occur only in March to October kick-offs. Tuning-range rows run from 23.0 to 24.25 h. A strict "at or before 24 h" rule would drop 262 FDUK evaluation rows that sit at 23.0 h.
- **Why the rule was needed.** The production `odds_analytics.feature_groups.TierSampler` also yields one row per event. For live-collected events, though, it picks the latest snapshot at least 12 h out that quotes a sharp bookmaker, whatever its horizon. When that snapshot lacks bet365, the event is dropped instead of falling back to an earlier snapshot.
- **The check.** The script fails unless the day-before sampler reproduces every production tuning-range row exactly: events, features and targets.

`experiments/results/epl_clv_rebaseline/sampling_comparison.csv` compares the production sampler's row with the day-before row, by event:

| Season | Production row | Day-before row | Events |
|---|---|---|---|
| 2024-25 | 23 to 25 h | yes | 380 |
| 2025-26 | 23 to 25 h | yes | 282 |
| 2025-26 | under 23 h (13.0 to 15.4 h) | yes | 4 |
| 2025-26 | none | yes | 19 |
| 2025-26 | under 23 h (14.9 to 22.9 h) | no | 4 |
| 2025-26 | over 25 h (41.5 to 44 h) | no | 3 |
| 2025-26 | none | no | 75 |
| 2026-27 | none | no | 43 |

Every FDUK event gets the same row from both samplers. The two disagree only on live-collected events (April and May 2026). There the production sampler mixes horizons from 13 h to 44 h. More often it drops the event, because the snapshot it picks is a Betfair API snapshot that has no bet365 price. That is the case for the 19 events that get only a day-before row.

## Event counts

`experiments/results/epl_clv_rebaseline/event_counts.csv` counts FINAL EPL events in the evaluation range:

| Season | Events | Distinct matches | Scored | With a day-before row | Rows per event |
|---|---|---|---|---|---|
| 2024-25 | 380 | 380 | 380 | 380 | 1 |
| 2025-26 | 387 | 379 | 354 | 305 | 1 |
| 2026-27 | 43 | 43 | 43 | 0 | none |

Run 1 had the same 2024-25 counts and 304 rows in 2025-26, and no 2026-27 events (`run1_local_copy/event_counts.csv`).

- **Twins.** In 2025-26, 8 matches in April 2026 exist under two event IDs, which is why 387 events are 379 matches.
- **Lost events** (`experiments/results/epl_clv_rebaseline/lost_events.csv`), 82 of 387 in 2025-26 and all 43 in 2026-27:
  - 33 in 2025-26 have no final score: the 28 `op_live_` events of 3 to 22 March 2026, and 5 April events, four of them `op_live_` twins of scored Odds API events. Production has no scores for these either, so this is a data gap, not a copy problem. Ten of the March events (3 to 5 March) have FDUK day-before snapshots and are lost only for the missing score.
  - 28 in 2025-26 and all 43 in 2026-27 have no closing snapshot that quotes both bet365 and a sharp bookmaker. The 2025-26 ones kick off from 11 April 2026, and 7 of them are Odds API twins. **No 2026-27 snapshot quotes bet365 at all:** that season has only Betfair API snapshots and Odds API snapshots that quote Pinnacle, so no bet365 target can be computed for it.
  - 21 in 2025-26 have no snapshot within 1 h of the day-before point that quotes bet365 and a sharp bookmaker. All kick off from 12 April 2026 and are live-collected.
- **FDUK coverage** ends with the 2026-03-05 kick-offs, in production as well as locally. Later day-before rows can only come from live-collected snapshots.
- **Pinnacle gap.** In 71 FDUK events from 2026-01-17 to 2026-03-01, the day-before snapshot has no Pinnacle price. These are not lost: the hybrid reference uses Betfair Exchange instead ([sharp benchmark](../modelling/sharp-benchmark.md)).
- **Tuning-range recount.** The tuning range has 1,900 events, the last kicking off at 2024-05-19 15:00 UTC, as ADR-0010 requires. Ten events share that kick-off. With ties ordered by event ID (`odds_lambda.storage.readers.OddsReader.get_events_by_date_range`), the 1,900th is `op_2023-2024_SHEUTD_TOT_2024-05-19`, not the `op_2023-2024_ARS_EVE_2024-05-19` that the #452 run's query-plan order gave. Either way, the first validation event is the first evaluation-range event, `op_2024-2025_MANUTD_FUL_2024-08-16`.
- **Scored rows.** The 685 evaluation rows give 13 folds, 650 events from 2024-08-16 to 2026-02-22 (`summary.json` `evaluation_range`). The last 35 events form the dropped partial block: 11 FDUK events up to 2026-03-01 and all 24 live-collected rows, 2026-04-18 to 2026-05-19. So **no live-collected row enters the keep test.**

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
| Training-fold mean (floor) | 830.2 | -15.3 [-39.8, 9.7] |

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
| FDUK rows | 650 | 809.0 | 808.9 | 814.9 | 830.2 | 5.9 [-6.7, 18.5] | 6.0 [-6.7, 18.8] |
| Live-collected rows | 0 | | | | | | |
| Partial last block (not a fold) | 35 | 613.0 | 613.6 | 603.6 | 609.4 | -9.3 [-54.9, 40.0] | -9.9 [-63.2, 48.4] |
| Partial block, FDUK rows | 11 | 364.5 | 360.1 | 362.8 | 347.6 | -1.7 [-84.1, 68.2] | 2.7 [-79.5, 71.0] |
| Partial block, live-collected rows | 24 | 726.8 | 729.7 | 714.0 | 729.4 | -12.8 [-67.5, 50.0] | -15.7 [-84.4, 60.9] |

- **Pooled R² against the training-fold mean:** 0.0256 for seed A, 0.0257 for seed B and 0.0185 for the OLS baseline.
- **By season:** in 2024-25 the model is no better than the OLS baseline (R² 0.032 and 0.033, against 0.035 for OLS). In 2025-26 it is about 17 × 10⁻⁶ better, with CIs that just include zero. There the OLS baseline scores worse than the floor (R² −0.006).
- **Pinnacle to Betfair Exchange switch:** the 60 scored rows whose decision snapshot uses Betfair Exchange as the sharp price fall from 2026-01-17 to 2026-02-22. They have higher MSE than the Pinnacle rows, and the model's gain over OLS there is similar to the Pinnacle rows' gain, with a CI about three times wider. Sixty events cannot separate the regimes.
- **FDUK against live-collected rows:** every scored row is an FDUK row, so the scored comparison has no live-collected rows. The only live-collected rows are the 24 in the partial last block (`tail_predictions.csv`). That block's 35 events are scored once, with training on every earlier event. On the 24 live rows the model is about 13 to 16 worse than OLS and level with the floor, with CIs about 130 wide. Their MSE is about double that of the FDUK rows in the same block, but 24 events cannot show more.

## Caveats

- **Single target.** Only bet365 was tested. A retarget, for example to Pinnacle, needs its own run under the same rules (#495).
- **No live-collected rows are scored.** The 2025-26 rows in the folds are all FDUK day-before snapshots. The live-collected regime (`oddsportal_live` and Betfair API snapshots, from April 2026) appears only in the dropped partial block. Live scoring reads live-collected snapshots, so this test says little about the model's live behaviour (inferred; [prediction scoring](../modelling/prediction-scoring.md)).
- **Data gaps shape the range.** The March 2026 scores are missing in production, FDUK coverage ends on 2026-03-05, and 2026-27 has no bet365 price in any snapshot. Together they leave the scored range ending on 2026-02-22, so the evaluation range on or after 2024-08-01 is scored only up to there. More data will extend it only if later events get scores, day-before bet365 prices and bet365 closing prices.
- **The deployed model is not this model.** `epl-clv-home` uses tabular and match_stats features with sliding-380 folds ([CLV model findings](clv-model-findings.md#deployed-model)). This test covers the tabular group under the ADR-0010 protocol only.
- **Seed B lands close to seed A.** The two seeds' tuned hyperparameters differ, but their pooled MSEs differ by 0.13 × 10⁻⁶. Condition 1 is therefore easy to pass. What decides the result is the event-level noise in condition 2.

## Related

- [CV protocol on a common validation range](exp-cv-protocol-common-range.md), [CLV model findings](clv-model-findings.md), [Cross-validation](../modelling/cross-validation.md), [Hyperparameter tuner](../modelling/hyperparameter-tuner.md)
